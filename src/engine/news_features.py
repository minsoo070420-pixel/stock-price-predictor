"""Point-in-time historical news-sentiment features, built from the NYT
Archive API and trained into the models (unlike news_pulse.py, which is
explicitly live-only context and never trained on -- see its docstring for
why). The Archive API returns every article published in a given
year/month, including how NYT tagged it at the time, which is what makes it
usable for training: at row t we only ever use articles published on or
before day t, so there is no look-ahead.

Two signals are built per trading day:
  - "market" sentiment: business/tech-desk coverage generally, applies to
    every ticker (SP500 included) as a shared risk-sentiment proxy.
  - "company" sentiment: coverage that specifically names that ticker's
    company, via NYT's own `organizations` keyword tags (high precision --
    avoids false hits like "apple" in a Food section article, or "amazon"
    in a World section article about the rainforest) with a text-fallback
    restricted to the same business/tech context.

Both are daily-aggregated with FinBERT (ProsusAI/finbert -- a BERT model
fine-tuned specifically on financial text, not a general-purpose lexicon) on
headline + abstract, then exposed in the same shape as macro_features.py --
one shared builder, plus align_macro_to_ticker (reused as-is) to join onto
each ticker's own trading calendar.

Why FinBERT over VADER (the original version of this module, and what
news_pulse.py's live-only readout still uses): VADER scores words it
recognizes from a general-purpose lexicon, so it has no notion that "cuts
guidance" is bad news or "beats estimates" is good news -- it just reacts to
words like "cut" and "beat" in isolation. FinBERT was fine-tuned on financial
text specifically and reliably gets exactly that kind of headline right (see
git history / commit message for a worked example). It's also far slower --
a BERT forward pass per article instead of a lexicon lookup -- which is why
this module now caches SCORED months (see SCORED_CACHE_DIR below), not just
raw fetched articles: rescoring ~50k historical articles on every pipeline
run would make every script take minutes just to start.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, ROOT

load_dotenv(ROOT / ".env")

NEWS_CACHE_DIR = DATA_DIR / "news_cache"
NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCORED_CACHE_DIR = DATA_DIR / "news_scored_cache"
SCORED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
REQUEST_SLEEP_SECONDS = 6  # conservative pacing between live NYT calls, cached afterward

RELEVANT_SECTIONS = {"Business Day", "Business", "Technology", "DealBook", "Your Money", "Media"}
RELEVANT_DESKS = {"Business", "Financial", "SundayBusiness", "Technology"}

# NYT's `organizations` keyword tag observed for each company (via sample checks
# against real archive months), lowercased substring match -- tolerant of the
# capitalization drift NYT itself isn't consistent about (e.g. "Nvidia Corp" vs
# "NVIDIA Corporation"). Text fallback is restricted to business/tech-context
# articles only (see RELEVANT_SECTIONS/RELEVANT_DESKS) to avoid generic-word
# collisions like "apple" (fruit) or "amazon" (rainforest).
COMPANY_KEYWORDS = {
    "AAPL": ["apple inc", "apple"],
    "PLTR": ["palantir"],
    "MSFT": ["microsoft"],
    "GOOGL": ["google", "alphabet inc"],
    "AMZN": ["amazon.com", "amazon inc"],
    "NVDA": ["nvidia"],
    "META": ["facebook inc", "meta platforms"],
}

_finbert_tokenizer = None
_finbert_model = None
_finbert_device = None
_finbert_pos_idx = None
_finbert_neg_idx = None


def _load_finbert():
    """Lazy singleton -- only loaded the first time a score is actually
    needed (many pipeline runs hit the scored-month cache and never need
    this at all). Uses the Apple Silicon MPS backend when available, which
    is meaningfully faster than CPU for batched BERT inference."""
    global _finbert_tokenizer, _finbert_model, _finbert_device, _finbert_pos_idx, _finbert_neg_idx
    if _finbert_model is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _finbert_device = "mps" if torch.backends.mps.is_available() else "cpu"
    _finbert_model.to(_finbert_device)
    _finbert_model.eval()
    label2id = {v: k for k, v in _finbert_model.config.id2label.items()}
    _finbert_pos_idx, _finbert_neg_idx = label2id["positive"], label2id["negative"]


def _finbert_sentiment_batch(texts: list[str], batch_size: int = 64) -> list[float]:
    """P(positive) - P(negative) per text, in [-1, 1] -- same range/shape as
    VADER's old compound score, so nothing downstream (daily_aggregate etc.)
    needs to know the scorer changed."""
    if not texts:
        return []
    _load_finbert()
    import torch

    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = _finbert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                     max_length=128).to(_finbert_device)
        with torch.no_grad():
            probs = torch.softmax(_finbert_model(**inputs).logits, dim=-1)
        scores.extend((probs[:, _finbert_pos_idx] - probs[:, _finbert_neg_idx]).cpu().tolist())
    return scores


def _api_key() -> str:
    key = os.environ.get("NYT_API_KEY")
    if not key:
        raise RuntimeError(
            "NYT_API_KEY not set. Put it in a .env file at the project root "
            "(NYT_API_KEY=...) or export it as an environment variable."
        )
    return key


def _slim_doc(doc: dict) -> dict:
    # NYT sometimes returns "keywords": null instead of [] for newer articles --
    # without the `or []`, that silently crashes the whole month's fetch.
    orgs = [k["value"] for k in (doc.get("keywords") or []) if k.get("name") == "organizations"]
    return {
        "pub_date": doc.get("pub_date"),
        "section_name": doc.get("section_name") or "",
        "news_desk": doc.get("news_desk") or "",
        "headline": (doc.get("headline") or {}).get("main") or "",
        "abstract": doc.get("abstract") or "",
        "organizations": "; ".join(orgs),
    }


def _cache_path(year: int, month: int) -> Path:
    return NEWS_CACHE_DIR / f"{year}_{month:02d}.csv"


def fetch_month(year: int, month: int, force: bool = False) -> pd.DataFrame:
    """One row per NYT article published that month, slimmed to the fields
    we use. Cached to disk (gitignored) so repeated runs don't re-hit the API."""
    path = _cache_path(year, month)
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["pub_date"])

    url = ARCHIVE_URL.format(year=year, month=month)
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params={"api-key": _api_key()}, timeout=30)
            if resp.status_code == 429:
                time.sleep(15)
                continue
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
            rows = [_slim_doc(d) for d in docs]
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
            time.sleep(REQUEST_SLEEP_SECONDS)
            return df
        except Exception as e:
            last_err = e
            time.sleep(5)
    print(f"  WARNING: failed to fetch NYT archive {year}-{month:02d} after retries: {last_err}")
    return pd.DataFrame(columns=["pub_date", "section_name", "news_desk", "headline", "abstract", "organizations"])


def _is_business_context(row: pd.Series) -> bool:
    return row["section_name"] in RELEVANT_SECTIONS or row["news_desk"] in RELEVANT_DESKS


def _mentions_company(row: pd.Series, keywords: list[str]) -> bool:
    orgs = str(row["organizations"]).lower()
    if any(kw in orgs for kw in keywords):
        return True
    if not row["_business_ctx"]:
        return False
    text = f"{row['headline']} {row['abstract']}".lower()
    return any(kw in text for kw in keywords)


def build_month_sentiment(month_df: pd.DataFrame) -> pd.DataFrame:
    """Adds business-context flag, FinBERT sentiment, and a per-company
    mention flag for every configured ticker to a raw month's articles.

    Only articles that actually end up used downstream -- business-context
    ones, plus anything that mentions a tracked company even outside that
    context -- get scored; a typical NYT month has ~4,000+ articles total but
    only ~300-450 are ever aggregated, and FinBERT is expensive enough
    (a BERT forward pass, not a lexicon lookup) that skipping the other ~90%
    matters."""
    if month_df.empty:
        return month_df
    df = month_df.copy()
    df["_business_ctx"] = df.apply(_is_business_context, axis=1)
    for ticker, keywords in COMPANY_KEYWORDS.items():
        df[f"_mentions_{ticker}"] = df.apply(lambda r: _mentions_company(r, keywords), axis=1)

    mention_cols = [f"_mentions_{t}" for t in COMPANY_KEYWORDS]
    relevant_mask = df["_business_ctx"] | df[mention_cols].any(axis=1)

    df["sentiment"] = float("nan")
    texts, text_idx = [], []
    for i in df.index[relevant_mask]:
        text = f"{df.at[i, 'headline']}. {df.at[i, 'abstract']}".strip()
        if text and text != ".":
            texts.append(text)
            text_idx.append(i)
    if texts:
        df.loc[text_idx, "sentiment"] = _finbert_sentiment_batch(texts)
    return df


def _scored_cache_path(year: int, month: int) -> Path:
    return SCORED_CACHE_DIR / f"{year}_{month:02d}.csv"


def score_month(year: int, month: int, force: bool = False) -> pd.DataFrame:
    """Fetches (via fetch_month's own cache) and scores one month, caching
    the SCORED result separately -- rescoring is the expensive step now, not
    refetching, so it gets its own cache keyed the same way."""
    path = _scored_cache_path(year, month)
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["pub_date"])
    raw = fetch_month(year, month, force=force)
    scored = build_month_sentiment(raw)
    if not scored.empty:
        scored.to_csv(path, index=False)
    return scored


def daily_aggregate(scored_df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date with market-wide and per-company daily
    sentiment means + article counts."""
    if scored_df.empty:
        return pd.DataFrame()
    df = scored_df.copy()
    # format="ISO8601" handles both the raw NYT string (millisecond-precision,
    # "Z"-suffixed) and the plain-datetime string pandas writes back out when a
    # month is cached and reloaded -- letting format-inference pick one from
    # whichever happens to be first in the batch silently breaks on the other.
    df["date"] = pd.to_datetime(df["pub_date"], format="ISO8601").dt.tz_localize(None).dt.normalize()

    market = df[df["_business_ctx"]]
    out = market.groupby("date")["sentiment"].agg(["mean", "count"])
    out.columns = ["news_market_sentiment_mean", "news_market_count"]

    for ticker in COMPANY_KEYWORDS:
        sub = df[df[f"_mentions_{ticker}"]]
        agg = sub.groupby("date")["sentiment"].agg(["mean", "count"])
        agg.columns = [f"news_{ticker}_sentiment_mean", f"news_{ticker}_count"]
        out = out.join(agg, how="outer")

    return out.sort_index()


def build_scored_articles(start_date: str, end_date: str) -> pd.DataFrame:
    """Loads (or fetches + scores, then caches) every month in
    [start_date, end_date] -- one row per article, not yet aggregated to
    daily. Exposed separately from build_news_history() so leakage_check.py
    can truncate at the article level by pub_date, the same way it truncates
    raw macro series. The current calendar month is always rescored (never
    served from either cache) since it's still accumulating articles."""
    months = pd.period_range(start=start_date, end=end_date, freq="M")
    current_period = pd.Timestamp.today().to_period("M")
    frames = []
    for i, p in enumerate(months, 1):
        print(f"  news archive {p.year}-{p.month:02d} ({i}/{len(months)})...")
        scored = score_month(p.year, p.month, force=(p == current_period))
        if not scored.empty:
            frames.append(scored)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_news_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches (or loads from cache) every month in [start_date, end_date],
    scores it, and returns one daily-aggregated table spanning the range."""
    scored = build_scored_articles(start_date, end_date)
    if scored.empty:
        return pd.DataFrame()
    return daily_aggregate(scored)


NEWS_FEATURE_COLUMNS = [
    "news_company_sentiment_mean", "news_company_count", "news_company_had_news",
    "news_market_sentiment_mean", "news_market_count",
]


def news_features_for_ticker(daily_df: pd.DataFrame, friendly_name: str) -> pd.DataFrame:
    """Selects/renames the generic news_company_* / news_market_* columns
    (see features.py) for one ticker out of the shared daily_df built by
    build_news_history(). SP500 has no company of its own, so its "company"
    signal is just the shared market signal."""
    if daily_df.empty:
        return pd.DataFrame(columns=NEWS_FEATURE_COLUMNS)

    out = pd.DataFrame(index=daily_df.index)
    out["news_market_sentiment_mean"] = daily_df.get("news_market_sentiment_mean")
    out["news_market_count"] = daily_df.get("news_market_count").fillna(0) if "news_market_count" in daily_df else 0

    if friendly_name in COMPANY_KEYWORDS and f"news_{friendly_name}_sentiment_mean" in daily_df:
        out["news_company_sentiment_mean"] = daily_df[f"news_{friendly_name}_sentiment_mean"]
        out["news_company_count"] = daily_df[f"news_{friendly_name}_count"].fillna(0)
    else:
        out["news_company_sentiment_mean"] = out["news_market_sentiment_mean"]
        out["news_company_count"] = out["news_market_count"]

    out["news_company_had_news"] = (out["news_company_count"] > 0).astype(int)
    return out[NEWS_FEATURE_COLUMNS]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build (or refresh the cache of) the NYT news-history table.")
    parser.add_argument("--start", default="2016-09", help="YYYY-MM")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m"), help="YYYY-MM")
    args = parser.parse_args()

    daily = build_news_history(args.start, args.end)
    out_path = DATA_DIR / "news_daily_aggregate.csv"
    daily.to_csv(out_path)
    print(f"\nSaved {len(daily)} days of news features to {out_path}")
    print(daily.tail(10))
