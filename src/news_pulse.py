"""Live news-sentiment readout, shown alongside predictions as context --
NOT fed into the trained models.

Why not train on this: yfinance's news feed (and every other free source we
found) only returns *current* headlines, with no queryable historical archive
of what a ticker's news looked like on any past date. Without point-in-time
history there's nothing to backtest against, so any "trained on sentiment"
claim built on this data would be fake -- it would really be reading today's
news and pretending that's how it always worked. A real historical news
feature would need a paid archive (e.g. NewsAPI's historical tier, Refinitiv,
RavenPack, Bloomberg). This module is intentionally scoped to what free data
actually supports: an honest, live "what does today's coverage look like"
snapshot using VADER (a lexicon-based sentiment scorer, runs offline, free).
"""
import sys
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TICKERS

_analyzer = SentimentIntensityAnalyzer()

# A few broad market/macro queries in addition to each ticker's own news,
# so "what's going on in the world" isn't limited to company-specific headlines.
MARKET_WIDE_SYMBOLS = ["^GSPC", "^VIX"]


def _headlines_for(symbol: str, max_items: int = 10) -> list[str]:
    import yfinance as yf
    try:
        news = yf.Ticker(symbol).news or []
    except Exception:
        return []
    out = []
    for item in news[:max_items]:
        content = item.get("content", item)
        title = content.get("title") or ""
        summary = content.get("summary") or ""
        text = (title + ". " + summary).strip()
        if text:
            out.append(text)
    return out


def sentiment_for_symbol(symbol: str) -> dict:
    headlines = _headlines_for(symbol)
    if not headlines:
        return {"symbol": symbol, "n_headlines": 0, "avg_compound": None, "label": "no recent news"}
    scores = [_analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg = sum(scores) / len(scores)
    if avg > 0.15:
        label = "positive"
    elif avg < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return {"symbol": symbol, "n_headlines": len(headlines), "avg_compound": avg, "label": label}


def market_pulse() -> pd.DataFrame:
    """One row per ticker (+ broad market symbols) with a live news-sentiment snapshot."""
    symbols = list(dict.fromkeys(list(TICKERS.keys()) + MARKET_WIDE_SYMBOLS))
    rows = [sentiment_for_symbol(sym) for sym in symbols]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = market_pulse()
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("Live news sentiment (context only -- not used by the trained models):")
    print(df.to_string(index=False))
