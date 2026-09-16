"""Download daily OHLCV history for each ticker and cache it as CSV under data/."""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    HISTORY_PERIOD,
    INTERVAL,
    INTRADAY_LIVE_INTERVAL,
    INTRADAY_LIVE_PERIOD,
    INTRADAY_TRAIN_INTERVAL,
    INTRADAY_TRAIN_PERIOD,
    MACRO_TICKERS,
    TICKERS,
)


def _download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    # Yahoo occasionally returns a trailing row with Volume populated but
    # OHLC still NaN -- a not-yet-fully-settled bar for that symbol (seen for
    # individual equities minutes to hours after close, even when the index
    # itself already shows a settled value for the same day). Drop it here,
    # at the source, so no downstream consumer has to defend against it individually.
    df = df.dropna(subset=["Close"])
    return df


# A fresh fetch coming back with far fewer rows than what's already cached is
# a strong sign of a provider-side outage for that specific symbol (seen
# live: ^VIX3M returning exactly 1 row -- today's date only -- across every
# period and on repeated retries, while every other ticker fetched fine at
# the same moment). Silently overwriting good historical data with that would
# collapse every downstream feature depending on it. Keep the cached file and
# warn instead of trusting a fetch that shrank this much.
_MIN_ROW_RATIO = 0.5


def _save_with_outage_guard(df: pd.DataFrame, out_path: Path, label: str) -> pd.DataFrame:
    if out_path.exists():
        cached = pd.read_csv(out_path, index_col=0, parse_dates=True)
        if len(cached) > 20 and len(df) < len(cached) * _MIN_ROW_RATIO:
            print(f"WARNING: {label} fetch returned {len(df)} rows vs. {len(cached)} cached -- "
                  f"looks like a provider-side outage for this symbol, not real history loss. "
                  f"Keeping the cached data instead of overwriting it.")
            return cached
    df.to_csv(out_path)
    return df


def fetch_one(symbol: str, name: str) -> pd.DataFrame:
    df = _download(symbol, HISTORY_PERIOD, INTERVAL)
    out_path = DATA_DIR / f"{name}.csv"
    df = _save_with_outage_guard(df, out_path, f"{symbol} ({name})")
    print(f"{symbol:>7} -> {name:<6}: {len(df):>5} rows  [{df.index.min().date()} .. {df.index.max().date()}]  saved to {out_path.relative_to(DATA_DIR.parent)}")
    return df


def fetch_all() -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for symbol, name in TICKERS.items():
        out[name] = fetch_one(symbol, name)
    return out


def fetch_intraday_train_one(symbol: str, name: str) -> pd.DataFrame:
    df = _download(symbol, INTRADAY_TRAIN_PERIOD, INTRADAY_TRAIN_INTERVAL)
    out_path = DATA_DIR / f"{name}_intraday_{INTRADAY_TRAIN_INTERVAL}.csv"
    df.to_csv(out_path)
    print(f"{symbol:>7} -> {name:<6}: {len(df):>5} {INTRADAY_TRAIN_INTERVAL} bars  "
          f"[{df.index.min()} .. {df.index.max()}]  saved to {out_path.relative_to(DATA_DIR.parent)}")
    return df


def fetch_intraday_train_all() -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for symbol, name in TICKERS.items():
        out[name] = fetch_intraday_train_one(symbol, name)
    return out


def fetch_macro_one(symbol: str, name: str) -> pd.DataFrame:
    df = _download(symbol, HISTORY_PERIOD, INTERVAL)
    out_path = DATA_DIR / f"macro_{name}.csv"
    df = _save_with_outage_guard(df, out_path, f"{symbol} (macro_{name})")
    print(f"{symbol:>7} -> macro_{name:<6}: {len(df):>5} rows  [{df.index.min().date()} .. {df.index.max().date()}]  saved to {out_path.relative_to(DATA_DIR.parent)}")
    return df


def fetch_macro_all() -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for symbol, name in MACRO_TICKERS.items():
        out[name] = fetch_macro_one(symbol, name)
    return out


def fetch_intraday_live(symbol: str) -> pd.DataFrame:
    """Fresh (uncached) recent intraday bars for live, same-day prediction."""
    return _download(symbol, INTRADAY_LIVE_PERIOD, INTRADAY_LIVE_INTERVAL)


if __name__ == "__main__":
    fetch_all()
