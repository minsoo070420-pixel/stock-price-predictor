"""Live single-stock options-implied-volatility snapshot, shown alongside
predictions as context -- NOT fed into the trained models.

Why not train on this: Yahoo Finance (and every other free source we found)
only exposes CURRENT options chains for individual equities, with no
queryable historical archive of what AAPL's or PLTR's implied vol looked like
on any past date. Without point-in-time history there's nothing to backtest
against -- the same constraint as news_pulse.py, for the same reason. A real
historical single-stock IV feature would need a paid options-data vendor
(OptionMetrics, CBOE DataShop, ORATS).

Index-level implied vol is a different story: VIX (S&P 500), VIX3M (its
3-month sibling), and VXN (Nasdaq-100) all have full historical daily
archives, ARE genuinely forward-looking, and ARE trained into the models --
see macro_features.py. This module only covers what free data can't give a
usable history for: AAPL's and PLTR's own options.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TICKERS

TARGET_DAYS_OUT = 30
# Only equities have single-stock options; the index has none tradable this way
# (and already has a real, backtestable proxy in VIX -- see macro_features.py).
EQUITY_SYMBOLS = [sym for sym in TICKERS if not sym.startswith("^")]


def _nearest_expiration(expirations: list[str], target_days: int = TARGET_DAYS_OUT) -> str | None:
    if not expirations:
        return None
    today = pd.Timestamp.now().normalize()
    diffs = [(abs((pd.Timestamp(e) - today).days - target_days), e) for e in expirations]
    return min(diffs, key=lambda x: x[0])[1]


def implied_vol_snapshot(symbol: str) -> dict:
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        spot = float(t.fast_info["last_price"])
        exp = _nearest_expiration(list(t.options))
        if exp is None:
            raise ValueError("no listed expirations")
        chain = t.option_chain(exp)
        calls, puts = chain.calls, chain.puts
    except Exception as e:
        return {
            "symbol": symbol, "status": f"unavailable ({type(e).__name__})",
            "expiration": None, "atm_iv": None, "iv_skew": None,
        }

    # Sanity-filter quotes: illiquid deep ITM/OTM strikes can produce nonsense
    # (e.g. >300% "implied vol") from a stale or one-sided bid/ask.
    calls_valid = calls[(calls["impliedVolatility"] > 0.01) & (calls["impliedVolatility"] < 5)]
    puts_valid = puts[(puts["impliedVolatility"] > 0.01) & (puts["impliedVolatility"] < 5)]
    if calls_valid.empty or puts_valid.empty:
        return {
            "symbol": symbol, "status": "no valid quotes",
            "expiration": exp, "atm_iv": None, "iv_skew": None,
        }

    atm_call = calls_valid.iloc[(calls_valid["strike"] - spot).abs().argsort()[:1]]
    atm_put = puts_valid.iloc[(puts_valid["strike"] - spot).abs().argsort()[:1]]
    atm_iv = float((atm_call["impliedVolatility"].iloc[0] + atm_put["impliedVolatility"].iloc[0]) / 2)

    # Skew: OTM put IV (~10% below spot) minus OTM call IV (~10% above spot).
    # Elevated skew = more demand for downside protection = a fear/hedging signal.
    otm_put = puts_valid.iloc[(puts_valid["strike"] - spot * 0.90).abs().argsort()[:1]]
    otm_call = calls_valid.iloc[(calls_valid["strike"] - spot * 1.10).abs().argsort()[:1]]
    iv_skew = float(otm_put["impliedVolatility"].iloc[0] - otm_call["impliedVolatility"].iloc[0])

    return {
        "symbol": symbol, "status": "ok", "expiration": exp,
        "atm_iv": atm_iv, "iv_skew": iv_skew,
    }


def implied_vol_snapshots() -> pd.DataFrame:
    rows = [implied_vol_snapshot(sym) for sym in EQUITY_SYMBOLS]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = implied_vol_snapshots()
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("Live single-stock implied volatility (context only -- not used by the trained models):")
    print(df.to_string(index=False))
