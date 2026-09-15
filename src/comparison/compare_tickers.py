"""Part 2: Descriptive comparison across all tickers configured in
src/engine/config.py's TICKERS (SP500, AAPL, PLTR, MSFT, GOOGL, AMZN, NVDA, META).

THIS IS NOT INVESTMENT ADVICE AND DOES NOT RECOMMEND ANY TRADE. It reports
historical, backtested statistics about how each ticker's model has performed
-- nothing here says what to buy, hold, or sell. Past accuracy is not a
promise of future accuracy; see the main README's "The 80-90% question" and
"How this compares to published research" sections for why these numbers
should not be treated as a trading edge regardless of which ticker looks
"best" below.

This is Part 2 of the two-piece project split:
  - Part 1 (src/engine/): the prediction/backtest/research engine.
  - Part 2 (this file): a neutral, side-by-side statistics view across
    tickers, built from Part 1's own outputs -- it does not train anything
    new or add any opinion about which ticker is a better investment.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
import yfinance as yf
from backtest_dates import backtest_ticker
from config import CLASSIFICATION_THRESHOLD, DATA_DIR, MODELS_DIR, REPORTS_DIR, TEST_FRACTION, TICKERS
from features import FEATURE_COLUMNS, make_dataset
from fetch_data import fetch_all, fetch_macro_all
from long_horizon_drift import analyze_ticker, most_recent_completed_window
from macro_features import build_macro_features
from predict_long_horizon import LONG_HORIZON_CALLS
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

RECENT_BACKTEST_DAYS = 30
FOUR_HORIZONS = ["1 month", "3 months", "6 months", "1 year"]

DISCLAIMER = (
    "=" * 78 + "\n"
    "NOT INVESTMENT ADVICE. Descriptive historical statistics only -- this\n"
    "does not recommend any trade, and no ticker being 'better' here means\n"
    "it will perform better going forward. See README: 'The 80-90% question'\n"
    "and 'How this compares to published research'.\n"
    + "=" * 78
)


def held_out_test_stats(name: str, df: pd.DataFrame, macro_df: pd.DataFrame) -> dict:
    """Re-evaluates the ticker's actual saved production models on the same
    held-out test split train.py uses -- the most reliable long-run number
    available, since it's hundreds of days rather than a handful."""
    with open(MODELS_DIR / f"{name}_features.json") as f:
        manifest = json.load(f)
    reg = joblib.load(MODELS_DIR / f"{name}_regressor.joblib")
    clf = joblib.load(MODELS_DIR / f"{name}_classifier.joblib")

    X, y_ret, y_dir, _ = make_dataset(df, macro_df=macro_df, feature_columns=FEATURE_COLUMNS)
    split = int(len(X) * (1 - TEST_FRACTION))
    X_test, y_ret_test, y_dir_test = X.iloc[split:], y_ret.iloc[split:], y_dir.iloc[split:]

    pred_ret = reg.predict(X_test[manifest["regressor"]])
    rmse = float(np.sqrt(mean_squared_error(y_ret_test, pred_ret)))

    proba = clf.predict_proba(X_test[manifest["classifier"]])[:, 1]
    pred_dir = (proba > CLASSIFICATION_THRESHOLD).astype(int)
    bal_acc = balanced_accuracy_score(y_dir_test, pred_dir)

    return {
        "held_out_test_rmse": rmse,
        "held_out_test_balanced_accuracy": bal_acc,
        "held_out_test_days": len(X_test),
        "regressor_type": type(reg).__name__,
        "classifier_type": type(clf).__name__,
    }


def recent_backtest_stats(name: str, macro_df: pd.DataFrame) -> dict:
    df = backtest_ticker(name, None, RECENT_BACKTEST_DAYS, macro_df)
    scored = df.dropna(subset=["direction_correct"])
    if scored.empty:
        return {"recent_accuracy": None, "recent_net_bps_per_trade": None, "recent_cumulative_pct": None}
    return {
        "recent_accuracy": float(scored["direction_correct"].mean()),
        "recent_net_bps_per_trade": float(scored["net_return_bps"].mean()),
        "recent_cumulative_pct": float(scored["net_return_bps"].sum() / 100),
    }


def fundamental_stats(symbol: str) -> dict:
    """Raw, unopinionated fundamental data -- no verdict attached, nothing
    labeled 'cheap' or 'expensive'. ^GSPC is an index, not a company, so most
    of these fields are structurally unavailable for it (not a data gap to
    fill in, just not a thing an index has)."""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        info = {}
    market_cap = info.get("marketCap")
    return {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "market_cap_billions": (market_cap / 1e9) if market_cap else None,
        "price_to_book": info.get("priceToBook"),
        "dividend_yield_pct": (info.get("dividendYield") or None),
        "beta": info.get("beta"),
        "profit_margin_pct": (info.get("profitMargins") * 100) if info.get("profitMargins") is not None else None,
        "revenue_growth_pct": (info.get("revenueGrowth") * 100) if info.get("revenueGrowth") is not None else None,
        "sector": info.get("sector") or ("Index, not a company" if symbol.startswith("^") else None),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
    }


def long_horizon_stats(name: str) -> dict:
    call = LONG_HORIZON_CALLS.get(name)
    if call is None:
        return {"long_horizon_call": "none reliable (see long_horizon_drift.py)", "long_horizon_oos_hit_rate": None}
    return {
        "long_horizon_call": f"UP over {call['horizon_label']}",
        "long_horizon_oos_hit_rate": call["oos_hit_rate"],
    }


def four_horizon_table() -> pd.DataFrame:
    """Out-of-sample UP hit rate at exactly 1/3/6/12 months, per ticker --
    NOT a ranking, NOT a recommendation. Reuses long_horizon_drift.py's own
    analysis (same walk-forward split, same overlapping-window caveats)
    rather than recomputing anything new."""
    all_rows = []
    for name in TICKERS.values():
        g = analyze_ticker(name)
        g = g[g["horizon"].isin(FOUR_HORIZONS)]
        all_rows.append(g)
    combined = pd.concat(all_rows, ignore_index=True)
    pivot = combined.pivot(index="ticker", columns="horizon", values="test_hit_rate")
    pivot = pivot[[h for h in FOUR_HORIZONS if h in pivot.columns]]  # keep requested order
    reliable = combined.pivot(index="ticker", columns="horizon", values="reliable")
    reliable = reliable[[h for h in FOUR_HORIZONS if h in reliable.columns]]
    indep_n = combined.pivot(index="ticker", columns="horizon", values="effective_independent_n")
    indep_n = indep_n[[h for h in FOUR_HORIZONS if h in indep_n.columns]]
    return pivot, reliable, indep_n


def main():
    print(DISCLAIMER)
    print("\nFetching data...")
    data = fetch_all()
    macro_df = build_macro_features(fetch_macro_all())

    rows = []
    for name in TICKERS.values():
        df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
        row = {"ticker": name}
        row.update(held_out_test_stats(name, df, macro_df))
        row.update(recent_backtest_stats(name, macro_df))
        row.update(long_horizon_stats(name))
        rows.append(row)

    result = pd.DataFrame(rows).set_index("ticker")
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n" + result.to_string())

    print(f"\n{'-'*78}\nReading this table:")
    print("  held_out_test_*        -- most reliable number (hundreds of days), what's actually deployed")
    print(f"  recent_*                -- last {RECENT_BACKTEST_DAYS} trading days (small sample, noisy -- see README sampling-noise math)")
    print("  long_horizon_*          -- a DIFFERENT, much weaker claim (buy-and-hold drift, not day-to-day prediction)")

    print(f"\n{'-'*78}")
    print("Out-of-sample UP hit rate at 1/3/6/12 months, per ticker.")
    print("This answers 'how often has always-betting-UP-and-holding worked historically,'")
    print("NOT 'which stock will go up.' No ticker is ranked or recommended here.\n")
    hit_rates, reliable, indep_n = four_horizon_table()
    printable = hit_rates.copy()
    for col in printable.columns:
        printable[col] = [
            f"{v:.1%}" + ("" if rel else " (unreliable, <20 windows)") + f"  [~{int(n)} indep.]"
            if pd.notna(v) else "n/a"
            for v, rel, n in zip(hit_rates[col], reliable[col], indep_n[col])
        ]
    print(printable.to_string())
    print("\nReminder: overlapping windows inflate apparent sample size -- '~N indep.' is the honest count.")

    print(f"\n{'-'*78}")
    print("Verification: 'run the algorithm on data from 3 months ago' -- the single most")
    print("recent completed 3-month window (price 63 trading days ago vs. today), per ticker.")
    print("One instance each, not a statistic -- an anecdote, not a validated rate.\n")
    recent3mo = [most_recent_completed_window(name, "3 months", 63) for name in TICKERS.values()]
    recent3mo_df = pd.DataFrame(recent3mo).set_index("ticker")
    print(recent3mo_df.to_string())
    n_correct_3mo = sum(1 for r in recent3mo if r.get("correct"))
    print(f"\n{n_correct_3mo}/{len(recent3mo)} tickers were actually UP over their most recent 3-month window.")

    print(f"\n{'-'*78}")
    print("Raw fundamental data -- purely descriptive, no verdict attached, nothing")
    print("here is labeled 'cheap' or 'expensive'. SP500 is an index, not a company,")
    print("so most fields are structurally n/a for it, not a missing data point.\n")
    fund_rows = []
    for symbol, name in TICKERS.items():
        row = {"ticker": name}
        row.update(fundamental_stats(symbol))
        fund_rows.append(row)
    fundamentals = pd.DataFrame(fund_rows).set_index("ticker")
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(fundamentals.to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "ticker_comparison.csv"
    result.to_csv(out_path)
    horizon_out_path = REPORTS_DIR / "four_horizon_comparison.csv"
    hit_rates.to_csv(horizon_out_path)
    recent3mo_out_path = REPORTS_DIR / "recent_3month_check.csv"
    recent3mo_df.to_csv(recent3mo_out_path)
    fund_out_path = REPORTS_DIR / "fundamentals_comparison.csv"
    fundamentals.to_csv(fund_out_path)
    print(f"\nSaved to {out_path}, {horizon_out_path}, {recent3mo_out_path}, and {fund_out_path}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
