"""Part 2: Descriptive comparison across SP500 / AAPL / PLTR.

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
from backtest_dates import backtest_ticker
from config import CLASSIFICATION_THRESHOLD, DATA_DIR, MODELS_DIR, REPORTS_DIR, TEST_FRACTION, TICKERS
from features import FEATURE_COLUMNS, make_dataset
from fetch_data import fetch_all, fetch_macro_all
from macro_features import build_macro_features
from predict_long_horizon import LONG_HORIZON_CALLS
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

RECENT_BACKTEST_DAYS = 30

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


def long_horizon_stats(name: str) -> dict:
    call = LONG_HORIZON_CALLS.get(name)
    if call is None:
        return {"long_horizon_call": "none reliable (see long_horizon_drift.py)", "long_horizon_oos_hit_rate": None}
    return {
        "long_horizon_call": f"UP over {call['horizon_label']}",
        "long_horizon_oos_hit_rate": call["oos_hit_rate"],
    }


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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "ticker_comparison.csv"
    result.to_csv(out_path)
    print(f"\nSaved to {out_path}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
