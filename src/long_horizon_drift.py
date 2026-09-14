"""Long-horizon "prediction": how often is UP over N trading days, in-sample
vs. out-of-sample?

READ THIS BEFORE TREATING ANYTHING HERE AS A FORECASTING RESULT: this answers
a fundamentally different and much weaker question than the rest of this
project. It is NOT "can the model read a signal about what's about to
happen" -- it's "if you always bet UP and hold for N trading days, how often
would that have worked historically?" A high hit rate here reflects the
equity market's well-known long-run upward drift (the equity risk premium),
not anything discovered about future information. It is functionally a
buy-and-hold backtest, reframed as an accuracy number. Predicting TOMORROW's
direction (the rest of this project) remains close to a coin flip -- this
script does not change that, and is not trying to.

Also note: windows here overlap (day 2's 252-day-forward window shares 251
days with day 1's), so a large window count is NOT a large *independent*
sample count. `effective_independent_n` = test_days / horizon is the more
honest sample size -- for anything beyond a few weeks, it's small.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, REPORTS_DIR, TEST_FRACTION, TICKERS

HORIZONS_TRADING_DAYS = {
    "1 week": 5,
    "2 weeks": 10,
    "1 month": 21,
    "2 months": 42,
    "3 months": 63,
    "6 months": 126,
    "9 months": 189,
    "1 year": 252,
    "18 months": 378,
    "2 years": 504,
}

MIN_TEST_WINDOWS = 20  # below this, don't trust the out-of-sample number at all


def hit_rate(close: pd.Series, horizon: int) -> tuple[float, int]:
    fwd_return = (close.shift(-horizon) / close - 1).dropna()
    n = len(fwd_return)
    if n == 0:
        return np.nan, 0
    return float((fwd_return > 0).mean()), n


def analyze_ticker(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
    close = df["Close"]
    n = len(close)
    split = int(n * (1 - TEST_FRACTION))
    train_close, test_close = close.iloc[:split], close.iloc[split:]

    rows = []
    for label, h in HORIZONS_TRADING_DAYS.items():
        train_hr, train_n = hit_rate(train_close, h)
        test_hr, test_n = hit_rate(test_close, h)
        rows.append({
            "ticker": name, "horizon": label, "trading_days": h,
            "train_hit_rate": train_hr, "train_n": train_n,
            "test_hit_rate": test_hr, "test_n_windows": test_n,
            "effective_independent_n": max(test_n // h, 0) if test_n else 0,
            "reliable": test_n >= MIN_TEST_WINDOWS,
        })
    return pd.DataFrame(rows)


def main():
    print("Long-horizon drift analysis -- NOT day-to-day forecasting. See module docstring.\n")

    all_rows = []
    for name in TICKERS.values():
        all_rows.append(analyze_ticker(name))
    result = pd.concat(all_rows, ignore_index=True)

    pd.set_option("display.width", 200)
    printable = result.copy()
    printable["train_hit_rate"] = printable["train_hit_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a")
    printable["test_hit_rate"] = printable["test_hit_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a")
    print(printable.to_string(index=False))

    print("\nShortest horizon crossing 80% out-of-sample, per ticker (only counting reliable windows, n>=20):")
    for name, g in result.groupby("ticker"):
        qualifying = g[(g["test_hit_rate"] >= 0.80) & g["reliable"]]
        if qualifying.empty:
            print(f"  {name}: no horizon reliably reached 80% out-of-sample")
        else:
            best = qualifying.iloc[0]
            print(f"  {name}: {best['horizon']} ({best['trading_days']}d) -> "
                  f"{best['test_hit_rate']:.1%} out-of-sample (train: {best['train_hit_rate']:.1%}), "
                  f"{int(best['test_n_windows'])} overlapping windows "
                  f"(~{int(best['effective_independent_n'])} independent)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "long_horizon_drift.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")


if __name__ == "__main__":
    main()
