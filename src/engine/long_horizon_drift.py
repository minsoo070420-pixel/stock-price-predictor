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


def individual_window_instances(name: str, horizon_label: str, horizon: int) -> pd.DataFrame:
    """Concrete, non-overlapping historical instances of 'predict UP, hold for
    `horizon` trading days' in the held-out test period -- actual dates and
    prices, not just the aggregate hit-rate percentage. Non-overlapping (unlike
    the rolling hit_rate() above) so each row is a genuinely independent
    instance, not 251 copies of the same window shifted by a day."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
    close = df["Close"]
    n = len(close)
    split = int(n * (1 - TEST_FRACTION))
    test_close = close.iloc[split:]

    rows = []
    i = 0
    while i + horizon < len(test_close):
        start_date, end_date = test_close.index[i], test_close.index[i + horizon]
        start_price, end_price = float(test_close.iloc[i]), float(test_close.iloc[i + horizon])
        ret = end_price / start_price - 1
        rows.append({
            "ticker": name, "horizon": horizon_label,
            "start_date": start_date.date(), "end_date": end_date.date(),
            "start_price": start_price, "end_price": end_price,
            "return_pct": ret * 100, "predicted": "UP", "correct": ret > 0,
        })
        i += horizon
    return pd.DataFrame(rows)


def most_recent_completed_window(name: str, horizon_label: str, horizon: int) -> dict:
    """The single most recent completed window: price `horizon` trading days
    ago vs. today's price. This directly answers 'what would the algorithm
    have said if run `horizon` trading days ago, and was it right' -- using
    the FULL price history (not just the held-out test slice), since the
    point is the most recent real instance, not an in-sample/out-of-sample
    split. One instance, not a statistic -- read it as an anecdote, not
    evidence of a rate."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
    close = df["Close"].dropna()  # defends against a stale CSV with a not-yet-settled trailing row
    if len(close) <= horizon:
        return {"ticker": name, "horizon": horizon_label, "note": "not enough history"}
    start_date, end_date = close.index[-horizon - 1], close.index[-1]
    start_price, end_price = float(close.iloc[-horizon - 1]), float(close.iloc[-1])
    ret = end_price / start_price - 1
    return {
        "ticker": name, "horizon": horizon_label,
        "start_date": start_date.date(), "end_date": end_date.date(),
        "start_price": start_price, "end_price": end_price,
        "return_pct": ret * 100, "predicted": "UP", "correct": ret > 0,
    }


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


EXPECTED_RETURN_HORIZONS = {"3 months": 63, "6 months": 126, "9 months": 189, "12 months": 252}


def expected_vs_actual_return(name: str, anchor_days_ago: int = 252) -> pd.DataFrame:
    """Walk-forward, no-lookahead check: using ONLY price history available
    strictly BEFORE the anchor date (`anchor_days_ago` trading days back --
    ~1 year, by default), compute the historical AVERAGE forward return at
    3/6/9/12 months as "the algorithm's expectation" -- then compare to what
    ACTUALLY happened over that exact same window, since every one of these
    windows (even the 12-month one, which lands on today) is now fully
    realized. This is the same "always bet the drift" algorithm as the rest
    of this module, just reporting the expected magnitude (mean historical
    return) instead of only the binary hit rate."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
    close = df["Close"]
    n = len(close)
    anchor_idx = n - 1 - anchor_days_ago
    if anchor_idx < 60:
        return pd.DataFrame()  # not enough pre-anchor history to form an expectation

    anchor_date = close.index[anchor_idx]
    anchor_price = float(close.iloc[anchor_idx])
    history_before_anchor = close.iloc[:anchor_idx]  # strictly before the anchor -- no lookahead

    rows = []
    for label, h in EXPECTED_RETURN_HORIZONS.items():
        fwd = (history_before_anchor.shift(-h) / history_before_anchor - 1).dropna()
        expected_return = float(fwd.mean()) if len(fwd) > 0 else None

        target_idx = anchor_idx + h
        if target_idx >= n or expected_return is None:
            rows.append({
                "ticker": name, "horizon": label, "anchor_date": anchor_date.date(),
                "anchor_price": anchor_price, "expected_return_pct": None,
                "expected_n_samples": len(fwd), "target_date": None, "target_price": None,
                "actual_return_pct": None, "error_pct_points": None, "direction_match": None,
            })
            continue

        target_date = close.index[target_idx]
        target_price = float(close.iloc[target_idx])
        actual_return = target_price / anchor_price - 1

        rows.append({
            "ticker": name, "horizon": label, "anchor_date": anchor_date.date(),
            "anchor_price": anchor_price, "expected_return_pct": expected_return * 100,
            "expected_n_samples": len(fwd), "target_date": target_date.date(),
            "target_price": target_price, "actual_return_pct": actual_return * 100,
            "error_pct_points": (actual_return - expected_return) * 100,
            "direction_match": (expected_return > 0) == (actual_return > 0),
        })
    return pd.DataFrame(rows)


VERIFY_HORIZONS = {  # the specific horizons checked against actual history below
    "1 month": 21, "3 months": 63, "6 months": 126, "9 months": 189, "1 year": 252,
}


def verify_predictions():
    """Check the 1/3/6/9/12-month 'always predict UP' calls against what
    ACTUALLY happened, instance by instance, in the held-out test period."""
    print(f"\n{'='*90}\nVerifying 1/3/6/9/12-month predictions against actual history "
          f"(non-overlapping, held-out test period)\n{'='*90}")
    for name in TICKERS.values():
        print(f"\n--- {name} ---")
        for label, h in VERIFY_HORIZONS.items():
            instances = individual_window_instances(name, label, h)
            if instances.empty:
                print(f"  {label:<10}: no complete non-overlapping window in the test period (too short)")
                continue
            n_correct = int(instances["correct"].sum())
            n_total = len(instances)
            print(f"  {label:<10}: {n_correct}/{n_total} correct")
            for _, r in instances.iterrows():
                mark = "correct" if r["correct"] else "WRONG"
                print(f"      {r['start_date']} (${r['start_price']:.2f}) -> {r['end_date']} "
                      f"(${r['end_price']:.2f})  {r['return_pct']:+.1f}%  predicted UP -> {mark}")


def run_expected_vs_actual(anchor_days_ago: int = 252):
    """Print + save the walk-forward 'expect the return using last year's
    data, then check against what actually happened' table for every ticker."""
    print(f"\n{'='*90}\nExpected vs. actual return at 3/6/9/12 months, using ONLY data available "
          f"~{anchor_days_ago} trading days ago (no lookahead)\n{'='*90}")
    all_rows = []
    for name in TICKERS.values():
        r = expected_vs_actual_return(name, anchor_days_ago)
        if r.empty:
            print(f"\n--- {name}: not enough pre-anchor history ---")
            continue
        print(f"\n--- {name} (anchor: {r['anchor_date'].iloc[0]}, ${r['anchor_price'].iloc[0]:.2f}) ---")
        for _, row in r.iterrows():
            if pd.isna(row["actual_return_pct"]):
                print(f"  {row['horizon']:<10}: target date not yet reached")
                continue
            mark = "direction matched" if row["direction_match"] else "direction WRONG"
            print(f"  {row['horizon']:<10}: expected {row['expected_return_pct']:+.2f}% "
                  f"(avg of {int(row['expected_n_samples'])} historical windows before anchor)  |  "
                  f"actual {row['actual_return_pct']:+.2f}% by {row['target_date']}  |  "
                  f"error {row['error_pct_points']:+.2f} pts  |  {mark}")
        all_rows.append(r)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / "expected_vs_actual_return.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")

        scored = combined.dropna(subset=["direction_match"])
        if not scored.empty:
            print(f"\nOverall direction match: {int(scored['direction_match'].sum())}/{len(scored)} "
                  f"({scored['direction_match'].mean():.1%})")
            mae = scored["error_pct_points"].abs().mean()
            print(f"Mean absolute error (expected vs. actual return): {mae:.2f} percentage points")


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

    verify_predictions()
    run_expected_vs_actual()


if __name__ == "__main__":
    main()
