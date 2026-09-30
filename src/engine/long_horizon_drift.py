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
from config import DATA_DIR, RECENCY_HALF_LIFE_TRADING_DAYS, REPORTS_DIR, TEST_FRACTION, TICKERS

# Every ticker gets its own, much longer price history here (see
# fetch_data.fetch_long_horizon_history() / config.LONG_HORIZON_HISTORY_PERIOD)
# so 6/9/12-month hit rates -- and the market-shrinkage prior every ticker's
# own estimate leans on -- rest on many genuinely independent windows
# spanning real bear markets, not the ~1 independent window the shared 10y
# daily-pipeline data allows. For IPO-limited tickers (PLTR) this file is a
# harmless near-duplicate of the regular {name}.csv, since 10 years already
# covers their whole listed history.
def _load_close(name: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / f"{name}_long_history.csv", index_col=0, parse_dates=True)
    return df["Close"]

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

# Two candidate refinements to the plain historical median, added as MORE
# estimators for run_expected_vs_actual()'s existing honest MAE comparison to
# judge -- not assumed to win just because they're more sophisticated.
# 1) Recency-weighted median: the same exponential half-life idea as the
#    daily models' recency_weights(), converted from trading days to calendar
#    days (a decade of history mixes very different regimes together; this
#    leans the median toward more recent, more relevant regimes instead of
#    treating all of it as equally informative). ~2 years, same as the daily models.
RECENCY_HALF_LIFE_CALENDAR_DAYS = RECENCY_HALF_LIFE_TRADING_DAYS * 365.25 / 252
# 2) Shrinkage toward the market (SP500)'s own recency-weighted median,
#    empirical-Bayes style: a ticker's own estimate is blended with SP500's
#    much more stable one, weighted by how much genuinely independent history
#    (non-overlapping windows, not raw overlapping-window count) actually
#    backs the ticker's own number. SHRINKAGE_PRIOR_STRENGTH is the
#    effective_independent_n at which a ticker's own estimate and the
#    market's get equal weight -- reuses MIN_TEST_WINDOWS (the same
#    "don't trust it below this" threshold already used for the reliability
#    flag) so a ticker just at the reliability cutoff also gets 50/50 weight.
SHRINKAGE_PRIOR_STRENGTH = MIN_TEST_WINDOWS


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
    close = _load_close(name)
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
    close = _load_close(name).dropna()  # defends against a stale CSV with a not-yet-settled trailing row
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
    close = _load_close(name)
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


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_weights = np.cumsum(weights)
    cutoff = weights.sum() / 2.0
    idx = int(np.searchsorted(cum_weights, cutoff))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def _recency_weighted_median(fwd: pd.Series, as_of_date: pd.Timestamp) -> float:
    """Median of historical forward returns, weighted toward windows that
    STARTED more recently (exponential decay, RECENCY_HALF_LIFE_CALENDAR_DAYS)
    instead of treating a decade of regime-mixed history as equally relevant --
    the same idea as the daily models' recency_weights(), applied here as a
    weighted median instead of a weighted mean/loss function."""
    if len(fwd) == 0:
        return float("nan")
    age_days = np.clip((as_of_date - fwd.index).days.to_numpy().astype(float), 0, None)
    weights = 0.5 ** (age_days / RECENCY_HALF_LIFE_CALENDAR_DAYS)
    return _weighted_median(fwd.to_numpy(dtype=float), weights)


def _shrink_toward_market(own_estimate: float, market_estimate: float, effective_n: int,
                           prior_strength: float = SHRINKAGE_PRIOR_STRENGTH) -> float:
    """Empirical-Bayes-style shrinkage: blend a ticker's own point estimate
    toward the market's (SP500's) much more stable one, weighted by how much
    genuinely independent history backs the ticker's own number. A ticker
    with effective_n == prior_strength gets equal weight on its own estimate
    and the market's; effective_n near 0 (common for short-history tickers or
    long horizons) defers almost entirely to the market. This is why SP500
    itself is dramatically more accurate than single-name tickers at these
    horizons (see README) -- shrinking a noisy single-stock estimate toward
    the much lower-variance, longer-history market estimate directly targets
    that same mechanism instead of trusting every ticker's own thin history
    equally."""
    if pd.isna(own_estimate) or pd.isna(market_estimate):
        return own_estimate
    own_weight = effective_n / (effective_n + prior_strength)
    return own_weight * own_estimate + (1 - own_weight) * market_estimate


def _point_estimators(fwd: pd.Series, as_of_date: pd.Timestamp | None = None,
                       market_fwd: pd.Series | None = None, effective_n: int | None = None) -> dict[str, float]:
    """Several candidate point estimates from the same historical sample of
    forward returns, computed the same no-lookahead way -- compared honestly
    rather than assuming one wins.

    Why these first four: the reported metric is MAE, and minimising MAE
    means forecasting the MEDIAN, not the mean (minimising RMSE forecasts the
    mean instead) -- a standard forecasting-theory result, not a guess. Financial
    return distributions are also fat-tailed (a few extreme historical windows
    can drag a raw mean far from the typical case, which is exactly what was
    seen with PLTR/NVDA's inflated averages), so a trimmed mean and a Tukey
    IQR-winsorized mean -- both standard Kaggle-competition techniques for
    outlier-heavy targets -- are included as robustness alternatives to the mean.

    Two more candidates are added when `as_of_date`/`market_fwd` are supplied
    (see _recency_weighted_median / _shrink_toward_market above) -- also just
    candidates for the same honest MAE comparison, not assumed to win."""
    if len(fwd) == 0:
        return {}
    mean = float(fwd.mean())
    median = float(fwd.median())
    if len(fwd) >= 10:
        sorted_fwd = fwd.sort_values()
        lo, hi = int(len(sorted_fwd) * 0.1), int(len(sorted_fwd) * 0.9)
        trimmed_mean = float(sorted_fwd.iloc[lo:hi].mean())
    else:
        trimmed_mean = mean
    q1, q3 = fwd.quantile(0.25), fwd.quantile(0.75)
    iqr = q3 - q1
    winsorized_mean = float(fwd.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr).mean())
    estimators = {
        "mean": mean, "median": median,
        "trimmed_mean_10pct": trimmed_mean, "winsorized_mean_iqr": winsorized_mean,
    }

    if as_of_date is not None:
        rw_median = _recency_weighted_median(fwd, as_of_date)
        estimators["recency_weighted_median"] = rw_median
        if market_fwd is not None and effective_n is not None:
            market_rw_median = _recency_weighted_median(market_fwd, as_of_date)
            estimators["recency_weighted_median_shrunk"] = _shrink_toward_market(
                rw_median, market_rw_median, effective_n
            )
    return estimators


def expected_vs_actual_return(name: str, anchor_days_ago: int = 252,
                               market_close: pd.Series | None = None) -> pd.DataFrame:
    """Walk-forward, no-lookahead check: using ONLY price history available
    strictly BEFORE the anchor date (`anchor_days_ago` trading days back --
    ~1 year, by default), compute several candidate point estimates of the
    historical forward return at 3/6/9/12 months (see _point_estimators) --
    then compare each to what ACTUALLY happened over that exact same window,
    since every one of these windows (even the 12-month one, which lands on
    today) is now fully realized. One row per (horizon, estimator).

    `market_close` (SP500's own Close series, passed by the caller so it's
    only loaded once) enables the recency_weighted_median_shrunk candidate --
    for SP500 itself, pass its own close series so shrinkage is a harmless
    no-op (shrinking toward itself)."""
    close = _load_close(name)
    n = len(close)
    anchor_idx = n - 1 - anchor_days_ago
    if anchor_idx < 60:
        return pd.DataFrame()  # not enough pre-anchor history to form an expectation

    anchor_date = close.index[anchor_idx]
    anchor_price = float(close.iloc[anchor_idx])
    history_before_anchor = close.iloc[:anchor_idx]  # strictly before the anchor -- no lookahead

    market_history_before_anchor = None
    if market_close is not None:
        market_history_before_anchor = market_close[market_close.index < anchor_date]

    rows = []
    for label, h in EXPECTED_RETURN_HORIZONS.items():
        fwd = (history_before_anchor.shift(-h) / history_before_anchor - 1).dropna()
        market_fwd = None
        if market_history_before_anchor is not None:
            market_fwd = (market_history_before_anchor.shift(-h) / market_history_before_anchor - 1).dropna()
        effective_n = len(fwd) // h if h else 0
        estimators = _point_estimators(fwd, as_of_date=anchor_date, market_fwd=market_fwd, effective_n=effective_n)

        target_idx = anchor_idx + h
        target_date = close.index[target_idx] if target_idx < n else None
        target_price = float(close.iloc[target_idx]) if target_idx < n else None
        actual_return = (target_price / anchor_price - 1) if target_price is not None else None

        if not estimators or actual_return is None:
            rows.append({
                "ticker": name, "horizon": label, "estimator": None, "anchor_date": anchor_date.date(),
                "anchor_price": anchor_price, "expected_return_pct": None,
                "expected_n_samples": len(fwd), "target_date": None, "target_price": None,
                "actual_return_pct": None, "error_pct_points": None, "direction_match": None,
            })
            continue

        for est_name, expected_return in estimators.items():
            rows.append({
                "ticker": name, "horizon": label, "estimator": est_name, "anchor_date": anchor_date.date(),
                "anchor_price": anchor_price, "expected_return_pct": expected_return * 100,
                "expected_n_samples": len(fwd), "target_date": target_date.date(),
                "target_price": target_price, "actual_return_pct": actual_return * 100,
                "error_pct_points": (actual_return - expected_return) * 100,
                "direction_match": (expected_return > 0) == (actual_return > 0),
            })
    return pd.DataFrame(rows)


LIVE_FORECAST_ESTIMATOR = "recency_weighted_median_shrunk"  # verified winner -- see README for the honest comparison


def live_horizon_forecast(name: str, market_close: pd.Series | None = None) -> pd.DataFrame:
    """Today-anchored version of expected_vs_actual_return() -- for the
    dashboard's horizon selector. Unlike that function, there's no realized
    "actual" to compare against yet (these windows haven't happened), so this
    just reports the historical forward-return distribution's point estimate
    (LIVE_FORECAST_ESTIMATOR -- see _point_estimators for why "median" is the
    default) plus that horizon's own out-of-sample hit rate from
    analyze_ticker(), same reliability framing (n>=20 windows) used
    everywhere else in this file. Still the SAME weaker claim as the rest of
    this module -- buy-and-hold drift, not a discovered signal -- not the
    day-to-day models the rest of the engine uses.

    `market_close` (SP500's own Close series) enables the
    recency_weighted_median_shrunk candidate the same way as
    expected_vs_actual_return; for SP500 itself pass its own close series."""
    close = _load_close(name)
    last_date = close.index[-1]
    last_price = float(close.iloc[-1])

    hit_rates = analyze_ticker(name).set_index("horizon")
    horizon_to_hitrate_label = {"3 months": "3 months", "6 months": "6 months",
                                 "9 months": "9 months", "12 months": "1 year"}

    rows = []
    for label, h in EXPECTED_RETURN_HORIZONS.items():
        fwd = (close.shift(-h) / close - 1).dropna()
        market_fwd = (market_close.shift(-h) / market_close - 1).dropna() if market_close is not None else None
        effective_n = len(fwd) // h if h else 0
        estimators = _point_estimators(fwd, as_of_date=last_date, market_fwd=market_fwd, effective_n=effective_n)
        median_return = estimators.get(LIVE_FORECAST_ESTIMATOR)
        target_date = (last_date + pd.tseries.offsets.BDay(h)).date()

        hr_label = horizon_to_hitrate_label[label]
        hr_row = hit_rates.loc[hr_label] if hr_label in hit_rates.index else None

        rows.append({
            "ticker": name,
            "horizon": label,
            "as_of_date": last_date.date(),
            "last_price": last_price,
            "expected_return_pct": median_return * 100 if median_return is not None else None,
            "predicted_price": last_price * (1 + median_return) if median_return is not None else None,
            "target_date": target_date,
            "n_historical_samples": len(fwd),
            "oos_hit_rate": float(hr_row["test_hit_rate"]) if hr_row is not None and pd.notna(hr_row["test_hit_rate"]) else None,
            "reliable": bool(hr_row["reliable"]) if hr_row is not None else False,
            "effective_independent_n": int(hr_row["effective_independent_n"]) if hr_row is not None else 0,
        })
    return pd.DataFrame(rows)


def run_live_horizon_forecast() -> pd.DataFrame:
    market_close = _load_close("SP500")
    all_rows = [live_horizon_forecast(name, market_close=market_close) for name in TICKERS.values()]
    result = pd.concat(all_rows, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "live_horizon_forecast.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved live (today-anchored) horizon forecast to {out_path}")
    return result


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
    data, then check against what actually happened' table for every ticker,
    comparing four candidate point estimators (see _point_estimators) rather
    than assuming the raw mean is the right one to report."""
    print(f"\n{'='*90}\nExpected vs. actual return at 3/6/9/12 months, using ONLY data available "
          f"~{anchor_days_ago} trading days ago (no lookahead)\n{'='*90}")
    market_close = _load_close("SP500")
    all_rows = []
    for name in TICKERS.values():
        r = expected_vs_actual_return(name, anchor_days_ago, market_close=market_close)
        if r.empty:
            print(f"\n--- {name}: not enough pre-anchor history ---")
            continue
        all_rows.append(r)

    if not all_rows:
        return

    combined = pd.concat(all_rows, ignore_index=True)
    scored = combined.dropna(subset=["direction_match"])

    print("\nMean absolute error by estimator (lower is better), across all tickers/horizons:")
    summary = scored.groupby("estimator").agg(
        mae_pct_points=("error_pct_points", lambda s: s.abs().mean()),
        direction_match_rate=("direction_match", "mean"),
        n=("error_pct_points", "size"),
    ).sort_values("mae_pct_points")
    print(summary.to_string())
    best_estimator = summary.index[0]
    print(f"\nBest estimator by MAE: '{best_estimator}' "
          f"({summary.loc[best_estimator, 'mae_pct_points']:.2f} pts vs. "
          f"{summary.loc['mean', 'mae_pct_points']:.2f} pts for the raw mean)")

    print(f"\nPer-ticker detail using '{best_estimator}':")
    best = scored[scored["estimator"] == best_estimator]
    for name, g in best.groupby("ticker"):
        anchor_date = g["anchor_date"].iloc[0]
        print(f"\n--- {name} (anchor: {anchor_date}) ---")
        for _, row in g.iterrows():
            mark = "direction matched" if row["direction_match"] else "direction WRONG"
            print(f"  {row['horizon']:<10}: expected {row['expected_return_pct']:+.2f}%  |  "
                  f"actual {row['actual_return_pct']:+.2f}% by {row['target_date']}  |  "
                  f"error {row['error_pct_points']:+.2f} pts  |  {mark}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "expected_vs_actual_return.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved full (all estimators) table to {out_path}")


def main():
    print("Long-horizon drift analysis -- NOT day-to-day forecasting. See module docstring.\n")

    from fetch_data import fetch_long_horizon_history
    fetch_long_horizon_history()

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
    run_live_horizon_forecast()


if __name__ == "__main__":
    main()
