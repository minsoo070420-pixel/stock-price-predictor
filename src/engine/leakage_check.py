"""Verify -- don't just assert -- that no feature at date D depends on data
after D. Method: rebuild features using only data truncated at several cutoff
dates, and confirm each cutoff's last row is bit-for-bit identical to that same
date's row computed from the full history. If truncating the future ever
changes a past row, that feature is leaking.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, TICKERS
from features import FEATURE_COLUMNS, OWN_FEATURE_COLUMNS, build_features
from fetch_data import fetch_macro_all
from macro_features import align_macro_to_ticker, build_macro_features
from news_features import build_scored_articles, daily_aggregate, news_features_for_ticker


def check_own_features(name: str, df: pd.DataFrame, cutoffs: list[int]) -> list[str]:
    problems = []
    full = build_features(df)
    for cut in cutoffs:
        truncated_df = df.iloc[:cut]
        if len(truncated_df) < 60:
            continue
        truncated = build_features(truncated_df)
        d = truncated.index[-1]
        row_full = full.loc[d, OWN_FEATURE_COLUMNS]
        row_trunc = truncated.loc[d, OWN_FEATURE_COLUMNS]
        diff = (row_full - row_trunc).abs()
        bad = diff[diff > 1e-9]
        if not bad.empty:
            problems.append(f"{name} own-features LEAK at {d.date()}: {list(bad.index)}")
    return problems


def check_macro_alignment(name: str, df: pd.DataFrame, macro_raw: dict, cutoffs: list[int]) -> list[str]:
    problems = []
    full_macro = build_macro_features(macro_raw)
    feats_full = build_features(df)
    aligned_full = align_macro_to_ticker(full_macro, feats_full.index)

    for cut in cutoffs:
        truncated_df = df.iloc[:cut]
        if len(truncated_df) < 60:
            continue
        feats_trunc = build_features(truncated_df)
        d = feats_trunc.index[-1]
        if d not in aligned_full.index:
            continue
        # Truncate EVERY macro series to the same cutoff date as the ticker,
        # simulating "what macro data existed at that point in time."
        macro_trunc_raw = {k: v[pd.to_datetime(v.index).tz_localize(None) <= d] for k, v in macro_raw.items()}
        macro_trunc = build_macro_features(macro_trunc_raw)
        aligned_trunc = align_macro_to_ticker(macro_trunc, feats_trunc.index)

        row_full = aligned_full.loc[d]
        row_trunc = aligned_trunc.loc[d]
        diff = (row_full - row_trunc).abs()
        bad = diff[diff > 1e-9]
        if not bad.empty:
            problems.append(f"{name} macro-alignment LEAK at {d.date()}: {list(bad.index)}")
    return problems


def check_news_alignment(name: str, df: pd.DataFrame, news_scored_full: pd.DataFrame, cutoffs: list[int]) -> list[str]:
    problems = []
    feats_full = build_features(df)
    full_daily = daily_aggregate(news_scored_full)
    full_news = news_features_for_ticker(full_daily, name)
    aligned_full = align_macro_to_ticker(full_news, feats_full.index)

    # Normalized to the calendar date, not the exact timestamp: the real
    # pipeline (daily_aggregate groups by pub_date's calendar day, with no
    # time-of-day cutoff) treats a full day's news as known by day's end --
    # legitimate for predicting the *next* day. Truncating at exact
    # timestamp <= midnight(d) would wrongly exclude same-day articles
    # published any time after 00:00:00, which is almost all of them.
    pub_dates = pd.to_datetime(news_scored_full["pub_date"], format="ISO8601").dt.tz_localize(None).dt.normalize()
    for cut in cutoffs:
        truncated_df = df.iloc[:cut]
        if len(truncated_df) < 60:
            continue
        feats_trunc = build_features(truncated_df)
        d = feats_trunc.index[-1]
        if d not in aligned_full.index:
            continue
        # Truncate the article table to only what was published on or before d,
        # simulating "what news existed at that point in time."
        news_trunc_scored = news_scored_full[pub_dates <= d]
        trunc_daily = daily_aggregate(news_trunc_scored)
        trunc_news = news_features_for_ticker(trunc_daily, name)
        aligned_trunc = align_macro_to_ticker(trunc_news, feats_trunc.index)

        row_full = aligned_full.loc[d]
        row_trunc = aligned_trunc.loc[d]
        diff = (row_full - row_trunc).abs()
        bad = diff[diff > 1e-9]
        if not bad.empty:
            problems.append(f"{name} news-alignment LEAK at {d.date()}: {list(bad.index)}")
    return problems


def main():
    print("Fetching data for leakage audit...")
    macro_raw = fetch_macro_all()

    all_problems = []
    news_scored_full = None
    for name in TICKERS.values():
        df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)
        n = len(df)
        cutoffs = sorted(set(int(n * f) for f in (0.3, 0.5, 0.7, 0.9, 0.99)))
        print(f"\n{name}: checking {len(cutoffs)} cutoff points out of {n} rows...")

        if news_scored_full is None:
            news_scored_full = build_scored_articles(df.index.min().strftime("%Y-%m"),
                                                       df.index.max().strftime("%Y-%m"))

        problems = check_own_features(name, df, cutoffs)
        problems += check_macro_alignment(name, df, macro_raw, cutoffs)
        problems += check_news_alignment(name, df, news_scored_full, cutoffs)

        if problems:
            all_problems.extend(problems)
            for p in problems:
                print(f"  FAIL: {p}")
        else:
            print(f"  OK: every checked date's features matched exactly whether computed "
                  f"from truncated or full history (own features + macro alignment + news alignment).")

    print()
    if all_problems:
        print(f"LEAKAGE CHECK FAILED: {len(all_problems)} issue(s) found.")
        sys.exit(1)
    else:
        print("LEAKAGE CHECK PASSED: no evidence that any feature depends on future data.")


if __name__ == "__main__":
    main()
