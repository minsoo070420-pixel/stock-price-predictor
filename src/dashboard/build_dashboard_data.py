"""Assembles reports/*.csv (Part 1 + Part 2 outputs) into one JSON blob
consumed by site/index.html -- the static, publishable snapshot of the
project's current numbers. Run this after train.py / compare_tickers.py /
backtest_dates.py / long_horizon_drift.py to refresh the published dashboard,
then re-run build_site.py to bake the new JSON into site/index.html.

This script only reads already-computed reports; it doesn't train or predict
anything new, same "descriptive, not advice" boundary as compare_tickers.py.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from config import REPORTS_DIR, TICKERS


def _nan_to_none(v):
    return None if pd.isna(v) else v


def build() -> dict:
    ticker_cmp = pd.read_csv(REPORTS_DIR / "ticker_comparison.csv", index_col="ticker")
    four_horizon = pd.read_csv(REPORTS_DIR / "four_horizon_comparison.csv", index_col="ticker")
    fundamentals = pd.read_csv(REPORTS_DIR / "fundamentals_comparison.csv", index_col="ticker")
    expected_vs_actual = pd.read_csv(REPORTS_DIR / "expected_vs_actual_return.csv")
    expected_vs_actual = expected_vs_actual[expected_vs_actual["estimator"] == "median"]
    backtest_recent = pd.read_csv(REPORTS_DIR / "backtest_recent.csv")
    live_forecast = pd.read_csv(REPORTS_DIR / "live_horizon_forecast.csv")

    out = {}
    for name in TICKERS.values():
        row = ticker_cmp.loc[name]

        recent = backtest_recent[backtest_recent["ticker"] == name].dropna(subset=["direction_correct"])
        recent5d = {
            "recent5d_accuracy": _nan_to_none(recent["direction_correct"].mean()) if not recent.empty else None,
            "recent5d_correct": int(recent["direction_correct"].sum()) if not recent.empty else None,
            "recent5d_n": len(recent) if not recent.empty else None,
            "recent5d_avg_net_bps": _nan_to_none(recent["net_return_bps"].mean()) if not recent.empty else None,
            "recent5d_cum_pct": _nan_to_none(recent["net_return_bps"].sum() / 100) if not recent.empty else None,
        }

        eva_rows = expected_vs_actual[expected_vs_actual["ticker"] == name]
        expected_vs_actual_list = [
            {
                "horizon": r["horizon"],
                "expected_pct": r["expected_return_pct"],
                "actual_pct": r["actual_return_pct"],
                "error_pts": r["error_pct_points"],
                "direction_match": bool(r["direction_match"]),
                "target_date": r["target_date"],
            }
            for _, r in eva_rows.iterrows()
        ]

        fund = None
        if name in fundamentals.index:
            f = fundamentals.loc[name]
            if pd.notna(f.get("trailing_pe")):
                fund = {
                    "trailing_pe": _nan_to_none(f.get("trailing_pe")),
                    "forward_pe": _nan_to_none(f.get("forward_pe")),
                    "market_cap_b": _nan_to_none(f.get("market_cap_billions")),
                    "price_to_book": _nan_to_none(f.get("price_to_book")),
                    "profit_margin_pct": _nan_to_none(f.get("profit_margin_pct")),
                    "revenue_growth_pct": _nan_to_none(f.get("revenue_growth_pct")),
                    "sector": f.get("sector"),
                }

        fh = four_horizon.loc[name] if name in four_horizon.index else pd.Series(dtype=float)

        horizon_forecast = {}
        for _, r in live_forecast[live_forecast["ticker"] == name].iterrows():
            horizon_forecast[r["horizon"]] = {
                "expected_return_pct": _nan_to_none(r["expected_return_pct"]),
                "predicted_price": _nan_to_none(r["predicted_price"]),
                "target_date": r["target_date"],
                "oos_hit_rate": _nan_to_none(r["oos_hit_rate"]),
                "reliable": bool(r["reliable"]),
                "effective_independent_n": int(r["effective_independent_n"]),
                "n_historical_samples": int(r["n_historical_samples"]),
            }

        out[name] = {
            "held_out_balanced_accuracy": _nan_to_none(row["held_out_test_balanced_accuracy"]),
            "held_out_days": int(row["held_out_test_days"]),
            "classifier_type": row["classifier_type"],
            "regressor_type": row["regressor_type"],
            **recent5d,
            "long_horizon_call": row["long_horizon_call"],
            "long_horizon_hit_rate": _nan_to_none(row["long_horizon_oos_hit_rate"]),
            "four_horizon": {
                "1 month": _nan_to_none(fh.get("1 month")),
                "3 months": _nan_to_none(fh.get("3 months")),
                "6 months": _nan_to_none(fh.get("6 months")),
                "1 year": _nan_to_none(fh.get("1 year")),
            },
            "expected_vs_actual": expected_vs_actual_list,
            "fundamentals": fund,
            "horizon_forecast": horizon_forecast,
            "last_price": float(live_forecast[live_forecast["ticker"] == name]["last_price"].iloc[0])
                if (live_forecast["ticker"] == name).any() else None,
        }
    return out


if __name__ == "__main__":
    data = build()
    out_path = Path(__file__).resolve().parent / "dashboard_data.json"
    out_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"Saved {out_path}")
