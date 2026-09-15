"""Long-horizon directional "forecast" -- read long_horizon_drift.py's
docstring first, this inherits all of its caveats. This is NOT day-to-day
prediction. It reports the historical, out-of-sample hit rate of "always bet
UP and hold for N trading days" -- a buy-and-hold statistic reframed as a
forecast, not a discovered signal. It says nothing about tomorrow's price,
which is what the rest of this project actually predicts.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TICKERS
from fetch_data import fetch_all

# Hand-picked from long_horizon_drift.py's output: the shortest horizon that
# reliably (>=20 overlapping test windows) crossed an 80% out-of-sample hit
# rate for that ticker. PLTR, META, and MSFT have no entry on purpose -- see
# the module docstring in long_horizon_drift.py: no horizon reliably reached
# 80% out-of-sample for them, so making a call here would be fabricated confidence.
LONG_HORIZON_CALLS = {
    "SP500": {"horizon_label": "2 months", "horizon_days": 42, "oos_hit_rate": 0.866, "independent_n": 7},
    "AAPL": {"horizon_label": "6 months", "horizon_days": 126, "oos_hit_rate": 0.976, "independent_n": 1},
    "AMZN": {"horizon_label": "6 months", "horizon_days": 126, "oos_hit_rate": 0.812, "independent_n": 1},
    "GOOGL": {"horizon_label": "6 months", "horizon_days": 126, "oos_hit_rate": 0.984, "independent_n": 1},
    "NVDA": {"horizon_label": "3 months", "horizon_days": 63, "oos_hit_rate": 0.815, "independent_n": 4},
}


def forecast(name: str, last_close: float, last_date: pd.Timestamp) -> dict:
    if name not in LONG_HORIZON_CALLS:
        return {
            "ticker": name,
            "call": "no confident long-horizon call",
            "historical_out_of_sample_hit_rate": None,
            "confidence": "n/a",
            "reason": "no horizon reliably reached 80% out-of-sample for this ticker "
                      "(see reports/long_horizon_drift.csv) -- at longer horizons the "
                      "held-out test period was actually a down-trending stretch",
        }
    c = LONG_HORIZON_CALLS[name]
    target_date = (last_date + pd.tseries.offsets.BDay(c["horizon_days"])).date()
    if c["independent_n"] <= 1:
        confidence = "LOW -- based on essentially one independent historical window; descriptive, not validated"
    else:
        confidence = f"based on ~{c['independent_n']} independent historical windows -- still thin by normal standards"
    return {
        "ticker": name,
        "call": f"UP by ~{target_date} ({c['horizon_label']} out)",
        "historical_out_of_sample_hit_rate": c["oos_hit_rate"],
        "confidence": confidence,
        "reason": "",
    }


def main():
    print(__doc__)
    data = fetch_all()
    rows = []
    for symbol, name in TICKERS.items():
        df = data[name]
        last_close = float(df["Close"].iloc[-1])
        last_date = df.index[-1]
        rows.append(forecast(name, last_close, last_date))

    df_out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df_out.to_string(index=False))
    print("\nThis is a buy-and-hold-style drift statistic, not a day-to-day forecast, and says "
          "nothing about tomorrow's price. Run long_horizon_drift.py for the full horizon-by-horizon analysis.")


if __name__ == "__main__":
    main()
