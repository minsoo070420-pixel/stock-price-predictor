"""Predict TODAY's 4pm ET closing price using intraday data available right now.

Run this any time during market hours (9:30am-4:00pm ET) to get an updated
estimate of where each ticker will close today.
"""
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, TICKERS
from fetch_data import fetch_all, fetch_intraday_live
from intraday_features import build_live_snapshot, SNAPSHOT_FEATURE_COLUMNS

warnings.filterwarnings("ignore")


def predict_one(symbol: str, name: str, daily_df: pd.DataFrame) -> dict | None:
    model_path = MODELS_DIR / f"{name}_intraday_regressor.joblib"
    if not model_path.exists():
        print(f"No trained intraday model for {name}. Run `python src/train_intraday.py` first.")
        return None

    intraday_live = fetch_intraday_live(symbol)
    intraday_live.index = pd.to_datetime(intraday_live.index)
    if intraday_live.index.tz is not None:
        intraday_live.index = intraday_live.index.tz_convert("America/New_York")

    today = pd.Timestamp.now(tz="America/New_York").normalize()
    today_bars = intraday_live[intraday_live.index.tz_localize(None).normalize() == today.tz_localize(None)]

    if today_bars.empty:
        last_session = intraday_live.index.tz_localize(None).normalize().max()
        last_close = float(intraday_live[intraday_live.index.tz_localize(None).normalize() == last_session]["Close"].iloc[-1])
        return {
            "ticker": name,
            "status": "market closed (no bars yet today)",
            "as_of": None,
            "current_price": None,
            "predicted_4pm_close": None,
            "predicted_change_pct": None,
            "last_session_close": last_close,
        }

    snapshot = build_live_snapshot(today_bars, daily_df)
    if snapshot is None:
        return {
            "ticker": name,
            "status": "not enough bars yet this session (try again in a few minutes)",
            "as_of": today_bars.index[-1],
            "current_price": float(today_bars["Close"].iloc[-1]),
            "predicted_4pm_close": None,
            "predicted_change_pct": None,
            "last_session_close": None,
        }

    model = joblib.load(model_path)
    pred_return = float(model.predict(snapshot[SNAPSHOT_FEATURE_COLUMNS])[0])
    price_now = snapshot.attrs["price_now"]
    pred_close = price_now * (1 + pred_return)
    minutes_left = float(snapshot["minutes_until_close"].iloc[0])

    return {
        "ticker": name,
        "status": "closed for the day" if minutes_left <= 0 else f"{minutes_left:.0f} min until close",
        "as_of": snapshot.attrs["as_of"],
        "current_price": price_now,
        "predicted_4pm_close": pred_close,
        "predicted_change_pct": pred_return * 100,
        "last_session_close": None,
    }


def main():
    print("Fetching daily context + live intraday data...")
    daily = fetch_all()

    rows = []
    for symbol, name in TICKERS.items():
        r = predict_one(symbol, name, daily[name])
        if r is not None:
            rows.append(r)

    if not rows:
        return

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\nToday's 4pm ET close predictions:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
