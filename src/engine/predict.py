"""Load the saved models and predict the next trading day for each ticker."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSIFICATION_THRESHOLD, MODELS_DIR, TICKERS
from features import build_features, FEATURE_COLUMNS, FEATURE_COLUMNS_WITH_NEWS
from fetch_data import fetch_all, fetch_macro_all
from implied_vol import implied_vol_snapshot
from macro_features import align_macro_to_ticker, build_macro_features
from news_features import build_news_history, news_features_for_ticker
from news_pulse import sentiment_for_symbol

warnings.filterwarnings("ignore")


def predict_next_day(name: str, symbol: str, df: pd.DataFrame, macro_df: pd.DataFrame, news_df: pd.DataFrame) -> dict:
    feats = build_features(df)
    aligned = align_macro_to_ticker(macro_df, feats.index)
    feats = feats.join(aligned)
    aligned_news = align_macro_to_ticker(news_df, feats.index)
    feats = feats.join(aligned_news)

    # The regressor and classifier may have been trained on different feature
    # sets (baseline / macro-enhanced / macro+news) -- use each model's own
    # recorded column list, and only require those specific columns to be
    # non-null, so a ticker whose winning model doesn't use news isn't broken
    # by a quiet news day.
    with open(MODELS_DIR / f"{name}_features.json") as f:
        manifest = json.load(f)
    required_cols = sorted(set(manifest["regressor"]) | set(manifest["classifier"]) | {"close"})
    latest = feats.dropna(subset=required_cols).iloc[[-1]]
    last_close = float(latest["close"].iloc[0])
    last_date = latest.index[0]

    reg = joblib.load(MODELS_DIR / f"{name}_regressor.joblib")
    clf = joblib.load(MODELS_DIR / f"{name}_classifier.joblib")

    pred_return = float(reg.predict(latest[manifest["regressor"]])[0])
    pred_close = last_close * (1 + pred_return)
    pred_dir_proba = float(clf.predict_proba(latest[manifest["classifier"]])[0][1])
    pred_dir = int(pred_dir_proba > CLASSIFICATION_THRESHOLD)

    # Live news sentiment and single-stock implied vol: shown for context, NOT
    # inputs to the trained model (see news_pulse.py / implied_vol.py for why --
    # no free source has point-in-time history for either to train on). Index-level
    # implied vol (VIX/VIX3M/VXN) IS trained in -- see macro_features.py.
    news = sentiment_for_symbol(symbol)
    iv = implied_vol_snapshot(symbol) if not symbol.startswith("^") else None

    return {
        "ticker": name,
        "last_close_date": last_date.date(),
        "last_close": last_close,
        "predicted_return_pct": pred_return * 100,
        "predicted_next_close": pred_close,
        "predicted_direction": "UP" if pred_dir == 1 else "DOWN",
        "up_probability": pred_dir_proba,
        "live_news_sentiment": news["label"],
        "live_atm_iv": iv["atm_iv"] if iv else None,
        "live_iv_skew": iv["iv_skew"] if iv else None,
    }


def main():
    data = fetch_all()
    macro_df = build_macro_features(fetch_macro_all())

    # Only need enough recent history to cover the longest news feature's rolling
    # lookback plus weekend/holiday gaps -- not the full multi-year training archive.
    recent_start = pd.Timestamp.today().to_period("M").strftime("%Y-%m")
    recent_start = (pd.Period(recent_start, freq="M") - 1).strftime("%Y-%m")
    recent_end = pd.Timestamp.today().strftime("%Y-%m")
    news_daily = build_news_history(recent_start, recent_end)

    rows = []
    for symbol, name in TICKERS.items():
        model_path = MODELS_DIR / f"{name}_regressor.joblib"
        if not model_path.exists():
            print(f"No trained model found for {name}. Run `python src/train.py` first.")
            continue
        news_df = news_features_for_ticker(news_daily, name)
        rows.append(predict_next_day(name, symbol, data[name], macro_df, news_df))

    if not rows:
        return

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\nNext trading day predictions:")
    print(df.to_string(index=False))
    print("\n(live_news_sentiment and live_atm_iv/live_iv_skew are shown for context only -- not used by "
          "the trained models; see news_pulse.py / implied_vol.py. Index-level implied vol IS trained in.)")


if __name__ == "__main__":
    main()
