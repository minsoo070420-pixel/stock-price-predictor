"""Load the saved models and predict the next trading day for each ticker."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSIFICATION_THRESHOLD, MODELS_DIR, TICKERS
from features import build_features, FEATURE_COLUMNS
from fetch_data import fetch_all, fetch_macro_all
from macro_features import align_macro_to_ticker, build_macro_features
from news_pulse import sentiment_for_symbol

warnings.filterwarnings("ignore")


def predict_next_day(name: str, symbol: str, df: pd.DataFrame, macro_df: pd.DataFrame) -> dict:
    feats = build_features(df)
    aligned = align_macro_to_ticker(macro_df, feats.index)
    feats = feats.join(aligned)
    latest = feats.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    last_close = float(latest["close"].iloc[0])
    last_date = latest.index[0]

    # The regressor and classifier may have been trained on different feature
    # sets (baseline vs. macro-enhanced) -- use each model's own recorded column list.
    with open(MODELS_DIR / f"{name}_features.json") as f:
        manifest = json.load(f)

    reg = joblib.load(MODELS_DIR / f"{name}_regressor.joblib")
    clf = joblib.load(MODELS_DIR / f"{name}_classifier.joblib")

    pred_return = float(reg.predict(latest[manifest["regressor"]])[0])
    pred_close = last_close * (1 + pred_return)
    pred_dir_proba = float(clf.predict_proba(latest[manifest["classifier"]])[0][1])
    pred_dir = int(pred_dir_proba > CLASSIFICATION_THRESHOLD)

    # Live news sentiment: shown for context, NOT an input to the trained model
    # (see news_pulse.py for why -- no free source has point-in-time history to train on).
    news = sentiment_for_symbol(symbol)

    return {
        "ticker": name,
        "last_close_date": last_date.date(),
        "last_close": last_close,
        "predicted_return_pct": pred_return * 100,
        "predicted_next_close": pred_close,
        "predicted_direction": "UP" if pred_dir == 1 else "DOWN",
        "up_probability": pred_dir_proba,
        "live_news_sentiment": news["label"],
        "news_headlines_seen": news["n_headlines"],
    }


def main():
    data = fetch_all()
    macro_df = build_macro_features(fetch_macro_all())
    rows = []
    for symbol, name in TICKERS.items():
        model_path = MODELS_DIR / f"{name}_regressor.joblib"
        if not model_path.exists():
            print(f"No trained model found for {name}. Run `python src/train.py` first.")
            continue
        rows.append(predict_next_day(name, symbol, data[name], macro_df))

    if not rows:
        return

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("\nNext trading day predictions:")
    print(df.to_string(index=False))
    print("\n(live_news_sentiment is shown for context only -- the model was not trained on it; see news_pulse.py)")


if __name__ == "__main__":
    main()
