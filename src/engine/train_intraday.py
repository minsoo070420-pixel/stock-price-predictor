"""Train models that predict TODAY's 4pm close from intraday price action so far.

Unlike train.py (which predicts tomorrow's close using yesterday's daily bar),
this trains on historical 'snapshots' taken at various times during each
trading day, and predicts the remaining return to that same day's close.
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TICKERS
from fetch_data import fetch_all, fetch_intraday_train_all
from intraday_features import SNAPSHOT_FEATURE_COLUMNS, build_snapshots

warnings.filterwarnings("ignore")

TEST_DAY_FRACTION = 0.15


def chrono_split_by_day(snapshots: pd.DataFrame, test_fraction=TEST_DAY_FRACTION):
    days = np.sort(snapshots["date"].unique())
    n_test_days = max(int(len(days) * test_fraction), 10)
    split_day = days[-n_test_days]
    train = snapshots[snapshots["date"] < split_day]
    test = snapshots[snapshots["date"] >= split_day]
    return train, test


def models():
    return {
        "baseline_zero_return": DummyRegressor(strategy="constant", constant=0.0),
        "linear_ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE,
        ),
    }


def accuracy_by_time_bucket(test_df: pd.DataFrame, pred_ret: np.ndarray) -> pd.DataFrame:
    df = test_df.copy()
    df["pred_ret"] = pred_ret
    df["abs_err"] = (df["pred_ret"] - df["target_return_to_close"]).abs()
    df["dir_hit"] = np.sign(df["pred_ret"]) == np.sign(df["target_return_to_close"])
    bins = [0, 60, 120, 180, 240, 300, 390]
    labels = ["<1h to close", "1-2h", "2-3h", "3-4h", "4-5h", ">5h to close"]
    df["bucket"] = pd.cut(df["minutes_until_close"], bins=bins, labels=labels)
    return df.groupby("bucket", observed=True).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        directional_accuracy=("dir_hit", "mean"),
    )


def train_for_ticker(name: str, intraday_df: pd.DataFrame, daily_df: pd.DataFrame, summary_rows: list):
    print(f"\n=== {name} (intraday, predict today's 4pm close) ===")
    snaps = build_snapshots(intraday_df, daily_df)
    if snaps.empty or len(snaps) < 200:
        print(f"  not enough intraday history for {name} yet ({len(snaps)} snapshot rows) -- skipping")
        return

    train_df, test_df = chrono_split_by_day(snaps)
    X_train, y_train = train_df[SNAPSHOT_FEATURE_COLUMNS], train_df["target_return_to_close"]
    X_test, y_test = test_df[SNAPSHOT_FEATURE_COLUMNS], test_df["target_return_to_close"]

    print(f"  snapshot rows: total={len(snaps)}  train={len(X_train)}  test={len(X_test)}  "
          f"days: train={train_df['date'].nunique()} test={test_df['date'].nunique()}  "
          f"({train_df['date'].min().date()}..{train_df['date'].max().date()} / "
          f"{test_df['date'].min().date()}..{test_df['date'].max().date()})")

    results = []
    best_model, best_name, best_pred, best_rmse = None, None, None, np.inf
    for mname, model in models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        dir_acc = float((np.sign(pred) == np.sign(y_test)).mean())
        results.append({"model": mname, "rmse": rmse, "mae": mae, "directional_accuracy": dir_acc})
        if mname != "baseline_zero_return" and rmse < best_rmse:
            best_rmse, best_model, best_name, best_pred = rmse, model, mname, pred

    res_df = pd.DataFrame(results).set_index("model")
    print("  -- overall (return remaining to close) --")
    print(res_df.round(5).to_string())

    bucket_df = accuracy_by_time_bucket(test_df, best_pred)
    print(f"  -- {best_name}: accuracy by time remaining until close --")
    print(bucket_df.round(4).to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / f"{name}_intraday_regressor.joblib")
    print(f"  saved best intraday model ({best_name}) to models/{name}_intraday_regressor.joblib")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bucket_df.index.astype(str), bucket_df["directional_accuracy"], color="#3d5a80")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Directional accuracy (close up/down)")
    ax.set_title(f"{name}: how prediction accuracy improves closer to the close")
    ax.set_ylim(0, 1)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    plot_path = REPORTS_DIR / f"{name}_intraday_accuracy.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"  saved plot to reports/{plot_path.name}")

    for r in results:
        summary_rows.append({"ticker": name, **r})


def main():
    print("Fetching daily context data...")
    daily = fetch_all()
    print("\nFetching intraday (60m) history -- this covers ~2 years, used only for training...")
    intraday = fetch_intraday_train_all()

    summary_rows = []
    for name in TICKERS.values():
        train_for_ticker(name, intraday[name], daily[name], summary_rows)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / "intraday_model_comparison.csv"
        summary.to_csv(out_path, index=False)
        print(f"\nSaved intraday model comparison to {out_path}")


if __name__ == "__main__":
    main()
