"""Train and evaluate next-day return/direction models for each ticker.

For each ticker we train two kinds of models on a chronological train/test
split (no shuffling, since this is time series):
  - Regression models predicting next-day % return
  - Classification models predicting next-day direction (up/down)

Both are compared against a naive baseline (predict 0% return / majority class)
because for daily stock returns, beating "no change" is the real bar to clear.

Optimization pipeline, in order, per ticker/task/feature-set:
  1. Feature selection (only when the feature set is large): a quick Random
     Forest importance ranking keeps only the top-K columns, to fight the
     curse of dimensionality on ~1-2k training rows.
  2. Hyperparameter tuning: RandomizedSearchCV with TimeSeriesSplit (not
     shuffled k-fold -- a fold's "future" must never leak into its own training
     data) tunes Random Forest and HistGradientBoosting.
  3. Ensembling: a soft-voting ensemble of the tuned RF + tuned HGB + the
     (untuned, already fast) linear model is added as one more candidate.
  4. Final selection: baseline vs. macro-enhanced feature sets, and every
     candidate model within each, are all compared on the SAME untouched
     held-out test window, and whichever wins is kept -- CV folds above are
     only ever used to pick hyperparameters, never to report the final metric.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASSIFICATION_THRESHOLD,
    FEATURE_SELECT_TOP_K,
    HYPERPARAM_CV_SPLITS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_FRACTION,
    TICKERS,
)
from features import FEATURE_COLUMNS, OWN_FEATURE_COLUMNS, make_dataset
from fetch_data import fetch_all, fetch_macro_all
from macro_features import build_macro_features

warnings.filterwarnings("ignore")

RF_PARAM_DIST = {
    "n_estimators": [150, 250, 350],
    "max_depth": [3, 4, 5, 6, 8],
    "min_samples_leaf": [3, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 0.8],
}
HGB_PARAM_DIST = {
    "max_iter": [100, 200, 300],
    "max_depth": [3, 4, 6, None],
    "learning_rate": [0.02, 0.05, 0.1],
    "l2_regularization": [0.0, 0.5, 1.0],
}
N_SEARCH_ITER = 10


def chrono_split(X, *ys, test_fraction=TEST_FRACTION):
    n_test = max(int(len(X) * test_fraction), 30)
    split = len(X) - n_test
    parts = [(X.iloc[:split], X.iloc[split:])]
    for y in ys:
        parts.append((y.iloc[:split], y.iloc[split:]))
    return parts, split


def select_top_features(X_train, y_train, task: str, top_k=FEATURE_SELECT_TOP_K) -> list[str]:
    """Fast importance-based feature selection to fight noise from a large feature set."""
    if X_train.shape[1] <= top_k:
        return list(X_train.columns)
    probe = (
        RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
        if task == "regression"
        else RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    )
    probe.fit(X_train, y_train)
    importances = pd.Series(probe.feature_importances_, index=X_train.columns)
    return list(importances.sort_values(ascending=False).head(top_k).index)


def tune_model(estimator_cls, param_dist, X_train, y_train, task: str):
    scoring = "neg_root_mean_squared_error" if task == "regression" else "accuracy"
    search = RandomizedSearchCV(
        estimator_cls(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        cv=TimeSeriesSplit(n_splits=HYPERPARAM_CV_SPLITS),
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def candidate_models_for(task: str, X_train, y_train):
    """Tune RF + HGB, keep the linear model fixed (fast/stable/low-variance),
    and add a voting ensemble of all three as one more candidate."""
    if task == "regression":
        linear = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        rf, rf_params = tune_model(RandomForestRegressor, RF_PARAM_DIST, X_train, y_train, task)
        hgb, hgb_params = tune_model(HistGradientBoostingRegressor, HGB_PARAM_DIST, X_train, y_train, task)
        ensemble = VotingRegressor(estimators=[("linear", linear), ("rf", rf), ("hgb", hgb)])
        return {
            "baseline_zero_return": DummyRegressor(strategy="constant", constant=0.0),
            "linear_ridge": linear,
            "random_forest_tuned": rf,
            "hist_gradient_boosting_tuned": hgb,
            "voting_ensemble": ensemble,
        }, {"random_forest_tuned": rf_params, "hist_gradient_boosting_tuned": hgb_params}
    else:
        linear = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
        rf, rf_params = tune_model(RandomForestClassifier, RF_PARAM_DIST, X_train, y_train, task)
        hgb, hgb_params = tune_model(HistGradientBoostingClassifier, HGB_PARAM_DIST, X_train, y_train, task)
        ensemble = VotingClassifier(estimators=[("linear", linear), ("rf", rf), ("hgb", hgb)], voting="soft")
        return {
            "baseline_majority": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": linear,
            "random_forest_tuned": rf,
            "hist_gradient_boosting_tuned": hgb,
            "voting_ensemble": ensemble,
        }, {"random_forest_tuned": rf_params, "hist_gradient_boosting_tuned": hgb_params}


def evaluate_regression(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred_ret = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_ret)))
    mae = float(mean_absolute_error(y_test, pred_ret))
    dir_acc = float((np.sign(pred_ret) == np.sign(y_test)).mean())
    return {"model": name, "rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}, model, pred_ret


def evaluate_classification(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    # Use the shared decision threshold (not sklearn's default 0.5) so what gets
    # reported/selected here matches exactly what predict.py/backtest_dates.py do.
    pred = (model.predict_proba(X_test)[:, 1] > CLASSIFICATION_THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, model


def plot_predictions(name, dates_test, close_test, pred_ret, out_path):
    predicted_close = close_test.values * (1 + pred_ret)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates_test, close_test.values, label="Actual close", color="#333333", linewidth=1.2)
    ax.plot(dates_test, predicted_close, label="Predicted next-day close", color="#d1495b", linewidth=1.0, alpha=0.8)
    ax.set_title(f"{name}: actual close vs. model's predicted next-day close (test period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_task(X_train_full, y_train, X_test_full, y_test, task: str, label: str):
    """Feature-select, tune, ensemble, and pick the best candidate for one
    (feature set, task) combination. Returns the winning model, its name, the
    exact feature columns it expects, and every candidate's metrics."""
    selected_cols = select_top_features(X_train_full, y_train, task)
    X_train, X_test = X_train_full[selected_cols], X_test_full[selected_cols]

    models, tuned_params = candidate_models_for(task, X_train, y_train)

    results, fitted_models, preds = [], {}, {}
    for mname, model in models.items():
        if task == "regression":
            metrics, fitted, pred = evaluate_regression(mname, model, X_train, y_train, X_test, y_test)
        else:
            metrics, fitted = evaluate_classification(mname, model, X_train, y_train, X_test, y_test)
            pred = None
        results.append(metrics)
        fitted_models[mname] = fitted
        preds[mname] = pred

    is_baseline = lambda n: n in ("baseline_zero_return", "baseline_majority")
    metric_key = "rmse" if task == "regression" else "accuracy"
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)
    best_name, best_val = None, (np.inf if task == "regression" else -np.inf)
    for r in results:
        if is_baseline(r["model"]):
            continue
        if better(r[metric_key], best_val):
            best_val, best_name = r[metric_key], r["model"]

    return {
        "results": results,
        "best_name": best_name,
        "best_model": fitted_models[best_name],
        "best_pred": preds.get(best_name),
        "best_metric": best_val,
        "feature_columns": selected_cols,
        "tuned_params": tuned_params,
    }


def run_feature_set(name: str, df: pd.DataFrame, macro_df, feature_columns: list[str], label: str):
    X, y_ret, y_dir, close = make_dataset(df, macro_df=macro_df, feature_columns=feature_columns)
    parts, split = chrono_split(X, y_ret, y_dir)
    (X_train, X_test), (y_ret_train, y_ret_test), (y_dir_train, y_dir_test) = parts
    close_test = close.iloc[split:]

    print(f"  [{label}] rows: total={len(X)}  train={len(X_train)}  test={len(X_test)}  "
          f"features={len(feature_columns)}  "
          f"({X_train.index.min().date()}..{X_train.index.max().date()} / "
          f"{X_test.index.min().date()}..{X_test.index.max().date()})")

    reg = run_task(X_train, y_ret_train, X_test, y_ret_test, "regression", label)
    print(f"  [{label}] best regressor: {reg['best_name']} (rmse={reg['best_metric']:.5f}, "
          f"{len(reg['feature_columns'])} features)")

    clf = run_task(X_train, y_dir_train, X_test, y_dir_test, "classification", label)
    print(f"  [{label}] best classifier: {clf['best_name']} (accuracy={clf['best_metric']:.4f}, "
          f"{len(clf['feature_columns'])} features)")

    return {
        "label": label,
        "reg": reg, "clf": clf,
        "X_test": X_test, "close_test": close_test,
    }


def train_for_ticker(name: str, df: pd.DataFrame, macro_df: pd.DataFrame, summary_rows: list):
    print(f"\n=== {name} ===")
    baseline = run_feature_set(name, df, None, OWN_FEATURE_COLUMNS, "baseline: own technical features only")
    enhanced = run_feature_set(name, df, macro_df, FEATURE_COLUMNS, "enhanced: + macro/cross-market features")

    reg_pick = enhanced if enhanced["reg"]["best_metric"] < baseline["reg"]["best_metric"] else baseline
    clf_pick = enhanced if enhanced["clf"]["best_metric"] > baseline["clf"]["best_metric"] else baseline

    print(f"  >> regressor RMSE:      {baseline['reg']['best_metric']:.5f} (baseline) -> "
          f"{enhanced['reg']['best_metric']:.5f} (with macro)  -- keeping '{reg_pick['label']}'")
    print(f"  >> classifier accuracy: {baseline['clf']['best_metric']:.1%} (baseline) -> "
          f"{enhanced['clf']['best_metric']:.1%} (with macro)  -- keeping '{clf_pick['label']}'")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_pick["reg"]["best_model"], MODELS_DIR / f"{name}_regressor.joblib")
    joblib.dump(clf_pick["clf"]["best_model"], MODELS_DIR / f"{name}_classifier.joblib")
    feature_manifest = {
        "regressor": reg_pick["reg"]["feature_columns"],
        "classifier": clf_pick["clf"]["feature_columns"],
    }
    with open(MODELS_DIR / f"{name}_features.json", "w") as f:
        json.dump(feature_manifest, f, indent=2)
    print(f"  saved best regressor ({reg_pick['reg']['best_name']}) and "
          f"classifier ({clf_pick['clf']['best_name']}) to models/")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = REPORTS_DIR / f"{name}_predictions.png"
    plot_predictions(name, reg_pick["X_test"].index, reg_pick["close_test"], reg_pick["reg"]["best_pred"], plot_path)
    print(f"  saved plot to reports/{plot_path.name}")

    for fs_label, fs in (("baseline", baseline), ("with_macro", enhanced)):
        for r in fs["reg"]["results"]:
            summary_rows.append({"ticker": name, "feature_set": fs_label, "task": "regression", **r})
        for r in fs["clf"]["results"]:
            summary_rows.append({"ticker": name, "feature_set": fs_label, "task": "classification", **r})

    return {
        "ticker": name,
        "baseline_reg_rmse": baseline["reg"]["best_metric"], "with_macro_reg_rmse": enhanced["reg"]["best_metric"],
        "baseline_clf_acc": baseline["clf"]["best_metric"], "with_macro_clf_acc": enhanced["clf"]["best_metric"],
        "regressor_kept": f"{reg_pick['label']} / {reg_pick['reg']['best_name']}",
        "classifier_kept": f"{clf_pick['label']} / {clf_pick['clf']['best_name']}",
    }


def main():
    print("Fetching daily price data...")
    data = fetch_all()
    print("Fetching macro / cross-market / global data (VIX, yields, dollar, oil, gold, "
          "other indices, credit spread, QQQ, Nikkei/FTSE/DAX, BTC)...")
    macro_raw = fetch_macro_all()
    macro_df = build_macro_features(macro_raw)

    summary_rows = []
    comparisons = []
    for name in TICKERS.values():
        comparisons.append(train_for_ticker(name, data[name], macro_df, summary_rows))

    summary = pd.DataFrame(summary_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "model_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved full model comparison table to {summary_path}")

    comp_df = pd.DataFrame(comparisons).set_index("ticker")
    print("\n=== Final result per ticker ===")
    print(comp_df.to_string())


if __name__ == "__main__":
    main()
