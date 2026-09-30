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
  4. Selection: every candidate model (within a feature set) and every
     feature set (baseline / +macro / +macro+news) is scored on
     WALK_FORWARD_STABILITY_FOLDS expanding chronological folds carved from
     the training region ONLY -- never the final test window -- and the
     winner is picked from that (see _selection_score()). This fixes a
     selection-bias gap the project's own walk-forward stability check
     found but couldn't correct on its own: picking a winner by the same
     window later quoted as its "held-out" score is a textbook winner's
     curse. The final test window is then scored exactly once, on the
     already-chosen winner -- a genuine, untouched confirmation number,
     never the thing that decided who won.
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
from sklearn.base import clone
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
    balanced_accuracy_score,
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
    RECENCY_HALF_LIFE_TRADING_DAYS,
    REPORTS_DIR,
    TEST_FRACTION,
    TICKERS,
    WALK_FORWARD_STABILITY_FOLDS,
)
from baselines import PersistenceClassifier, PersistenceRegressor
from features import FEATURE_COLUMNS, FEATURE_COLUMNS_WITH_NEWS, OWN_FEATURE_COLUMNS, make_dataset
from fetch_data import fetch_all, fetch_macro_all
from macro_features import build_macro_features
from news_features import build_news_history, news_features_for_ticker

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


def recency_weights(n: int, half_life: int = RECENCY_HALF_LIFE_TRADING_DAYS) -> np.ndarray:
    """Exponential recency weighting: the most recent training row gets
    weight 1.0, a row `half_life` trading days older gets 0.5, one twice
    that old gets 0.25, etc. Addresses non-stationarity -- markets in 2016
    aren't guaranteed to behave like markets in 2026 -- without discarding
    older data outright the way a hard lookback window would."""
    age_in_days = np.arange(n)[::-1]
    return 0.5 ** (age_in_days / half_life)


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


def tune_model(estimator_cls, param_dist, X_train, y_train, task: str,
                extra_params: dict | None = None, sample_weight: np.ndarray | None = None):
    scoring = "neg_root_mean_squared_error" if task == "regression" else "balanced_accuracy"
    search = RandomizedSearchCV(
        estimator_cls(random_state=RANDOM_STATE, **(extra_params or {})),
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        cv=TimeSeriesSplit(n_splits=HYPERPARAM_CV_SPLITS),
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    search.fit(X_train, y_train, **fit_kwargs)
    return search.best_estimator_, search.best_params_


def candidate_models_for(task: str, X_train, y_train, recency_weight: np.ndarray):
    """Tune RF + HGB, keep the linear model fixed (fast/stable/low-variance),
    add a voting ensemble of all three, and add recency-weighted RF + HGB
    variants (see recency_weights()) as two more candidates.

    Classification candidates use class_weight="balanced" and are tuned/scored
    on balanced accuracy rather than raw accuracy -- raw accuracy is what let a
    model that just echoes the training data's ~56% historical up-day rate look
    "good" without learning any real day-to-day signal. class_weight="balanced"
    penalizes missing either class equally during training; balanced accuracy
    (mean of per-class recall) scores that equally during selection. A model
    that always predicts one class now scores exactly 50% on this metric,
    matching the majority-class baseline instead of beating it for free.
    (class_weight and sample_weight compose multiplicatively in sklearn, so
    the recency-weighted classifiers keep the balanced-class correction too.)"""
    if task == "regression":
        linear = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        rf, rf_params = tune_model(RandomForestRegressor, RF_PARAM_DIST, X_train, y_train, task)
        hgb, hgb_params = tune_model(HistGradientBoostingRegressor, HGB_PARAM_DIST, X_train, y_train, task)
        ensemble = VotingRegressor(estimators=[("linear", linear), ("rf", rf), ("hgb", hgb)])
        rf_rw, rf_rw_params = tune_model(RandomForestRegressor, RF_PARAM_DIST, X_train, y_train, task,
                                          sample_weight=recency_weight)
        hgb_rw, hgb_rw_params = tune_model(HistGradientBoostingRegressor, HGB_PARAM_DIST, X_train, y_train, task,
                                            sample_weight=recency_weight)
        models = {
            "baseline_zero_return": DummyRegressor(strategy="constant", constant=0.0),
            "linear_ridge": linear,
            "random_forest_tuned": rf,
            "hist_gradient_boosting_tuned": hgb,
            "voting_ensemble": ensemble,
            "random_forest_recency_weighted": rf_rw,
            "hist_gradient_boosting_recency_weighted": hgb_rw,
        }
        tuned_params = {
            "random_forest_tuned": rf_params, "hist_gradient_boosting_tuned": hgb_params,
            "random_forest_recency_weighted": rf_rw_params, "hist_gradient_boosting_recency_weighted": hgb_rw_params,
        }
        sample_weights = {
            "random_forest_recency_weighted": recency_weight,
            "hist_gradient_boosting_recency_weighted": recency_weight,
        }
        return models, tuned_params, sample_weights
    else:
        linear = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        rf, rf_params = tune_model(RandomForestClassifier, RF_PARAM_DIST, X_train, y_train, task, {"class_weight": "balanced"})
        hgb, hgb_params = tune_model(HistGradientBoostingClassifier, HGB_PARAM_DIST, X_train, y_train, task, {"class_weight": "balanced"})
        ensemble = VotingClassifier(estimators=[("linear", linear), ("rf", rf), ("hgb", hgb)], voting="soft")
        rf_rw, rf_rw_params = tune_model(RandomForestClassifier, RF_PARAM_DIST, X_train, y_train, task,
                                          {"class_weight": "balanced"}, sample_weight=recency_weight)
        hgb_rw, hgb_rw_params = tune_model(HistGradientBoostingClassifier, HGB_PARAM_DIST, X_train, y_train, task,
                                            {"class_weight": "balanced"}, sample_weight=recency_weight)
        models = {
            "baseline_majority": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": linear,
            "random_forest_tuned": rf,
            "hist_gradient_boosting_tuned": hgb,
            "voting_ensemble": ensemble,
            "random_forest_recency_weighted": rf_rw,
            "hist_gradient_boosting_recency_weighted": hgb_rw,
        }
        tuned_params = {
            "random_forest_tuned": rf_params, "hist_gradient_boosting_tuned": hgb_params,
            "random_forest_recency_weighted": rf_rw_params, "hist_gradient_boosting_recency_weighted": hgb_rw_params,
        }
        sample_weights = {
            "random_forest_recency_weighted": recency_weight,
            "hist_gradient_boosting_recency_weighted": recency_weight,
        }
        return models, tuned_params, sample_weights


def evaluate_regression(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    pred_ret = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_ret)))
    mae = float(mean_absolute_error(y_test, pred_ret))
    dir_acc = float((np.sign(pred_ret) == np.sign(y_test)).mean())
    return {"model": name, "rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}, model, pred_ret


def evaluate_classification(name, model, X_train, y_train, X_test, y_test, sample_weight=None):
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    # Use the shared decision threshold (not sklearn's default 0.5) so what gets
    # reported/selected here matches exactly what predict.py/backtest_dates.py do.
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > CLASSIFICATION_THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    pct_predicted_up = float(pred.mean())
    return {
        "model": name, "accuracy": acc, "balanced_accuracy": bal_acc,
        "precision": prec, "recall": rec, "f1": f1, "pct_predicted_up": pct_predicted_up,
    }, model


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


def _selection_score(wf: dict, task: str) -> float | None:
    """Pessimistic, pre-declared selection criterion so model/feature-set
    selection isn't decided by a single lucky held-out window -- the same
    winner's-curse problem walk_forward_stability() already surfaces
    (PLTR's 58.7% single-window balanced accuracy was really 52.0% +/- 3.3%
    once checked across folds -- but that check only ran AFTER a winner was
    already picked, so it could never have stopped a candidate chosen partly
    by luck from shipping). Penalizes inconsistency across folds, not just
    rewards a good average: fixed here, before looking at any per-ticker
    result, so it can't be quietly re-tuned later to make a preferred
    candidate win. Regression: lower is better (mean + 0.5*std). Classification:
    higher is better (mean - 0.5*std)."""
    if wf["mean"] is None:
        return None
    return wf["mean"] + 0.5 * wf["std"] if task == "regression" else wf["mean"] - 0.5 * wf["std"]


def run_task(X_train_full, y_train, X_test_full, y_test, task: str, label: str):
    """Feature-select, tune, ensemble, and pick the best candidate for one
    (feature set, task) combination. Returns the winning model, its name, the
    exact feature columns it expects, and every candidate's metrics."""
    selected_cols = select_top_features(X_train_full, y_train, task)
    X_train, X_test = X_train_full[selected_cols], X_test_full[selected_cols]

    recency_weight = recency_weights(len(X_train))
    models, tuned_params, sample_weights = candidate_models_for(task, X_train, y_train, recency_weight)

    results, fitted_models, preds = [], {}, {}
    for mname, model in models.items():
        sw = sample_weights.get(mname)
        if task == "regression":
            metrics, fitted, pred = evaluate_regression(mname, model, X_train, y_train, X_test, y_test, sample_weight=sw)
        else:
            metrics, fitted = evaluate_classification(mname, model, X_train, y_train, X_test, y_test, sample_weight=sw)
            pred = None
        results.append(metrics)
        fitted_models[mname] = fitted
        preds[mname] = pred

    # "Predict yesterday's value" is a famously strong finance baseline -- add
    # a momentum-persistence baseline alongside the zero-return/majority-class
    # ones. Uses the FULL (pre-feature-selection) frame so "ret_1d" is always
    # present, since feature selection could otherwise have dropped it.
    if task == "regression":
        metrics, fitted, pred = evaluate_regression(
            "baseline_persistence", PersistenceRegressor(), X_train_full, y_train, X_test_full, y_test
        )
    else:
        metrics, fitted = evaluate_classification(
            "baseline_persistence", PersistenceClassifier(), X_train_full, y_train, X_test_full, y_test
        )
        pred = None
    results.append(metrics)
    fitted_models["baseline_persistence"] = fitted
    preds["baseline_persistence"] = pred

    # Only the "know-nothing" floors (predict zero / predict the majority
    # class) are excluded from actually winning -- they exist purely to show
    # the bar. baseline_persistence ("tomorrow repeats today") is a real
    # forecasting strategy and is deliberately left eligible to win: if a
    # one-line momentum rule beats every tuned/ensembled model on held-out
    # data, deploying the complex model anyway would mean shipping something
    # provably worse than a dumb baseline, just because it's fancier.
    is_floor_baseline = lambda n: n in ("baseline_zero_return", "baseline_majority")
    metric_key = "rmse" if task == "regression" else "balanced_accuracy"
    better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)

    # Selection-time walk-forward scoring: picking the winner by the SAME
    # single window that then gets quoted as its "held-out" score is a
    # textbook selection-bias setup (the winner's curse) -- whichever
    # candidate got lucky on that one window wins, and its luck becomes the
    # reported number. Score every non-floor candidate across
    # WALK_FORWARD_STABILITY_FOLDS expanding folds carved ONLY from the
    # training region (X_train_full/y_train) -- X_test_full/y_test are never
    # touched here, so the single-window metric computed above stays a
    # genuine, untouched confirmation number instead of also being what
    # picked the winner.
    for r in results:
        mname = r["model"]
        if is_floor_baseline(mname):
            r["selection_fold_mean"] = r["selection_fold_std"] = r["selection_n_folds"] = None
            continue
        cols = ["ret_1d"] if mname == "baseline_persistence" else selected_cols
        wf = walk_forward_stability(X_train_full, y_train, task, mname, fitted_models[mname], cols)
        r["selection_fold_mean"] = wf["mean"]
        r["selection_fold_std"] = wf["std"]
        r["selection_n_folds"] = wf["n_folds"]

    non_floor = [r for r in results if not is_floor_baseline(r["model"])]
    any_fold_based = any(r["selection_fold_mean"] is not None for r in non_floor)
    selection_basis = "walk_forward" if any_fold_based else "single_holdout_fallback"

    best_name, best_key = None, (np.inf if task == "regression" else -np.inf)
    for r in non_floor:
        r["selection_basis"] = selection_basis
        if any_fold_based:
            wf = {"mean": r["selection_fold_mean"], "std": r["selection_fold_std"]}
            key = _selection_score(wf, task)
            if key is None:
                continue  # too little training-region data to fold this candidate fairly this round
        else:
            key = r[metric_key]
        if better(key, best_key):
            best_key, best_name = key, r["model"]

    # best_val is deliberately the single-window metric (results[best_name][metric_key]),
    # NOT best_key -- it's now a confirmation number the winner never got picked by,
    # not the thing that decided the winner.
    best_val = next(r[metric_key] for r in results if r["model"] == best_name)
    winning_selection = next(r for r in results if r["model"] == best_name)

    # baseline_persistence was fit on X_*_full and only ever looks at "ret_1d" --
    # it must be served that column at inference time regardless of what the
    # OTHER candidates' feature selection kept, or a later predict() KeyErrors
    # on a column that got dropped from the top-K selection.
    winning_feature_columns = ["ret_1d"] if best_name == "baseline_persistence" else selected_cols

    return {
        "results": results,
        "best_name": best_name,
        "best_model": fitted_models[best_name],
        "best_pred": preds.get(best_name),
        "best_metric": best_val,
        "selection_fold_mean": winning_selection["selection_fold_mean"],
        "selection_fold_std": winning_selection["selection_fold_std"],
        "selection_basis": selection_basis,
        "feature_columns": winning_feature_columns,
        "tuned_params": tuned_params,
    }


def walk_forward_stability(X: pd.DataFrame, y: pd.Series, task: str, model_name: str,
                            model_template, feature_columns: list[str],
                            n_folds: int = WALK_FORWARD_STABILITY_FOLDS) -> dict:
    """Answers a question a single train/test split cannot: is a metric a
    stable, time-consistent effect, or does it swing around depending on
    which window happens to get tested? Re-fits the given model -- same
    class and hyperparameters, via sklearn's clone() -- across several
    expanding chronological windows of whatever (X, y) the caller passes in,
    and reports the metric's mean and standard deviation across folds.

    Two call sites, same mechanism, different data slice:
      - run_task() calls this on EVERY non-floor candidate, fed only the
        training region (X_train_full/y_train) -- this is what now DECIDES
        the winner (via _selection_score()), fixing the selection bias of
        picking a winner by the same window later quoted as its score. See
        README: PLTR's classifier looked like 58.7% balanced accuracy on a
        single window; walk-forward checking found 52.0% +/- 3.3%.
      - train_for_ticker(), post-selection, calls this again on the
        ALREADY-CHOSEN model over its FULL history (train+test) purely to
        report how stable the shipped model is over time -- this second use
        never feeds back into what's shipped, it's a confirmation report.
    A high std relative to the mean is itself an important, honest finding
    either way, not a bug to explain away."""
    Xc = X[feature_columns]
    splitter = TimeSeriesSplit(n_splits=n_folds)
    fold_scores = []
    for train_idx, test_idx in splitter.split(Xc):
        if len(test_idx) < 10:
            continue
        X_train, X_test = Xc.iloc[train_idx], Xc.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = clone(model_template)
        sw = recency_weights(len(X_train)) if "recency_weighted" in model_name else None
        if task == "regression":
            metrics, _, _ = evaluate_regression(model_name, model, X_train, y_train, X_test, y_test, sample_weight=sw)
            fold_scores.append(metrics["rmse"])
        else:
            metrics, _ = evaluate_classification(model_name, model, X_train, y_train, X_test, y_test, sample_weight=sw)
            fold_scores.append(metrics["balanced_accuracy"])

    if not fold_scores:
        return {"n_folds": 0, "mean": None, "std": None, "fold_scores": []}
    return {
        "n_folds": len(fold_scores),
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
    }


def _feature_set_selection_key(task_result: dict, task: str) -> float:
    """Same fix as run_task()'s model selection, one level up: pick which
    feature set (baseline / +macro / +macro+news) wins using the same
    walk-forward score that picked the winning model inside it, not the
    single-window confirmation number -- otherwise the selection bias just
    moves up one level instead of actually going away."""
    if task_result["selection_basis"] == "walk_forward":
        wf = {"mean": task_result["selection_fold_mean"], "std": task_result["selection_fold_std"]}
        score = _selection_score(wf, task)
        if score is not None:
            return score
    return task_result["best_metric"]


def run_feature_set(name: str, df: pd.DataFrame, macro_df, feature_columns: list[str], label: str, news_df=None):
    X, y_ret, y_dir, close = make_dataset(df, macro_df=macro_df, news_df=news_df, feature_columns=feature_columns)
    parts, split = chrono_split(X, y_ret, y_dir)
    (X_train, X_test), (y_ret_train, y_ret_test), (y_dir_train, y_dir_test) = parts
    close_test = close.iloc[split:]

    print(f"  [{label}] rows: total={len(X)}  train={len(X_train)}  test={len(X_test)}  "
          f"features={len(feature_columns)}  "
          f"({X_train.index.min().date()}..{X_train.index.max().date()} / "
          f"{X_test.index.min().date()}..{X_test.index.max().date()})")

    reg = run_task(X_train, y_ret_train, X_test, y_ret_test, "regression", label)
    reg_persist_rmse = next(r for r in reg["results"] if r["model"] == "baseline_persistence")["rmse"]
    print(f"  [{label}] best regressor: {reg['best_name']} (rmse={reg['best_metric']:.5f} vs. "
          f"persistence baseline {reg_persist_rmse:.5f}, {len(reg['feature_columns'])} features)")

    clf = run_task(X_train, y_dir_train, X_test, y_dir_test, "classification", label)
    clf_row = next(r for r in clf["results"] if r["model"] == clf["best_name"])
    clf_persist_bal_acc = next(r for r in clf["results"] if r["model"] == "baseline_persistence")["balanced_accuracy"]
    print(f"  [{label}] best classifier: {clf['best_name']} (balanced_accuracy={clf['best_metric']:.4f} vs. "
          f"persistence baseline {clf_persist_bal_acc:.4f}, accuracy={clf_row['accuracy']:.4f}, "
          f"predicts UP {clf_row['pct_predicted_up']:.0%} of the time, {len(clf['feature_columns'])} features)")

    return {
        "label": label,
        "reg": reg, "clf": clf,
        "X_test": X_test, "close_test": close_test,
        "X": X, "y_ret": y_ret, "y_dir": y_dir,
    }


def train_for_ticker(name: str, df: pd.DataFrame, macro_df: pd.DataFrame, news_df: pd.DataFrame, summary_rows: list,
                      stability_rows: list):
    print(f"\n=== {name} ===")
    baseline = run_feature_set(name, df, None, OWN_FEATURE_COLUMNS, "baseline: own technical features only")
    enhanced = run_feature_set(name, df, macro_df, FEATURE_COLUMNS, "enhanced: + macro/cross-market features")
    with_news = run_feature_set(name, df, macro_df, FEATURE_COLUMNS_WITH_NEWS,
                                 "with_news: + macro + NYT news sentiment", news_df=news_df)

    candidates = [baseline, enhanced, with_news]
    reg_pick = min(candidates, key=lambda c: _feature_set_selection_key(c["reg"], "regression"))
    clf_pick = max(candidates, key=lambda c: _feature_set_selection_key(c["clf"], "classification"))

    print(f"  >> regressor RMSE:              baseline={baseline['reg']['best_metric']:.5f}  "
          f"macro={enhanced['reg']['best_metric']:.5f}  "
          f"macro+news={with_news['reg']['best_metric']:.5f}  -- keeping '{reg_pick['label']}'")
    print(f"  >> classifier balanced accuracy: baseline={baseline['clf']['best_metric']:.1%}  "
          f"macro={enhanced['clf']['best_metric']:.1%}  "
          f"macro+news={with_news['clf']['best_metric']:.1%}  -- keeping '{clf_pick['label']}'")

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

    reg_stability = walk_forward_stability(
        reg_pick["X"], reg_pick["y_ret"], "regression",
        reg_pick["reg"]["best_name"], reg_pick["reg"]["best_model"], reg_pick["reg"]["feature_columns"],
    )
    clf_stability = walk_forward_stability(
        clf_pick["X"], clf_pick["y_dir"], "classification",
        clf_pick["clf"]["best_name"], clf_pick["clf"]["best_model"], clf_pick["clf"]["feature_columns"],
    )
    if reg_stability["n_folds"]:
        print(f"  walk-forward stability (regressor, {reg_stability['n_folds']} folds over full history): "
              f"RMSE {reg_stability['mean']:.5f} +/- {reg_stability['std']:.5f}  "
              f"(single-window: {reg_pick['reg']['best_metric']:.5f})")
    if clf_stability["n_folds"]:
        print(f"  walk-forward stability (classifier, {clf_stability['n_folds']} folds over full history): "
              f"balanced accuracy {clf_stability['mean']:.1%} +/- {clf_stability['std']:.1%}  "
              f"(single-window: {clf_pick['clf']['best_metric']:.1%})")
    stability_rows.append({
        "ticker": name, "task": "regression", "model": reg_pick["reg"]["best_name"],
        "single_window_metric": reg_pick["reg"]["best_metric"],
        "walk_forward_mean": reg_stability["mean"], "walk_forward_std": reg_stability["std"],
        "n_folds": reg_stability["n_folds"],
    })
    stability_rows.append({
        "ticker": name, "task": "classification", "model": clf_pick["clf"]["best_name"],
        "single_window_metric": clf_pick["clf"]["best_metric"],
        "walk_forward_mean": clf_stability["mean"], "walk_forward_std": clf_stability["std"],
        "n_folds": clf_stability["n_folds"],
    })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = REPORTS_DIR / f"{name}_predictions.png"
    plot_predictions(name, reg_pick["X_test"].index, reg_pick["close_test"], reg_pick["reg"]["best_pred"], plot_path)
    print(f"  saved plot to reports/{plot_path.name}")

    for fs_label, fs in (("baseline", baseline), ("with_macro", enhanced), ("with_macro_and_news", with_news)):
        for r in fs["reg"]["results"]:
            summary_rows.append({"ticker": name, "feature_set": fs_label, "task": "regression", **r})
        for r in fs["clf"]["results"]:
            summary_rows.append({"ticker": name, "feature_set": fs_label, "task": "classification", **r})

    return {
        "ticker": name,
        "baseline_reg_rmse": baseline["reg"]["best_metric"], "with_macro_reg_rmse": enhanced["reg"]["best_metric"],
        "with_news_reg_rmse": with_news["reg"]["best_metric"],
        "baseline_clf_balanced_acc": baseline["clf"]["best_metric"], "with_macro_clf_balanced_acc": enhanced["clf"]["best_metric"],
        "with_news_clf_balanced_acc": with_news["clf"]["best_metric"],
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

    any_ticker_df = next(iter(data.values()))
    news_start = any_ticker_df.index.min().strftime("%Y-%m")
    news_end = any_ticker_df.index.max().strftime("%Y-%m")
    print(f"Fetching NYT news archive ({news_start}..{news_end}, cached locally after first run)...")
    news_daily = build_news_history(news_start, news_end)

    summary_rows = []
    stability_rows = []
    comparisons = []
    for name in TICKERS.values():
        news_df = news_features_for_ticker(news_daily, name)
        comparisons.append(train_for_ticker(name, data[name], macro_df, news_df, summary_rows, stability_rows))

    summary = pd.DataFrame(summary_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "model_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved full model comparison table to {summary_path}")

    stability = pd.DataFrame(stability_rows)
    stability_path = REPORTS_DIR / "walk_forward_stability.csv"
    stability.to_csv(stability_path, index=False)
    print(f"Saved walk-forward stability table to {stability_path}")

    comp_df = pd.DataFrame(comparisons).set_index("ticker")
    print("\n=== Final result per ticker ===")
    print(comp_df.to_string())

    print(f"\n=== Walk-forward stability ({WALK_FORWARD_STABILITY_FOLDS} folds each, full history) ===")
    print("(is the single-window metric above a stable effect, or does it swing across time windows?)")
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
