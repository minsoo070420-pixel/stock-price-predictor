"""Finance-specific dumb baselines, kept in their own module (not inside
train.py) so joblib can unpickle a saved model built from one of these
classes regardless of which script loads it later -- pickle resolves a class
by its module path, so PersistenceRegressor must live somewhere importable
under the same name from any caller, not inside train.py's __main__ context.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class PersistenceRegressor(BaseEstimator, RegressorMixin):
    """Predicts tomorrow's return = today's return (momentum continuation).
    "Predict yesterday's value" is a famously strong baseline in finance --
    this is that idea applied to returns instead of price levels (predicting
    the raw price itself would trivially "win" by tracking the drift, which is
    exactly the trap of evaluating on price levels instead of returns)."""

    def __init__(self, feature_name: str = "ret_1d"):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.feature_name].to_numpy()


class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """Predicts tomorrow's direction = today's direction."""

    def __init__(self, feature_name: str = "ret_1d"):
        self.feature_name = feature_name
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return (X[self.feature_name].to_numpy() > 0).astype(int)

    def predict_proba(self, X):
        pred = self.predict(X)
        proba = np.full((len(pred), 2), 0.02)
        proba[np.arange(len(pred)), pred] = 0.98
        return proba
