"""Probabilistic forecasting: an ensemble of statistical, tree-based and neural models.

Each model estimates P(price goes up by more than trading costs within `horizon` bars).
The ensemble averages the probabilities. A rule-based SMA-crossover baseline is kept
for comparison, so the calibration report can show whether the ML actually adds value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class _ConstantModel:
    """Fallback when the training labels contain only one class."""

    def fit(self, X, y):
        self.p = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


class Ensemble:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models: dict[str, object] = {
            # statistical / probabilistic: regularised logistic regression
            "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
            # gradient-boosted trees (non-linear interactions)
            "gbm": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=seed),
            # small neural network (deep-learning style, scikit-learn implementation)
            "mlp": make_pipeline(StandardScaler(),
                                 MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                               early_stopping=True, max_iter=300,
                                               random_state=seed)),
        }
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ensemble":
        yv = np.asarray(y).astype(int)
        if len(np.unique(yv)) < 2:
            self.models = {"const": _ConstantModel().fit(X, yv)}
        else:
            for m in self.models.values():
                m.fit(X.values, yv)
        self.fitted = True
        return self

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: m.predict_proba(X.values)[:, 1] for name, m in self.models.items()},
                            index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Ensemble probability of an upward move (simple average of model probabilities)."""
        if not self.fitted:
            raise RuntimeError("model not fitted")
        return self.predict_components(X).mean(axis=1)


def sma_crossover_prob(features: pd.DataFrame, bull: float = 0.65, bear: float = 0.35) -> pd.Series:
    """Classic rule-based baseline expressed as a 'probability': bullish when fast SMA > slow SMA."""
    trend = features["sma10_sma30"]
    return pd.Series(np.where(trend > 0, bull, bear), index=features.index)
