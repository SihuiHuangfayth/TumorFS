"""
selectors/wrapper.py
--------------------
Wrapper and hybrid feature selectors: ReliefF, Leshy, Boruta, RFE.
"""

import numpy as np
import pandas as pd
from .base import BaseSelector


class ReliefFSelector(BaseSelector):
    """ReliefF — instance-based, handles multiclass and interactions."""
    def _fit(self, X, y):
        from skrebate import ReliefF
        k = min(self.kwargs.get("n_neighbors", 10), len(X) - 1)
        model = ReliefF(n_neighbors=k, n_jobs=-1)
        model.fit(X.values, y.values)
        self._rank_by_importance(model.feature_importances_)


class LeshySelector(BaseSelector):
    """
    Leshy — Boruta-style all-relevant selection from the ARFS package.
    """
    def _fit(self, X, y):
        try:
            from arfs.feature_selection import Leshy
        except ImportError as e:
            raise ImportError(
                "arfs is required for LeshySelector.  pip install arfs"
            ) from e
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(
            n_estimators=self.kwargs.get("n_estimators", 100),
            random_state=self.random_state,
            n_jobs=-1,
        )
        sel = Leshy(
            estimator=estimator,
            n_estimators="auto",
            max_iter=self.kwargs.get("max_iter", 30),
            random_state=self.random_state,
            verbose=0,
        )
        sel.fit(X, y)
        # Leshy exposes selected features via support_
        selected = X.columns[sel.support_].tolist()
        # Build importances: 1 if selected, 0 otherwise
        importances = np.array([1.0 if c in selected else 0.0 for c in X.columns])
        self._rank_by_importance(importances)


class BorutaSelector(BaseSelector):
    """Boruta — shadow-feature permutation wrapper around Random Forest."""
    def _fit(self, X, y):
        try:
            from boruta import BorutaPy
        except ImportError as e:
            raise ImportError(
                "boruta is required for BorutaSelector.  pip install boruta"
            ) from e
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_jobs=-1,
            max_depth=self.kwargs.get("max_depth", 5),
            random_state=self.random_state,
        )
        sel = BorutaPy(
            rf,
            n_estimators=self.kwargs.get("n_estimators", "auto"),
            max_iter=self.kwargs.get("max_iter", 50),
            random_state=self.random_state,
            verbose=0,
        )
        sel.fit(X.values, y.values)
        # Use Boruta's ranking directly (lower rank = more important)
        boruta_ranking = sel.ranking_  # shape (n_features,)
        # Invert so that rank 1 has highest importance
        importances = 1.0 / boruta_ranking
        self._rank_by_importance(importances)


class RFESelector(BaseSelector):
    """
    Recursive Feature Elimination with a configurable base estimator.

    Parameters (via kwargs)
    -----------------------
    estimator : sklearn estimator
        Defaults to RandomForestClassifier.
    step : int or float
        Number or fraction of features removed at each step.
    """
    def _fit(self, X, y):
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        estimator = self.kwargs.get(
            "estimator",
            RandomForestClassifier(n_jobs=-1, random_state=self.random_state),
        )
        n_sel = self.n_features or max(1, X.shape[1] // 2)
        sel = RFE(
            estimator=estimator,
            n_features_to_select=n_sel,
            step=self.kwargs.get("step", 0.05),
        )
        sel.fit(X.values, y.values)
        # ranking_: 1 = selected, higher = eliminated earlier
        importances = 1.0 / sel.ranking_
        self._rank_by_importance(importances)
