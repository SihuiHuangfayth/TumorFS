"""
selectors/statistical.py
------------------------
Statistical / filter-based feature selectors.
"""

import numpy as np
import pandas as pd
from .base import BaseSelector


class FisherScoreSelector(BaseSelector):
    """Fisher score using scikit-feature (skfeature)."""
    def _fit(self, X, y):
        try:
            from skfeature.function.similarity_based import fisher_score
            scores = fisher_score.fisher_score(X.values, y.values)
        except ImportError:
            # Fallback: manual Fisher criterion
            scores = self._manual_fisher(X.values, y.values)
        self._rank_by_importance(scores)

    @staticmethod
    def _manual_fisher(X, y):
        classes = np.unique(y)
        overall_mean = X.mean(axis=0)
        between = np.zeros(X.shape[1])
        within = np.zeros(X.shape[1])
        for c in classes:
            Xc = X[y == c]
            nc = len(Xc)
            diff = Xc.mean(axis=0) - overall_mean
            between += nc * diff ** 2
            within += Xc.var(axis=0) * nc
        within = np.where(within == 0, 1e-10, within)
        return between / within


class FScoreSelector(BaseSelector):
    """ANOVA F-value (sklearn SelectKBest wrapper)."""
    def _fit(self, X, y):
        from sklearn.feature_selection import f_classif
        scores, _ = f_classif(X.values, y.values)
        scores = np.nan_to_num(scores, nan=0.0)
        self._rank_by_importance(scores)


class MutualInfoSelector(BaseSelector):
    """Mutual information between features and target."""
    def _fit(self, X, y):
        from sklearn.feature_selection import mutual_info_classif
        scores = mutual_info_classif(
            X.values, y.values,
            random_state=self.random_state,
            n_neighbors=self.kwargs.get("n_neighbors", 3),
        )
        self._rank_by_importance(scores)


class ChiSquareSelector(BaseSelector):
    """Chi-square test (requires non-negative features)."""
    def _fit(self, X, y):
        from sklearn.feature_selection import chi2
        # Shift negative values to make all values non-negative
        X_nn = X.values - X.values.min()
        scores, _ = chi2(X_nn, y.values)
        scores = np.nan_to_num(scores, nan=0.0)
        self._rank_by_importance(scores)


class LassoSelector(BaseSelector):
    """L1-regularised logistic regression; |coef| as importance."""
    def _fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X.values)
        model = LogisticRegression(
            penalty="l1",
            C=self.kwargs.get("C", 0.1),
            solver="liblinear",
            max_iter=self.kwargs.get("max_iter", 1000),
            random_state=self.random_state,
        )
        model.fit(Xs, y.values)
        importances = np.abs(model.coef_).mean(axis=0)
        self._rank_by_importance(importances)


class ElasticNetSelector(BaseSelector):
    """ElasticNet (L1+L2); |coef| as importance."""
    def _fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X.values)
        model = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=self.kwargs.get("l1_ratio", 0.5),
            C=self.kwargs.get("C", 0.1),
            solver="saga",
            max_iter=self.kwargs.get("max_iter", 2000),
            random_state=self.random_state,
        )
        model.fit(Xs, y.values)
        importances = np.abs(model.coef_).mean(axis=0)
        self._rank_by_importance(importances)


class LinearRegressionSelector(BaseSelector):
    """Linear regression coefficient magnitude as importance."""
    def _fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X.values)
        model = LinearRegression()
        model.fit(Xs, y.values)
        self._rank_by_importance(np.abs(model.coef_))
