"""
selectors/base.py
-----------------
Unified FeatureSelector factory and base class.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Registry: method name → (module, class)
_SELECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    # Tree-based
    "lightgbm":          ("tumorfs.selectors.tree_based",  "LightGBMSelector"),
    "xgboost":           ("tumorfs.selectors.tree_based",  "XGBoostSelector"),
    "catboost":          ("tumorfs.selectors.tree_based",  "CatBoostSelector"),
    "random_forest":     ("tumorfs.selectors.tree_based",  "RandomForestSelector"),
    "adaboost":          ("tumorfs.selectors.tree_based",  "AdaBoostSelector"),
    "decision_tree":     ("tumorfs.selectors.tree_based",  "DecisionTreeSelector"),
    "extra_trees":       ("tumorfs.selectors.tree_based",  "ExtraTreesSelector"),
    "gradient_boosting": ("tumorfs.selectors.tree_based",  "GradientBoostingSelector"),
    # Statistical / filter
    "fisher_score":      ("tumorfs.selectors.statistical", "FisherScoreSelector"),
    "f_score":           ("tumorfs.selectors.statistical", "FScoreSelector"),
    "mutual_info":       ("tumorfs.selectors.statistical", "MutualInfoSelector"),
    "chi2":              ("tumorfs.selectors.statistical", "ChiSquareSelector"),
    "lasso":             ("tumorfs.selectors.statistical", "LassoSelector"),
    "elastic_net":       ("tumorfs.selectors.statistical", "ElasticNetSelector"),
    "linear_regression": ("tumorfs.selectors.statistical", "LinearRegressionSelector"),
    # Graph-based
    "laplacian_score":   ("tumorfs.selectors.graph_based", "LaplacianScoreSelector"),
    "spec":              ("tumorfs.selectors.graph_based", "SPECSelector"),
    "mcfs":              ("tumorfs.selectors.graph_based", "MCFSSelector"),
    # Wrapper
    "relieff":           ("tumorfs.selectors.wrapper",     "ReliefFSelector"),
    "leshy":             ("tumorfs.selectors.wrapper",     "LeshySelector"),
    "boruta":            ("tumorfs.selectors.wrapper",     "BorutaSelector"),
    "rfe":               ("tumorfs.selectors.wrapper",     "RFESelector"),
}

AVAILABLE_METHODS = sorted(_SELECTOR_REGISTRY.keys())


class BaseSelector(ABC):
    """
    Abstract base class that every selector must implement.

    Subclasses only need to implement ``_fit`` and set
    ``self.feature_importances_`` (shape = n_features,) and
    ``self.ranked_features_`` (list of feature names, best first).
    """

    def __init__(self, n_features: Optional[int] = None, random_state: int = 42, **kwargs):
        self.n_features = n_features
        self.random_state = random_state
        self.kwargs = kwargs
        self.feature_importances_: Optional[np.ndarray] = None
        self.ranked_features_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSelector":
        self._feature_names = list(X.columns)
        self._fit(X, y)
        # Derive selected_features_ from ranked list
        n = self.n_features or len(self.ranked_features_)
        self.selected_features_ = self.ranked_features_[:n]
        logger.info(
            "%s selected %d / %d features.",
            self.__class__.__name__,
            len(self.selected_features_),
            len(self._feature_names),
        )
        return self

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute importances and set self.ranked_features_."""

    def _rank_by_importance(self, importances: np.ndarray) -> None:
        """Helper: sort feature names by descending importance."""
        self.feature_importances_ = importances
        order = np.argsort(importances)[::-1]
        self.ranked_features_ = [self._feature_names[i] for i in order]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() before transform().")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_ranking(self) -> pd.Series:
        """Return a Series mapping gene → rank (1 = best)."""
        return pd.Series(
            range(1, len(self.ranked_features_) + 1),
            index=self.ranked_features_,
            name=self.__class__.__name__,
        )


class FeatureSelector:
    """
    Factory / facade that instantiates the correct selector by name.

    Parameters
    ----------
    method : str
        One of :data:`AVAILABLE_METHODS`.
    n_features : int or None
        Number of top features to keep.
    random_state : int
    **kwargs
        Passed verbatim to the underlying selector.

    Examples
    --------
    >>> fs = FeatureSelector("lightgbm", n_features=100)
    >>> fs.fit(X_train, y_train)
    >>> top_genes = fs.selected_features_
    """

    def __init__(
        self,
        method: str,
        n_features: Optional[int] = None,
        random_state: int = 42,
        **kwargs,
    ):
        if method not in _SELECTOR_REGISTRY:
            raise ValueError(
                f"Unknown method '{method}'.  "
                f"Available: {AVAILABLE_METHODS}"
            )
        self.method = method
        self._selector = self._build(method, n_features, random_state, **kwargs)

    @staticmethod
    def _build(method, n_features, random_state, **kwargs) -> BaseSelector:
        import importlib
        mod_name, cls_name = _SELECTOR_REGISTRY[method]
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(n_features=n_features, random_state=random_state, **kwargs)

    # Delegate everything to the inner selector
    def fit(self, X, y):
        self._selector.fit(X, y)
        return self

    def transform(self, X):
        return self._selector.transform(X)

    def fit_transform(self, X, y):
        return self._selector.fit_transform(X, y)

    def get_ranking(self):
        return self._selector.get_ranking()

    @property
    def selected_features_(self):
        return self._selector.selected_features_

    @property
    def ranked_features_(self):
        return self._selector.ranked_features_

    @property
    def feature_importances_(self):
        return self._selector.feature_importances_
