"""
selectors/graph_based.py
------------------------
Graph-Laplacian and spectral feature selectors from scikit-feature.
"""

import numpy as np
import pandas as pd
from .base import BaseSelector


def _check_skfeature():
    try:
        import skfeature  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scikit-feature is required for graph-based selectors.  "
            "Install it with:\n"
            "  pip install git+https://github.com/jundongl/scikit-feature.git"
        ) from e


class LaplacianScoreSelector(BaseSelector):
    """Laplacian Score — preserves local geometric structure."""
    def _fit(self, X, y):
        _check_skfeature()
        from skfeature.function.similarity_based import lap_score
        from skfeature.utility import construct_W
        kwargs_W = dict(
            metric="euclidean",
            neighborMode="knn",
            weightMode="heatKernel",
            k=self.kwargs.get("k", 5),
            t=self.kwargs.get("t", 1),
        )
        W = construct_W.construct_W(X.values, **kwargs_W)
        scores = lap_score.lap_score(X.values, W=W)
        # Lower Laplacian score = better; invert for ranking
        self._rank_by_importance(-scores)


class SPECSelector(BaseSelector):
    """SPEC — spectral feature selection."""
    def _fit(self, X, y):
        _check_skfeature()
        from skfeature.function.similarity_based import SPEC
        from skfeature.utility import construct_W
        W = construct_W.construct_W(X.values, k=self.kwargs.get("k", 5))
        scores = SPEC.spec(X.values, W=W)
        self._rank_by_importance(-scores)  # lower = better


class MCFSSelector(BaseSelector):
    """MCFS — Multi-Cluster Feature Selection via sparse regression."""
    def _fit(self, X, y):
        _check_skfeature()
        from skfeature.function.sparse_learning_based import MCFS
        n_selected = self.n_features or min(200, X.shape[1])
        score = MCFS.mcfs(
            X.values,
            n_selected_features=n_selected,
            clusters=self.kwargs.get("clusters", len(np.unique(y))),
            k=self.kwargs.get("k", 5),
        )
        self._rank_by_importance(score)
