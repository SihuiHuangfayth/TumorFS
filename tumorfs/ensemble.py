"""
ensemble.py
-----------
Combine rankings from multiple selectors into a single robust gene ranking.

Three aggregation strategies are supported:

* ``"rank_aggregation"``  (default) — Borda count: convert each selector's
  per-gene rank into a score (higher rank → higher score), sum across
  selectors, and sort descending.

* ``"vote"`` — majority vote: a gene is kept if it appears in the top-N list
  of at least ``min_votes`` selectors.

* ``"weighted"`` — like rank_aggregation but each selector is weighted by its
  standalone cross-validation accuracy on the training data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from .selectors.base import BaseSelector, FeatureSelector, AVAILABLE_METHODS

logger = logging.getLogger(__name__)


class EnsembleRanker:
    """
    Run multiple feature selectors and combine their rankings.

    Parameters
    ----------
    methods : list of str
        Selector names (see ``tumorfs.selectors.AVAILABLE_METHODS``).
    n_features : int or None
        How many top features each individual selector considers.
    strategy : {'rank_aggregation', 'vote', 'weighted'}
        How to combine individual rankings.
    min_votes : int
        For ``strategy='vote'``: minimum number of selectors that must agree.
    weights : dict or None
        For ``strategy='weighted'``: manual weights per method.
        If None and ``strategy='weighted'``, weights are estimated by CV.
    random_state : int
    n_jobs : int
        Parallel selector fitting (-1 = all cores).
    selector_kwargs : dict
        Per-method extra kwargs, e.g.
        ``{"lightgbm": {"n_estimators": 300}, "lasso": {"C": 0.05}}``.

    Examples
    --------
    >>> ranker = EnsembleRanker(
    ...     methods=["lightgbm", "relieff", "lasso", "laplacian_score"],
    ...     n_features=100,
    ...     strategy="rank_aggregation",
    ... )
    >>> ranker.fit(X_train, y_train)
    >>> print(ranker.top_features_[:10])
    """

    def __init__(
        self,
        methods: List[str],
        n_features: Optional[int] = 200,
        strategy: str = "rank_aggregation",
        min_votes: int = 2,
        weights: Optional[Dict[str, float]] = None,
        random_state: int = 42,
        n_jobs: int = 1,
        selector_kwargs: Optional[Dict[str, dict]] = None,
    ):
        unknown = set(methods) - set(AVAILABLE_METHODS)
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}.  Available: {AVAILABLE_METHODS}")

        self.methods = methods
        self.n_features = n_features
        self.strategy = strategy
        self.min_votes = min_votes
        self.weights = weights or {}
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selector_kwargs = selector_kwargs or {}

        # Populated after fit()
        self.selectors_: Dict[str, BaseSelector] = {}
        self.rankings_: Optional[pd.DataFrame] = None   # genes × methods
        self.ensemble_score_: Optional[pd.Series] = None
        self.top_features_: Optional[List[str]] = None
        self.method_weights_: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleRanker":
        """Fit all selectors and compute ensemble ranking."""
        self._fit_selectors(X, y)

        rankings_dict = {}
        for name, sel in self.selectors_.items():
            if sel.ranked_features_ is not None:
                rankings_dict[name] = sel.get_ranking()

        # Align on all genes
        self.rankings_ = pd.DataFrame(rankings_dict)  # genes × methods

        if self.strategy == "rank_aggregation":
            self.ensemble_score_ = self._borda_count(self.rankings_)
        elif self.strategy == "vote":
            self.ensemble_score_ = self._vote(self.rankings_)
        elif self.strategy == "weighted":
            weights = self._estimate_weights(X, y) if not self.weights else self.weights
            self.method_weights_ = weights
            self.ensemble_score_ = self._weighted_borda(self.rankings_, weights)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'.")

        self.top_features_ = (
            self.ensemble_score_.sort_values(ascending=False).index.tolist()
        )
        logger.info(
            "EnsembleRanker fitted with %d methods (%s).  Top gene: %s",
            len(self.selectors_),
            self.strategy,
            self.top_features_[0] if self.top_features_ else "N/A",
        )
        return self

    # ------------------------------------------------------------------
    def get_top_features(self, n: int) -> List[str]:
        """Return the top-n genes from the ensemble ranking."""
        if self.top_features_ is None:
            raise RuntimeError("Call fit() first.")
        return self.top_features_[:n]

    def get_ranking_table(self) -> pd.DataFrame:
        """
        Return a DataFrame with ensemble score, per-method ranks, and vote count.
        """
        df = self.rankings_.copy()
        df["ensemble_score"] = self.ensemble_score_
        df["vote_count"] = (df[self.methods] <= (self.n_features or len(df))).sum(axis=1)
        return df.sort_values("ensemble_score", ascending=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_selectors(self, X, y):
        """Fit each selector, optionally in parallel."""
        if self.n_jobs == 1:
            for name in self.methods:
                logger.info("Fitting %s …", name)
                kwargs = self.selector_kwargs.get(name, {})
                try:
                    sel = FeatureSelector(
                        name,
                        n_features=self.n_features,
                        random_state=self.random_state,
                        **kwargs,
                    )
                    sel.fit(X, y)
                    self.selectors_[name] = sel._selector
                except Exception as exc:
                    logger.warning("Selector '%s' failed: %s — skipped.", name, exc)
        else:
            from joblib import Parallel, delayed

            def _fit_one(name):
                kwargs = self.selector_kwargs.get(name, {})
                try:
                    sel = FeatureSelector(
                        name,
                        n_features=self.n_features,
                        random_state=self.random_state,
                        **kwargs,
                    )
                    sel.fit(X, y)
                    return name, sel._selector
                except Exception as exc:
                    logger.warning("Selector '%s' failed: %s — skipped.", name, exc)
                    return name, None

            results = Parallel(n_jobs=self.n_jobs)(delayed(_fit_one)(m) for m in self.methods)
            self.selectors_ = {n: s for n, s in results if s is not None}

    @staticmethod
    def _borda_count(rankings: pd.DataFrame) -> pd.Series:
        """Higher rank = higher Borda score.  Max rank = n_genes."""
        n = len(rankings)
        # Invert: rank 1 → score n, rank n → score 1; NaN → 0
        scores = rankings.apply(lambda col: (n + 1 - col).fillna(0))
        return scores.sum(axis=1)

    def _vote(self, rankings: pd.DataFrame) -> pd.Series:
        """Count how many selectors placed each gene in the top-N list."""
        threshold = self.n_features or len(rankings)
        votes = (rankings <= threshold).sum(axis=1)
        return votes.astype(float)

    @staticmethod
    def _weighted_borda(rankings: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        n = len(rankings)
        scores = pd.Series(0.0, index=rankings.index)
        for method, col in rankings.items():
            w = weights.get(method, 1.0)
            scores += w * (n + 1 - col).fillna(0)
        return scores

    def _estimate_weights(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Estimate per-method weight via 5-fold CV accuracy on top-50 genes."""
        weights: Dict[str, float] = {}
        clf = SVC(kernel="linear", random_state=self.random_state)
        for name, sel in self.selectors_.items():
            try:
                top = (sel.selected_features_ or sel.ranked_features_)[:50]
                score = cross_val_score(clf, X[top], y, cv=5, scoring="accuracy").mean()
                weights[name] = max(score, 0.01)
            except Exception:
                weights[name] = 1.0
        logger.info("Estimated method weights: %s", weights)
        return weights
