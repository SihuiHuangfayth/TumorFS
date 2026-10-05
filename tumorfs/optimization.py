"""
optimization.py
---------------
Hyperparameter search and algorithm evaluation utilities.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


def optimize_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    param_grid: Dict,
    n_features_list: Optional[List[int]] = None,
    cv: int = 5,
    classifier=None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Grid-search over ``param_grid`` for a single selector, evaluating
    downstream classifier CV accuracy for each parameter combination.

    Parameters
    ----------
    X              : pd.DataFrame
    y              : pd.Series
    method         : str  selector name
    param_grid     : dict  {param_name: [values]}
    n_features_list : list of int  gene-subset sizes to evaluate
    cv             : int
    classifier     : sklearn estimator (default: scaled SVM)
    random_state   : int

    Returns
    -------
    pd.DataFrame with all (params, n_features, accuracy_mean, accuracy_std) combos,
    sorted by accuracy_mean descending.
    """
    from itertools import product
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from tumorfs.selectors.base import FeatureSelector

    if classifier is None:
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=random_state)),
        ])

    n_features_list = n_features_list or [10, 20, 50, 100, 200]
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    records = []
    for combo in product(*param_values):
        kw = dict(zip(param_names, combo))
        for n_feat in n_features_list:
            try:
                sel = FeatureSelector(
                    method,
                    n_features=max(n_features_list),
                    random_state=random_state,
                    **kw,
                )
                sel.fit(X, y)
                genes = sel.selected_features_[:n_feat]
                scores = cross_val_score(
                    classifier, X[genes], y,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
                    scoring="accuracy",
                    n_jobs=-1,
                )
                record = {**kw, "n_features": n_feat,
                          "accuracy_mean": scores.mean(), "accuracy_std": scores.std()}
                records.append(record)
                logger.debug("  %s n=%d → %.4f", kw, n_feat, scores.mean())
            except Exception as exc:
                logger.warning("Failed for %s n=%d: %s", kw, n_feat, exc)

    return pd.DataFrame(records).sort_values("accuracy_mean", ascending=False)


def evaluate_algorithms(
    X: pd.DataFrame,
    y: pd.Series,
    methods: List[str],
    n_features: int = 50,
    cv: int = 5,
    classifier=None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Benchmark multiple selectors side-by-side: for each method, select
    ``n_features`` genes and measure downstream CV accuracy.

    Returns
    -------
    pd.DataFrame  (method → accuracy_mean, accuracy_std, selected_genes)
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from tumorfs.selectors.base import FeatureSelector

    if classifier is None:
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=random_state)),
        ])

    records = []
    for method in methods:
        try:
            sel = FeatureSelector(method, n_features=n_features, random_state=random_state)
            sel.fit(X, y)
            genes = sel.selected_features_
            scores = cross_val_score(
                classifier, X[genes], y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
                scoring="accuracy",
                n_jobs=-1,
            )
            records.append({
                "method": method,
                "n_features": len(genes),
                "accuracy_mean": scores.mean(),
                "accuracy_std": scores.std(),
                "selected_genes": ",".join(genes[:10]) + ("…" if len(genes) > 10 else ""),
            })
            logger.info("%-20s → %.4f ± %.4f", method, scores.mean(), scores.std())
        except Exception as exc:
            logger.warning("Method '%s' failed: %s", method, exc)
            records.append({"method": method, "accuracy_mean": np.nan, "accuracy_std": np.nan})

    return pd.DataFrame(records).sort_values("accuracy_mean", ascending=False)


def ensemble_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    methods: List[str],
    n_features_list: Optional[List[int]] = None,
    strategy: str = "rank_aggregation",
    cv: int = 5,
    classifier=None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate an EnsembleRanker across several top-N gene-count thresholds.

    Returns
    -------
    pd.DataFrame  (n_features → accuracy_mean, accuracy_std)
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from tumorfs.ensemble import EnsembleRanker

    if classifier is None:
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=random_state)),
        ])

    n_features_list = n_features_list or [10, 20, 50, 100, 200]
    ranker = EnsembleRanker(methods=methods, n_features=max(n_features_list),
                            strategy=strategy, random_state=random_state)
    ranker.fit(X, y)

    records = []
    for n in n_features_list:
        genes = ranker.get_top_features(n)
        scores = cross_val_score(
            classifier, X[genes], y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            scoring="accuracy",
            n_jobs=-1,
        )
        records.append({"n_features": n, "accuracy_mean": scores.mean(), "accuracy_std": scores.std()})
        logger.info("Ensemble n=%3d → %.4f ± %.4f", n, scores.mean(), scores.std())

    return pd.DataFrame(records)
