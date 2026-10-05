"""
feature_selection.py
--------------------
Utility functions for post-hoc gene ranking, scoring, and stepwise
feature-count optimisation (the "step-in / step-out" accuracy curve).

These are internal utilities consumed by EnsembleRanker and the pipeline.
They are also importable directly for custom workflows.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gene counting / ranking helpers
# ---------------------------------------------------------------------------

def Counting(importance_list: pd.DataFrame, num: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count how often each gene appears in the top-``num`` positions
    across all selector columns.

    Parameters
    ----------
    importance_list : pd.DataFrame
        Rows = rank positions, columns = selectors, values = gene names.
    num : int
        Only consider the top *num* rows.

    Returns
    -------
    df_genecounts : pd.DataFrame  (gene → Counts)
    rankremain    : pd.DataFrame  (top-num slice)
    """
    num = min(num, len(importance_list))
    rankremain = importance_list.iloc[:num]
    gene_counts = rankremain.stack().value_counts().rename("Counts")
    df_genecounts = gene_counts.to_frame()
    return df_genecounts, rankremain


def RerankGenes(df_genecounts: pd.DataFrame, rankremain: pd.DataFrame) -> pd.DataFrame:
    """
    Build a gene × algorithm matrix showing at what rank each gene appeared
    in each selector.

    Parameters
    ----------
    df_genecounts : pd.DataFrame  (index = gene names)
    rankremain    : pd.DataFrame  (rows = rank positions, columns = algorithms)

    Returns
    -------
    Rerank : pd.DataFrame  (genes × algorithms, values = rank strings)
    """
    genes = df_genecounts.index.tolist()
    algorithms = rankremain.columns.tolist()
    Rerank = pd.DataFrame(np.nan, index=genes, columns=algorithms)

    for col in rankremain.columns:
        for rank_pos, gene in rankremain[col].items():
            if pd.notna(gene) and gene in Rerank.index:
                current = Rerank.at[gene, col]
                if pd.isna(current):
                    Rerank.at[gene, col] = str(rank_pos)
                else:
                    Rerank.at[gene, col] = f"{current}, {rank_pos}"
    return Rerank


def convert_to_numeric(x) -> float:
    """
    Coerce a value to numeric; return 0 for NaN, keep first int if comma-separated.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    # Handle "12, 45" — take the first occurrence rank
    try:
        return float(str(x).split(",")[0].strip())
    except (ValueError, AttributeError):
        return np.nan


def transform_Scoring(x, total: int = 201) -> float:
    """
    Convert rank position to a score (rank 1 → highest score).

    Parameters
    ----------
    x     : rank value (float or NaN)
    total : score assigned to rank-1 is ``total - 1``  (default 200)
    """
    if pd.isnull(x) or np.isnan(x):
        return 0.0
    return float(total - x)


def process_data(importance_list: pd.DataFrame, num: int = 200) -> pd.DataFrame:
    """
    Full processing pipeline: count → rerank → score.

    Parameters
    ----------
    importance_list : pd.DataFrame  (rows = rank positions, cols = selectors)
    num             : int  number of top genes to consider

    Returns
    -------
    Scoring : pd.DataFrame  (algorithms × genes, values = scores)
    """
    df_genecounts, rankremain = Counting(importance_list, num=num)
    Rerank = RerankGenes(df_genecounts, rankremain)

    # pandas ≥ 2.1 deprecated applymap → use map() with element-wise callable
    _map = Rerank.map if hasattr(Rerank, "map") else Rerank.applymap  # compat shim

    Rerankint = _map(convert_to_numeric)
    Scoring = _map.__self__.copy() if False else Rerankint  # type annotation helper
    Scoring = Rerankint.map(transform_Scoring) if hasattr(Rerankint, "map") \
        else Rerankint.applymap(transform_Scoring)

    return Scoring.T  # algorithms × genes


# ---------------------------------------------------------------------------
# Normalisation and ordering
# ---------------------------------------------------------------------------

def scale_order(
    Scoring: pd.DataFrame,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> Tuple[pd.Index, pd.DataFrame]:
    """
    Min-max normalise each selector column and derive an ensemble gene order.

    Parameters
    ----------
    Scoring  : pd.DataFrame  (algorithms × genes)
    new_min  : float  target minimum after normalisation
    new_max  : float  target maximum after normalisation

    Returns
    -------
    order         : pd.Index  genes sorted by summed normalised score (desc)
    StandardScore : pd.DataFrame  normalised scoring matrix
    """
    StandardScore = pd.DataFrame(index=Scoring.index, columns=Scoring.columns, dtype=float)

    for col in Scoring.columns:
        col_min = Scoring[col].min()
        col_max = Scoring[col].max()
        denom = col_max - col_min
        if denom == 0:
            StandardScore[col] = 0.0
        else:
            StandardScore[col] = (
                (Scoring[col] - col_min) / denom * (new_max - new_min) + new_min
            )

    gene_total = StandardScore.sum(axis=0)
    order = gene_total.sort_values(ascending=False).index
    return order, StandardScore


# ---------------------------------------------------------------------------
# Step-in / step-out accuracy curve
# ---------------------------------------------------------------------------

def Step_in_out_ACC(
    X: pd.DataFrame,
    y: pd.Series,
    gene_order: List[str],
    step: int = 1,
    min_features: int = 1,
    max_features: Optional[int] = None,
    cv: int = 5,
    classifier=None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute cross-validated accuracy as features are added one-by-one
    (step-in) and removed one-by-one (step-out) following ``gene_order``.

    Parameters
    ----------
    X           : pd.DataFrame  full expression matrix
    y           : pd.Series  labels
    gene_order  : list of str  genes in descending importance order
    step        : int  evaluate every ``step`` genes
    min_features : int  start from this many features
    max_features : int or None  stop here (default = len(gene_order))
    cv          : int  cross-validation folds
    classifier  : sklearn-compatible estimator or None (defaults to scaled SVM)
    random_state : int

    Returns
    -------
    pd.DataFrame with columns [n_features, accuracy_mean, accuracy_std]
    """
    if classifier is None:
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=random_state)),
        ])

    max_features = max_features or len(gene_order)
    feature_counts = range(min_features, max_features + 1, step)

    records = []
    for n in feature_counts:
        genes = gene_order[:n]
        # Guard: skip genes not in X
        genes = [g for g in genes if g in X.columns]
        if not genes:
            continue
        try:
            scores = cross_val_score(
                classifier, X[genes], y, cv=cv, scoring="accuracy", n_jobs=-1
            )
            records.append({
                "n_features": n,
                "accuracy_mean": scores.mean(),
                "accuracy_std": scores.std(),
            })
        except Exception as exc:
            logger.warning("Skipping n=%d due to error: %s", n, exc)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convenience: find the optimal feature count from the step-in curve
# ---------------------------------------------------------------------------

def find_optimal_n(
    step_results: pd.DataFrame,
    criterion: str = "max_mean",
) -> int:
    """
    Given the output of :func:`Step_in_out_ACC`, return the optimal number
    of features.

    Parameters
    ----------
    criterion : {'max_mean', 'elbow'}
    """
    if criterion == "max_mean":
        return int(step_results.loc[step_results["accuracy_mean"].idxmax(), "n_features"])
    elif criterion == "elbow":
        # Simple elbow: largest drop in improvement
        acc = step_results["accuracy_mean"].values
        diffs = np.diff(acc)
        elbow_idx = np.argmin(diffs) + 1  # point after biggest deceleration
        return int(step_results.iloc[elbow_idx]["n_features"])
    else:
        raise ValueError(f"Unknown criterion '{criterion}'.")
