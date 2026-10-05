"""
evaluation.py
-------------
Model evaluation and publication-ready plotting utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_model(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation and return a results DataFrame.

    Parameters
    ----------
    clf        : sklearn-compatible estimator
    X, y       : data
    cv         : number of folds
    scoring    : list of sklearn scoring strings

    Returns
    -------
    pd.DataFrame  (one row per fold, columns = metrics)
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    scoring = scoring or ["accuracy", "f1_macro", "roc_auc_ovr"]
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = cross_validate(clf, X, y, cv=cv_strat, scoring=scoring,
                             return_train_score=True, n_jobs=-1)
    return pd.DataFrame(results)


def plot_results(
    results: pd.DataFrame,
    title: str = "Model Performance",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot CV accuracy / F1 / AUC across folds as a bar chart with error bars.

    Parameters
    ----------
    results   : output of evaluate_model()
    title     : plot title
    save_path : if set, save the figure to this path
    show      : if True, call plt.show()
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib / seaborn not installed; cannot plot.")
        return

    test_cols = [c for c in results.columns if c.startswith("test_")]
    means = results[test_cols].mean()
    stds = results[test_cols].std()
    labels = [c.replace("test_", "") for c in test_cols]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means.values, yerr=stds.values, capsize=5,
                  color=sns.color_palette("muted", len(labels)), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    for bar, mean in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    if show:
        plt.show()
    return fig


def plot_accuracy_vs_n_features(
    step_df: pd.DataFrame,
    optimal_n: Optional[int] = None,
    title: str = "Accuracy vs. Number of Features",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot the step-in accuracy curve from :func:`~tumorfs.feature_selection.Step_in_out_ACC`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step_df["n_features"], step_df["accuracy_mean"], "o-", color="#2196F3",
            linewidth=2, markersize=4, label="CV Accuracy")
    ax.fill_between(
        step_df["n_features"],
        step_df["accuracy_mean"] - step_df["accuracy_std"],
        step_df["accuracy_mean"] + step_df["accuracy_std"],
        alpha=0.2, color="#2196F3",
    )
    if optimal_n is not None:
        opt_acc = step_df.loc[step_df["n_features"] == optimal_n, "accuracy_mean"].values
        if len(opt_acc):
            ax.axvline(optimal_n, linestyle="--", color="red", alpha=0.7,
                       label=f"Optimal n={optimal_n}")
            ax.scatter([optimal_n], opt_acc, color="red", zorder=5)

    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("CV Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_feature_importance(
    ranked_genes: List[str],
    importances: np.ndarray,
    top_n: int = 30,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Horizontal bar chart of the top-n genes by importance."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib / seaborn not installed; cannot plot.")
        return

    n = min(top_n, len(ranked_genes))
    genes = ranked_genes[:n][::-1]
    imps = importances[:n][::-1]

    fig, ax = plt.subplots(figsize=(7, n * 0.3 + 1))
    ax.barh(genes, imps, color=sns.color_palette("Blues_r", n))
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
