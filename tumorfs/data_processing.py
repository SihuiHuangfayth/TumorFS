"""
data_processing.py
------------------
Data loading and preprocessing utilities for RNA-seq tumor classification.

Key class
---------
DataLoader : unified interface replacing the scattered data_import / train / final functions.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary class (recommended interface)
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Load, preprocess, and split RNA-seq expression data for tumor classification.

    Parameters
    ----------
    file_path : str or Path
        Path to a CSV file.  Two layouts are supported:

        *Wide layout* (genes as rows, samples as columns — common TCGA format):
            First column = gene names; remaining columns = sample IDs.
            Set ``layout="wide"`` (default).

        *Tidy layout* (samples as rows, genes as columns):
            First column = sample IDs; last column = labels.
            Set ``layout="tidy"``.

    label_col : str
        Name given to the first column (gene-name / sample-ID column).
    target_col : str
        Name of the label column.  In wide layout the label is inferred from
        the last character(s) of the sample ID (``'01'`` → tumour, ``'11'`` →
        normal by TCGA convention) unless ``label_map`` is provided.
    n_features : int or None
        Maximum number of gene features to retain.  ``None`` keeps all.
    test_size : float
        Fraction of samples reserved for the held-out test set.
    random_state : int
        Reproducibility seed.
    balance : bool
        If True, down-sample the majority class so both classes are equal.
        Only applied to the training split (test set is left imbalanced).
    layout : {'wide', 'tidy'}
        CSV layout (see above).
    label_map : dict or None
        Mapping from raw suffix strings to integer labels, e.g.
        ``{'01': 1, '11': 0}``.  If None, the last column is used as-is.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        label_col: str = "sample",
        target_col: str = "ylable",
        n_features: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 111,
        balance: bool = True,
        layout: str = "wide",
        label_map: Optional[dict] = None,
    ):
        self.file_path = Path(file_path)
        self.label_col = label_col
        self.target_col = target_col
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
        self.balance = balance
        self.layout = layout
        self.label_map = label_map or {"01": 1, "11": 0}

        # Populated after load()
        self.gene_names_: Optional[pd.Index] = None
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    def load(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index]:
        """
        Read the CSV, preprocess, and produce train/test splits.

        Returns
        -------
        X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
        gene_names : pd.Index
        """
        if self.layout == "wide":
            X, y, gene_names = self._load_wide()
        else:
            X, y, gene_names = self._load_tidy()

        self.X_ = X
        self.y_ = y
        self.gene_names_ = gene_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        if self.balance:
            X_train, y_train = self._balance(X_train, y_train)

        logger.info(
            "Loaded %d samples, %d features.  Train=%d  Test=%d  Classes=%s",
            len(X), X.shape[1], len(X_train), len(X_test),
            dict(y_train.value_counts()),
        )
        return X_train, X_test, y_train, y_test, gene_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_wide(self) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
        """Wide layout: genes × samples CSV (standard TCGA download format)."""
        raw = pd.read_csv(self.file_path, sep=",", index_col=0)
        gene_names = raw.index  # genes are rows

        # Transpose → samples × genes
        exp = raw.T
        exp.index.name = self.label_col

        # Derive labels from sample-ID suffixes
        suffixes = exp.index.str.split("-").str[-1].str[:2]
        y = suffixes.map(self.label_map)

        if y.isna().any():
            unknown = suffixes[y.isna()].unique().tolist()
            warnings.warn(
                f"Could not map sample-ID suffixes {unknown} to labels. "
                "Those samples will be dropped.  Pass a custom ``label_map`` to fix this."
            )
            mask = y.notna()
            exp, y = exp.loc[mask], y.loc[mask]

        y = y.astype(int)

        # Restrict to n_features
        if self.n_features is not None:
            gene_names = gene_names[: self.n_features]
            exp = exp.iloc[:, : self.n_features]

        return exp, y, gene_names

    def _load_tidy(self) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
        """Tidy layout: samples × genes CSV with an explicit label column."""
        raw = pd.read_csv(self.file_path, sep=",", index_col=0)

        if self.target_col not in raw.columns:
            raise ValueError(
                f"target_col='{self.target_col}' not found.  "
                f"Available columns: {raw.columns.tolist()[:10]}"
            )

        y = raw[self.target_col].astype(int)
        X = raw.drop(columns=[self.target_col])

        if self.n_features is not None:
            X = X.iloc[:, : self.n_features]

        gene_names = X.columns
        return X, y, gene_names

    @staticmethod
    def _balance(
        X: pd.DataFrame, y: pd.Series, random_state: int = 111
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Down-sample the majority class to match the minority class size."""
        counts = y.value_counts()
        min_size = counts.min()
        parts_X, parts_y = [], []
        for cls in counts.index:
            idx = y[y == cls].index
            sampled = idx.to_series().sample(n=min_size, random_state=random_state)
            parts_X.append(X.loc[sampled])
            parts_y.append(y.loc[sampled])
        return pd.concat(parts_X), pd.concat(parts_y)

    # ------------------------------------------------------------------
    # Convenience: save / reload splits
    # ------------------------------------------------------------------

    def save_splits(self, X_train, X_test, y_train, y_test, out_dir: Union[str, Path] = "."):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{self.target_col: y_train}).to_csv(out_dir / "train.csv")
        X_test.assign(**{self.target_col: y_test}).to_csv(out_dir / "test.csv")
        logger.info("Splits saved to %s", out_dir)

    @staticmethod
    def load_split(file_path: Union[str, Path], target_col: str = "ylable"):
        """Reload a previously saved split CSV."""
        df = pd.read_csv(file_path, index_col=0)
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
        return X, y


# ---------------------------------------------------------------------------
# Legacy functional API  (kept for backward compatibility)
# ---------------------------------------------------------------------------

def data_import(
    file_path,
    label_col="sample",
    target_col="ylable",
    n_features=30557,
    test_size=0.2,
    random_state=111,
):
    """
    Legacy function — wraps DataLoader.

    .. deprecated::
        Use :class:`DataLoader` instead.
    """
    warnings.warn(
        "data_import() is deprecated; use DataLoader instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    loader = DataLoader(
        file_path,
        label_col=label_col,
        target_col=target_col,
        n_features=n_features,
        test_size=test_size,
        random_state=random_state,
        layout="wide",
    )
    X_train, X_test, y_train, y_test, gene_names = loader.load()

    # Reconstruct the old 9-tuple return signature
    X = loader.X_
    y = loader.y_
    data_df = pd.concat([X_train, y_train], axis=1)
    final_df = pd.concat([X_test, y_test], axis=1)
    return X, y, data_df, final_df, X_train, y_train, X_test, y_test, gene_names


def train(file_path: str, target_col: str = "ylable"):
    """Legacy: load training split from a saved CSV."""
    return DataLoader.load_split(file_path, target_col=target_col)


def final(file_path: str, target_col: str = "ylable"):
    """Legacy: load test split from a saved CSV."""
    return DataLoader.load_split(file_path, target_col=target_col)


def Counting(importance_list: pd.DataFrame, num: int = 200):
    """
    Count how often each gene appears in the top-``num`` positions
    across all selector columns.

    Parameters
    ----------
    importance_list : pd.DataFrame
        Each column is a selector; each row (0-indexed) is a rank position;
        values are gene names.
    num : int
        Consider only the top *num* rows.

    Returns
    -------
    df_genecounts : pd.DataFrame  (gene → Counts)
    rankremain    : pd.DataFrame  (top-num slice)
    """
    if num > len(importance_list):
        num = len(importance_list)
        logger.warning("num=%d exceeds list length; using %d", num, num)

    rankremain = importance_list.iloc[:num]
    gene_counts = rankremain.stack().value_counts()
    df_genecounts = gene_counts.rename("Counts").to_frame()
    return df_genecounts, rankremain
