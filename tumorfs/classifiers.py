"""
classifiers.py
--------------
Downstream classification models for validating selected gene subsets.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, callable] = {}


def _register_models():
    """Lazy-load models to avoid import overhead at package load time."""
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import lightgbm as lgb
    import xgboost as xgb

    registry = {
        "svm":               lambda rs: SVC(kernel="rbf", probability=True, random_state=rs),
        "svm_linear":        lambda rs: SVC(kernel="linear", probability=True, random_state=rs),
        "random_forest":     lambda rs: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=rs),
        "extra_trees":       lambda rs: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=rs),
        "gradient_boosting": lambda rs: GradientBoostingClassifier(n_estimators=100, random_state=rs),
        "adaboost":          lambda rs: AdaBoostClassifier(n_estimators=100, random_state=rs),
        "logistic":          lambda rs: LogisticRegression(max_iter=1000, random_state=rs, n_jobs=-1),
        "knn":               lambda rs: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "naive_bayes":       lambda _:  GaussianNB(),
        "mlp":               lambda rs: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=rs),
        "lda":               lambda _:  LinearDiscriminantAnalysis(),
        "lightgbm":          lambda rs: lgb.LGBMClassifier(n_estimators=200, n_jobs=-1, verbose=-1, random_state=rs),
        "xgboost":           lambda rs: xgb.XGBClassifier(n_estimators=200, use_label_encoder=False,
                                                           eval_metric="logloss", n_jobs=-1, random_state=rs),
    }
    try:
        from catboost import CatBoostClassifier
        registry["catboost"] = lambda rs: CatBoostClassifier(iterations=200, verbose=0, random_seed=rs)
    except ImportError:
        pass

    return registry


AVAILABLE_MODELS = [
    "svm", "svm_linear", "random_forest", "extra_trees", "gradient_boosting",
    "adaboost", "logistic", "knn", "naive_bayes", "mlp", "lda",
    "lightgbm", "xgboost", "catboost",
]


class TumorClassifier:
    """
    Train and cross-validate a classifier on the selected gene subset.

    Parameters
    ----------
    model : str
        One of ``AVAILABLE_MODELS``.
    cv : int
        Number of stratified cross-validation folds.
    scale : bool
        If True, prepend a StandardScaler (recommended for SVM, KNN, MLP).
    random_state : int
    scoring : list of str
        Metrics to compute.  See sklearn ``scoring`` parameter.
    **kwargs
        Passed directly to the model constructor (overrides defaults).

    Examples
    --------
    >>> clf = TumorClassifier("svm", cv=5)
    >>> results = clf.evaluate(X_train[top_genes], y_train)
    >>> print(results["test_accuracy"].mean())
    """

    _SCALE_BY_DEFAULT = {"svm", "svm_linear", "knn", "mlp", "lda", "logistic"}

    def __init__(
        self,
        model: str = "svm",
        cv: int = 5,
        scale: Optional[bool] = None,
        random_state: int = 42,
        scoring: Optional[List[str]] = None,
        **kwargs,
    ):
        if model not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{model}'.  Available: {AVAILABLE_MODELS}")
        self.model_name = model
        self.cv = cv
        self.scale = scale if scale is not None else (model in self._SCALE_BY_DEFAULT)
        self.random_state = random_state
        self.scoring = scoring or ["accuracy", "f1_macro", "roc_auc_ovr"]
        self.kwargs = kwargs

        self._registry = _register_models()
        self._estimator = self._build()

    def _build(self):
        base = self._registry[self.model_name](self.random_state)
        # Apply any user-supplied kwargs
        for k, v in self.kwargs.items():
            setattr(base, k, v)
        if self.scale:
            return Pipeline([("scaler", StandardScaler()), ("clf", base)])
        return base

    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Run stratified k-fold CV and return per-fold metrics.

        Returns
        -------
        pd.DataFrame with columns = metrics and rows = folds.
        """
        cv_strategy = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        results = cross_validate(
            self._estimator, X, y,
            cv=cv_strategy,
            scoring=self.scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        df = pd.DataFrame(results)
        logger.info(
            "%s CV results — %s",
            self.model_name,
            {k: f"{v.mean():.4f}±{v.std():.4f}" for k, v in df.items() if "test_" in k},
        )
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TumorClassifier":
        """Fit the classifier on the full training data."""
        self._estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict_proba(X)

    def summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """Compute mean ± std for each metric across folds."""
        test_cols = [c for c in results.columns if c.startswith("test_")]
        rows = []
        for col in test_cols:
            metric = col.replace("test_", "")
            rows.append({
                "metric": metric,
                "mean": results[col].mean(),
                "std": results[col].std(),
                "min": results[col].min(),
                "max": results[col].max(),
            })
        return pd.DataFrame(rows).set_index("metric")
