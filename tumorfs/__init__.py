"""
TumorFS: Multi-Algorithm Feature Selection for Tumor RNA-seq Classification
===========================================================================

A comprehensive toolkit for identifying tumor-specific gene subsets from
RNA-seq data using an ensemble of feature selection algorithms and ML classifiers.

Basic usage
-----------
>>> from tumorfs import DataLoader, FeatureSelector, EnsembleRanker, TumorClassifier
>>> loader = DataLoader("expression.csv", label_col="sample", target_col="ylable")
>>> X_train, X_test, y_train, y_test, gene_names = loader.load()
>>> ranker = EnsembleRanker(methods=["lightgbm", "relieff", "lasso"], n_features=50)
>>> ranker.fit(X_train, y_train)
>>> clf = TumorClassifier(model="svm", cv=5)
>>> results = clf.evaluate(X_train[ranker.top_features_], y_train)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data_processing import DataLoader
from .selectors import FeatureSelector
from .ensemble import EnsembleRanker
from .classifiers import TumorClassifier
from .pipeline import TumorFSPipeline
from .evaluation import evaluate_model, plot_results

__all__ = [
    "DataLoader",
    "FeatureSelector",
    "EnsembleRanker",
    "TumorClassifier",
    "TumorFSPipeline",
    "evaluate_model",
    "plot_results",
]
