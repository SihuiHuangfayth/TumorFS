"""
pipeline.py
-----------
End-to-end TumorFS pipeline: load → select → ensemble → classify → report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class TumorFSPipeline:
    """
    High-level convenience class that wires together all TumorFS components.

    Parameters
    ----------
    config : dict or str/Path to a YAML config file

    The config dict / YAML may contain:

    .. code-block:: yaml

        data:
          expression_path: data/expression_matrix.csv
          labels_path: data/labels.csv   # optional; for tidy layout only
          layout: wide                   # 'wide' or 'tidy'
          test_size: 0.2
          random_state: 42

        feature_selection:
          methods: [lightgbm, xgboost, relieff, lasso]
          n_features: 200
          strategy: rank_aggregation

        classification:
          models: [svm, random_forest, lightgbm]
          cv_folds: 5
          scoring: [accuracy, f1_macro]

        output:
          results_dir: results/
          plots: true

    Examples
    --------
    >>> pipeline = TumorFSPipeline("configs/experiment.yaml")
    >>> pipeline.run()
    >>> print(pipeline.summary_)
    """

    def __init__(self, config: Union[dict, str, Path]):
        if isinstance(config, (str, Path)):
            import yaml
            with open(config) as f:
                config = yaml.safe_load(f)
        self.config = config

        # Results populated by run()
        self.loader_ = None
        self.ranker_ = None
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_test_: Optional[pd.DataFrame] = None
        self.y_train_: Optional[pd.Series] = None
        self.y_test_: Optional[pd.Series] = None
        self.top_genes_: Optional[List[str]] = None
        self.cv_results_: Optional[Dict[str, pd.DataFrame]] = None
        self.summary_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def run(self) -> "TumorFSPipeline":
        """Execute all pipeline stages in order."""
        self._load_data()
        self._select_features()
        self._classify()
        self._report()
        return self

    # ------------------------------------------------------------------
    def _load_data(self):
        from tumorfs.data_processing import DataLoader
        cfg = self.config.get("data", {})
        self.loader_ = DataLoader(
            file_path=cfg["expression_path"],
            layout=cfg.get("layout", "wide"),
            test_size=cfg.get("test_size", 0.2),
            random_state=cfg.get("random_state", 42),
            balance=cfg.get("balance", True),
        )
        (self.X_train_, self.X_test_,
         self.y_train_, self.y_test_, _) = self.loader_.load()
        logger.info("Data loaded.  Train=%d  Test=%d", len(self.X_train_), len(self.X_test_))

    def _select_features(self):
        from tumorfs.ensemble import EnsembleRanker
        cfg = self.config.get("feature_selection", {})
        self.ranker_ = EnsembleRanker(
            methods=cfg.get("methods", ["lightgbm", "relieff"]),
            n_features=cfg.get("n_features", 200),
            strategy=cfg.get("strategy", "rank_aggregation"),
            random_state=self.config.get("data", {}).get("random_state", 42),
        )
        self.ranker_.fit(self.X_train_, self.y_train_)
        self.top_genes_ = self.ranker_.top_features_
        logger.info("Feature selection done.  Top gene: %s", self.top_genes_[0])

    def _classify(self):
        from tumorfs.classifiers import TumorClassifier
        cfg = self.config.get("classification", {})
        models = cfg.get("models", ["svm"])
        cv = cfg.get("cv_folds", 5)
        scoring = cfg.get("scoring", ["accuracy", "f1_macro"])

        self.cv_results_ = {}
        for model_name in models:
            clf = TumorClassifier(model=model_name, cv=cv, scoring=scoring)
            results = clf.evaluate(
                self.X_train_[self.top_genes_[:cfg.get("n_final_features", 50)]],
                self.y_train_,
            )
            self.cv_results_[model_name] = results
            logger.info(
                "%s  acc=%.4f ± %.4f",
                model_name,
                results["test_accuracy"].mean(),
                results["test_accuracy"].std(),
            )

    def _report(self):
        out_dir = Path(self.config.get("output", {}).get("results_dir", "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        do_plots = self.config.get("output", {}).get("plots", False)

        # Gene ranking table
        ranking_table = self.ranker_.get_ranking_table()
        ranking_table.to_csv(out_dir / "ensemble_ranking.csv")

        # Top genes list
        (out_dir / "top_genes.txt").write_text("\n".join(self.top_genes_))

        # CV results per model
        rows = []
        for model, df in self.cv_results_.items():
            for col in df.columns:
                if col.startswith("test_"):
                    rows.append({
                        "model": model,
                        "metric": col.replace("test_", ""),
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                    })
        self.summary_ = pd.DataFrame(rows)
        self.summary_.to_csv(out_dir / "cv_summary.csv", index=False)
        logger.info("Results saved to %s", out_dir)

        if do_plots:
            try:
                from tumorfs.evaluation import plot_results
                for model, df in self.cv_results_.items():
                    plot_results(
                        df, title=f"{model} CV Results",
                        save_path=str(out_dir / f"{model}_cv.png"),
                        show=False,
                    )
            except Exception as exc:
                logger.warning("Plotting failed: %s", exc)
