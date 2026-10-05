"""
selectors/tree_based.py
-----------------------
Tree-based and boosting feature selectors.
"""

import numpy as np
import pandas as pd
from .base import BaseSelector


class LightGBMSelector(BaseSelector):
    def _fit(self, X, y):
        import lightgbm as lgb
        params = dict(
            n_estimators=self.kwargs.get("n_estimators", 200),
            learning_rate=self.kwargs.get("learning_rate", 0.05),
            num_leaves=self.kwargs.get("num_leaves", 31),
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)


class XGBoostSelector(BaseSelector):
    def _fit(self, X, y):
        import xgboost as xgb
        params = dict(
            n_estimators=self.kwargs.get("n_estimators", 200),
            learning_rate=self.kwargs.get("learning_rate", 0.05),
            max_depth=self.kwargs.get("max_depth", 6),
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)


class CatBoostSelector(BaseSelector):
    def _fit(self, X, y):
        from catboost import CatBoostClassifier
        params = dict(
            iterations=self.kwargs.get("iterations", 200),
            learning_rate=self.kwargs.get("learning_rate", 0.05),
            depth=self.kwargs.get("depth", 6),
            random_seed=self.random_state,
            verbose=0,
        )
        model = CatBoostClassifier(**params)
        model.fit(X, y)
        self._rank_by_importance(model.get_feature_importance())


class RandomForestSelector(BaseSelector):
    def _fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=self.kwargs.get("n_estimators", 200),
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)


class AdaBoostSelector(BaseSelector):
    def _fit(self, X, y):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(
            n_estimators=self.kwargs.get("n_estimators", 100),
            random_state=self.random_state,
        )
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)


class DecisionTreeSelector(BaseSelector):
    def _fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(
            max_depth=self.kwargs.get("max_depth", None),
            random_state=self.random_state,
        )
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)


class ExtraTreesSelector(BaseSelector):
    def _fit(self, X, y):
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(
            n_estimators=self.kwargs.get("n_estimators", 200),
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)


class GradientBoostingSelector(BaseSelector):
    def _fit(self, X, y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=self.kwargs.get("n_estimators", 100),
            learning_rate=self.kwargs.get("learning_rate", 0.1),
            max_depth=self.kwargs.get("max_depth", 3),
            random_state=self.random_state,
        )
        model.fit(X, y)
        self._rank_by_importance(model.feature_importances_)
