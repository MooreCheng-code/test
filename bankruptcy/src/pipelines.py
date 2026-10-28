"""
Pipeline Assembly Module for Bankruptcy Prediction.

Constructs imbalanced-learn Pipelines integrating:
1. Feature selection (FeatureSelector)
2. Constant and duplicate column elimination (DropConstantAndDuplicate)
3. Outlier winsorization (Winsorizer)
4. Extreme value binary indicators (LargeValueIndicator)
5. Normal / standard feature scaling (QuantileScaler)
6. Collinear feature filtering via training fold AUC (CorrelationFilter)
7. Resampling strategy (S0 - S6)
8. Classifier (M0 - M5)
"""

import sys
from pathlib import Path
from typing import Any, Sequence, Union

# Ensure config is importable regardless of working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import SEED
from src.preprocessing import (
    CorrelationFilter,
    DropConstantAndDuplicate,
    FeatureSelector,
    LargeValueIndicator,
    QuantileScaler,
    TrainingMedianImputer,
    Winsorizer,
)


class AdaptiveScaleWeightClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper for gradient boosting classifiers (XGBoost, LightGBM) under strategy S1.

    Dynamically calculates:
        scale_pos_weight = (n_negative / n_positive)
    strictly from the training fold's y passed to fit(). This guarantees zero leakage
    from out-of-fold validation or holdout test sets while keeping the Pipeline self-contained.
    """

    def __init__(self, estimator=None, enabled: bool = True):
        self.estimator = estimator
        self.enabled = enabled

    def fit(self, X, y, **fit_params):
        if self.estimator is None:
            raise ValueError("An underlying estimator must be provided.")

        self.estimator_ = clone(self.estimator)

        if self.enabled:
            y_arr = np.asarray(y).ravel()
            n_neg = int((y_arr == 0).sum())
            n_pos = int((y_arr == 1).sum())
            scale_weight = float(n_neg / max(1, n_pos))
            self.scale_pos_weight_ = scale_weight
            self.estimator_.set_params(scale_pos_weight=scale_weight)
        else:
            self.scale_pos_weight_ = getattr(self.estimator, "scale_pos_weight", 1.0)

        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    def decision_function(self, X):
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        raise AttributeError(
            f"{self.estimator_.__class__.__name__} does not implement decision_function."
        )

    @property
    def feature_importances_(self):
        if hasattr(self, "estimator_") and hasattr(self.estimator_, "feature_importances_"):
            return self.estimator_.feature_importances_
        raise AttributeError(
            f"Underlying estimator {getattr(self, 'estimator_', self.estimator).__class__.__name__} has no feature_importances_."
        )

    @property
    def coef_(self):
        if hasattr(self, "estimator_") and hasattr(self.estimator_, "coef_"):
            return self.estimator_.coef_
        raise AttributeError(
            f"Underlying estimator {getattr(self, 'estimator_', self.estimator).__class__.__name__} has no coef_."
        )


def _get_resampler(imbalance: str, seed: int = SEED):
    """Return appropriate resampler based on strategy code S0 - S6."""
    if imbalance == "S0" or imbalance == "S1":
        return None
    elif imbalance == "S2":
        return SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)
    elif imbalance == "S3":
        return SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=seed)
    elif imbalance == "S4":
        return SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=seed)
    elif imbalance == "S5":
        return BorderlineSMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=seed)
    elif imbalance == "S6":
        smote_inner = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=seed)
        return SMOTETomek(smote=smote_inner, random_state=seed)
    else:
        raise ValueError(f"Unknown imbalance strategy: '{imbalance}'. Expected S0 to S6.")


def _get_classifier(
    model_name: str,
    imbalance: str,
    model_params: Union[dict[str, Any], None] = None,
    seed: int = SEED,
):
    """Instantiate classifier with base parameters and class balancing if S1."""
    params = model_params.copy() if model_params is not None else {}

    if model_name == "M0":
        # Dummy baseline
        clf_params = {"strategy": "prior", "random_state": seed}
        clf_params.update(params)
        return DummyClassifier(**clf_params)

    elif model_name == "M1":
        # Logistic Regression
        clf_params = {
            "max_iter": 5000,
            "random_state": seed,
            "class_weight": "balanced" if imbalance == "S1" else None,
        }
        clf_params.update(params)
        return LogisticRegression(**clf_params)

    elif model_name == "M2":
        # Random Forest
        clf_params = {
            "n_estimators": 300,
            "random_state": seed,
            "n_jobs": -1,
            "class_weight": "balanced" if imbalance == "S1" else None,
        }
        clf_params.update(params)
        return RandomForestClassifier(**clf_params)

    elif model_name == "M3":
        # Extra Trees
        clf_params = {
            "n_estimators": 300,
            "random_state": seed,
            "n_jobs": -1,
            "class_weight": "balanced" if imbalance == "S1" else None,
        }
        clf_params.update(params)
        return ExtraTreesClassifier(**clf_params)

    elif model_name == "M4":
        # XGBoost
        clf_params = {
            "n_estimators": 300,
            "random_state": seed,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        clf_params.update(params)
        base_xgb = XGBClassifier(**clf_params)
        if imbalance == "S1":
            return AdaptiveScaleWeightClassifier(estimator=base_xgb, enabled=True)
        return base_xgb

    elif model_name == "M5":
        # LightGBM
        clf_params = {
            "n_estimators": 300,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        clf_params.update(params)
        base_lgbm = LGBMClassifier(**clf_params)
        if imbalance == "S1":
            return AdaptiveScaleWeightClassifier(estimator=base_lgbm, enabled=True)
        return base_lgbm

    else:
        raise ValueError(
            f"Model '{model_name}' not recognized. Currently supported: M0, M1, M2, M3, M4, M5."
        )


def build_pipeline(
    model_name: str,
    imbalance: str = "S0",
    scaler: str = "quantile",
    use_winsorizer: bool = True,
    use_corr_filter: bool = True,
    use_large_indicator: bool = False,
    features: Union[Sequence[str], None] = None,
    model_params: Union[dict[str, Any], None] = None,
    seed: int = SEED,
    use_imputer: bool = False,
) -> Pipeline:
    """
    Assemble the complete imblearn Pipeline.

    Execution order:
    1. FeatureSelector
    2. Imputer (TrainingMedianImputer, if use_imputer=True)
    3. DropConstantAndDuplicate
    4. Winsorizer
    5. LargeValueIndicator
    6. QuantileScaler
    7. CorrelationFilter
    8. Resampler (S2 - S6; skipped for S0 and S1)
    9. Classifier
    """
    steps = [
        ("feature_selector", FeatureSelector(features=features)),
    ]
    if use_imputer:
        steps.append(("imputer", TrainingMedianImputer()))

    steps.extend([
        ("drop_const_dup", DropConstantAndDuplicate()),
        ("winsorizer", Winsorizer(lower=0.01, upper=0.99, enabled=use_winsorizer)),
        ("large_value_indicator", LargeValueIndicator(enabled=use_large_indicator)),
        ("scaler", QuantileScaler(kind=scaler)),
        ("corr_filter", CorrelationFilter(threshold=0.95, enabled=use_corr_filter)),
    ])

    # Resampling step (only for S2 - S6)
    resampler = _get_resampler(imbalance, seed=seed)
    if resampler is not None:
        steps.append(("resampler", resampler))

    # Classifier
    clf = _get_classifier(model_name, imbalance, model_params=model_params, seed=seed)
    steps.append(("classifier", clf))

    return Pipeline(steps)


def get_score(pipeline: Pipeline, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Extract continuous risk scores for the positive class.

    Prioritizes predict_proba[:, 1]; falls back to decision_function if unavailable.
    """
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    elif hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X)
    else:
        raise AttributeError(
            f"The fitted pipeline {pipeline} does not implement predict_proba or decision_function."
        )


def get_feature_names_after_preprocessing(fitted_pipeline: Pipeline) -> list[str]:
    """
    Extract the final list of feature names immediately before the classifier step.
    """
    corr_step = fitted_pipeline.named_steps.get("corr_filter")
    if corr_step is not None and hasattr(corr_step, "get_feature_names_out"):
        return list(corr_step.get_feature_names_out())

    # Fallback to scanning steps in reverse
    for name, step in reversed(fitted_pipeline.steps[:-1]):
        if name != "resampler" and hasattr(step, "get_feature_names_out"):
            return list(step.get_feature_names_out())

    raise ValueError("Could not determine feature names from fitted pipeline.")
