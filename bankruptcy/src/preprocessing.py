"""
Custom Preprocessing Transformers for Bankruptcy Prediction.

All transformers inherit from BaseEstimator and TransformerMixin, maintaining full
compatibility with scikit-learn and imbalanced-learn pipelines (clone, get_params,
get_feature_names_out).

Design Principles:
1. Strict Leakage Prevention: All fitting parameters (quantiles, means, correlations,
   and univariate AUC rankings) are learned exclusively on the training fold passed to fit().
2. DataFrame Input/Output: All transformations accept and return pandas DataFrames,
   preserving column names throughout the pipeline.
3. Feature Names Tracking: Implements get_feature_names_out() on every transformer.
"""

import sys
from pathlib import Path
from typing import Sequence, Union

# Ensure config is importable regardless of working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from config import SEED


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects a predefined subset of features.

    If features is None, all columns are retained.
    If specified, raises ValueError if any requested column is missing.
    """

    def __init__(self, features: Union[Sequence[str], None] = None):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.features is None:
            self.selected_features_ = list(X.columns)
        else:
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise ValueError(
                    f"The following required features are missing in input data: {missing}"
                )
            self.selected_features_ = list(self.features)

        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        missing = [f for f in self.selected_features_ if f not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required features during transform: {missing}"
            )
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.selected_features_, dtype=object)


class DropConstantAndDuplicate(BaseEstimator, TransformerMixin):
    """
    Identifies and removes constant columns and completely identical duplicate columns.

    During fit():
    - Identifies constant columns (nunique <= 1).
    - Identifies identical duplicate column pairs and marks subsequent columns for removal,
      preserving the column appearing earlier in the DataFrame.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        cols = list(X.columns)
        # 1. Constant columns
        constant_cols = [c for c in cols if X[c].nunique(dropna=False) <= 1]

        # 2. Duplicate columns (only check amongst non-constant columns)
        non_const = [c for c in cols if c not in constant_cols]
        dup_cols = []
        seen = []
        for c1 in non_const:
            is_dup = False
            for c2 in seen:
                if X[c1].equals(X[c2]):
                    dup_cols.append(c1)
                    is_dup = True
                    break
            if not is_dup:
                seen.append(c1)

        self.constant_columns_ = constant_cols
        self.duplicate_columns_ = dup_cols
        self.dropped_ = constant_cols + dup_cols
        self.selected_features_ = [c for c in cols if c not in self.dropped_]
        self.n_features_in_ = len(cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.selected_features_, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizes (clips) continuous features based on training fold quantiles.

    Rules:
    - Only features determined to be continuous at fit time (nunique > 2) are processed.
    - Binary flag features (nunique <= 2) and constants are left untouched.
    - If enabled=False, passes input through unchanged (for ablation experiments).
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99, enabled: bool = True):
        self.lower = lower
        self.upper = upper
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.continuous_cols_ = []

        if self.enabled:
            for c in X.columns:
                # Only continuous features
                if X[c].nunique(dropna=True) > 2:
                    self.continuous_cols_.append(c)
                    self.lower_bounds_[c] = float(X[c].quantile(self.lower))
                    self.upper_bounds_[c] = float(X[c].quantile(self.upper))

        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.enabled or not self.continuous_cols_:
            return X.copy()

        X_out = X.copy()
        for c in self.continuous_cols_:
            if c in X_out.columns:
                X_out[c] = X_out[c].clip(
                    lower=self.lower_bounds_[c],
                    upper=self.upper_bounds_[c]
                )
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_names_in_, dtype=object)


class LargeValueIndicator(BaseEstimator, TransformerMixin):
    """
    Ablation tool for extreme values.

    During fit(), identifies features where the proportion of values > 1.0 is between
    1% and 99% in the training data (bimodal coding variables).
    During transform(), appends binary flag columns (<col>__gt1) while retaining the original column.

    Default enabled=False.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)
        self.indicator_cols_ = []

        if self.enabled:
            for c in X.columns:
                prop_gt_1 = float((X[c] > 1.0).mean())
                if 0.01 <= prop_gt_1 <= 0.99:
                    self.indicator_cols_.append(c)

        new_cols = [f"{c}__gt1" for c in self.indicator_cols_]
        self.feature_names_out_ = list(X.columns) + new_cols
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.enabled or not self.indicator_cols_:
            return X.copy()

        X_out = X.copy()
        for c in self.indicator_cols_:
            if c in X_out.columns:
                X_out[f"{c}__gt1"] = (X_out[c] > 1.0).astype(float)
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_names_out_, dtype=object)


class QuantileScaler(BaseEstimator, TransformerMixin):
    """
    Wraps QuantileTransformer or StandardScaler and maintains a DataFrame output with column names.

    Parameters:
        kind: 'quantile' (QuantileTransformer mapped to standard normal) or 'standard' (StandardScaler).

    Note:
        Binary flag features (such as Liability-Assets Flag) are scaled alongside continuous features
        without special exclusion. This allows downstream distance-based resamplers (SMOTE) to treat
        all dimensions in a normalized euclidean space.
    """

    def __init__(self, kind: str = "quantile"):
        self.kind = kind

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)

        if self.kind == "quantile":
            n_quantiles = min(1000, len(X))
            self.scaler_ = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=SEED,
            )
        elif self.kind == "standard":
            self.scaler_ = StandardScaler()
        else:
            raise ValueError(
                f"Unknown scaler kind: {self.kind}. Expected 'quantile' or 'standard'."
            )

        self.scaler_.fit(X)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        vals = self.scaler_.transform(X)
        return pd.DataFrame(vals, columns=self.feature_names_in_, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_names_in_, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Filters collinear features based on training fold correlation and univariate AUC.

    During fit(X, y):
    1. Computes the Pearson correlation matrix strictly on X.
    2. Computes the univariate ROC-AUC for every feature against y: max(AUC, 1 - AUC).
    3. For pairs with |r| > threshold:
       - Retains the feature with higher univariate predictive AUC.
       - If AUCs are tied, retains the feature appearing earlier in column order.
    4. Drops the redundant feature and records removed column names in `dropped_`.

    If enabled=False, passes input through without changes.
    """

    def __init__(self, threshold: float = 0.95, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)

        if not self.enabled or len(X.columns) <= 1:
            self.selected_features_ = list(X.columns)
            self.dropped_ = []
            self.auc_scores_ = {}
            self.n_features_in_ = len(X.columns)
            return self

        if y is None:
            raise ValueError(
                "CorrelationFilter requires target 'y' to evaluate univariate AUC for collinear pairs."
            )

        y_arr = np.asarray(y).ravel()

        # 1. Pearson correlation matrix
        corr = X.corr(method="pearson").abs()

        # 2. Univariate adjusted AUC for each feature
        auc_dict = {}
        for c in X.columns:
            vals = X[c].values
            if np.all(vals == vals[0]):
                auc_dict[c] = 0.5
            else:
                try:
                    raw_auc = roc_auc_score(y_arr, vals)
                    auc_dict[c] = float(max(raw_auc, 1.0 - raw_auc))
                except Exception:
                    auc_dict[c] = 0.5
        self.auc_scores_ = auc_dict

        # 3. Greedy maximum-predictive feature selection
        # Priority: higher AUC first, earlier original column index as deterministic tie-breaker
        col_order = {c: i for i, c in enumerate(X.columns)}
        ranked_cols = sorted(X.columns, key=lambda c: (-auc_dict[c], col_order[c]))

        dropped = set()
        selected = []
        for c in ranked_cols:
            if c in dropped:
                continue
            selected.append(c)
            # Find any remaining correlated features to drop
            for other in ranked_cols:
                if other != c and other not in dropped and other not in selected:
                    corr_val = corr.loc[c, other]
                    if corr_val > self.threshold:
                        dropped.add(other)

        # Preserve original column order in selected features
        self.selected_features_ = [c for c in X.columns if c in selected]
        self.dropped_ = [c for c in X.columns if c in dropped]
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.enabled:
            return X.copy()
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not self.enabled:
            return np.array(self.feature_names_in_, dtype=object)
        return np.array(self.selected_features_, dtype=object)
