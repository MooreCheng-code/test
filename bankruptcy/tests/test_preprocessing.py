"""
Unit Tests for Custom Preprocessing Transformers and Pipeline Assembly.

Covers:
1. Leak-free parameter estimation (fit on A, transform B using only A's parameters).
2. Transform B invariance under extreme outliers in B.
3. Winsorizer behavior on continuous features vs binary flags, and enabled=False passthrough.
4. DropConstantAndDuplicate on real data (verifies Net Income Flag and duplicate pairs).
5. CorrelationFilter logic (keeps higher-AUC feature on synthetic correlated data, tracks dropped_).
6. Full Pipeline functionality (cloneable, predict_proba, get_score, feature name tracking).
7. SMOTE validation fold preservation (resampling does not touch validation fold data).
8. AdaptiveScaleWeightClassifier dynamic weight calculation without leakage.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from config import DATA_PATH, TARGET_COL
from src.pipelines import (
    AdaptiveScaleWeightClassifier,
    build_pipeline,
    get_feature_names_after_preprocessing,
    get_score,
)
from src.preprocessing import (
    CorrelationFilter,
    DropConstantAndDuplicate,
    FeatureSelector,
    LargeValueIndicator,
    QuantileScaler,
    Winsorizer,
)


# ==============================================================================
# Test 1 & 2: Leakage Prevention (Fit on A, Transform B)
# ==============================================================================
def test_winsorizer_leakage_and_invariance():
    """
    Verify Winsorizer learns bounds strictly on dataset A, and applying to B
    clips B to A's bounds without being affected by extreme outliers in B.
    """
    df_a = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    # Lower 10%, Upper 90%
    winsor = Winsorizer(lower=0.10, upper=0.90, enabled=True)
    winsor.fit(df_a)

    expected_lower = df_a["feat"].quantile(0.10)  # 1.9
    expected_upper = df_a["feat"].quantile(0.90)  # 9.1
    assert winsor.lower_bounds_["feat"] == pytest.approx(expected_lower)
    assert winsor.upper_bounds_["feat"] == pytest.approx(expected_upper)

    # Dataset B with massive extreme values
    df_b = pd.DataFrame({"feat": [-999999.0, 5.0, 999999.0]})
    b_trans = winsor.transform(df_b)

    # Values must be clipped strictly to A's bounds
    assert b_trans["feat"].iloc[0] == pytest.approx(expected_lower)
    assert b_trans["feat"].iloc[1] == pytest.approx(5.0)
    assert b_trans["feat"].iloc[2] == pytest.approx(expected_upper)

    # Confirm A's bounds did not change
    assert winsor.lower_bounds_["feat"] == pytest.approx(expected_lower)
    assert winsor.upper_bounds_["feat"] == pytest.approx(expected_upper)


# ==============================================================================
# Test 3: Winsorizer Continuous vs Binary Flag
# ==============================================================================
def test_winsorizer_binary_flag_and_disabled():
    """
    Verify Winsorizer modifies continuous columns (nunique > 2), leaves binary flags
    (nunique <= 2) untouched, and passes all data unchanged when enabled=False.
    """
    df = pd.DataFrame({
        "cont": np.linspace(0, 100, 50),
        "flag": [0, 1] * 25,
        "const": [1] * 50,
    })

    winsor = Winsorizer(lower=0.05, upper=0.95, enabled=True)
    winsor.fit(df)

    assert "cont" in winsor.continuous_cols_
    assert "flag" not in winsor.continuous_cols_
    assert "const" not in winsor.continuous_cols_

    df_trans = winsor.transform(df)
    assert df_trans["flag"].equals(df["flag"])
    assert df_trans["const"].equals(df["const"])
    assert df_trans["cont"].min() == pytest.approx(df["cont"].quantile(0.05))

    # Disabled Winsorizer
    winsor_disabled = Winsorizer(enabled=False)
    winsor_disabled.fit(df)
    df_disabled_trans = winsor_disabled.transform(df)
    assert df_disabled_trans.equals(df)


# ==============================================================================
# Test 4: DropConstantAndDuplicate on Real Data
# ==============================================================================
def test_drop_constant_and_duplicate_real_data():
    """
    Verify DropConstantAndDuplicate correctly drops 'Net Income Flag' and the
    two duplicate column pairs on real bankruptcy dataset.
    """
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    X = df.drop(columns=[TARGET_COL])

    transformer = DropConstantAndDuplicate()
    transformer.fit(X)

    # Check constant column
    assert "Net Income Flag" in transformer.constant_columns_

    # Check identical duplicate pairs
    assert "Current Liability to Liability" in transformer.duplicate_columns_ or "Current Liabilities/Liability" in transformer.duplicate_columns_
    assert "Current Liability to Equity" in transformer.duplicate_columns_ or "Current Liabilities/Equity" in transformer.duplicate_columns_

    X_trans = transformer.transform(X)
    assert "Net Income Flag" not in X_trans.columns
    assert len(X_trans.columns) == len(X.columns) - 3  # 1 constant + 2 duplicates
    assert len(transformer.get_feature_names_out()) == len(X_trans.columns)


# ==============================================================================
# Test 5: CorrelationFilter Logic
# ==============================================================================
def test_correlation_filter_logic():
    """
    Verify CorrelationFilter drops the feature with lower univariate AUC
    when |r| > threshold, passes y properly, and maintains determinism.
    """
    # Create two highly collinear features, but feat_high has stronger predictive signal with y
    rng = np.random.RandomState(42)
    y = np.array([0] * 200 + [1] * 50)

    # feat_high has strong predictive signal with y
    feat_high = y * 1.5 + rng.normal(0, 0.6, 250)
    # feat_low is highly collinear (corr ~ 0.98 > 0.95), but slightly degraded for a few points
    feat_low = feat_high * 0.99 + rng.normal(0, 0.05, 250)
    feat_low[np.where(y == 0)[0][:8]] += 1.0  # Slightly degrade predictive ranking
    feat_indep = rng.normal(0, 1, 250)

    df = pd.DataFrame({
        "feat_high": feat_high,
        "feat_low": feat_low,
        "feat_indep": feat_indep,
    })

    corr_filter = CorrelationFilter(threshold=0.95, enabled=True)
    corr_filter.fit(df, y)

    # feat_high should have higher adjusted AUC than feat_low
    assert corr_filter.auc_scores_["feat_high"] > corr_filter.auc_scores_["feat_low"]

    # feat_low should be dropped, feat_high and feat_indep retained
    assert "feat_low" in corr_filter.dropped_
    assert "feat_high" in corr_filter.selected_features_
    assert "feat_indep" in corr_filter.selected_features_

    df_trans = corr_filter.transform(df)
    assert list(df_trans.columns) == ["feat_high", "feat_indep"]


# ==============================================================================
# Test 6: Full Pipeline Assembly and sklearn Compatibility
# ==============================================================================
def test_pipeline_assembly_and_clone():
    """
    Verify build_pipeline integrates all steps, is cloneable, supports predict_proba,
    get_score, and tracks feature names.
    """
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "num1": np.linspace(0, 10, 60),
        "num2": rng.normal(0, 1, 60),  # Not collinear with num1
        "const": [1.0] * 60,
    })
    y = np.array([0] * 50 + [1] * 10)

    pipe = build_pipeline(model_name="M1", imbalance="S0", scaler="standard")

    # Verify cloneability
    cloned_pipe = clone(pipe)
    assert cloned_pipe is not pipe

    # Fit pipeline
    pipe.fit(df, y)

    # Feature names after preprocessing
    feat_names = get_feature_names_after_preprocessing(pipe)
    assert "const" not in feat_names  # DropConstantAndDuplicate removed it
    assert "num1" in feat_names and "num2" in feat_names

    # Scores extraction
    scores = get_score(pipe, df)
    assert len(scores) == 60
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


# ==============================================================================
# Test 7: SMOTE Resampling Only in Training
# ==============================================================================
def test_smote_resampling_only_in_training():
    """
    Verify that in imblearn Pipeline, SMOTE only resamples during fit, and transform/predict
    on validation data leaves sample count and distributions unaltered.
    """
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame({
        "f1": rng.randn(100),
        "f2": rng.randn(100),
    })
    # SMOTE with k_neighbors=5 needs at least 6 positive samples
    y_train = np.array([0] * 85 + [1] * 15)

    X_val = pd.DataFrame({
        "f1": rng.randn(30),
        "f2": rng.randn(30),
    })
    y_val = np.array([0] * 28 + [1] * 2)

    # Build S3 pipeline (SMOTE sampling_strategy=0.3)
    pipe = build_pipeline(model_name="M1", imbalance="S3")
    pipe.fit(X_train, y_train)

    # When predicting on X_val, the output length must remain strictly 30
    scores = get_score(pipe, X_val)
    assert len(scores) == len(X_val)
    assert len(X_val) == 30  # X_val has not been modified


# ==============================================================================
# Test 8: AdaptiveScaleWeightClassifier Dynamic Calculation
# ==============================================================================
def test_adaptive_scale_weight_classifier():
    """
    Verify AdaptiveScaleWeightClassifier computes scale_pos_weight from y at fit time.
    """
    from xgboost import XGBClassifier

    X = pd.DataFrame({"f": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 negatives, 2 positives -> ratio = 4.0

    base_xgb = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    adaptive_clf = AdaptiveScaleWeightClassifier(estimator=base_xgb, enabled=True)

    adaptive_clf.fit(X, y)
    assert adaptive_clf.scale_pos_weight_ == pytest.approx(8.0 / 2.0)
    assert adaptive_clf.estimator_.get_params()["scale_pos_weight"] == pytest.approx(4.0)

    # Test predict_proba
    probas = adaptive_clf.predict_proba(X)
    assert probas.shape == (10, 2)


# ==============================================================================
# Test 9: FeatureSelector Fit on A, Transform on B
# ==============================================================================
def test_feature_selector_fit_a_transform_b():
    """
    Verify FeatureSelector learns selected features exclusively from A or parameters,
    and transforming B keeps only those features even if B has extraneous columns.
    """
    df_a = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    df_b = pd.DataFrame({"col1": [10, 20], "col2": [30, 40], "col_extra": [99, 99]})

    # Case 1: Specific features specified
    selector = FeatureSelector(features=["col1", "col2"])
    selector.fit(df_a)
    assert selector.selected_features_ == ["col1", "col2"]

    b_trans = selector.transform(df_b)
    assert list(b_trans.columns) == ["col1", "col2"]
    assert "col_extra" not in b_trans.columns

    # Case 2: features=None (learn all columns from A)
    selector_all = FeatureSelector(features=None)
    selector_all.fit(df_a)
    assert selector_all.selected_features_ == ["col1", "col2", "col3"]

    # If B is missing 'col3', transform must raise ValueError
    with pytest.raises(ValueError, match="Missing required features"):
        selector_all.transform(df_b)


# ==============================================================================
# Test 10: QuantileScaler Fit on A, Transform on B
# ==============================================================================
def test_quantile_scaler_fit_a_transform_b():
    """
    Verify QuantileScaler learns empirical distribution quantiles strictly from A.
    Transforming B maps values through A's quantiles without modifying A's learned quantiles.
    """
    rng = np.random.RandomState(42)
    df_a = pd.DataFrame({
        "feat": rng.normal(0.0, 1.0, 200),
    })
    scaler = QuantileScaler(kind="quantile")
    scaler.fit(df_a)

    learned_quantiles = scaler.scaler_.quantiles_.copy()

    # Dataset B with different distribution and massive outliers
    df_b = pd.DataFrame({
        "feat": [-1000.0, 0.0, 1000.0],
    })
    b_trans = scaler.transform(df_b)

    # Output maintains DataFrame format and column names
    assert isinstance(b_trans, pd.DataFrame)
    assert list(b_trans.columns) == ["feat"]
    assert len(b_trans) == 3

    # Ensure scaler learned quantiles are completely untouched by transforming B
    assert np.array_equal(scaler.scaler_.quantiles_, learned_quantiles)


# ==============================================================================
# Test 11: LargeValueIndicator Fit on A, Transform on B
# ==============================================================================
def test_large_value_indicator_fit_a_transform_b():
    """
    Verify LargeValueIndicator:
    1. enabled=False passes input through identical and unchanged.
    2. enabled=True learns indicator columns strictly from A (ratio > 1 between 1% and 99%).
    3. Transforming B applies indicators only to columns identified in A.
    """
    # In A:
    # col_bimodal: 50% > 1 -> should get indicator
    # col_normal: 0% > 1 -> no indicator
    # col_all_large: 100% > 1 -> no indicator (not between 1% and 99%)
    df_a = pd.DataFrame({
        "col_bimodal": [0.5, 0.5, 2.0, 3.0],
        "col_normal": [0.1, 0.2, 0.3, 0.4],
        "col_all_large": [10.0, 20.0, 30.0, 40.0],
    })

    # Test disabled
    ind_disabled = LargeValueIndicator(enabled=False)
    ind_disabled.fit(df_a)
    a_trans_dis = ind_disabled.transform(df_a)
    assert a_trans_dis.equals(df_a)

    # Test enabled
    ind_enabled = LargeValueIndicator(enabled=True)
    ind_enabled.fit(df_a)
    assert ind_enabled.indicator_cols_ == ["col_bimodal"]
    assert "col_bimodal__gt1" in ind_enabled.feature_names_out_
    assert "col_normal__gt1" not in ind_enabled.feature_names_out_

    # Transform B
    df_b = pd.DataFrame({
        "col_bimodal": [0.1, 5.0],
        "col_normal": [5.0, 5.0],  # Even if B has >1 values here, A didn't flag it!
        "col_all_large": [1.0, 2.0],
    })
    b_trans = ind_enabled.transform(df_b)
    assert list(b_trans.columns) == ["col_bimodal", "col_normal", "col_all_large", "col_bimodal__gt1"]
    assert list(b_trans["col_bimodal__gt1"]) == [0.0, 1.0]


# ==============================================================================
# Test 12: AdaptiveScaleWeightClassifier Feature Importances & Pipeline
# ==============================================================================
def test_adaptive_scale_weight_classifier_feature_importances_and_pipeline():
    """
    Verify AdaptiveScaleWeightClassifier forwards feature_importances_ and integrates
    seamlessly in an imblearn Pipeline.
    """
    from xgboost import XGBClassifier
    from src.pipelines import build_pipeline

    df = pd.DataFrame({
        "f1": np.random.randn(60),
        "f2": np.random.randn(60),
    })
    y = np.array([0] * 50 + [1] * 10)

    pipe = build_pipeline(model_name="M4", imbalance="S1")
    pipe.fit(df, y)

    # The classifier step is the AdaptiveScaleWeightClassifier
    clf_step = pipe.named_steps["classifier"]
    assert isinstance(clf_step, AdaptiveScaleWeightClassifier)

    # feature_importances_ must be accessible and forwarded
    importances = clf_step.feature_importances_
    assert len(importances) == 2
    assert np.all(importances >= 0.0)


# ==============================================================================
# Test 13: S1 Single Training Fold Scale Weight Calculation
# ==============================================================================
def test_s1_single_training_fold_weight_calculation():
    """
    Verify on real fold 0 (4,364 samples), scale_pos_weight is calculated strictly
    from that training fold (n_neg=4,224, n_pos=140 -> ratio ≈ 30.1714).
    """
    from src.cv import get_train_data, load_splits

    X_train, y_train = get_train_data()
    splits = load_splits()["cv_splits"]
    fold0_split = splits[0]

    tr_idx = fold0_split["train_idx"]
    # Map to X_train row positions
    train_orig_idx = load_splits()["train_idx"]
    orig_to_rel = {orig_i: rel_i for rel_i, orig_i in enumerate(train_orig_idx)}
    rel_tr_idx = np.array([orig_to_rel[i] for i in tr_idx])

    X_fold0 = X_train.iloc[rel_tr_idx]
    y_fold0 = y_train.iloc[rel_tr_idx]

    assert len(y_fold0) == 4364
    n_neg = int((y_fold0 == 0).sum())
    n_pos = int((y_fold0 == 1).sum())
    expected_weight = n_neg / n_pos

    assert n_neg == 4224
    assert n_pos == 140
    assert expected_weight == pytest.approx(30.17142857)

    # Fit M4 under S1 on this fold
    pipe = build_pipeline(model_name="M4", imbalance="S1")
    pipe.fit(X_fold0, y_fold0)

    clf_step = pipe.named_steps["classifier"]
    assert clf_step.scale_pos_weight_ == pytest.approx(expected_weight)
    assert clf_step.estimator_.get_params()["scale_pos_weight"] == pytest.approx(expected_weight)
