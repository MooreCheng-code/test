"""
Unit Tests for Cross-Validation and Data Splitting (src/cv.py).

Covers:
1. Non-overlapping train/val indices across all folds.
2. Complete exclusion of holdout test indices from all CV folds.
3. Full coverage guarantee: Every training sample validated exactly once per repeat.
4. Save and load integrity with dataset SHA-256 hash checks.
5. Error raising on label hash or sample size mismatch.
6. Test set shielding (get_test_data raises PermissionError by default).
7. Deterministic inner CV splits for nested tuning.
8. Smoke test: 5-fold CV with M1 (Logistic Regression) + S3 (SMOTE) on real train data.
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

from config import CV_FOLDS, DATA_PATH, TARGET_COL
from src.cv import (
    compute_y_hash,
    cross_validate_pipeline,
    get_test_data,
    get_train_data,
    load_splits,
    make_cv_splits,
    make_holdout_split,
    make_inner_splits,
    save_splits,
)
from src.pipelines import build_pipeline


# ==============================================================================
# Test 1 & 2 & 3: CV Splits Disjointness and Coverage
# ==============================================================================
def test_cv_splits_integrity_and_coverage():
    """
    Verify:
    1. In each fold, train_idx and val_idx are strictly disjoint.
    2. Holdout test set does not appear in any CV training or validation fold.
    3. In each repeat, the union of val_idx covers the training set exactly once without duplicates.
    """
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    train_idx, test_idx = make_holdout_split(df)
    test_set = set(test_idx)
    train_set = set(train_idx)

    # Test set and train set must be mutually exclusive
    assert len(test_set.intersection(train_set)) == 0
    assert len(test_set) + len(train_set) == len(df)

    y_train = df.iloc[train_idx][TARGET_COL].values
    cv_splits = make_cv_splits(y_train, train_orig_indices=train_idx)

    # Check each fold
    for split in cv_splits:
        tr_fold = set(split["train_idx"])
        val_fold = set(split["val_idx"])

        # Train and val in fold are disjoint
        assert len(tr_fold.intersection(val_fold)) == 0
        # Neither contains any test sample
        assert len(tr_fold.intersection(test_set)) == 0
        assert len(val_fold.intersection(test_set)) == 0

    # Check coverage per repeat
    repeats = sorted(list(set(s["repeat"] for s in cv_splits)))
    for r in repeats:
        val_indices_r = np.concatenate([s["val_idx"] for s in cv_splits if s["repeat"] == r])
        assert len(val_indices_r) == len(train_idx)
        assert set(val_indices_r) == train_set


# ==============================================================================
# Test 4 & 5: Save/Load Integrity and Hash Guardrail
# ==============================================================================
def test_splits_save_load_and_tamper_detection(tmp_path):
    """
    Verify splits serialization and ensure ValueError is raised if row count or
    target hash is altered.
    """
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    tmp_splits_file = tmp_path / "splits_test.joblib"

    # Save
    saved_payload = save_splits(df, splits_path=tmp_splits_file, overwrite=True)
    assert tmp_splits_file.exists()

    # Load and verify match
    loaded_payload = load_splits(df=df, splits_path=tmp_splits_file)
    assert np.array_equal(saved_payload["test_idx"], loaded_payload["test_idx"])
    assert saved_payload["metadata"]["y_hash"] == loaded_payload["metadata"]["y_hash"]

    # Tampered DataFrame 1: Row count mismatch
    df_short = df.iloc[:1000].copy()
    with pytest.raises(ValueError, match="Data row count mismatch"):
        load_splits(df=df_short, splits_path=tmp_splits_file)

    # Tampered DataFrame 2: Same length, but flipped one label
    df_tampered = df.copy()
    df_tampered.loc[0, TARGET_COL] = 1 - df_tampered.loc[0, TARGET_COL]
    with pytest.raises(ValueError, match="Target label hash mismatch"):
        load_splits(df=df_tampered, splits_path=tmp_splits_file)


# ==============================================================================
# Test 6: Test Set Shielding
# ==============================================================================
def test_test_set_access_protection():
    """
    Verify get_test_data() raises PermissionError by default to prevent premature leakage.
    """
    with pytest.raises(PermissionError, match="restricted until Phase E6"):
        get_test_data()

    with pytest.raises(PermissionError, match="restricted until Phase E6"):
        get_test_data(i_know_this_is_final_evaluation=False)

    # Explicit confirmation allows access
    X_test, y_test = get_test_data(i_know_this_is_final_evaluation=True)
    assert len(X_test) == len(y_test)
    assert len(X_test) > 0


# ==============================================================================
# Test 7: Deterministic Inner Splits for Nested CV
# ==============================================================================
def test_make_inner_splits_determinism():
    """
    Verify make_inner_splits produces identical splits on repeated calls with the same input.
    """
    outer_idx = np.arange(100)
    y = np.array([0] * 90 + [1] * 10)

    splits_1 = make_inner_splits(outer_idx, y, outer_id="r0_f0", n_splits=3)
    splits_2 = make_inner_splits(outer_idx, y, outer_id="r0_f0", n_splits=3)

    assert len(splits_1) == 3
    for (tr1, val1), (tr2, val2) in zip(splits_1, splits_2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(val1, val2)

    # Different outer_id should generate different splits
    splits_diff = make_inner_splits(outer_idx, y, outer_id="r0_f1", n_splits=3)
    assert not np.array_equal(splits_1[0][0], splits_diff[0][0])


# ==============================================================================
# Test 8: Smoke Test on Real Training Data (5 Folds, 1 Repeat)
# ==============================================================================
def test_cross_validate_smoke_test():
    """
    Smoke test: Run a single repeat of 5-fold CV using Logistic Regression with S3 (SMOTE)
    on real training data. Verifies execution, evaluates metrics, and checks for leakage signs.
    """
    X_train, y_train = get_train_data()
    splits_data = load_splits()

    # Filter for first repeat only (repeat 0, 5 folds)
    splits_r0 = [s for s in splits_data["cv_splits"] if s["repeat"] == 0]
    assert len(splits_r0) == CV_FOLDS

    # Map outer indices to X_train row positions (0 to len(X_train)-1)
    train_orig_idx = splits_data["train_idx"]
    orig_to_rel = {orig_i: rel_i for rel_i, orig_i in enumerate(train_orig_idx)}

    rel_splits = []
    for s in splits_r0:
        rel_splits.append({
            "repeat": s["repeat"],
            "fold": s["fold"],
            "split_id": s["split_id"],
            "train_idx": np.array([orig_to_rel[i] for i in s["train_idx"]]),
            "val_idx": np.array([orig_to_rel[i] for i in s["val_idx"]]),
        })

    # Factory function for fresh M1 + S3 pipeline
    def pipeline_factory():
        return build_pipeline(
            model_name="M1",
            imbalance="S3",
            scaler="quantile",
            use_winsorizer=True,
            use_corr_filter=True,
        )

    # Run CV
    cv_output = cross_validate_pipeline(
        pipeline_or_factory=pipeline_factory,
        X=X_train,
        y=y_train,
        splits=rel_splits,
    )

    fold_df = cv_output["fold_results"]
    assert len(fold_df) == CV_FOLDS
    assert "pr_auc" in fold_df.columns
    assert "alert_ratio@recall_0.95" in fold_df.columns
    assert "roc_auc" in fold_df.columns

    mean_pr_auc = fold_df["pr_auc"].mean()
    mean_roc_auc = fold_df["roc_auc"].mean()
    mean_alert_ratio_95 = fold_df["alert_ratio@recall_0.95"].mean()

    # Check that ROC-AUC is healthy and not unnaturally close to 1.0 (leakage check)
    assert 0.70 < mean_roc_auc < 0.98, f"Suspicious ROC-AUC: {mean_roc_auc:.4f}"
    assert mean_pr_auc > 0.10
    assert mean_alert_ratio_95 > 1.0
