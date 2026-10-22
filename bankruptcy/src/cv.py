"""
Cross-Validation and Dataset Splitting Module for Bankruptcy Prediction.

Ensures leak-free evaluation:
1. Outer Holdout Split: Fixed 80/20 stratified split locked until final evaluation (E6).
2. Repeated Stratified K-Fold: 5 folds x 3 repeats (15 outer validation folds).
3. Splits Serialization: Stored in outputs/models/splits.joblib with dataset hash guardrails.
4. Test Set Shielding: get_test_data() requires explicit confirmation flag.
5. Deterministic Inner Splits: Procedurally generated for nested hyperparameter tuning.
6. cross_validate_pipeline: Executes multi-repeat CV, records fold metrics, pooled OOF scores,
   and pooled repeat metrics.
"""

import hashlib
import sys
from pathlib import Path
from typing import Callable, Union

# Ensure config is importable regardless of working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from config import (
    CV_FOLDS,
    CV_REPEATS,
    DATA_PATH,
    MODELS_DIR,
    SEED,
    TARGET_COL,
    TEST_SIZE,
)
from src.metrics import evaluate_scores
from src.pipelines import get_score


SPLITS_PATH = MODELS_DIR / "splits.joblib"


def compute_y_hash(y: Union[np.ndarray, pd.Series, Sequence]) -> str:
    """Compute SHA-256 hash of target label array."""
    y_bytes = np.ascontiguousarray(np.asarray(y).ravel()).tobytes()
    return hashlib.sha256(y_bytes).hexdigest()


def make_holdout_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate outer holdout train and test indices using stratified split.

    Returns:
        tuple[np.ndarray, np.ndarray]: (train_idx, test_idx)
    """
    y = df[TARGET_COL].values
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED,
    )
    return np.sort(train_idx), np.sort(test_idx)


def make_cv_splits(
    y_train: Union[np.ndarray, pd.Series],
    train_orig_indices: Union[np.ndarray, None] = None,
) -> list[dict]:
    """
    Generate Repeated Stratified K-Fold CV splits on the training set.

    Parameters:
        y_train: Target labels of the training set.
        train_orig_indices: Original row indices in the full dataset.

    Returns:
        list[dict]: List of split dictionaries, each containing:
            - 'repeat': Repeat index (0 to CV_REPEATS - 1)
            - 'fold': Fold index (0 to CV_FOLDS - 1)
            - 'split_id': Unique split identifier (e.g. 'r0_f1')
            - 'train_idx': Array of training indices (mapped to original df rows)
            - 'val_idx': Array of validation indices (mapped to original df rows)
    """
    y_arr = np.asarray(y_train).ravel()
    n_train = len(y_arr)

    if train_orig_indices is None:
        train_orig_indices = np.arange(n_train)
    else:
        train_orig_indices = np.asarray(train_orig_indices)

    rskf = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=SEED,
    )

    splits = []
    for split_idx, (sub_train_idx, sub_val_idx) in enumerate(rskf.split(np.zeros(n_train), y_arr)):
        repeat_num = split_idx // CV_FOLDS
        fold_num = split_idx % CV_FOLDS

        splits.append({
            "repeat": int(repeat_num),
            "fold": int(fold_num),
            "split_id": f"r{repeat_num}_f{fold_num}",
            "train_idx": train_orig_indices[sub_train_idx],
            "val_idx": train_orig_indices[sub_val_idx],
        })

    return splits


def save_splits(
    df: pd.DataFrame,
    splits_path: Path = SPLITS_PATH,
    overwrite: bool = False,
) -> dict:
    """
    Generate and save test indices and CV fold splits with integrity metadata.
    """
    splits_path = Path(splits_path)
    if splits_path.exists() and not overwrite:
        return load_splits(df=df, splits_path=splits_path)

    splits_path.parent.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = make_holdout_split(df)
    y_train = df.iloc[train_idx][TARGET_COL].values
    cv_splits = make_cv_splits(y_train, train_orig_indices=train_idx)

    y_all = df[TARGET_COL].values
    metadata = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_repeats": CV_REPEATS,
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "y_hash": compute_y_hash(y_all),
    }

    payload = {
        "metadata": metadata,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "cv_splits": cv_splits,
    }

    joblib.dump(payload, splits_path)
    return payload


def load_splits(
    df: Union[pd.DataFrame, None] = None,
    splits_path: Path = SPLITS_PATH,
) -> dict:
    """
    Load saved splits and verify data integrity against df.

    Raises:
        FileNotFoundError: If splits file does not exist.
        ValueError: If row count or target hash do not match.
    """
    splits_path = Path(splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found at {splits_path}. Run save_splits() first."
        )

    payload = joblib.load(splits_path)
    meta = payload["metadata"]

    if df is not None:
        if len(df) != meta["n_total"]:
            raise ValueError(
                f"Data row count mismatch: saved splits expected {meta['n_total']} rows, "
                f"but provided DataFrame has {len(df)} rows."
            )

        current_hash = compute_y_hash(df[TARGET_COL].values)
        if current_hash != meta["y_hash"]:
            raise ValueError(
                "Target label hash mismatch! The provided dataset does not match the data "
                "used to generate splits.joblib."
            )

    return payload


def get_train_data(data_path: Union[Path, str, None] = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retrieve X_train and y_train for development and cross-validation.
    Guarantees that test samples are never returned.
    """
    if data_path is None:
        data_path = DATA_PATH
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    if not SPLITS_PATH.exists():
        save_splits(df)

    splits_data = load_splits(df=df)
    train_idx = splits_data["train_idx"]

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    return X_train, y_train


def get_test_data(
    i_know_this_is_final_evaluation: bool = False,
    data_path: Union[Path, str, None] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retrieve X_test and y_test strictly for Phase E6 final evaluation.

    Raises:
        PermissionError: If called without i_know_this_is_final_evaluation=True.
    """
    if not i_know_this_is_final_evaluation:
        raise PermissionError(
            "Access to the holdout test set is restricted until Phase E6 final evaluation! "
            "Call get_test_data(i_know_this_is_final_evaluation=True) only in final_eval.py."
        )

    if data_path is None:
        data_path = DATA_PATH
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    if not SPLITS_PATH.exists():
        save_splits(df)

    splits_data = load_splits(df=df)
    test_idx = splits_data["test_idx"]

    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_test, y_test


def make_inner_splits(
    outer_train_idx: np.ndarray,
    y: Union[np.ndarray, pd.Series],
    outer_id: Union[int, str],
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Deterministically generate inner CV splits for nested hyperparameter tuning.

    Parameters:
        outer_train_idx: Array of indices of the outer training fold.
        y: Full target array.
        outer_id: Unique identifier for the outer fold (e.g. integer or 'r0_f1').
        n_splits: Number of inner folds (default 3).

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of (inner_train_idx, inner_val_idx),
            with indices relative to outer_train_idx.
    """
    outer_train_idx = np.asarray(outer_train_idx)
    y_outer = np.asarray(y)[outer_train_idx]

    # Deterministic seed unique to this outer fold
    id_hash = int(hashlib.md5(str(outer_id).encode("utf-8")).hexdigest()[:7], 16)
    inner_seed = (SEED + id_hash) % (2**31 - 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=inner_seed)

    inner_splits = []
    dummy_x = np.zeros(len(outer_train_idx))
    for in_train_sub, in_val_sub in skf.split(dummy_x, y_outer):
        inner_splits.append((
            outer_train_idx[in_train_sub],
            outer_train_idx[in_val_sub],
        ))

    return inner_splits


def cross_validate_pipeline(
    pipeline_or_factory: Union[Callable[[], Any], Any],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    splits: list[dict],
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Execute cross-validation across all provided splits.

    Verifications:
    - In each repeat, every training sample must be validated exactly once.
    - OOF scores are pooled across all folds per repeat.

    Returns:
        dict containing:
            - 'fold_results': DataFrame of individual fold metrics.
            - 'oof_scores': ndarray of shape (n_repeats, n_samples).
            - 'oof_summary': DataFrame of pooled OOF metrics for each repeat.
    """
    X_df = X.copy().reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_ser = y.copy().reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y)
    n_samples = len(X_df)

    # Determine repeats count
    repeats_list = sorted(list(set(s["repeat"] for s in splits)))
    n_repeats = len(repeats_list)

    # 1. Verify coverage: In each repeat, each sample must be validated exactly once
    for r in repeats_list:
        val_indices_r = np.concatenate([s["val_idx"] for s in splits if s["repeat"] == r])
        if len(val_indices_r) != n_samples or len(set(val_indices_r)) != n_samples:
            raise AssertionError(
                f"Repeat {r} does not validate every sample exactly once! "
                f"Total validated: {len(val_indices_r)}, Unique: {len(set(val_indices_r))}, Expected: {n_samples}."
            )

    oof_scores = np.full((n_repeats, n_samples), fill_value=np.nan, dtype=float)
    fold_records = []

    for split in splits:
        r = split["repeat"]
        f = split["fold"]
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

        X_train_f = X_df.iloc[train_idx]
        y_train_f = y_ser.iloc[train_idx]
        X_val_f = X_df.iloc[val_idx]
        y_val_f = y_ser.iloc[val_idx]

        # Build fresh pipeline instance
        if callable(pipeline_or_factory):
            pipe = pipeline_or_factory()
        else:
            pipe = clone(pipeline_or_factory)

        # Fit exclusively on this training fold
        pipe.fit(X_train_f, y_train_f)

        # Predict continuous risk scores on validation fold
        val_scores = get_score(pipe, X_val_f)

        # Store in OOF array
        repeat_idx = repeats_list.index(r)
        oof_scores[repeat_idx, val_idx] = val_scores

        # Evaluate fold-level metrics
        fold_metrics = evaluate_scores(y_val_f, val_scores)
        fold_metrics["repeat"] = r
        fold_metrics["fold"] = f
        fold_records.append(fold_metrics)

    # Verify no unpredicted holes in OOF scores
    if np.isnan(oof_scores).any():
        raise AssertionError("OOF scores array contains unpredicted NaN values!")

    fold_results_df = pd.DataFrame(fold_records)

    # Compute pooled OOF metrics for each repeat
    oof_records = []
    for r_idx, r in enumerate(repeats_list):
        pooled_scores = oof_scores[r_idx]
        pooled_metrics = evaluate_scores(y_ser.values, pooled_scores)
        pooled_metrics["repeat"] = r
        oof_records.append(pooled_metrics)

    oof_summary_df = pd.DataFrame(oof_records)

    return {
        "fold_results": fold_results_df,
        "oof_scores": oof_scores,
        "oof_summary": oof_summary_df,
    }
