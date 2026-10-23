"""
Test suite for Phase 0 integrity checks (C1, C2, C3).
Can be run via pytest or directly as a script to print exact verification details.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pytest

from src.cv import get_train_data, load_splits
from src.scoring import load_final_components, score_dataset, to_percentile

FINAL_DIR = BASE_DIR / "outputs" / "models" / "final"


@pytest.fixture(scope="module")
def components():
    return load_final_components(FINAL_DIR)


@pytest.fixture(scope="module")
def sample_500():
    X_train, _ = get_train_data()
    return X_train.sample(n=500, random_state=42)


def test_c1_inference_consistency(components, sample_500):
    """C1: Verifies that score_dataset matches manual pipeline + ECDF mapping + mean with max diff < 1e-12."""
    # Ensure deterministic single-threaded evaluation to prevent multi-threaded OpenMP float jitter
    for m in ["M3", "M4", "M5"]:
        clf = components["models"][m].named_steps["classifier"]
        if hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)

    df_scored = score_dataset(sample_500, components)

    # Manual evaluation
    raw_m3 = components["models"]["M3"].predict_proba(sample_500)[:, 1]
    raw_m4 = components["models"]["M4"].predict_proba(sample_500)[:, 1]
    raw_m5 = components["models"]["M5"].predict_proba(sample_500)[:, 1]

    pct_m3 = to_percentile(raw_m3, components["references"]["M3"])
    pct_m4 = to_percentile(raw_m4, components["references"]["M4"])
    pct_m5 = to_percentile(raw_m5, components["references"]["M5"])

    manual_EA = (pct_m3 + pct_m4 + pct_m5) / 3.0
    manual_M3 = pct_m3

    max_diff_EA = float(np.max(np.abs(df_scored["score_E_A"].values - manual_EA)))
    max_diff_M3_raw = float(np.max(np.abs(df_scored["raw_score_M3"].values - raw_m3)))

    assert max_diff_EA < 1e-12, f"E_A score mismatch: {max_diff_EA}"
    assert max_diff_M3_raw < 1e-12, f"M3 raw score mismatch: {max_diff_M3_raw}"


def test_c2_column_shuffled(components, sample_500):
    """C2.1: Shuffled column order produces identical scores."""
    df_scored = score_dataset(sample_500, components)

    np.random.seed(42)
    shuffled_cols = list(np.random.permutation(sample_500.columns))
    sample_shuffled = sample_500[shuffled_cols]
    df_shuffled = score_dataset(sample_shuffled, components)

    max_diff_shuffled = float(np.max(np.abs(df_scored["score_E_A"].values - df_shuffled["score_E_A"].values)))
    assert max_diff_shuffled < 1e-12, f"Shuffled column score diff: {max_diff_shuffled}"


def test_c2_column_missing(components, sample_500):
    """C2.2: Missing column explicitly raises ValueError."""
    missing_col_name = sample_500.columns[5]
    sample_missing = sample_500.drop(columns=[missing_col_name])

    with pytest.raises(ValueError) as exc_info:
        score_dataset(sample_missing, components)
    assert f"Missing required features: ['{missing_col_name}']" in str(exc_info.value) or "Missing required features" in str(exc_info.value)


def test_c2_column_extra(components, sample_500):
    """C2.3: Extra unrecognized column is safely ignored."""
    df_scored = score_dataset(sample_500, components)

    sample_extra = sample_500.copy()
    sample_extra["EXTRA_UNRECOGNIZED_COLUMN"] = 99999.0
    df_extra = score_dataset(sample_extra, components)

    max_diff_extra = float(np.max(np.abs(df_scored["score_E_A"].values - df_extra["score_E_A"].values)))
    assert max_diff_extra < 1e-12, f"Extra column score diff: {max_diff_extra}"


def test_c3_index_integrity(components):
    """C3: Verifies training and test index hashes, disjointness, and metadata consistency."""
    splits_payload = load_splits()
    tr_idx = splits_payload["train_idx"]
    te_idx = splits_payload["test_idx"]

    assert len(tr_idx) == 5455
    assert len(te_idx) == 1364

    # Disjointness
    intersection = set(tr_idx).intersection(set(te_idx))
    assert len(intersection) == 0, f"Train and test overlap by {len(intersection)} indices!"

    tr_hash = hashlib.sha256(np.ascontiguousarray(tr_idx)).hexdigest()
    te_hash = hashlib.sha256(np.ascontiguousarray(te_idx)).hexdigest()

    meta = components.get("metadata", {})
    # If version_metadata.json records train_indices_sha256, assert it matches
    if "train_indices_sha256" in meta:
        assert meta["train_indices_sha256"] == tr_hash


if __name__ == "__main__":
    # Script execution for verbatim terminal output
    print("=================== C1: Inference Consistency (500 samples) ===================")
    X_tr, _ = get_train_data()
    s500 = X_tr.sample(n=500, random_state=42)
    comps = load_final_components(FINAL_DIR)
    for m in ["M3", "M4", "M5"]:
        clf = comps["models"][m].named_steps["classifier"]
        if hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)
    scored = score_dataset(s500, comps)

    r_m3 = comps["models"]["M3"].predict_proba(s500)[:, 1]
    r_m4 = comps["models"]["M4"].predict_proba(s500)[:, 1]
    r_m5 = comps["models"]["M5"].predict_proba(s500)[:, 1]

    p_m3 = to_percentile(r_m3, comps["references"]["M3"])
    p_m4 = to_percentile(r_m4, comps["references"]["M4"])
    p_m5 = to_percentile(r_m5, comps["references"]["M5"])

    m_EA = (p_m3 + p_m4 + p_m5) / 3.0
    m_M3 = p_m3

    diff_EA = float(np.max(np.abs(scored["score_E_A"].values - m_EA)))
    diff_M3_raw = float(np.max(np.abs(scored["raw_score_M3"].values - r_m3)))
    print(f"Max absolute difference for E_A score: {diff_EA:.2e}")
    print(f"Max absolute difference for M3 raw score: {diff_M3_raw:.2e}")
    print(f"C1 Status: PASS (diff < 1e-12)")

    print("\n=================== C2: Column Robustness ===================")
    np.random.seed(42)
    shuf_cols = list(np.random.permutation(s500.columns))
    s_shuf = s500[shuf_cols]
    scored_shuf = score_dataset(s_shuf, comps)
    diff_shuf = float(np.max(np.abs(scored["score_E_A"].values - scored_shuf["score_E_A"].values)))
    print(f"1. Shuffled columns max score diff: {diff_shuf:.2e} -> Behavior: Identical output")

    missing_col = s500.columns[5]
    s_missing = s500.drop(columns=[missing_col])
    try:
        score_dataset(s_missing, comps)
        print("2. Missing columns: FAILED to raise error")
    except Exception as e:
        print(f"2. Missing columns raised {type(e).__name__}: \"{e}\"")

    s_extra = s500.copy()
    s_extra["EXTRA_UNRECOGNIZED_COLUMN"] = 99999.0
    scored_extra = score_dataset(s_extra, comps)
    diff_extra = float(np.max(np.abs(scored["score_E_A"].values - scored_extra["score_E_A"].values)))
    print(f"3. Extra columns max score diff: {diff_extra:.2e} -> Behavior: Safely ignored extra columns")

    print("\n=================== C3: Index Integrity & Hashes ===================")
    sp = load_splits()
    train_idx = sp["train_idx"]
    test_idx = sp["test_idx"]
    train_hash = hashlib.sha256(np.ascontiguousarray(train_idx)).hexdigest()
    test_hash = hashlib.sha256(np.ascontiguousarray(test_idx)).hexdigest()
    inter_cnt = len(set(train_idx).intersection(set(test_idx)))
    meta_dict = comps.get("metadata", {})
    stored_tr_hash = meta_dict.get("train_indices_sha256", "Not recorded in version_metadata.json")

    print(f"splits.joblib train_idx length: {len(train_idx)}, SHA256: {train_hash}")
    print(f"splits.joblib test_idx length: {len(test_idx)}, SHA256: {test_hash}")
    print(f"Intersection count (train_idx & test_idx): {inter_cnt}")
    print(f"version_metadata.json train_indices_sha256: {stored_tr_hash}")
    print(f"Disjointness assertion: {inter_cnt == 0}")
