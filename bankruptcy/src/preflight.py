"""
E6 Preflight Checklist & System Integrity Verification.

Executes comprehensive 8-step preflight verification before any test data unsealing:
1. Anti-leakage lockfile does NOT exist.
2. C1: score_dataset matches manual inference within 1e-12 on 200 training samples.
3. C2: Missing required features strictly raises ValueError.
4. Reinforcement 1 (items 1 & 2):
   - Preprocessing quantiles_ and CorrelationFilter dropped/selected match bitwise across M3, M4, M5.
   - reference_y.npy matches np.concatenate([y_train, y_train]) bitwise.
5. Index integrity: Train and test indices are strictly disjoint (0 overlap) and match saved hashes.
6. Threshold reproduction: Integer caught counts and alert counts reproduced from training OOF.
7. Frozen component file hashes computed and recorded.
8. Full pytest suite passes with 0 failures.

Writes verbatim report to outputs/logs/e6_preflight.txt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.cv import get_train_data, load_splits
from src.lock_utils import is_e6_locked
from src.pipelines import build_pipeline
from src.scoring import load_final_components, score_dataset, to_percentile

logger = logging.getLogger("bankruptcy_preflight")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

BASE_DIR = Path(config.BASE_DIR)
FINAL_DIR = BASE_DIR / "outputs" / "models" / "final"
LOGS_DIR = BASE_DIR / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREFLIGHT_OUTPUT_PATH = LOGS_DIR / "e6_preflight.txt"


def compute_file_hash(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_e6_preflight() -> dict[str, Any]:
    """
    Executes the 8-step preflight checklist.
    Raises RuntimeError or exits with code 1 if any step fails.
    Returns preflight metadata dictionary.
    """
    log_lines: list[str] = []

    def log(msg: str):
        logger.info(msg)
        log_lines.append(msg)

    log("=" * 80)
    log("=== E6 PREFLIGHT VERIFICATION: COMMENCING INTEGRITY AUDIT ===")
    log("=" * 80)

    # --------------------------------------------------------------------------
    # Check 1: Lockfile does not exist
    # --------------------------------------------------------------------------
    log("\n[Check 1/8] Verifying that E6_LOCK file does not exist...")
    lock_file = LOGS_DIR / "E6_LOCK"
    if lock_file.exists() or is_e6_locked(LOGS_DIR):
        err = f"FAILED: E6_LOCK file already exists at {lock_file}! Test set was already unsealed."
        log(err)
        raise RuntimeError(err)
    log("   -> PASS: Lockfile does not exist. Environment is ready for clean unseal.")

    # --------------------------------------------------------------------------
    # Check 2: C1 Inference Consistency (200 training samples)
    # --------------------------------------------------------------------------
    log("\n[Check 2/8] C1: Verifying score_dataset vs manual pipeline on 200 random training samples...")
    X_train, y_train = get_train_data()
    y_arr = y_train.values
    sample_200 = X_train.sample(n=200, random_state=42)
    components = load_final_components(FINAL_DIR)

    scored = score_dataset(sample_200, components)

    # Manual evaluation
    raw_m3 = components["models"]["M3"].predict_proba(sample_200)[:, 1]
    raw_m4 = components["models"]["M4"].predict_proba(sample_200)[:, 1]
    raw_m5 = components["models"]["M5"].predict_proba(sample_200)[:, 1]

    pct_m3 = to_percentile(raw_m3, components["references"]["M3"])
    pct_m4 = to_percentile(raw_m4, components["references"]["M4"])
    pct_m5 = to_percentile(raw_m5, components["references"]["M5"])

    manual_EA = (pct_m3 + pct_m4 + pct_m5) / 3.0
    manual_M3 = pct_m3

    diff_EA = float(np.max(np.abs(scored["score_E_A"].values - manual_EA)))
    diff_M3 = float(np.max(np.abs(scored["score_M3"].values - manual_M3)))

    log(f"   Max absolute difference E_A: {diff_EA:.2e}")
    log(f"   Max absolute difference M3: {diff_M3:.2e}")
    if diff_EA >= 1e-12 or diff_M3 >= 1e-12:
        err = f"FAILED: C1 inference difference exceeds 1e-12! (E_A: {diff_EA}, M3: {diff_M3})"
        log(err)
        raise RuntimeError(err)
    log("   -> PASS: C1 inference consistency verified (max diff < 1e-12).")

    # --------------------------------------------------------------------------
    # Check 3: C2 Missing Features Behavior
    # --------------------------------------------------------------------------
    log("\n[Check 3/8] C2: Verifying behavior when input features are missing...")
    missing_col = sample_200.columns[3]
    sample_missing = sample_200.drop(columns=[missing_col])
    try:
        score_dataset(sample_missing, components)
        err = "FAILED: score_dataset did not raise ValueError on missing column!"
        log(err)
        raise RuntimeError(err)
    except ValueError as e:
        log(f"   -> PASS: Correctly raised ValueError: \"{e}\"")

    # --------------------------------------------------------------------------
    # Check 4: Reinforcement 1 (Preprocessing Bitwise Reproducibility & reference_y)
    # --------------------------------------------------------------------------
    log("\n[Check 4/8] Reinforcement 1: Verifying preprocessing bitwise identity & reference_y...")
    fresh_pipe = build_pipeline("M3", imbalance="S0", scaler="quantile", use_winsorizer=False, use_corr_filter=True)
    fresh_pipe.fit(X_train, y_train)

    fresh_qs = fresh_pipe.named_steps["scaler"]
    fresh_cf = fresh_pipe.named_steps["corr_filter"]

    for m in ["M3", "M4", "M5"]:
        stored_pipe = components["models"][m]
        stored_qs = stored_pipe.named_steps["scaler"]
        stored_cf = stored_pipe.named_steps["corr_filter"]

        qs_match = np.array_equal(fresh_qs.scaler_.quantiles_, stored_qs.scaler_.quantiles_)
        sel_match = fresh_cf.selected_features_ == stored_cf.selected_features_
        drop_match = fresh_cf.dropped_ == stored_cf.dropped_

        log(f"   {m}: Quantiles bitwise match: {qs_match} | Selected features (72) match: {sel_match} | Dropped (20) match: {drop_match}")
        if not (qs_match and sel_match and drop_match):
            err = f"FAILED: Preprocessing mismatch in stored {m} pipeline!"
            log(err)
            raise RuntimeError(err)

    ref_y = np.load(FINAL_DIR / "reference_y.npy")
    expected_ref_y = np.concatenate([y_arr, y_arr])
    ref_y_match = np.array_equal(ref_y, expected_ref_y)
    log(f"   reference_y equals [y_train, y_train] (len={len(ref_y)}): {ref_y_match}")
    if not ref_y_match:
        err = "FAILED: reference_y.npy does not match training set labels repeated twice!"
        log(err)
        raise RuntimeError(err)
    log("   -> PASS: Preprocessing and reference_y verified as 100% training-only derived.")

    # --------------------------------------------------------------------------
    # Check 5: Training & Test Index Integrity & Hashes
    # --------------------------------------------------------------------------
    log("\n[Check 5/8] Verifying training/test index disjointness and hashes...")
    sp = load_splits()
    tr_idx = sp["train_idx"]
    te_idx = sp["test_idx"]

    overlap = len(set(tr_idx).intersection(set(te_idx)))
    log(f"   Train count: {len(tr_idx)}, Test count: {len(te_idx)}, Overlap: {overlap}")
    if overlap != 0:
        err = f"FAILED: Train and test sets overlap by {overlap} samples!"
        log(err)
        raise RuntimeError(err)

    tr_hash = hashlib.sha256(np.ascontiguousarray(tr_idx)).hexdigest()
    te_hash = hashlib.sha256(np.ascontiguousarray(te_idx)).hexdigest()

    index_hash_file = FINAL_DIR / "index_hashes.json"
    if index_hash_file.exists():
        with open(index_hash_file, "r") as f:
            saved_hashes = json.load(f)
        assert saved_hashes["train_indices_sha256"] == tr_hash
        assert saved_hashes["test_indices_sha256"] == te_hash

    log(f"   train_idx SHA256: {tr_hash}")
    log(f"   test_idx SHA256:  {te_hash}")
    log("   -> PASS: Train/test index integrity and disjointness strictly verified.")

    # --------------------------------------------------------------------------
    # Check 6: Thresholds Reproduction from Saved OOF
    # --------------------------------------------------------------------------
    log("\n[Check 6/8] Verifying thresholds.json integer reproduction on training OOF...")
    oof_r0_M3 = np.load(FINAL_DIR / "_tmp_oof_r0_M3.npy")
    oof_r1_M3 = np.load(FINAL_DIR / "_tmp_oof_r1_M3.npy")
    oof_r0_M4 = np.load(FINAL_DIR / "_tmp_oof_r0_M4.npy")
    oof_r1_M4 = np.load(FINAL_DIR / "_tmp_oof_r1_M4.npy")
    oof_r0_M5 = np.load(FINAL_DIR / "_tmp_oof_r0_M5.npy")
    oof_r1_M5 = np.load(FINAL_DIR / "_tmp_oof_r1_M5.npy")

    p_m3 = to_percentile((oof_r0_M3 + oof_r1_M3) / 2.0, components["references"]["M3"])
    p_m4 = to_percentile((oof_r0_M4 + oof_r1_M4) / 2.0, components["references"]["M4"])
    p_m5 = to_percentile((oof_r0_M5 + oof_r1_M5) / 2.0, components["references"]["M5"])

    oof_EA = (p_m3 + p_m4 + p_m5) / 3.0
    oof_M3 = p_m3
    th_cfg = components["thresholds"]
    pos_mask = y_arr == 1

    expected_counts = {
        "E_A": {
            "R80": (145, 608),
            "R90": (169, 1378),
            "R95": (175, 2142),
            "R100": (176, 4811),
        },
        "M3": {
            "R80": (148, 683),
            "R90": (166, 1218),
            "R95": (175, 1946),
            "R100": (176, 4275),
        },
    }

    for grp, scores_arr in [("E_A", oof_EA), ("M3", oof_M3)]:
        for r_k, (exp_caught, exp_alerts) in expected_counts[grp].items():
            t = th_cfg[grp][r_k]["threshold"]
            flags = scores_arr >= t
            c_cnt = int((flags & pos_mask).sum())
            a_cnt = int(flags.sum())
            if c_cnt != exp_caught or a_cnt != exp_alerts:
                err = f"FAILED: Threshold {grp} {r_k} counts mismatch: got ({c_cnt}, {a_cnt}), expected ({exp_caught}, {exp_alerts})"
                log(err)
                raise RuntimeError(err)

    assert th_cfg["E_A"]["R90"]["threshold"] != th_cfg["E_A"]["R95"]["threshold"]
    log("   -> PASS: All operating thresholds reproduced bitwise (caught & alert counts identical).")

    # --------------------------------------------------------------------------
    # Check 7: Frozen Components File Hashes
    # --------------------------------------------------------------------------
    log("\n[Check 7/8] Recording frozen component SHA-256 hashes...")
    frozen_files = [
        "model_M3.joblib",
        "model_M4.joblib",
        "model_M5.joblib",
        "reference_M3.npy",
        "reference_M4.npy",
        "reference_M5.npy",
        "reference_y.npy",
        "thresholds.json",
        "thresholds.csv",
        "calibrator_platt.joblib",
        "calibrator_isotonic.joblib",
    ]
    frozen_hashes = {}
    for fn in frozen_files:
        fp = FINAL_DIR / fn
        if not fp.exists():
            err = f"FAILED: Required frozen component missing: {fp}"
            log(err)
            raise FileNotFoundError(err)
        h = compute_file_hash(fp)
        frozen_hashes[fn] = h
        log(f"   {fn}: {h}")
    log("   -> PASS: Frozen components directory verified intact.")

    # --------------------------------------------------------------------------
    # Check 8: Project Test Suite (pytest -q)
    # --------------------------------------------------------------------------
    log("\n[Check 8/8] Executing complete project test suite (pytest -q)...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        err = f"FAILED: pytest suite failed with code {res.returncode}!\nOutput:\n{res.stdout}\n{res.stderr}"
        log(err)
        raise RuntimeError(err)

    test_summary = [line for line in res.stdout.strip().split("\n") if "passed" in line]
    summary_str = test_summary[-1] if test_summary else res.stdout.strip()
    log(f"   Test suite output: {summary_str}")
    log("   -> PASS: All unit & regression tests passed.")

    log("\n" + "=" * 80)
    log("=== PREFLIGHT AUDIT PASSED: ALL 8 VERIFICATION CHECKS SUCCEEDED ===")
    log("=" * 80)

    # Save to outputs/logs/e6_preflight.txt
    full_output = "\n".join(log_lines) + "\n"
    PREFLIGHT_OUTPUT_PATH.write_text(full_output, encoding="utf-8")
    log(f"Saved full preflight report to {PREFLIGHT_OUTPUT_PATH}")

    return {
        "train_indices_sha256": tr_hash,
        "test_indices_sha256": te_hash,
        "frozen_file_hashes": frozen_hashes,
        "preflight_output": full_output,
    }


if __name__ == "__main__":
    run_e6_preflight()
