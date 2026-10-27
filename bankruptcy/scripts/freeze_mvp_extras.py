"""
scripts/freeze_mvp_extras.py

Generates and freezes MVP expansion artifacts in outputs/models/mvp_extras/:
1. feature_manifest.json (95 raw features, 72 model features, 95 training medians)
2. model_M1.joblib (full training set L2-regularized Logistic Regression)
3. reference_M1.npy (10-fold Repeated Stratified 5-Fold OOF reference distribution, length 10,910)
4. calibrator_M1_platt.joblib, calibrator_M3_platt.joblib, calibrator_M4_platt.joblib
5. thresholds_extras.json (M1 and M4 thresholds, plus E_A, M3, and demo_r0_f0_E_A inner thresholds)
6. demo_r0_f0_M4.joblib, demo_r0_f0_M1.joblib (out-of-fold demo models for zero runtime cache dependency)
7. oof_demo_batch.csv (1,091 validation samples from r0_f0 with true labels and pre-computed OOF scores)
8. mvp_extras_hashes.json (SHA-256 hash manifest)

NEVER touches outputs/models/final/ or the holdout test set.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.cv import get_train_data, load_splits
from src.pipelines import build_pipeline
from src.thresholding import derive_operating_threshold, fit_single_model_platt
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("freeze_mvp_extras")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "outputs" / "models" / "final"
BEST_PARAMS_DIR = PROJECT_ROOT / "outputs" / "models" / "best_params"
OOF_DIR = PROJECT_ROOT / "outputs" / "models" / "oof"
E5_CACHE_DIR = PROJECT_ROOT / "outputs" / "models" / "e5_cache"
E2_CACHE_DIR = PROJECT_ROOT / "outputs" / "logs" / "e2_cache"
E4A_TRANSFER_PATH = PROJECT_ROOT / "outputs" / "tables" / "e4a_threshold_transfer_per_fold.csv"
DEFAULTS_JSON_PATH = PROJECT_ROOT / "app" / "assets" / "defaults.json"
MVP_EXTRAS_DIR = PROJECT_ROOT / "outputs" / "models" / "mvp_extras"

SEED = config.SEED
TARGET_RECALLS = [0.80, 0.90, 0.95, 1.00]


def main() -> None:
    logger.info("=== Starting freeze_mvp_extras.py ===")
    MVP_EXTRAS_DIR.mkdir(parents=True, exist_ok=True)

    # Load 5,455 training data (never test data)
    X_train, y_train = get_train_data()
    y_arr = y_train.values
    n_samples = len(X_train)
    logger.info(f"Loaded training data: N={n_samples}, bankrupt={int(np.sum(y_arr == 1))}")

    # =========================================================================
    # 1. Feature Manifest (95 raw features, 72 selected features, 95 medians)
    # =========================================================================
    logger.info("[1/7] Generating feature_manifest.json ...")
    raw_features_95 = list(X_train.columns)
    assert len(raw_features_95) == 95, f"Expected 95 raw features, got {len(raw_features_95)}"

    # Fit a standard pipeline on full training set to get the 72 selected features
    temp_pipe = build_pipeline(
        model_name="M1",
        imbalance="S0",
        scaler="quantile",
        use_winsorizer=False,
        use_corr_filter=True,
        use_large_indicator=False,
    )
    temp_pipe.fit(X_train, y_train)
    selected_features_72 = list(temp_pipe.named_steps["corr_filter"].selected_features_)
    assert len(selected_features_72) == 72, f"Expected 72 model features, got {len(selected_features_72)}"

    train_medians_95 = {col: float(val) for col, val in X_train.median().to_dict().items()}
    assert len(train_medians_95) == 95, f"Expected 95 medians, got {len(train_medians_95)}"

    # Cross-check with defaults.json
    if DEFAULTS_JSON_PATH.exists():
        with open(DEFAULTS_JSON_PATH, "r", encoding="utf-8") as f:
            defaults_data = json.load(f)
        assert raw_features_95 == defaults_data["raw_columns"], "Raw features mismatch defaults.json"
        assert selected_features_72 == defaults_data["model_columns"], "Model features mismatch defaults.json"
        for col in raw_features_95:
            assert np.isclose(train_medians_95[col], defaults_data["medians"][col]), (
                f"Median mismatch for {col}: {train_medians_95[col]} vs {defaults_data['medians'][col]}"
            )
        logger.info("  Verified 100% equality against defaults.json")

    manifest_payload = {
        "created_at": datetime.datetime.now().isoformat(),
        "n_raw_features": len(raw_features_95),
        "raw_features_95": raw_features_95,
        "n_selected_features": len(selected_features_72),
        "selected_features_72": selected_features_72,
        "train_medians_95": train_medians_95,
    }
    manifest_path = MVP_EXTRAS_DIR / "feature_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {manifest_path}")

    # =========================================================================
    # 2. Final M1 Model & 5x2 OOF Reference Distribution (N=10,910)
    # =========================================================================
    logger.info("[2/7] Generating model_M1.joblib and reference_M1.npy ...")
    with open(BEST_PARAMS_DIR / "M1.json", "r", encoding="utf-8") as f:
        m1_cfg = json.load(f)
    m1_imb = m1_cfg["imbalance"]
    m1_params = m1_cfg["best_params"]

    # Fit final M1 on full training set
    pipe_m1_final = build_pipeline(
        model_name="M1",
        imbalance=m1_imb,
        scaler="quantile",
        use_winsorizer=False,
        use_corr_filter=True,
        use_large_indicator=False,
        model_params=m1_params,
    )
    pipe_m1_final.fit(X_train, y_train)
    clf_m1 = pipe_m1_final.named_steps.get("classifier")
    if clf_m1 is not None and hasattr(clf_m1, "n_jobs"):
        clf_m1.set_params(n_jobs=1)
    model_m1_path = MVP_EXTRAS_DIR / "model_M1.joblib"
    joblib.dump(pipe_m1_final, model_m1_path)
    logger.info(f"  Saved {model_m1_path}")

    # 10-fold OOF predictions on full training set (Repeated Stratified 5-Fold x 2 Repeats)
    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    splits_r0 = list(skf1.split(np.zeros(n_samples), y_arr))
    splits_r1 = list(skf2.split(np.zeros(n_samples), y_arr))

    oof_r0_m1 = np.zeros(n_samples, dtype=float)
    oof_r1_m1 = np.zeros(n_samples, dtype=float)

    for tr_i, val_i in splits_r0:
        p = build_pipeline(
            model_name="M1",
            imbalance=m1_imb,
            scaler="quantile",
            use_winsorizer=False,
            use_corr_filter=True,
            use_large_indicator=False,
            model_params=m1_params,
        )
        p.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
        oof_r0_m1[val_i] = p.predict_proba(X_train.iloc[val_i])[:, 1]

    for tr_i, val_i in splits_r1:
        p = build_pipeline(
            model_name="M1",
            imbalance=m1_imb,
            scaler="quantile",
            use_winsorizer=False,
            use_corr_filter=True,
            use_large_indicator=False,
            model_params=m1_params,
        )
        p.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
        oof_r1_m1[val_i] = p.predict_proba(X_train.iloc[val_i])[:, 1]

    ref_m1 = np.concatenate([oof_r0_m1, oof_r1_m1])
    assert len(ref_m1) == 10910, f"Expected reference length 10910, got {len(ref_m1)}"
    ref_m1_path = MVP_EXTRAS_DIR / "reference_M1.npy"
    np.save(ref_m1_path, ref_m1)
    logger.info(f"  Saved {ref_m1_path} (N={len(ref_m1)})")

    mean_oof_m1 = (oof_r0_m1 + oof_r1_m1) / 2.0

    # =========================================================================
    # 3. Platt Calibrators for M1, M3, M4
    # =========================================================================
    logger.info("[3/7] Generating Platt calibrators for M1, M3, M4 ...")
    cal_m1 = fit_single_model_platt(mean_oof_m1, y_arr, seed=SEED)
    cal_m1_path = MVP_EXTRAS_DIR / "calibrator_M1_platt.joblib"
    joblib.dump(cal_m1, cal_m1_path)
    logger.info(f"  Saved {cal_m1_path}")

    # Load cached 5x2 OOF for M3 and M4 from final/_tmp_oof_* (read-only)
    oof_r0_m3 = np.load(FINAL_DIR / "_tmp_oof_r0_M3.npy")
    oof_r1_m3 = np.load(FINAL_DIR / "_tmp_oof_r1_M3.npy")
    mean_oof_m3 = (oof_r0_m3 + oof_r1_m3) / 2.0
    cal_m3 = fit_single_model_platt(mean_oof_m3, y_arr, seed=SEED)
    cal_m3_path = MVP_EXTRAS_DIR / "calibrator_M3_platt.joblib"
    joblib.dump(cal_m3, cal_m3_path)
    logger.info(f"  Saved {cal_m3_path}")

    oof_r0_m4 = np.load(FINAL_DIR / "_tmp_oof_r0_M4.npy")
    oof_r1_m4 = np.load(FINAL_DIR / "_tmp_oof_r1_M4.npy")
    mean_oof_m4 = (oof_r0_m4 + oof_r1_m4) / 2.0
    cal_m4 = fit_single_model_platt(mean_oof_m4, y_arr, seed=SEED)
    cal_m4_path = MVP_EXTRAS_DIR / "calibrator_M4_platt.joblib"
    joblib.dump(cal_m4, cal_m4_path)
    logger.info(f"  Saved {cal_m4_path}")

    # =========================================================================
    # 4. Thresholds Extras (M1, M4, E_A, M3, and demo_r0_f0_E_A)
    # =========================================================================
    logger.info("[4/7] Generating thresholds_extras.json ...")
    with open(FINAL_DIR / "thresholds.json", "r", encoding="utf-8") as f:
        final_thresholds = json.load(f)

    thresholds_extras = {
        "E_A": final_thresholds["E_A"],
        "M3": final_thresholds["M3"],
        "M4": {},
        "M1": {},
        "demo_r0_f0_E_A": {},
    }

    # Derive empirical training OOF thresholds for M4 and M1 (margin=0.0)
    for R in TARGET_RECALLS:
        key_name = f"R{int(R*100)}"
        thresholds_extras["M4"][key_name] = derive_operating_threshold(
            mean_oof_m4, y_arr, target_recall=R, margin=0.0
        )
        thresholds_extras["M1"][key_name] = derive_operating_threshold(
            mean_oof_m1, y_arr, target_recall=R, margin=0.0
        )

    # Load r0_f0 inner thresholds for E_A from e4a_threshold_transfer_per_fold.csv matching recommended margins
    MARGIN_REC_PATH = PROJECT_ROOT / "outputs" / "tables" / "e4a_margin_recommendation.csv"
    assert MARGIN_REC_PATH.exists(), f"Missing {MARGIN_REC_PATH}"
    rec_df = pd.read_csv(MARGIN_REC_PATH)
    rec_ea = rec_df[rec_df["group"] == "E_A"].set_index("target_recall")["recommended_margin"].to_dict()

    assert E4A_TRANSFER_PATH.exists(), f"Missing {E4A_TRANSFER_PATH}"
    e4a_df = pd.read_csv(E4A_TRANSFER_PATH)
    r0_f0_ea = e4a_df[(e4a_df["repeat"] == 0) & (e4a_df["fold"] == 0) & (e4a_df["group"] == "E_A")]
    for R, m_rec in rec_ea.items():
        match_rows = r0_f0_ea[
            np.isclose(r0_f0_ea["target_recall"], R) & np.isclose(r0_f0_ea["margin"], m_rec)
        ]
        assert len(match_rows) == 1, f"Could not match unique row for R={R}, margin={m_rec}"
        row = match_rows.iloc[0]
        key_name = f"R{int(R*100)}"
        thresholds_extras["demo_r0_f0_E_A"][key_name] = {
            "target_recall": float(R),
            "recommended_margin": float(m_rec),
            "threshold": float(row["threshold"]),
            "training_oof_actual_recall": float(row["actual_recall"]),
            "training_oof_precision": float(row["precision"]),
            "alert_count": int(row["alert_count"]),
            "note": "採選項 A：r0_f0 折內門檻（推薦安全邊際來自 e4a_margin_recommendation.csv，數值記錄於 e4a_threshold_transfer_per_fold.csv）",
        }

    thresh_extras_path = MVP_EXTRAS_DIR / "thresholds_extras.json"
    with open(thresh_extras_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_extras, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {thresh_extras_path}")

    # =========================================================================
    # 5. Demo Models: demo_r0_f0_M4 and demo_r0_f0_M1
    # =========================================================================
    logger.info("[5/7] Preparing demo models in mvp_extras/ ...")
    # Copy M4 demo pipeline from e5_cache
    src_m4_cache = E5_CACHE_DIR / "pipeline_M4_r0_f0.joblib"
    assert src_m4_cache.exists(), f"Missing {src_m4_cache}"
    dst_m4_demo = MVP_EXTRAS_DIR / "demo_r0_f0_M4.joblib"
    shutil.copyfile(src_m4_cache, dst_m4_demo)
    # Enforce n_jobs=1 on copy
    pipe_m4_demo = joblib.load(dst_m4_demo)
    clf_m4 = pipe_m4_demo.named_steps.get("classifier")
    if clf_m4 is not None and hasattr(clf_m4, "n_jobs"):
        clf_m4.set_params(n_jobs=1)
    joblib.dump(pipe_m4_demo, dst_m4_demo)
    logger.info(f"  Saved {dst_m4_demo}")

    # Build M1 demo pipeline on r0_f0 training split
    splits_payload = load_splits()
    s0 = splits_payload["cv_splits"][0]
    train_orig_idx = splits_payload["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}
    out_tr_idx = np.array([orig_to_rel[i] for i in s0["train_idx"]])
    out_val_idx = np.array([orig_to_rel[i] for i in s0["val_idx"]])
    assert len(out_val_idx) == 1091, f"Expected 1091 validation samples, got {len(out_val_idx)}"

    # Load r0_f0 hyperparameters for M1
    m1_r0_f0_cache = E2_CACHE_DIR / "M1_r0_f0.joblib"
    assert m1_r0_f0_cache.exists(), f"Missing {m1_r0_f0_cache}"
    m1_r0_f0_info = joblib.load(m1_r0_f0_cache)
    demo_m1_params = m1_r0_f0_info["model_params"]
    demo_m1_imb = m1_r0_f0_info["best_imbalance"]

    pipe_m1_demo = build_pipeline(
        model_name="M1",
        imbalance=demo_m1_imb,
        scaler="quantile",
        use_winsorizer=False,
        use_corr_filter=True,
        use_large_indicator=False,
        model_params=demo_m1_params,
    )
    pipe_m1_demo.fit(X_train.iloc[out_tr_idx], y_train.iloc[out_tr_idx])
    clf_m1_demo = pipe_m1_demo.named_steps.get("classifier")
    if clf_m1_demo is not None and hasattr(clf_m1_demo, "n_jobs"):
        clf_m1_demo.set_params(n_jobs=1)
    dst_m1_demo = MVP_EXTRAS_DIR / "demo_r0_f0_M1.joblib"
    joblib.dump(pipe_m1_demo, dst_m1_demo)
    logger.info(f"  Saved {dst_m1_demo}")

    # Verify bitwise/close agreement with official OOF predictions
    pred_m1_demo = pipe_m1_demo.predict_proba(X_train.iloc[out_val_idx])[:, 1]
    e2_m1_oof = np.load(OOF_DIR / "e2_M1.npy")[0, out_val_idx]
    assert np.allclose(pred_m1_demo, e2_m1_oof), "demo_r0_f0_M1 does not match e2_M1.npy[0]"
    logger.info("  Verified demo_r0_f0_M1 matches e2_M1.npy[0] exactly (np.allclose=True)")

    # =========================================================================
    # 6. Demo Batch Dataset: oof_demo_batch.csv (N=1,091)
    # =========================================================================
    logger.info("[6/7] Generating oof_demo_batch.csv ...")
    demo_df = X_train.iloc[out_val_idx].copy().reset_index(drop=True)
    demo_df["Bankrupt?"] = y_train.iloc[out_val_idx].values
    assert int(demo_df["Bankrupt?"].sum()) == 36, f"Expected 36 bankrupt firms, got {demo_df['Bankrupt?'].sum()}"

    # Pre-stored official OOF risk scores
    demo_df["oof_score_E_A"] = np.load(OOF_DIR / "e3_ensemble_E_A.npy")[0, out_val_idx]
    demo_df["oof_score_M4"] = np.load(OOF_DIR / "e2_M4.npy")[0, out_val_idx]
    demo_df["oof_score_M3"] = np.load(OOF_DIR / "e2_M3.npy")[0, out_val_idx]
    demo_df["oof_score_M1"] = e2_m1_oof

    demo_csv_path = MVP_EXTRAS_DIR / "oof_demo_batch.csv"
    demo_df.to_csv(demo_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  Saved {demo_csv_path} (shape: {demo_df.shape})")

    # =========================================================================
    # 7. Hash Manifest: mvp_extras_hashes.json
    # =========================================================================
    logger.info("[7/7] Generating mvp_extras_hashes.json ...")
    expected_files = [
        "feature_manifest.json",
        "model_M1.joblib",
        "reference_M1.npy",
        "calibrator_M1_platt.joblib",
        "calibrator_M3_platt.joblib",
        "calibrator_M4_platt.joblib",
        "thresholds_extras.json",
        "demo_r0_f0_M4.joblib",
        "demo_r0_f0_M1.joblib",
        "oof_demo_batch.csv",
    ]

    hashes = {}
    for fname in expected_files:
        fpath = MVP_EXTRAS_DIR / fname
        assert fpath.exists(), f"Expected extra file missing: {fpath}"
        sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        size_bytes = fpath.stat().st_size
        hashes[fname] = {
            "sha256": sha,
            "size_bytes": size_bytes,
        }
        logger.info(f"  {fname:30s} : {sha} ({size_bytes:,} bytes)")

    hash_manifest_path = MVP_EXTRAS_DIR / "mvp_extras_hashes.json"
    with open(hash_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.datetime.now().isoformat(),
                "n_files": len(hashes),
                "files": hashes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"  Saved {hash_manifest_path}")
    logger.info("=== freeze_mvp_extras.py completed successfully ===")


if __name__ == "__main__":
    main()
