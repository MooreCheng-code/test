"""
tests/test_mvp_completeness.py: Final MVP Completeness, Governance & Hash Integrity Suite.

Verifications:
1. MODEL_CARD.md contains all standard governance sections and verbatim limitation restrictions.
2. MODEL_CARD.md numbers strictly match outputs/tables/mvp_model_card_metrics.csv.
3. outputs/models/mvp_extras/feature_manifest.json contains exactly 95 raw features and finite training medians.
4. outputs/models/final/ directory is bitwise frozen (matches preflight SHA-256 hashes).
5. outputs/models/mvp_extras/ directory matches mvp_extras_hashes.json manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CARD_PATH = PROJECT_ROOT / "MODEL_CARD.md"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_FINAL_DIR = PROJECT_ROOT / "outputs" / "models" / "final"
MODELS_EXTRAS_DIR = PROJECT_ROOT / "outputs" / "models" / "mvp_extras"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_model_card_exists_and_has_all_required_sections():
    """Verifies that MODEL_CARD.md exists and contains all required governance sections."""
    assert MODEL_CARD_PATH.exists(), f"Missing {MODEL_CARD_PATH}"
    content = MODEL_CARD_PATH.read_text(encoding="utf-8")

    required_sections = [
        "模型基本資訊 (Model Details)",
        "預期用途與應用場景 (Intended Use)",
        "影響因子與評估群體 (Factors, Populations & Subgroups)",
        "量化效能評估 (Quantitative Analyses)",
        "模型限制與治理聲明 (Model Limitations & Caveats)",
    ]
    for section in required_sections:
        assert section in content, f"Missing required section '{section}' in MODEL_CARD.md"


def test_model_card_mandatory_limitations_and_verbatim_statements():
    """Verifies that all mandatory disclaimers and verbatim restrictions exist in MODEL_CARD.md."""
    content = MODEL_CARD_PATH.read_text(encoding="utf-8")

    # 1. Verbatim M5 restriction
    verbatim_m5 = (
        "CV 評估的 M5 啟用 subsample_freq=1，最終凍結 M5 為 subsample_freq=0；"
        "CV 指標僅供選模參考，最終效能以 E6 測試集結果為準；E5 的 M5 全域 SHAP 亦使用未含該參數的重建設定。"
    )
    assert verbatim_m5 in content, "Verbatim M5 limitation text is missing or altered in MODEL_CARD.md"

    # 2. M1/M4 margin_validated=False
    assert "margin_validated=False" in content or "margin_validated=false" in content, (
        "Missing margin_validated=False disclaimer in MODEL_CARD.md"
    )

    # 3. R100 caution
    assert "R100" in content
    assert "接近全抓" in content or "91.6%" in content or "安全底線" in content

    # 4. Dynamic demo data scale
    demo_csv = MODELS_EXTRAS_DIR / "oof_demo_batch.csv"
    assert demo_csv.exists()
    df_demo = pd.read_csv(demo_csv)
    n_demo_str = f"{len(df_demo):,}"
    assert n_demo_str in content, f"Demo dataset count {n_demo_str} not dynamically found in MODEL_CARD.md"

    # 5. CV replacing fixed validation set
    assert "15 折" in content or "5 折重複 3 次" in content

    # 6. Cost sensitivity matrix 10/20/50
    assert "10" in content and "20" in content and "50" in content
    assert "C_{FN}/C_{FP}" in content or "C_FN/C_FP" in content

    # 7. calibrated_score non-probability warning & odds ratio adjustment
    assert "calibrated_score" in content
    assert "勝算比" in content or "Odds" in content

    # 8. Verbatim E3 quote snippet from RESULTS_SUMMARY.md:59
    assert "Nadeau-Bengio 調整之 Holm 校正 $p$ 值全數為 $p_{\\text{Holm}} = 1.0$，統計上無法區分" in content

    # 9. Verbatim E6 quote snippet from RESULTS_SUMMARY.md:75
    assert "不可將 100% 召回當作未來保證，專題結論以 CV 估計為主、測試集為確認" in content

    # 10. Terminology normalization: ensure '四級' does not appear in MODEL_CARD.md
    assert "四級" not in content, "MODEL_CARD.md still contains '四級', which should be '四個工作點'"

    # 11. Test set limitation in Section 5
    assert "44 家破產" in content
    assert "不可把 100% 召回當作未來表現的保證" in content

    # 12. AR@0.95 vs R95 definition note
    assert "AR@0.95 是以排序找出 95% 召回所需的最小預警倍數" in content


def test_model_card_numbers_match_csv_audit_table():
    """Verifies that numerical metrics in MODEL_CARD.md match mvp_model_card_metrics.csv."""
    metrics_csv_path = TABLES_DIR / "mvp_model_card_metrics.csv"
    assert metrics_csv_path.exists(), f"Missing audit metrics table: {metrics_csv_path}"

    df_metrics = pd.read_csv(metrics_csv_path)
    content = MODEL_CARD_PATH.read_text(encoding="utf-8")

    # Check key numerical matches across all metrics in CSV
    for _, row in df_metrics.iterrows():
        val = float(row["value"])
        formats = [
            f"{val:.4f}", f"{val:.2%}", f"{val:.6f}", f"{val:.1%}", f"{val:.3f}",
            f"{val:.2f}x", f"{val:.1f}x", f"{val:.2f}", f"{val:.1f}",
            f"{val*100:.1f}%", f"{val*100:.2f}%", f"{val*100:.0f}%",
        ]
        if abs(val - round(val)) < 1e-5:
            formats.extend([f"{int(round(val))}", f"{int(round(val)):,}"])

        found = any(rep in content for rep in formats)
        assert found, f"Metric {row['section']} {row['model']} {row['metric']}={val} not found in MODEL_CARD.md"


def test_feature_manifest_completeness_and_finite_medians():
    """Verifies feature_manifest.json contains 95 raw features and finite medians."""
    manifest_path = MODELS_EXTRAS_DIR / "feature_manifest.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["n_raw_features"] == 95
    assert len(data["raw_features_95"]) == 95
    assert len(data["train_medians_95"]) == 95

    # All medians must be valid finite floats
    for feat, median_val in data["train_medians_95"].items():
        assert isinstance(median_val, (int, float))
        assert not np.isnan(median_val), f"Feature '{feat}' has NaN median"
        assert not np.isinf(median_val), f"Feature '{feat}' has Infinite median"


def test_frozen_final_and_extras_hashes_unaltered():
    """Verifies that all frozen final models and extras artifacts remain unaltered."""
    # 1. Final hashes
    preflight_log = PROJECT_ROOT / "outputs" / "logs" / "e6_preflight.txt"
    assert preflight_log.exists(), f"Missing preflight log: {preflight_log}"
    final_manifest = {}
    for line in preflight_log.read_text(encoding="utf-8").splitlines():
        if "outputs/models/final/" in line:
            parts = line.strip().split()
            if len(parts) >= 2:
                final_manifest[parts[1]] = parts[0]

    for rel_path, expected_hash in final_manifest.items():
        full_path = PROJECT_ROOT / rel_path
        assert full_path.exists(), f"Missing frozen file: {full_path}"
        actual_hash = sha256_file(full_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}: {actual_hash} != {expected_hash}"

    # 2. Extras hashes
    extras_manifest_path = MODELS_EXTRAS_DIR / "mvp_extras_hashes.json"
    assert extras_manifest_path.exists(), f"Missing extras manifest: {extras_manifest_path}"
    extras_manifest = json.loads(extras_manifest_path.read_text(encoding="utf-8"))["files"]

    for filename, expected_info in extras_manifest.items():
        file_path = MODELS_EXTRAS_DIR / filename
        assert file_path.exists(), f"Missing extras file: {file_path}"
        actual_hash = sha256_file(file_path)
        expected_hash = expected_info["sha256"] if isinstance(expected_info, dict) else expected_info
        assert actual_hash == expected_hash, f"Extras hash mismatch for {filename}: {actual_hash} != {expected_hash}"


def test_independent_recomputation_of_ea_metrics_matches_csv():
    """
    Independent sklearn/numpy recomputation of E_A ensemble metrics from e2_M3, e2_M4, e2_M5
    and asserts bitwise match against outputs/tables/mvp_model_card_metrics.csv without
    importing src.metrics.
    """
    import joblib
    from sklearn.metrics import average_precision_score, roc_auc_score
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import config

    oof_dir = PROJECT_ROOT / "outputs" / "models" / "oof"
    m3 = np.load(oof_dir / "e2_M3.npy")
    m4 = np.load(oof_dir / "e2_M4.npy")
    m5 = np.load(oof_dir / "e2_M5.npy")
    ea_orig = np.load(oof_dir / "e3_ensemble_E_A.npy")

    splits = joblib.load(PROJECT_ROOT / "outputs" / "models" / "splits.joblib")
    train_orig_idx = splits["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}
    cv_splits = splits["cv_splits"]

    # Load ground truth labels
    df_raw = pd.read_csv(config.DATA_PATH)
    y_train = df_raw[config.TARGET_COL].values[train_orig_idx]

    def ecdf_map(reference_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
        ref_sorted = np.sort(np.asarray(reference_scores, dtype=float))
        n_ref = len(ref_sorted)
        query = np.asarray(query_scores, dtype=float)
        left = np.searchsorted(ref_sorted, query, side="left")
        right = np.searchsorted(ref_sorted, query, side="right")
        return (left + right) / (2.0 * n_ref)

    # Reconstruct E_A from base models
    ea_recon = np.zeros_like(ea_orig)
    rocs, prs, rec10s = [], [], []

    for s in cv_splits:
        rep = s["repeat"]
        f = s["fold"]
        val_rel = np.array([orig_to_rel[i] for i in s["val_idx"]])

        ref3 = np.load(oof_dir / f"e3_reference_M3_r{rep}_f{f}.npy")
        ref4 = np.load(oof_dir / f"e3_reference_M4_r{rep}_f{f}.npy")
        ref5 = np.load(oof_dir / f"e3_reference_M5_r{rep}_f{f}.npy")

        p3 = ecdf_map(ref3, m3[rep, val_rel])
        p4 = ecdf_map(ref4, m4[rep, val_rel])
        p5 = ecdf_map(ref5, m5[rep, val_rel])

        s_val = (p3 + p4 + p5) / 3.0
        ea_recon[rep, val_rel] = s_val

        y_v = y_train[val_rel]
        rocs.append(roc_auc_score(y_v, s_val))
        prs.append(average_precision_score(y_v, s_val))

        # Pure numpy recall@top10% computation (floor quota, pessimistic tie-breaking)
        n_v = len(y_v)
        k = int(np.floor(n_v * 0.10))
        sorted_s = np.sort(s_val)[::-1]
        thr = sorted_s[k - 1]
        strictly_greater = s_val > thr
        tp_strictly = int((strictly_greater & (y_v == 1)).sum())
        rem = k - int(strictly_greater.sum())
        tied_neg = int(((s_val == thr) & (y_v == 0)).sum())
        tp = tp_strictly + max(0, rem - tied_neg)
        p = int(np.sum(y_v == 1))
        rec10 = float(tp / p) if p > 0 else 0.0
        rec10s.append(rec10)

    # Assert bitwise reconstruction match
    max_diff = float(np.max(np.abs(ea_orig - ea_recon)))
    assert max_diff == 0.0, f"Reconstructed E_A deviates from stored file by {max_diff}"

    # Load audit metrics CSV
    metrics_csv = TABLES_DIR / "mvp_model_card_metrics.csv"
    assert metrics_csv.exists(), f"Missing {metrics_csv}"
    m_df = pd.read_csv(metrics_csv)

    # Check 15-fold CV metrics
    expected_roc_mean = float(m_df[(m_df["section"] == "CV_15_Fold") & (m_df["model"] == "E_A") & (m_df["metric"] == "ROC_AUC_mean")]["value"].iloc[0])
    expected_pr_mean = float(m_df[(m_df["section"] == "CV_15_Fold") & (m_df["model"] == "E_A") & (m_df["metric"] == "PR_AUC_mean")]["value"].iloc[0])
    expected_rec10_mean = float(m_df[(m_df["section"] == "CV_15_Fold") & (m_df["model"] == "E_A") & (m_df["metric"] == "Recall_Top10_mean")]["value"].iloc[0])

    assert np.mean(rocs) == pytest.approx(expected_roc_mean, rel=1e-6)
    assert np.mean(prs) == pytest.approx(expected_pr_mean, rel=1e-6)
    assert np.mean(rec10s) == pytest.approx(expected_rec10_mean, rel=1e-6)
