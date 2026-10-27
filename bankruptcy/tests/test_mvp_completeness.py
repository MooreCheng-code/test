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
        "AI 倫理與治理公約 (Ethical Considerations)",
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


def test_model_card_numbers_match_csv_audit_table():
    """Verifies that numerical metrics in MODEL_CARD.md match mvp_model_card_metrics.csv."""
    metrics_csv_path = TABLES_DIR / "mvp_model_card_metrics.csv"
    assert metrics_csv_path.exists(), f"Missing audit metrics table: {metrics_csv_path}"

    df_metrics = pd.read_csv(metrics_csv_path)
    content = MODEL_CARD_PATH.read_text(encoding="utf-8")

    # Check key numerical matches
    for _, row in df_metrics.iterrows():
        val = float(row["value"])
        val_str_4f = f"{val:.4f}"
        val_str_pct = f"{val:.2%}"
        val_str_6f = f"{val:.6f}"

        # At least one formatted representation should exist in MODEL_CARD.md
        found = (val_str_4f in content) or (val_str_pct in content) or (val_str_6f in content)
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
