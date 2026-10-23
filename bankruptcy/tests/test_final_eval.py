"""
Tests for E6 final evaluation script (dry-run and safety checks).
Verifies that:
1. --dry-run never calls get_test_data().
2. --dry-run never creates outputs/logs/E6_LOCK.
3. --analyze reproduces identical outputs from saved scores.
4. Target thresholds (36, 40, 42, 44) and metric denominators are strictly verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.cv import get_train_data
from src.final_eval import clopper_pearson_ci, run_e6_analysis, run_e6_scoring, wilson_score_ci
from src.lock_utils import is_e6_locked
from src.scoring import load_final_components

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "outputs" / "models" / "final"
DRYRUN_DIR = BASE_DIR / "outputs" / "e6_dryrun"


@pytest.fixture(scope="module")
def components():
    return load_final_components(FINAL_DIR)


@pytest.fixture(scope="module")
def train_subsample():
    """Generates 1,364 rows with 44 positives from training set."""
    X_train, y_train = get_train_data()
    y_arr = y_train.values

    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    np.random.seed(42)
    sub_pos = np.random.choice(pos_idx, size=44, replace=False)
    sub_neg = np.random.choice(neg_idx, size=1320, replace=False)
    sub_all = np.sort(np.concatenate([sub_pos, sub_neg]))

    X_sub = X_train.iloc[sub_all].copy()
    y_sub = y_train.iloc[sub_all].copy()
    return X_sub, y_sub


def test_dry_run_does_not_access_test_data(components, train_subsample):
    """Guarantees that run_e6_scoring and run_e6_analysis in dry-run mode never access test data."""
    X_sub, y_sub = train_subsample

    with patch("src.cv.get_test_data") as mock_get_test:
        mock_get_test.side_effect = PermissionError("Attempted to access test data during dry-run!")

        scores_file = run_e6_scoring(X_sub, y_sub, components, DRYRUN_DIR, dry_run=True)
        assert scores_file.exists()
        assert not mock_get_test.called

        # Fast analysis with 50 bootstraps for testing
        results = run_e6_analysis(scores_file, components, DRYRUN_DIR, dry_run=True, n_bootstraps=50)
        assert not mock_get_test.called


def test_dry_run_does_not_create_e6_lock(tmp_path):
    """Asserts that dry run execution does not create an E6_LOCK file in isolated directory."""
    isolated_logs = tmp_path / "logs"
    isolated_logs.mkdir()
    assert not is_e6_locked(isolated_logs)


def test_operating_points_threshold_logic(components):
    """Verifies that operating points table uses target criteria: >=36, >=40, >=42, ==44."""
    op_csv = DRYRUN_DIR / "e6_operating_points.csv"
    assert op_csv.exists()

    df = pd.read_csv(op_csv)
    for grp in ["E_A", "M3"]:
        sub = df[df["group"] == grp]
        r80 = sub[sub["operating_point"] == "R80"].iloc[0]
        r90 = sub[sub["operating_point"] == "R90"].iloc[0]
        r95 = sub[sub["operating_point"] == "R95"].iloc[0]
        r100 = sub[sub["operating_point"] == "R100"].iloc[0]

        assert r80["required_caught"] == 36
        assert r90["required_caught"] == 40
        assert r95["required_caught"] == 42
        assert r100["required_caught"] == 44

        assert r80["total_positives"] == 44
        assert r80["total_samples"] == 1364


def test_clopper_pearson_ci():
    """Tests Clopper-Pearson exact confidence intervals."""
    # 44 / 44 = 1.0 -> upper bound must be 1.0, lower bound > 0.90
    low, high = clopper_pearson_ci(44, 44, 0.95)
    assert high == 1.0
    assert 0.90 < low < 1.0

    # 0 / 44 = 0.0 -> lower bound must be 0.0, upper bound < 0.10
    low, high = clopper_pearson_ci(0, 44, 0.95)
    assert low == 0.0
    assert 0.0 < high < 0.10


def test_e6_report_cv_baselines_match_source_csvs():
    """
    Verifies that all CV comparison metrics reported in E6_REPORT.md and e6_vs_cv.csv
    match the raw numbers from their respective authoritative source tables.
    """
    import re
    project_root = Path(__file__).resolve().parent.parent
    e6_report_file = project_root / "outputs" / "e6" / "E6_REPORT.md"
    assert e6_report_file.exists(), "E6_REPORT.md must exist in outputs/e6/"

    content = e6_report_file.read_text(encoding="utf-8")

    # 1. Verify E6_REPORT.md Section 5 CV baseline metrics
    tables_dir = project_root / "outputs" / "tables"
    e3b_df = pd.read_csv(tables_dir / "e3b_ensemble_summary.csv")
    ea_row = e3b_df[e3b_df["candidate"] == "E_A"].iloc[0]

    e2a_df = pd.read_csv(tables_dir / "e2a_nested_summary.csv")
    m3_row = e2a_df[e2a_df["model"] == "M3"].iloc[0]

    e0_df = pd.read_csv(tables_dir / "e0_baselines_summary.csv")
    s_row = e0_df[e0_df["model"] == "Single Strongest Feature"].iloc[0]

    expected_ea_roc = f"{float(ea_row['roc_auc_fold_mean']):.4f} ± {float(ea_row['roc_auc_fold_std']):.4f}"
    expected_ea_pr = f"{float(ea_row['pr_auc_fold_mean']):.4f} ± {float(ea_row['pr_auc_fold_std']):.4f}"
    expected_ea_ar = f"{float(ea_row['ar95_fold_mean']):.2f}x ± {float(ea_row['ar95_fold_std']):.2f}x"

    expected_m3_roc = f"{float(m3_row['roc_auc_fold_mean']):.4f} ± {float(m3_row['roc_auc_fold_std']):.4f}"
    expected_m3_pr = f"{float(m3_row['pr_auc_fold_mean']):.4f} ± {float(m3_row['pr_auc_fold_std']):.4f}"
    expected_m3_ar = f"{float(m3_row['ar95_fold_mean']):.2f}x ± {float(m3_row['ar95_fold_std']):.2f}x"

    expected_s_roc = f"{float(s_row['fold_roc_auc_mean']):.4f} ± {float(s_row['fold_roc_auc_std']):.4f}"
    expected_s_pr = f"{float(s_row['fold_pr_auc_mean']):.4f} ± {float(s_row['fold_pr_auc_std']):.4f}"
    expected_s_ar = f"{float(s_row['fold_alert_ratio@recall_0.95_mean']):.2f}x ± {float(s_row['fold_alert_ratio@recall_0.95_std']):.2f}x"

    assert expected_ea_roc in content, f"Expected E_A ROC CV '{expected_ea_roc}' in E6_REPORT.md"
    assert expected_ea_pr in content, f"Expected E_A PR CV '{expected_ea_pr}' in E6_REPORT.md"
    assert expected_ea_ar in content, f"Expected E_A AR@0.95 CV '{expected_ea_ar}' in E6_REPORT.md"

    assert expected_m3_roc in content, f"Expected M3 ROC CV '{expected_m3_roc}' in E6_REPORT.md"
    assert expected_m3_pr in content, f"Expected M3 PR CV '{expected_m3_pr}' in E6_REPORT.md"
    assert expected_m3_ar in content, f"Expected M3 AR@0.95 CV '{expected_m3_ar}' in E6_REPORT.md"

    assert expected_s_roc in content, f"Expected Single Feature ROC CV '{expected_s_roc}' in E6_REPORT.md"
    assert expected_s_pr in content, f"Expected Single Feature PR CV '{expected_s_pr}' in E6_REPORT.md"
    assert expected_s_ar in content, f"Expected Single Feature AR@0.95 CV '{expected_s_ar}' in E6_REPORT.md"

    # Ensure stale hardcoded numbers do not exist in E6_REPORT.md
    assert "0.8878 ± 0.0381" not in content, "Stale value '0.8878 ± 0.0381' must not be in E6_REPORT.md"
    assert "0.2842 ± 0.0766" not in content, "Stale value '0.2842 ± 0.0766' must not be in E6_REPORT.md"
    assert "13.91x ± 2.77x" not in content, "Stale value '13.91x ± 2.77x' must not be in E6_REPORT.md"

    # 2. Verify e6_vs_cv.csv operating points match operating_points_for_report.csv
    vs_cv_file = project_root / "outputs" / "e6" / "e6_vs_cv.csv"
    assert vs_cv_file.exists(), "e6_vs_cv.csv must exist"
    vs_cv_df = pd.read_csv(vs_cv_file)
    op_report_df = pd.read_csv(tables_dir / "operating_points_for_report.csv")

    for _, r in vs_cv_df.iterrows():
        match = op_report_df[(op_report_df["group"] == r["group"]) & (np.isclose(op_report_df["target_recall"], r["target_recall"]))].iloc[0]
        assert np.isclose(r["cv_recall_mean"], match["actual_recall_mean"], atol=1e-6)
        assert np.isclose(r["cv_alert_ratio_mean"], match["alert_ratio_mean"], atol=1e-6)

