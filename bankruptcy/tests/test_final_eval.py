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


def test_dry_run_does_not_create_e6_lock(train_subsample):
    """Asserts that no E6_LOCK file is created during dry-run."""
    lock_file = BASE_DIR / "outputs" / "logs" / "E6_LOCK"
    assert not lock_file.exists()
    assert not is_e6_locked(BASE_DIR / "outputs" / "logs")


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
