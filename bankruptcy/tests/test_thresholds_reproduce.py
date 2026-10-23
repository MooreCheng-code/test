"""
Unit test to strictly reproduce frozen thresholds from saved OOF predictions.
Guarantees bitwise reproduction of positive cases caught and total alert counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.scoring import to_percentile

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "outputs" / "models" / "final"


@pytest.fixture(scope="module")
def oof_data():
    """Loads saved training OOF predictions and labels used during component freezing."""
    ref_y = np.load(FINAL_DIR / "reference_y.npy")
    # First 5,455 elements correspond to the training set labels
    y_train = ref_y[:5455]
    assert len(y_train) == 5455
    assert int((y_train == 1).sum()) == 176

    oof_r0_M3 = np.load(FINAL_DIR / "_tmp_oof_r0_M3.npy")
    oof_r1_M3 = np.load(FINAL_DIR / "_tmp_oof_r1_M3.npy")
    oof_r0_M4 = np.load(FINAL_DIR / "_tmp_oof_r0_M4.npy")
    oof_r1_M4 = np.load(FINAL_DIR / "_tmp_oof_r1_M4.npy")
    oof_r0_M5 = np.load(FINAL_DIR / "_tmp_oof_r0_M5.npy")
    oof_r1_M5 = np.load(FINAL_DIR / "_tmp_oof_r1_M5.npy")

    ref_M3 = np.load(FINAL_DIR / "reference_M3.npy")
    ref_M4 = np.load(FINAL_DIR / "reference_M4.npy")
    ref_M5 = np.load(FINAL_DIR / "reference_M5.npy")

    # Final training OOF ensemble score calculation (as implemented in thresholding.py:run_e4c)
    mean_M3 = (oof_r0_M3 + oof_r1_M3) / 2.0
    mean_M4 = (oof_r0_M4 + oof_r1_M4) / 2.0
    mean_M5 = (oof_r0_M5 + oof_r1_M5) / 2.0

    pct_M3 = to_percentile(mean_M3, ref_M3)
    pct_M4 = to_percentile(mean_M4, ref_M4)
    pct_M5 = to_percentile(mean_M5, ref_M5)

    scores_EA = (pct_M3 + pct_M4 + pct_M5) / 3.0
    scores_M3 = pct_M3

    with open(FINAL_DIR / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds_cfg = json.load(f)

    return {
        "y_train": y_train,
        "scores_EA": scores_EA,
        "scores_M3": scores_M3,
        "thresholds_cfg": thresholds_cfg,
    }


def test_ea_threshold_reproduction(oof_data):
    """Verifies that E_A operating thresholds reproduce exact integer counts of caught positives and alerts."""
    y = oof_data["y_train"]
    scores = oof_data["scores_EA"]
    cfg = oof_data["thresholds_cfg"]["E_A"]
    pos_mask = y == 1

    expected_counts = {
        "R80": {"caught": 145, "alerts": 608},
        "R90": {"caught": 169, "alerts": 1378},
        "R95": {"caught": 175, "alerts": 2142},
        "R100": {"caught": 176, "alerts": 4811},
    }

    for r_key, exp in expected_counts.items():
        t = cfg[r_key]["threshold"]
        flags = scores >= t
        caught = int((flags & pos_mask).sum())
        alerts = int(flags.sum())

        assert caught == exp["caught"], f"E_A {r_key} caught mismatch: got {caught}, expected {exp['caught']}"
        assert alerts == exp["alerts"], f"E_A {r_key} alerts mismatch: got {alerts}, expected {exp['alerts']}"

        # Cross-check with stored fields in thresholds.json
        expected_rec = exp["caught"] / 176.0
        expected_ar = exp["alerts"] / 176.0
        assert np.isclose(cfg[r_key]["training_oof_actual_recall"], expected_rec)
        assert np.isclose(cfg[r_key]["training_oof_alert_ratio"], expected_ar)


def test_m3_threshold_reproduction(oof_data):
    """Verifies that M3 operating thresholds reproduce exact integer counts of caught positives and alerts."""
    y = oof_data["y_train"]
    scores = oof_data["scores_M3"]
    cfg = oof_data["thresholds_cfg"]["M3"]
    pos_mask = y == 1

    expected_counts = {
        "R80": {"caught": 148, "alerts": 683},
        "R90": {"caught": 166, "alerts": 1218},
        "R95": {"caught": 175, "alerts": 1946},
        "R100": {"caught": 176, "alerts": 4275},
    }

    for r_key, exp in expected_counts.items():
        t = cfg[r_key]["threshold"]
        flags = scores >= t
        caught = int((flags & pos_mask).sum())
        alerts = int(flags.sum())

        assert caught == exp["caught"], f"M3 {r_key} caught mismatch: got {caught}, expected {exp['caught']}"
        assert alerts == exp["alerts"], f"M3 {r_key} alerts mismatch: got {alerts}, expected {exp['alerts']}"

        expected_rec = exp["caught"] / 176.0
        expected_ar = exp["alerts"] / 176.0
        assert np.isclose(cfg[r_key]["training_oof_actual_recall"], expected_rec)
        assert np.isclose(cfg[r_key]["training_oof_alert_ratio"], expected_ar)


def test_r90_and_r95_are_distinct_thresholds(oof_data):
    """Explicitly asserts that R=0.90 and R=0.95 operating thresholds are distinct."""
    cfg = oof_data["thresholds_cfg"]

    # E_A
    t_ea_90 = cfg["E_A"]["R90"]["threshold"]
    t_ea_95 = cfg["E_A"]["R95"]["threshold"]
    assert t_ea_90 != t_ea_95
    assert t_ea_95 < t_ea_90, "Higher target recall must have lower threshold"
    assert abs(t_ea_90 - t_ea_95) > 0.10

    # M3
    t_m3_90 = cfg["M3"]["R90"]["threshold"]
    t_m3_95 = cfg["M3"]["R95"]["threshold"]
    assert t_m3_90 != t_m3_95
    assert t_m3_95 < t_m3_90, "Higher target recall must have lower threshold"
    assert abs(t_m3_90 - t_m3_95) > 0.10
