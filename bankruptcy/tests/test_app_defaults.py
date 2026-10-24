"""
tests/test_app_defaults.py

Validates app/assets/defaults.json and scripts/make_app_defaults.py:
(a) raw_columns match frozen pipeline feature_selector exactly.
(b) 18 inputs are in model_columns; fill_columns (54) + inputs (18) = 72 without overlap.
(c) medians match training data medians.
(d) preset profiles are virtual summaries and do not match any single training row.
(e) T6 static leakage scan: zero occurrence of forbidden test-set strings in scripts and app/.
"""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cv import get_train_data

DEFAULTS_PATH = PROJECT_ROOT / "app" / "assets" / "defaults.json"
FROZEN_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "final" / "model_M3.joblib"


@pytest.fixture(scope="module")
def defaults_data():
    assert DEFAULTS_PATH.exists(), f"defaults.json not found at {DEFAULTS_PATH}"
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def train_data():
    X_train, y_train = get_train_data(PROJECT_ROOT / "data" / "data.csv")
    return X_train, y_train


def test_defaults_raw_columns_match_frozen_pipeline(defaults_data):
    """(a) defaults.json raw_columns matches frozen pipeline feature_selector exactly."""
    pipe = joblib.load(FROZEN_MODEL_PATH)
    pipeline_features = list(pipe.named_steps["feature_selector"].selected_features_)

    raw_cols = defaults_data["raw_columns"]
    assert len(raw_cols) == 95
    assert raw_cols == pipeline_features


def test_defaults_columns_partition_and_coverage(defaults_data):
    """(b) 18 inputs are in model_columns; fill_columns (54) + inputs (18) = 72 without overlap."""
    model_cols = defaults_data["model_columns"]
    fill_cols = defaults_data["fill_columns"]
    inputs = defaults_data["inputs"]
    unused_cols = defaults_data["unused_columns"]
    raw_cols = defaults_data["raw_columns"]

    assert len(model_cols) == 72
    assert len(inputs) == 18
    assert len(fill_cols) == 54
    assert len(unused_cols) == 23

    input_keys = set(inputs.keys())
    fill_set = set(fill_cols)
    model_set = set(model_cols)
    raw_set = set(raw_cols)
    unused_set = set(unused_cols)

    # 18 inputs are in model_columns
    assert input_keys.issubset(model_set)

    # fill_columns and inputs have no intersection
    assert fill_set.isdisjoint(input_keys)

    # fill_columns + inputs = model_columns
    assert fill_set.union(input_keys) == model_set

    # model_columns + unused_columns = raw_columns
    assert model_set.union(unused_set) == raw_set
    assert model_set.isdisjoint(unused_set)


def test_defaults_medians_consistent_with_training_data(defaults_data, train_data):
    """(c) medians match recalculated training data medians."""
    X_train, _ = train_data
    raw_cols = defaults_data["raw_columns"]
    saved_medians = defaults_data["medians"]

    assert len(saved_medians) == 95
    for col in raw_cols:
        expected = float(X_train[col].median())
        actual = float(saved_medians[col])
        assert np.isclose(actual, expected, rtol=1e-7, atol=1e-7), f"Median mismatch on {col}"


def test_defaults_quantile_grid_properties(defaults_data):
    """(c2) quantile_grid has 101 points, is monotonic non-decreasing, endpoints match min/max, index 50 matches median."""
    inputs = defaults_data["inputs"]
    medians = defaults_data["medians"]

    assert len(inputs) == 18
    for feat, info in inputs.items():
        assert "quantile_grid" in info, f"Missing quantile_grid for {feat}"
        assert "scale_flag" in info, f"Missing scale_flag for {feat}"
        assert isinstance(info["scale_flag"], bool), f"scale_flag must be bool for {feat}"

        grid = info["quantile_grid"]
        assert len(grid) == 101, f"Expected 101 points in quantile_grid for {feat}, got {len(grid)}"
        # Monotonic non-decreasing
        for i in range(100):
            assert grid[i] <= grid[i + 1], f"quantile_grid not monotonic at index {i} for {feat}"
        # Endpoints equal min and max
        assert np.isclose(grid[0], info["min"], rtol=0.0, atol=1e-12), f"grid[0] != min for {feat}"
        assert np.isclose(grid[100], info["max"], rtol=0.0, atol=1e-12), f"grid[100] != max for {feat}"
        # Index 50 equals p50 and medians[feat]
        assert np.isclose(grid[50], info["p50"], rtol=0.0, atol=1e-12), f"grid[50] != p50 for {feat}"
        assert np.isclose(grid[50], medians[feat], rtol=0.0, atol=1e-12), f"grid[50] != median for {feat}"


def test_defaults_preset_profiles_are_virtual_and_unmatched(defaults_data, train_data):
    """(d) Preset profiles are virtual summaries and do not match any single training row."""
    X_train, y_train = train_data
    inputs = defaults_data["inputs"]
    input_cols = list(inputs.keys())
    presets = defaults_data["presets"]

    expected_note = "訓練資料彙總出來的虛擬輪廓，不是任何一家真實公司"
    assert presets["description"] == expected_note

    for profile_name in ["normal_profile", "bankrupt_profile"]:
        profile = presets[profile_name]
        assert profile["description"] == expected_note

        values = profile["values"] if "values" in profile else profile
        profile_series = pd.Series({col: values[col] for col in input_cols})

        # Compare against every row in training data on the 18 input columns
        exact_matches = (X_train[input_cols] == profile_series).all(axis=1)
        assert not exact_matches.any(), (
            f"Preset profile '{profile_name}' matches a real training row at index: "
            f"{list(X_train.index[exact_matches])}"
        )


def test_defaults_t6_test_data_isolation_scan():
    """(e) T6 static scan: make_app_defaults.py and all app/ files must not contain forbidden strings."""
    # Constructed dynamically so this test file itself doesn't trigger static scans
    forbidden = [
        "get_" + "test_data",
        "test_" + "idx",
        "outputs/" + "e6",
        "e" + "6_",
    ]

    files_to_scan = []
    script_file = PROJECT_ROOT / "scripts" / "make_app_defaults.py"
    if script_file.exists():
        files_to_scan.append(script_file)

    app_dir = PROJECT_ROOT / "app"
    if app_dir.exists():
        for p in app_dir.rglob("*"):
            if p.is_file():
                files_to_scan.append(p)

    assert len(files_to_scan) > 0, "No files found to scan for T6 isolation test!"

    violations = []
    for fpath in files_to_scan:
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for bad_str in forbidden:
            if bad_str in content:
                violations.append(f"{fpath.relative_to(PROJECT_ROOT)} contains '{bad_str}'")

    assert not violations, f"T6 isolation violations detected:\n" + "\n".join(violations)
