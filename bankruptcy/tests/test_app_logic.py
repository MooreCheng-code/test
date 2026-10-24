"""
tests/test_app_logic.py

Unit and integration tests for app/logic.py covering T1, T2, T3, T4, T7, T10, T11, T12, and T13.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import logic
from src import scoring
from src.cv import get_train_data


@pytest.fixture(scope="module")
def defaults():
    return logic.load_defaults()


@pytest.fixture(scope="module")
def components():
    return logic.load_components()


@pytest.fixture(scope="module")
def train_df():
    X_train, _ = get_train_data(PROJECT_ROOT / "data" / "data.csv")
    return X_train


# ------------------------------------------------------------------------------
# T1: build_full_row
# ------------------------------------------------------------------------------
def test_t1_build_full_row_columns_order_and_median_fill(defaults):
    user_in = {"Quick Ratio": 0.025, "Borrowing dependency": 0.37}
    df = logic.build_full_row(user_in, defaults)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 95)
    assert list(df.columns) == defaults["raw_columns"]

    assert df["Quick Ratio"].iloc[0] == 0.025
    assert df["Borrowing dependency"].iloc[0] == 0.37

    for col in defaults["raw_columns"]:
        if col not in user_in:
            assert df[col].iloc[0] == float(defaults["medians"][col])


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"Unknown_Feature_XYZ": 0.5},
        {"Quick Ratio": float("nan")},
        {"Quick Ratio": float("inf")},
        {"Quick Ratio": float("-inf")},
        {"Quick Ratio": "0.025"},
        {"Quick Ratio": None},
        {"Quick Ratio": True},
    ],
)
def test_t1_build_full_row_invalid_inputs_raise(defaults, bad_inputs):
    with pytest.raises((ValueError, TypeError)):
        logic.build_full_row(bad_inputs, defaults)


# ------------------------------------------------------------------------------
# T2: score_rows bitwise consistency with scoring.score_dataset
# ------------------------------------------------------------------------------
def test_t2_score_rows_exact_bitwise_match(defaults, components, train_df):
    normal_vals = defaults["presets"]["normal_profile"]["values"]
    row_df = logic.build_full_row(normal_vals, defaults)

    res_app = logic.score_rows(row_df, components)
    res_direct = scoring.score_dataset(row_df, components)

    assert list(res_app.columns) == list(res_direct.columns)
    for col in ["score_E_A", "calibrated_prob_platt", "flag_R80", "flag_R90", "flag_R95", "flag_R100"]:
        assert col in res_app.columns

    for col in res_app.columns:
        if res_app[col].dtype == bool:
            assert (res_app[col].values == res_direct[col].values).all()
        else:
            max_diff = float(np.max(np.abs(res_app[col].values - res_direct[col].values)))
            assert max_diff == 0.0, f"Bitwise difference {max_diff} on column {col}"

    # Full 95-column training row verification
    sample_row_idx = 42
    full_train_row_dict = train_df.iloc[sample_row_idx].to_dict()
    built_train_row = logic.build_full_row(full_train_row_dict, defaults)
    scored_via_logic = logic.score_rows(built_train_row, components)
    scored_direct_train = scoring.score_dataset(train_df.iloc[[sample_row_idx]], components)

    diff_ea = abs(
        float(scored_via_logic["score_E_A"].iloc[0]) - float(scored_direct_train["score_E_A"].iloc[0])
    )
    assert diff_ea == 0.0


# ------------------------------------------------------------------------------
# T3: single_feature_effects
# ------------------------------------------------------------------------------
def test_t3_single_feature_effects_zero_at_medians_and_manual_match(defaults, components):
    # 1. All inputs at median -> all deltas must be 0.0
    median_inputs = {f: defaults["medians"][f] for f in defaults["inputs"]}
    eff_med = logic.single_feature_effects(median_inputs, defaults, components)
    assert len(eff_med) == 18
    assert np.all(eff_med["delta"].values == 0.0)
    assert np.all(eff_med["abs_delta"].values == 0.0)

    # 2. Bankrupt profile: verify 3 sampled features against manual score_dataset calls
    bankrupt_inputs = defaults["presets"]["bankrupt_profile"]["values"]
    eff_bank = logic.single_feature_effects(bankrupt_inputs, defaults, components)

    # Verify sorted descending by abs_delta
    abs_deltas = eff_bank["abs_delta"].values
    assert np.all(abs_deltas[:-1] >= abs_deltas[1:])

    base_row = logic.build_full_row(bankrupt_inputs, defaults)
    base_score_manual = float(scoring.score_dataset(base_row, components)["score_E_A"].iloc[0])

    sample_features = [
        "Interest-bearing debt interest rate",
        "Borrowing dependency",
        "Persistent EPS in the Last Four Seasons",
    ]
    for feat in sample_features:
        mod_inputs = dict(bankrupt_inputs)
        mod_inputs[feat] = float(defaults["medians"][feat])
        mod_row = logic.build_full_row(mod_inputs, defaults)
        rep_score_manual = float(scoring.score_dataset(mod_row, components)["score_E_A"].iloc[0])
        expected_delta = base_score_manual - rep_score_manual

        row_in_eff = eff_bank[eff_bank["feature"] == feat].iloc[0]
        assert np.isclose(row_in_eff["base_score"], base_score_manual, rtol=0.0, atol=1e-15)
        assert np.isclose(row_in_eff["replaced_score"], rep_score_manual, rtol=0.0, atol=1e-15)
        assert np.isclose(row_in_eff["delta"], expected_delta, rtol=0.0, atol=1e-15)


# ------------------------------------------------------------------------------
# T4: what_if_curve
# ------------------------------------------------------------------------------
def test_t4_what_if_curve_matches_pointwise_scoring(defaults, components):
    user_in = defaults["presets"]["normal_profile"]["values"]
    target_feat = "Borrowing dependency"
    res = logic.what_if_curve(target_feat, user_in, defaults, components, n=40)

    curve_df = res["curve_df"]
    assert len(curve_df) == 40
    assert set(res["thresholds"].keys()) == {"R80", "R90", "R95", "R100"}

    # Check 5 sampled indices against individual score_dataset calls
    sample_indices = [0, 9, 19, 29, 39]
    for idx in sample_indices:
        val = float(curve_df["feature_value"].iloc[idx])
        score_from_curve = float(curve_df["score_E_A"].iloc[idx])

        pt_inputs = dict(user_in)
        pt_inputs[target_feat] = val
        pt_row = logic.build_full_row(pt_inputs, defaults)
        score_manual = float(scoring.score_dataset(pt_row, components)["score_E_A"].iloc[0])

        assert np.isclose(score_from_curve, score_manual, rtol=0.0, atol=1e-15)


# ------------------------------------------------------------------------------
# T7: input_warnings
# ------------------------------------------------------------------------------
def test_t7_input_warnings_levels(defaults):
    base_in = {f: defaults["medians"][f] for f in defaults["inputs"]}
    assert logic.input_warnings(base_in, defaults) == []

    # Test 'range' warning: outside [p01, p99] but inside [min, max]
    feat = "Borrowing dependency"
    info = defaults["inputs"][feat]
    assert info["p99"] < info["max"]
    range_in = dict(base_in)
    range_in[feat] = 0.5 * (float(info["p99"]) + float(info["max"]))
    w_range = logic.input_warnings(range_in, defaults)
    assert len(w_range) == 1
    assert w_range[0]["feature"] == feat
    assert w_range[0]["level"] == "range"

    # Test 'out_of_range' warning: outside [min, max]
    oor_in = dict(base_in)
    oor_in[feat] = float(info["max"]) + 0.1
    w_oor = logic.input_warnings(oor_in, defaults)
    assert len(w_oor) == 1
    assert w_oor[0]["feature"] == feat
    assert w_oor[0]["level"] == "out_of_range"

    oor_low_in = dict(base_in)
    oor_low_in[feat] = float(info["min"]) - 0.1
    w_oor_low = logic.input_warnings(oor_low_in, defaults)
    assert len(w_oor_low) == 1
    assert w_oor_low[0]["feature"] == feat
    assert w_oor_low[0]["level"] == "out_of_range"


# ------------------------------------------------------------------------------
# T10: operating_point_table
# ------------------------------------------------------------------------------
def test_t10_operating_point_table_consistency_and_tamper_detection(components):
    op_df = logic.operating_point_table(
        components,
        flags={"R80": False, "R90": True, "R95": True, "R100": True},
    )
    assert len(op_df) == 4
    assert list(op_df["operating_point"]) == ["R80", "R90", "R95", "R100"]
    assert list(op_df["flagged"]) == [False, True, True, True]

    for _, row in op_df.iterrows():
        op = row["operating_point"]
        expected_t = float(components["thresholds"]["E_A"][op]["threshold"])
        assert np.isclose(row["threshold"], expected_t, rtol=0.0, atol=1e-12)
        assert np.isclose(row["fixed_threshold"], expected_t, rtol=0.0, atol=1e-12)

    # Tamper with threshold in components -> must raise ValueError
    tampered_comp = copy.deepcopy(components)
    tampered_comp["thresholds"]["E_A"]["R90"]["threshold"] += 0.01
    with pytest.raises(ValueError, match="Threshold mismatch"):
        logic.operating_point_table(tampered_comp)


# ------------------------------------------------------------------------------
# T11: position_table
# ------------------------------------------------------------------------------
def test_t11_position_table_min_median_max_and_clamping(defaults):
    min_inputs = {f: float(info["min"]) for f, info in defaults["inputs"].items()}
    med_inputs = {f: float(defaults["medians"][f]) for f in defaults["inputs"]}
    max_inputs = {f: float(info["max"]) for f, info in defaults["inputs"].items()}

    pos_min = logic.position_table(min_inputs, defaults)
    pos_med = logic.position_table(med_inputs, defaults)
    pos_max = logic.position_table(max_inputs, defaults)

    assert np.all(pos_min["percentile"].values == 0.0)
    assert np.allclose(pos_med["percentile"].values, 50.0, atol=1.0)
    assert np.all(pos_max["percentile"].values == 100.0)

    # Below min and above max clamping
    below_inputs = {f: float(info["min"]) - 10.0 for f, info in defaults["inputs"].items()}
    above_inputs = {f: float(info["max"]) + 10.0 for f, info in defaults["inputs"].items()}

    pos_below = logic.position_table(below_inputs, defaults)
    pos_above = logic.position_table(above_inputs, defaults)
    assert np.all(pos_below["percentile"].values == 0.0)
    assert np.all(pos_above["percentile"].values == 100.0)


# ------------------------------------------------------------------------------
# T12: Static scan of app/logic.py and scripts/make_app_defaults.py
# ------------------------------------------------------------------------------
def test_t12_static_scan_no_streamlit_no_test_data_no_hardcoded_numbers():
    target_files = [
        PROJECT_ROOT / "app" / "logic.py",
        PROJECT_ROOT / "scripts" / "make_app_defaults.py",
    ]
    forbidden_strings = [
        "get_" + "test_data",
        "test_" + "idx",
        "outputs/" + "e6",
        "e" + "6_",
    ]
    # Hardcoded threshold / prevalence patterns that must never appear as literals
    forbidden_numeric_substrings = [
        "0.8845",
        "0.7399",
        "0.7400",
        "0.5983",
        "0.1708",
        "0.8743",
        "0.7801",
        "0.6449",
        "0.2673",
        "0.0322",
        "0.0323",
        "3.22%",
        "3.23%",
    ]

    for fpath in target_files:
        assert fpath.exists(), f"Missing file {fpath}"
        source = fpath.read_text(encoding="utf-8")

        # 1. Check AST imports for streamlit
        tree = ast.parse(source, filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("streamlit"), f"{fpath.name} imports streamlit"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("streamlit"), f"{fpath.name} imports from streamlit"

        # 2. Check forbidden test-set strings
        for bad in forbidden_strings:
            assert bad not in source, f"{fpath.name} contains forbidden string {bad!r}"

        # 3. Check forbidden hardcoded thresholds or prevalence literals
        for bad_num in forbidden_numeric_substrings:
            assert bad_num not in source, f"{fpath.name} contains hardcoded literal {bad_num!r}"


# ------------------------------------------------------------------------------
# T13: Batch inference performance benchmark (1 + 18 + 40 = 59 rows)
# ------------------------------------------------------------------------------
def test_t13_full_interaction_batch_timing(defaults, components):
    user_in = defaults["presets"]["bankrupt_profile"]["values"]
    target_feat = "Borrowing dependency"

    t0 = time.perf_counter()
    row_df = logic.build_full_row(user_in, defaults)
    _ = logic.score_rows(row_df, components)  # 1 row
    _ = logic.single_feature_effects(user_in, defaults, components)  # 19 rows (or 18 replacements)
    _ = logic.what_if_curve(target_feat, user_in, defaults, components, n=40)  # 40 rows
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"Full interaction took {elapsed:.3f}s, exceeding 3.0s limit"
