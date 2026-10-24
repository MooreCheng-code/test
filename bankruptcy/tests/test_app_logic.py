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
    # 1. All inputs at median -> all deltas must be 0.0 for both drop and add
    median_inputs = {f: defaults["medians"][f] for f in defaults["inputs"]}
    eff_med = logic.single_feature_effects(median_inputs, defaults, components)
    assert len(eff_med) == 18
    assert np.all(eff_med["delta_drop"].values == 0.0)
    assert np.all(eff_med["delta_add"].values == 0.0)
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

    all_med_row = logic.build_full_row({}, defaults)
    med_base_score_manual = float(scoring.score_dataset(all_med_row, components)["score_E_A"].iloc[0])

    sample_features = [
        "Interest-bearing debt interest rate",
        "Borrowing dependency",
        "Persistent EPS in the Last Four Seasons",
    ]
    for feat in sample_features:
        # Drop manual verification
        mod_drop_inputs = dict(bankrupt_inputs)
        mod_drop_inputs[feat] = float(defaults["medians"][feat])
        mod_drop_row = logic.build_full_row(mod_drop_inputs, defaults)
        rep_score_manual = float(scoring.score_dataset(mod_drop_row, components)["score_E_A"].iloc[0])
        expected_delta_drop = base_score_manual - rep_score_manual

        # Add manual verification
        mod_add_inputs = {feat: bankrupt_inputs[feat]}
        mod_add_row = logic.build_full_row(mod_add_inputs, defaults)
        add_score_manual = float(scoring.score_dataset(mod_add_row, components)["score_E_A"].iloc[0])
        expected_delta_add = add_score_manual - med_base_score_manual

        row_in_eff = eff_bank[eff_bank["feature"] == feat].iloc[0]
        assert np.isclose(row_in_eff["base_score"], base_score_manual, rtol=0.0, atol=1e-15)
        assert np.isclose(row_in_eff["replaced_score"], rep_score_manual, rtol=0.0, atol=1e-15)
        assert np.isclose(row_in_eff["delta_drop"], expected_delta_drop, rtol=0.0, atol=1e-15)
        assert np.isclose(row_in_eff["delta_add"], expected_delta_add, rtol=0.0, atol=1e-15)


# ------------------------------------------------------------------------------
# T4: what_if_curve
# ------------------------------------------------------------------------------
def test_t4_what_if_curve_matches_pointwise_scoring(defaults, components):
    user_in = defaults["presets"]["normal_profile"]["values"]
    target_feat = "Borrowing dependency"

    # Default grid="percentile"
    res_pct = logic.what_if_curve(target_feat, user_in, defaults, components, n=40, grid="percentile")
    curve_df_pct = res_pct["curve_df"]
    assert "percentile" in curve_df_pct.columns
    assert "feature_value" in curve_df_pct.columns
    assert "score_E_A" in curve_df_pct.columns
    # Verify x-axis (percentile) is monotonic non-decreasing
    pcts = curve_df_pct["percentile"].values
    assert np.all(pcts[:-1] <= pcts[1:])

    # Pointwise check for grid="percentile"
    for idx in [0, 10, 20, len(curve_df_pct) - 1]:
        val = float(curve_df_pct["feature_value"].iloc[idx])
        score_curve = float(curve_df_pct["score_E_A"].iloc[idx])
        pt_row = logic.build_full_row({target_feat: val}, defaults)
        score_manual = float(scoring.score_dataset(pt_row, components)["score_E_A"].iloc[0])
        assert np.isclose(score_curve, score_manual, rtol=0.0, atol=1e-15)

    # grid="value" backward compatibility
    res_val = logic.what_if_curve(target_feat, user_in, defaults, components, n=40, grid="value")
    curve_df_val = res_val["curve_df"]
    assert len(curve_df_val) == 40
    for idx in [0, 9, 19, 29, 39]:
        val = float(curve_df_val["feature_value"].iloc[idx])
        score_from_curve = float(curve_df_val["score_E_A"].iloc[idx])
        pt_row = logic.build_full_row({target_feat: val}, defaults)
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
# D-1 enhancements: values_from_percentiles, scale_warning, describe_score, training_share
# ------------------------------------------------------------------------------
def test_d1_values_from_percentiles_properties(defaults):
    for feat, info in defaults["inputs"].items():
        v0 = logic.values_from_percentiles(0.0, defaults, feature=feat)
        v50 = logic.values_from_percentiles(50.0, defaults, feature=feat)
        v100 = logic.values_from_percentiles(100.0, defaults, feature=feat)

        assert np.isclose(v0, info["min"], rtol=0.0, atol=1e-12)
        assert np.isclose(v50, info["p50"], rtol=0.0, atol=1e-12)
        assert np.isclose(v100, info["max"], rtol=0.0, atol=1e-12)

        # Monotonic non-decreasing test
        test_pcts = np.linspace(0.0, 100.0, 101)
        vals = [logic.values_from_percentiles(p, defaults, feature=feat) for p in test_pcts]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    # Dict input test
    p_dict = {f: 50.0 for f in defaults["inputs"]}
    vals_dict = logic.values_from_percentiles(p_dict, defaults)
    assert len(vals_dict) == 18
    for f in defaults["inputs"]:
        assert np.isclose(vals_dict[f], defaults["inputs"][f]["p50"], rtol=0.0, atol=1e-12)


def test_d1_scale_warning(defaults):
    for feat, info in defaults["inputs"].items():
        msg = logic.scale_warning(feat, defaults)
        if info["scale_flag"]:
            assert msg is not None
            assert "跨越" in msg
        else:
            assert msg is None


def test_d1_describe_score():
    s = logic.describe_score(0.85)
    assert "平均百分位分數 0.85" in s
    assert "三個模型各自把這筆輸入排在訓練資料參考分布的百分位" in s


def test_d1_training_share_at_or_above_diff_less_than_0005(components):
    th = components["thresholds"]["E_A"]
    for op_key, info in th.items():
        t = float(info["threshold"])
        expected_prop = float(info["training_oof_alert_proportion"])
        calc_prop = logic.training_share_at_or_above(t, components)
        diff = abs(calc_prop - expected_prop)
        assert diff < 0.005, f"Operating point {op_key}: diff {diff} >= 0.005"


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
