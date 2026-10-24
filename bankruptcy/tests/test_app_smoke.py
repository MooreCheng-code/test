"""
tests/test_app_smoke.py

Automated smoke, consistency, isolation, and AppTest checks for the Streamlit app.
Covers requirements T5, T6, T7, and T8.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ==============================================================================
# T5: Consistency & No Hardcoded Numbers
# ==============================================================================
def test_t5_static_check_no_hardcoded_numbers():
    """Verify no hardcoded thresholds or prevalence numbers exist in app/ python files."""
    import ast

    prohibited_patterns = [
        r"0\.8845",
        r"0\.7400",
        r"0\.5983",
        r"0\.1708",
        r"0\.03226",
        r"3\.2%",
        r"3\.226%",
        r"3\.23%",
        r"百分之三點二",
        r"百分之三",
    ]

    py_files = list(APP_DIR.glob("*.py"))
    assert len(py_files) >= 3, f"Expected at least 3 python files in app/, found {len(py_files)}"

    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        for pat in prohibited_patterns:
            matches = re.findall(pat, content)
            assert not matches, (
                f"File {py_file.name} contains hardcoded pattern '{pat}': {matches}"
            )

    # Strict AST scan on ui_text.py for any numeric percent or decimal literals
    ui_path = APP_DIR / "ui_text.py"
    tree = ast.parse(ui_path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree)

    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == docstring:
                continue
            strings.append((node.value, getattr(node, "lineno", 0)))

    pat_pct = re.compile(r"\b\d+(?:\.\d+)?%")
    pat_decimal = re.compile(r"\b\d+\.\d+\b")
    pat_chinese_pct = re.compile(r"百分之[一二三四五六七八九十百千點\d]+")

    violations = []
    for s, line in strings:
        clean_s = re.sub(r"\{[^}]*:[^}]*\}%?", "", s)
        clean_s = re.sub(r"%\.?\d*[a-zA-Z]", "", clean_s)

        m_pct = pat_pct.findall(clean_s)
        m_dec = pat_decimal.findall(clean_s)
        m_cpct = pat_chinese_pct.findall(clean_s)

        if m_pct or m_dec or m_cpct:
            violations.append((line, s[:60], m_pct, m_dec, m_cpct))

    assert not violations, f"Strict AST scan found violations in ui_text.py:\n{violations}"


def test_t5_page_display_matches_json_and_csv():
    """Verify App display values match thresholds.json and operating_points_for_report.csv."""
    thresholds_path = PROJECT_ROOT / "outputs" / "models" / "final" / "thresholds.json"
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_data = json.load(f)

    report_csv_path = PROJECT_ROOT / "outputs" / "tables" / "operating_points_for_report.csv"
    report_df = pd.read_csv(report_csv_path)

    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Click calculate on initial (or normal preset)
    calc_btn = [b for b in at.button if "計算" in b.label][0]
    calc_btn.click().run()
    assert not at.exception

    # Check that dataframes are rendered
    assert len(at.dataframe) >= 2, f"Expected dataframes rendered, got {len(at.dataframe)}"
    
    # Tab 1 operating points dataframe is DF 1
    df_tab1 = at.dataframe[1].value
    assert isinstance(df_tab1, pd.DataFrame)
    assert len(df_tab1) == 4

    for op in ["R80", "R90", "R95", "R100"]:
        row_ui = df_tab1[df_tab1["工作點代碼"] == op].iloc[0]
        expected_thresh = thresholds_data["E_A"][op]["threshold"]
        ui_thresh = float(row_ui["固化預警門檻 (t)"])
        assert abs(ui_thresh - expected_thresh) < 1e-4

        rec_val = {"R80": 0.80, "R90": 0.90, "R95": 0.95, "R100": 1.00}[op]
        expected_csv_row = report_df[
            (report_df["group"] == "E_A") & (report_df["target_recall"] == rec_val)
        ].iloc[0]
        # alert_proportion_mean
        expected_prop = f"{expected_csv_row['alert_proportion_mean']*100:.2f}%"
        assert row_ui["CV 預警比例"] == expected_prop
        # actual_recall_mean
        expected_recall = f"{expected_csv_row['actual_recall_mean']*100:.2f}%"
        assert row_ui["CV 15 折預期召回"] == expected_recall
        # alert_ratio_mean
        expected_ratio = f"{expected_csv_row['alert_ratio_mean']:.2f}x"
        assert row_ui["CV 預警倍數 (AR)"] == expected_ratio
        # precision_mean
        expected_prec = f"{expected_csv_row['precision_mean']*100:.2f}%"
        assert row_ui["CV 精準率"] == expected_prec

    # Tab 3 cost table dataframe (contains '目標召回 (Target Recall)' column)
    cost_dfs = [
        d.value
        for d in at.dataframe
        if isinstance(d.value, pd.DataFrame)
        and "目標召回 (Target Recall)" in d.value.columns
    ]
    assert len(cost_dfs) >= 1, "Cost table dataframe not found in at.dataframe"
    df_tab3 = cost_dfs[0]
    assert len(df_tab3) == 4
    for op in ["R80", "R90", "R95", "R100"]:
        rec_val = {"R80": 0.80, "R90": 0.90, "R95": 0.95, "R100": 1.00}[op]
        expected_csv_row = report_df[
            (report_df["group"] == "E_A") & (report_df["target_recall"] == rec_val)
        ].iloc[0]
        row_tab3 = df_tab3[df_tab3["目標召回 (Target Recall)"] == f"{int(rec_val*100)}%"].iloc[0]
        assert abs(float(row_tab3["固化門檻 (Fixed Threshold)"]) - expected_csv_row["fixed_threshold"]) < 1e-4


# ==============================================================================
# T6: Test Data Isolation Static Scan
# ==============================================================================
def test_t6_isolation_scan():
    """Verify app/ and scripts/make_app_defaults.py do not access holdout test data."""
    target_files = list(APP_DIR.glob("*.py")) + [SCRIPTS_DIR / "make_app_defaults.py"]
    prohibited_tokens = [
        "get_test_data",
        "test_idx",
        "outputs/e6",
        "outputs\\e6",
        "e6_",
    ]

    for fpath in target_files:
        assert fpath.exists(), f"Target file does not exist: {fpath}"
        content = fpath.read_text(encoding="utf-8")
        for tok in prohibited_tokens:
            assert tok not in content, (
                f"Prohibited token '{tok}' found in {fpath.relative_to(PROJECT_ROOT)}"
            )


# ==============================================================================
# T7: Warning Mechanism AppTest
# ==============================================================================
def test_t7_apptest_warnings():
    """Verify range warnings (yellow) and out-of-range errors (red) work via AppTest."""
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Switch to exact input mode
    at.radio[0].set_value(at.radio[0].options[1]).run()
    assert not at.exception

    # Find Borrowing dependency widget (bounded normal scale: p99 ~ 0.4009, max ~ 0.9548)
    w_bd = [w for w in at.number_input if "Borrowing dependency" in w.key][0]

    # Test 1: Value within [min, max] but outside [p01, p99] -> range warning
    w_bd.set_value(0.5).run()
    assert len(at.warning) >= 1
    assert any("常見範圍警示" in str(w.value) for w in at.warning)

    # Test 2: Value outside [min, max] -> out of range error
    w_bd.set_value(2.0).run()
    assert len(at.error) >= 1
    assert any("全距超出警示" in str(e.value) for e in at.error)


# ==============================================================================
# T8: End-to-End Smoke Test
# ==============================================================================
def test_t8_apptest_smoke():
    """End-to-end smoke test verifying presets, calculations, and tab renders."""
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4

    # 1. Normal profile preset
    btn_norm = [b for b in at.button if "典型正常公司" in b.label][0]
    btn_norm.click().run()
    assert not at.exception

    calc_btn = [b for b in at.button if "計算" in b.label][0]
    calc_btn.click().run()
    assert not at.exception

    metric_labels = [m.label for m in at.metric]
    assert any("Platt" in lbl for lbl in metric_labels)
    normal_prob_str = [m.value for m in at.metric if "Platt" in m.label][0]
    normal_prob = float(normal_prob_str.replace("%", ""))

    # 2. Bankrupt profile preset
    btn_bank = [b for b in at.button if "破產" in b.label][0]
    btn_bank.click().run()
    assert not at.exception

    calc_btn = [b for b in at.button if "計算" in b.label][0]
    calc_btn.click().run()
    assert not at.exception

    bankrupt_prob_str = [m.value for m in at.metric if "Platt" in m.label][0]
    bankrupt_prob = float(bankrupt_prob_str.replace("%", ""))

    # Bankrupt profile risk must be substantially higher than normal profile
    assert bankrupt_prob > normal_prob
    assert bankrupt_prob > 10.0  # ~20.59%
    assert normal_prob < 1.0     # ~0.06%


def test_t8_tab2_renders_with_bankrupt_profile_and_matches_logic():
    """Verify Tab 2 (effects, position table, what-if) renders without exception and matches logic."""
    from app import logic

    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Apply bankrupt preset
    btn_bank = [b for b in at.button if "破產" in b.label][0]
    btn_bank.click().run()
    assert not at.exception

    calc_btn = [b for b in at.button if "計算" in b.label][0]
    calc_btn.click().run()
    assert not at.exception

    defaults = logic.load_defaults()
    components = logic.load_components()
    user_inputs = defaults["presets"]["bankrupt_profile"]["values"]

    # Verify single feature effects logic matches
    eff_df = logic.single_feature_effects(user_inputs, defaults, components)
    assert len(eff_df) == 18
    assert "delta_drop" in eff_df.columns
    assert "delta_add" in eff_df.columns

    # Verify position table in Tab 2 exists
    pos_dfs = [
        d.value
        for d in at.dataframe
        if isinstance(d.value, pd.DataFrame) and "落點百分位 (0~100)" in d.value.columns
    ]
    assert len(pos_dfs) == 1
    ui_pos_df = pos_dfs[0]
    assert len(ui_pos_df) == 18

    # Verify what-if curve logic matches
    feat = list(defaults["inputs"].keys())[0]
    curve_res = logic.what_if_curve(feat, user_inputs, defaults, components, n=40, grid="percentile")
    assert 1 <= len(curve_res["curve_df"]) <= 40
    assert "score_E_A" in curve_res["curve_df"].columns
