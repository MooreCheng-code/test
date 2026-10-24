"""
app/app.py

Main interactive Streamlit application for bankruptcy early warning demonstration.
Execution:
    streamlit run app/app.py (from repository root bankruptcy/)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    import logic  # type: ignore
    import ui_text  # type: ignore
except ImportError:
    from app import logic, ui_text

# ==============================================================================
# Page configuration & Caching
# ==============================================================================
st.set_page_config(
    page_title=ui_text.PAGE_TITLE,
    page_icon=ui_text.PAGE_ICON,
    layout="wide",
)


@st.cache_resource(show_spinner="正在載入固化模型元件...")
def get_cached_components():
    return logic.load_components()


@st.cache_data(show_spinner="正在載入訓練資料設定檔...")
def get_cached_defaults():
    return logic.load_defaults()


def format_value_smart(val: float) -> str:
    """Format float cleanly in scientific notation if extreme, else fixed decimal."""
    abs_v = abs(val)
    if (abs_v != 0.0 and (abs_v < 1e-4 or abs_v >= 1e5)):
        return f"{val:.4e}"
    return f"{val:.4f}"


def main():
    components = get_cached_components()
    defaults = get_cached_defaults()

    st.title(ui_text.APP_HEADER_TITLE)
    st.caption(ui_text.APP_HEADER_SUBTITLE)

    tab1, tab2, tab3, tab4 = st.tabs(ui_text.TAB_LABELS)

    # ==========================================================================
    # Tab 1: 輸入與結果
    # ==========================================================================
    with tab1:
        st.subheader("公司核心財務特徵設定")

        # Session state initialization
        if "inputs_state" not in st.session_state:
            st.session_state.inputs_state = {
                f: float(defaults["inputs"][f]["p50"]) for f in defaults["inputs"]
            }
        if "slider_state" not in st.session_state:
            st.session_state.slider_state = {f: 50.0 for f in defaults["inputs"]}

        col_mode, col_actions = st.columns([1, 2])
        with col_mode:
            input_mode = st.radio(
                ui_text.INPUT_MODE_LABEL,
                [ui_text.MODE_SLIDER_LABEL, ui_text.MODE_EXACT_LABEL],
                index=0,
                horizontal=True,
            )

        with col_actions:
            st.write("輪廓範例快速套用：")
            b_norm, b_bank, b_reset = st.columns(3)
            if b_norm.button(ui_text.PRESET_BUTTON_NORMAL, use_container_width=True):
                norm_vals = defaults["presets"]["normal_profile"]["values"]
                pos_df = logic.position_table(norm_vals, defaults)
                pct_map = dict(zip(pos_df["feature"], pos_df["percentile"]))
                for f in defaults["inputs"]:
                    val = float(norm_vals[f])
                    pct = float(pct_map.get(f, 50.0))
                    st.session_state[f"num_{f}"] = val
                    st.session_state[f"slider_{f}"] = pct
                    st.session_state.inputs_state[f] = val
                    st.session_state.slider_state[f] = pct
                st.rerun()

            if b_bank.button(ui_text.PRESET_BUTTON_BANKRUPT, use_container_width=True):
                bank_vals = defaults["presets"]["bankrupt_profile"]["values"]
                pos_df = logic.position_table(bank_vals, defaults)
                pct_map = dict(zip(pos_df["feature"], pos_df["percentile"]))
                for f in defaults["inputs"]:
                    val = float(bank_vals[f])
                    pct = float(pct_map.get(f, 50.0))
                    st.session_state[f"num_{f}"] = val
                    st.session_state[f"slider_{f}"] = pct
                    st.session_state.inputs_state[f] = val
                    st.session_state.slider_state[f] = pct
                st.rerun()

            if b_reset.button(ui_text.PRESET_BUTTON_RESET, use_container_width=True):
                for f in defaults["inputs"]:
                    val = float(defaults["inputs"][f]["p50"])
                    st.session_state[f"num_{f}"] = val
                    st.session_state[f"slider_{f}"] = 50.0
                    st.session_state.inputs_state[f] = val
                    st.session_state.slider_state[f] = 50.0
                st.rerun()

        st.caption(ui_text.VIRTUAL_PROFILE_NOTICE)

        user_current_inputs: dict[str, float] = {}

        # Render feature inputs divided into Main and Advanced tiers
        main_feats = [f for f, info in defaults["inputs"].items() if info["tier"] == "main"]
        adv_feats = [f for f, info in defaults["inputs"].items() if info["tier"] == "advanced"]

        with st.expander(ui_text.EXPANDER_MAIN_TITLE, expanded=True):
            cols = st.columns(2)
            for idx, feat in enumerate(main_feats):
                info = defaults["inputs"][feat]
                col = cols[idx % 2]
                with col:
                    f_label = f"**{info['importance_rank']}. {feat}** {ui_text.FEATURE_NOTE_SUFFIX}"
                    warn_msg = logic.scale_warning(feat, defaults)
                    if warn_msg:
                        st.caption(f"⚠️ {warn_msg}")

                    if input_mode == ui_text.MODE_SLIDER_LABEL:
                        slider_key = f"slider_{feat}"
                        if slider_key not in st.session_state:
                            st.session_state[slider_key] = 50.0
                        current_pct = st.slider(
                            f_label,
                            min_value=0.0,
                            max_value=100.0,
                            step=1.0,
                            key=slider_key,
                        )
                        st.session_state.slider_state[feat] = current_pct
                        calc_val = float(
                            logic.values_from_percentiles(current_pct, defaults, feature=feat)
                        )
                        st.session_state.inputs_state[feat] = calc_val
                        user_current_inputs[feat] = calc_val
                        st.caption(f"對應訓練分布數值：`{format_value_smart(calc_val)}`")
                    else:
                        num_key = f"num_{feat}"
                        if num_key not in st.session_state:
                            st.session_state[num_key] = float(info["p50"])
                        curr_num = st.number_input(
                            f_label,
                            format="%.6g",
                            key=num_key,
                        )
                        st.session_state.inputs_state[feat] = float(curr_num)
                        user_current_inputs[feat] = float(curr_num)

        with st.expander(ui_text.EXPANDER_ADVANCED_TITLE, expanded=False):
            cols_adv = st.columns(2)
            for idx, feat in enumerate(adv_feats):
                info = defaults["inputs"][feat]
                col = cols_adv[idx % 2]
                with col:
                    f_label = f"**{info['importance_rank']}. {feat}** {ui_text.FEATURE_NOTE_SUFFIX}"
                    warn_msg = logic.scale_warning(feat, defaults)
                    if warn_msg:
                        st.caption(f"⚠️ {warn_msg}")

                    if input_mode == ui_text.MODE_SLIDER_LABEL:
                        slider_key = f"slider_{feat}"
                        if slider_key not in st.session_state:
                            st.session_state[slider_key] = 50.0
                        current_pct = st.slider(
                            f_label,
                            min_value=0.0,
                            max_value=100.0,
                            step=1.0,
                            key=slider_key,
                        )
                        st.session_state.slider_state[feat] = current_pct
                        calc_val = float(
                            logic.values_from_percentiles(current_pct, defaults, feature=feat)
                        )
                        st.session_state.inputs_state[feat] = calc_val
                        user_current_inputs[feat] = calc_val
                        st.caption(f"對應訓練分布數值：`{format_value_smart(calc_val)}`")
                    else:
                        num_key = f"num_{feat}"
                        if num_key not in st.session_state:
                            st.session_state[num_key] = float(info["p50"])
                        curr_num = st.number_input(
                            f_label,
                            format="%.6g",
                            key=num_key,
                        )
                        st.session_state.inputs_state[feat] = float(curr_num)
                        user_current_inputs[feat] = float(curr_num)

        # Check and display input warnings
        active_warnings = logic.input_warnings(user_current_inputs, defaults)
        if active_warnings:
            st.write("---")
            for w in active_warnings:
                if w["level"] == "out_of_range":
                    st.error(
                        ui_text.format_out_of_range_message(
                            w["feature"], w["value"], w["min"], w["max"]
                        )
                    )
                else:
                    st.warning(
                        ui_text.format_range_warning_message(
                            w["feature"], w["value"], w["p01"], w["p99"]
                        )
                    )

        # 54 filled columns explanation
        with st.expander(ui_text.EXPANDER_FILL_TITLE, expanded=False):
            st.markdown(ui_text.format_fill_explanation(len(defaults["fill_columns"])))
            fill_table_data = [
                {
                    "未手動輸入之模型特徵": c,
                    "訓練中位數補值": format_value_smart(defaults["medians"][c]),
                }
                for c in defaults["fill_columns"]
            ]
            st.dataframe(pd.DataFrame(fill_table_data), width="stretch", height=240)

        # Calculation Section
        st.write("---")
        calc_clicked = st.button(ui_text.RESULTS_CALCULATE_BTN, type="primary", use_container_width=True)

        if calc_clicked:
            st.subheader(ui_text.RESULTS_SECTION_TITLE)

            row_df = logic.build_full_row(user_current_inputs, defaults)
            scored_df = logic.score_rows(row_df, components)

            score_ea = float(scored_df["score_E_A"].iloc[0])
            score_desc = logic.describe_score(score_ea)
            share_ge = logic.training_share_at_or_above(score_ea, components)

            st.info(ui_text.format_score_summary(score_desc, share_ge))

            col_res_l, col_res_r = st.columns([3, 2])
            with col_res_l:
                st.markdown(f"#### {ui_text.TABLE_OPERATING_POINTS_TITLE}")
                op_df = logic.operating_point_table(components, flags=scored_df)
                display_op_df = pd.DataFrame(
                    {
                        "工作點代碼": op_df["operating_point"],
                        "目標召回要求": op_df["target_recall"].apply(lambda x: f"{x*100:.0f}%"),
                        "建議安全邊際 (m)": op_df["recommended_margin"].apply(lambda x: f"+{x:.2f}"),
                        "固化預警門檻 (t)": op_df["fixed_threshold"].apply(lambda x: f"{x:.4f}"),
                        "本樣本是否預警": op_df["flagged"].apply(lambda x: "🚨 預警" if x else "✅ 正常"),
                        "CV 15 折預期召回": op_df["actual_recall_mean"].apply(lambda x: f"{x*100:.2f}%"),
                        "CV 預警倍數 (AR)": op_df["alert_ratio_mean"].apply(lambda x: f"{x:.2f}x"),
                        "CV 預警比例": op_df["alert_proportion_mean"].apply(lambda x: f"{x*100:.2f}%"),
                        "CV 精準率": op_df["precision_mean"].apply(lambda x: f"{x*100:.2f}%"),
                        "CV 達標率": op_df["attainment_rate"].apply(lambda x: f"{x*100:.0f}%"),
                    }
                )
                st.dataframe(display_op_df, width="stretch", hide_index=True)

            with col_res_r:
                st.markdown(f"#### {ui_text.CALIBRATED_PROB_TITLE}")
                prob_platt = float(scored_df["calibrated_prob_platt"].iloc[0])
                base_rate = float(defaults["meta"]["train_bankruptcy_rate"])
                st.metric("Platt 校準後驗破產機率", f"{prob_platt*100:.2f}%")
                st.caption(ui_text.format_platt_note(prob_platt, base_rate))

    # ==========================================================================
    # Tab 2: 為什麼（下一版加入）
    # ==========================================================================
    with tab2:
        st.subheader(ui_text.TAB2_TITLE)
        st.markdown(ui_text.TAB2_NEXT_PHASE_NOTICE)

    # ==========================================================================
    # Tab 3: 模型如何運作
    # ==========================================================================
    with tab3:
        st.subheader(ui_text.TAB3_TITLE)
        st.markdown(ui_text.TAB3_ARCHITECTURE_TEXT)

        # Display Operating Points Cost Table
        st.markdown("#### 工作點固化基準表（取自 operating_points_for_report.csv）")
        cv_csv_path = PROJECT_ROOT / "outputs" / "tables" / "operating_points_for_report.csv"
        if cv_csv_path.exists():
            raw_cv_df = pd.read_csv(cv_csv_path)
            ea_report_df = raw_cv_df[raw_cv_df["group"] == "E_A"].copy()
            cost_table = pd.DataFrame(
                {
                    "模型組別": ea_report_df["group"],
                    "目標召回 (Target Recall)": ea_report_df["target_recall"].apply(lambda x: f"{x*100:.0f}%"),
                    "安全邊際 (Margin)": ea_report_df["recommended_margin"].apply(lambda x: f"+{x:.2f}"),
                    "固化門檻 (Fixed Threshold)": ea_report_df["fixed_threshold"].apply(lambda x: f"{x:.4f}"),
                    "15 折 CV 實際召回 (均值)": ea_report_df["actual_recall_mean"].apply(lambda x: f"{x*100:.2f}%"),
                    "預期預警倍數 (Alert Ratio)": ea_report_df["alert_ratio_mean"].apply(lambda x: f"{x:.2f}x"),
                    "預期預警比例 (Proportion)": ea_report_df["alert_proportion_mean"].apply(lambda x: f"{x*100:.2f}%"),
                    "預期精準率 (Precision)": ea_report_df["precision_mean"].apply(lambda x: f"{x*100:.2f}%"),
                }
            )
            st.dataframe(cost_table, width="stretch", hide_index=True)

        st.markdown(f"### {ui_text.TAB3_PREVALENCE_SECTION_TITLE}")
        st.markdown(ui_text.TAB3_PREVALENCE_EXPLANATION)

        e7a_path = PROJECT_ROOT / "outputs" / "tables" / "e7a_prevalence_sensitivity.csv"
        if e7a_path.exists():
            e7a_df = pd.read_csv(e7a_path)
            e7a_filtered = e7a_df[
                (e7a_df["group"] == "E_A") & (e7a_df["operating_point"].isin(["R90", "R95"]))
            ].copy()
            prev_display = pd.DataFrame(
                {
                    "工作點": e7a_filtered["operating_point"],
                    "假設市場盛行率": e7a_filtered["prevalence"].apply(lambda x: f"{x*100:.2f}%"),
                    "CV 預期精準率": e7a_filtered["cv_precision_mean"].apply(lambda x: f"{x*100:.2f}%"),
                    "每捕獲 1 家破產需審查家數 (CV)": e7a_filtered["cv_effort_per_positive"].apply(lambda x: f"{x:.1f} 家"),
                    "每萬家預期預警家數": e7a_filtered["cv_expected_alerts_per_10k"].apply(lambda x: f"{x:.0f} 家"),
                }
            )
            st.dataframe(prev_display, width="stretch", hide_index=True)

    # ==========================================================================
    # Tab 4: 限制與免責
    # ==========================================================================
    with tab4:
        st.subheader(ui_text.TAB4_TITLE)
        st.markdown(ui_text.TAB4_DISCLAIMER_TEXT)


if __name__ == "__main__":
    main()
