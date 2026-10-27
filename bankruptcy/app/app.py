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

import config
from src.batch_ui_helper import (
    compute_batch_summary_metrics,
    filter_and_sort_batch,
    format_operating_point_option,
    format_top_k_header,
    format_top_k_pct_text,
    format_pct,
    prepare_export_csv_bytes,
)
from src.local_explain import explain_case, get_local_explainer
from src.scoring import load_mvp_components, score_mvp_batch

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

@st.cache_resource(show_spinner="正在載入 MVP 評分組件...")
def get_cached_mvp_components():
    return load_mvp_components()


@st.cache_resource(show_spinner="正在載入特徵歸因解釋器...")
def get_cached_explainer():
    return get_local_explainer()


@st.cache_data(show_spinner="正在載入 OOF 示範資料...")
def get_cached_demo_data():
    csv_path = PROJECT_ROOT / "outputs" / "models" / "mvp_extras" / "oof_demo_batch.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(ui_text.TAB_LABELS)

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
    # Tab 2: 為什麼（特徵影響與假設分析）
    # ==========================================================================
    with tab2:
        import altair as alt

        st.subheader(ui_text.TAB2_TITLE)

        # 1. Single Feature Replacement Effects
        st.markdown(f"### {ui_text.TAB2_EFFECTS_TITLE}")
        st.markdown(ui_text.TAB2_EFFECTS_EXPLANATION)

        eff_view = st.radio(
            ui_text.TAB2_EFFECTS_RADIO_LABEL,
            [ui_text.TAB2_EFFECTS_RADIO_DROP, ui_text.TAB2_EFFECTS_RADIO_ADD],
            horizontal=True,
        )
        target_col = "delta_drop" if eff_view == ui_text.TAB2_EFFECTS_RADIO_DROP else "delta_add"
        target_title = (
            "Delta Drop（替換為訓練中位數後分數下降量）"
            if eff_view == ui_text.TAB2_EFFECTS_RADIO_DROP
            else "Delta Add（僅加入該特徵後分數上升量）"
        )

        eff_df = logic.single_feature_effects(user_current_inputs, defaults, components)

        bar_chart = (
            alt.Chart(eff_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{target_col}:Q", title=target_title),
                y=alt.Y(
                    "feature:N",
                    sort=alt.EncodingSortField(field=target_col, op="sum", order="descending"),
                    title="財務特徵",
                ),
                color=alt.condition(
                    alt.datum[target_col] > 0,
                    alt.value("#d9534f"),
                    alt.value("#5cb85c"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="特徵名稱"),
                    alt.Tooltip(f"{target_col}:Q", title="影響量 (Delta)", format=".4f"),
                    alt.Tooltip("current_value:Q", title="目前數值", format=".4g"),
                    alt.Tooltip("median_value:Q", title="訓練中位數", format=".4g"),
                ],
            )
            .properties(height=480)
        )
        st.altair_chart(bar_chart, width="stretch")

        st.write("---")

        # 2. Position Table
        st.markdown(f"### {ui_text.TAB2_POSITION_TITLE}")
        st.markdown(ui_text.TAB2_POSITION_EXPLANATION)

        pos_df = logic.position_table(user_current_inputs, defaults)
        pos_df["scale_flag"] = pos_df["feature"].apply(
            lambda f: bool(defaults["inputs"][f].get("scale_flag", False))
        )
        pos_display = pd.DataFrame(
            {
                "財務特徵": pos_df["feature"],
                "目前輸入值": pos_df["value"].apply(format_value_smart),
                "落點百分位 (0~100)": pos_df["percentile"],
                "訓練中位數 (p50)": pos_df["p50"].apply(format_value_smart),
                "常見下界 (p01)": pos_df["p01"].apply(format_value_smart),
                "常見上界 (p99)": pos_df["p99"].apply(format_value_smart),
                "尺度特性": pos_df["scale_flag"].apply(
                    lambda x: "⚠️ 跨數量級特徵" if x else "常規尺度"
                ),
            }
        )
        st.dataframe(
            pos_display,
            width="stretch",
            hide_index=True,
            column_config={
                "落點百分位 (0~100)": st.column_config.ProgressColumn(
                    "落點百分位 (0~100)",
                    format="%.1f",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
        )

        st.write("---")

        # 3. What-If Curve
        st.markdown(f"### {ui_text.TAB2_WHATIF_TITLE}")
        st.markdown(ui_text.TAB2_WHATIF_EXPLANATION)

        col_w1, col_w2 = st.columns([2, 2])
        with col_w1:
            selected_feat = st.selectbox(
                ui_text.TAB2_WHATIF_SELECT_FEATURE,
                list(defaults["inputs"].keys()),
                index=0,
            )
        with col_w2:
            grid_choice = st.radio(
                ui_text.TAB2_WHATIF_GRID_LABEL,
                [ui_text.TAB2_WHATIF_GRID_PCT, ui_text.TAB2_WHATIF_GRID_VAL],
                horizontal=True,
            )

        grid_mode = "percentile" if grid_choice == ui_text.TAB2_WHATIF_GRID_PCT else "value"
        res_curve = logic.what_if_curve(
            selected_feat, user_current_inputs, defaults, components, n=40, grid=grid_mode
        )

        curve_data = res_curve["curve_df"].copy()
        x_field = "percentile" if grid_mode == "percentile" else "feature_value"
        x_title = (
            f"{selected_feat} 訓練百分位 (1~99)"
            if grid_mode == "percentile"
            else f"{selected_feat} 數值 (p01~p99)"
        )

        line_chart = (
            alt.Chart(curve_data)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x=alt.X(f"{x_field}:Q", title=x_title),
                y=alt.Y(
                    "score_E_A:Q",
                    title="集成風險百分位分數 (Score E_A)",
                    scale=alt.Scale(domain=[0.0, 1.0]),
                ),
                tooltip=[
                    alt.Tooltip(f"{x_field}:Q", title="X 軸位置", format=".4g"),
                    alt.Tooltip("score_E_A:Q", title="評估分數", format=".4f"),
                ],
            )
        )

        # Threshold horizontal rules
        thresh_items = [
            {"工作點": op_k, "門檻值": float(op_v)}
            for op_k, op_v in res_curve["thresholds"].items()
        ]
        thresh_df_chart = pd.DataFrame(thresh_items)
        rule_chart = (
            alt.Chart(thresh_df_chart)
            .mark_rule(strokeDash=[4, 4])
            .encode(
                y="門檻值:Q",
                color=alt.Color("工作點:N", title="固化預警門檻"),
                tooltip=["工作點:N", alt.Tooltip("門檻值:Q", format=".4f")],
            )
        )

        # Mark current user input location
        curr_x_val = (
            float(
                logic.position_table({selected_feat: user_current_inputs[selected_feat]}, defaults)[
                    "percentile"
                ].iloc[0]
            )
            if grid_mode == "percentile"
            else float(user_current_inputs[selected_feat])
        )
        # Calculate current row score directly
        curr_scored_row = logic.score_rows(
            logic.build_full_row(user_current_inputs, defaults), components
        )
        curr_score = float(curr_scored_row["score_E_A"].iloc[0])

        current_pt = pd.DataFrame(
            [
                {
                    "curr_x": curr_x_val,
                    "curr_y": curr_score,
                    "標記": "目前輸入位置",
                }
            ]
        )
        curr_point_chart = (
            alt.Chart(current_pt)
            .mark_point(size=140, color="#d9534f", filled=True)
            .encode(
                x="curr_x:Q",
                y="curr_y:Q",
                tooltip=[
                    alt.Tooltip("curr_x:Q", title="目前 X 座標", format=".4g"),
                    alt.Tooltip("curr_y:Q", title="目前評估分數", format=".4f"),
                ],
            )
        )

        combined_whatif = (line_chart + rule_chart + curr_point_chart).properties(
            height=400
        )
        st.altair_chart(combined_whatif, width="stretch")

    # ==========================================================================
    # Tab 3: 模型如何運作
    # ==========================================================================
    with tab3:
        st.subheader(ui_text.TAB3_TITLE)
        n_raw = int(defaults["meta"]["n_raw_columns"])
        n_model = int(defaults["meta"]["n_model_columns"])
        st.markdown(ui_text.format_tab3_architecture(n_raw, n_model))

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
        st.markdown(ui_text.format_tab3_prevalence_explanation())

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
        train_rate = float(defaults["meta"]["train_bankruptcy_rate"])
        st.markdown(ui_text.format_tab4_disclaimer(train_rate))


    # ==========================================================================
    # Tab 5: 批次評分與示範
    # ==========================================================================
    with tab5:
        st.subheader(ui_text.TAB5_TITLE)

        mvp_comp = get_cached_mvp_components()
        explainer = get_cached_explainer()
        thresholds_meta = mvp_comp.get("thresholds", {})

        mode_choice = st.radio(
            ui_text.TAB5_MODE_RADIO_LABEL,
            [ui_text.TAB5_MODE_DEMO, ui_text.TAB5_MODE_UPLOAD],
            horizontal=True,
        )

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            model_selected = st.selectbox(
                ui_text.TAB5_MODEL_SELECT_LABEL,
                ["E_A", "M3", "M4", "M1"],
                index=0,
            )
        with col_m2:
            op_choices = ["R80", "R90", "R95", "R100"]
            op_selected = st.selectbox(
                ui_text.TAB5_OP_SELECT_LABEL,
                op_choices,
                index=1,
                format_func=lambda op: format_operating_point_option(
                    op, thresholds_meta.get(model_selected, {})
                ),
            )

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            high_risk_only = st.checkbox(ui_text.TAB5_HIGH_RISK_FILTER_LABEL, value=False)
        with col_f2:
            sort_by_risk = st.checkbox(ui_text.TAB5_SORT_BY_RISK_LABEL, value=True)

        if mode_choice == ui_text.TAB5_MODE_DEMO:
            st.info(ui_text.TAB5_DEMO_EXPLANATION)
            df_demo_raw = get_cached_demo_data()

            if df_demo_raw.empty:
                st.warning("查無 OOF 示範資料檔案。")
            else:
                demo_df = df_demo_raw.copy()
                if "company_id" not in demo_df.columns:
                    demo_df.insert(0, "company_id", [f"DEMO_{i+1:04d}" for i in range(len(demo_df))])

                score_col = f"oof_score_{model_selected}"
                if score_col in demo_df.columns:
                    demo_df["risk_score"] = demo_df[score_col]
                else:
                    demo_df["risk_score"] = demo_df.get("oof_score_E_A", np.nan)

                demo_df["status"] = "ok"

                # Threshold and alert evaluation
                if model_selected == "E_A":
                    thresh_dict = thresholds_meta.get("demo_r0_f0_E_A", {}).get(op_selected, {})
                    thresh_val = float(thresh_dict.get("threshold", 0.7235))
                    demo_df["threshold"] = thresh_val
                    demo_df["alert"] = (demo_df["risk_score"] >= thresh_val)
                    cal_platt = mvp_comp.get("calibrators", {}).get("platt")
                    if cal_platt is not None:
                        demo_df["calibrated_score"] = cal_platt.predict_proba(demo_df["risk_score"].values.reshape(-1, 1))[:, 1]
                    else:
                        demo_df["calibrated_score"] = np.nan
                else:
                    thresh_dict = thresholds_meta.get(model_selected, {}).get(op_selected, {})
                    thresh_val = float(thresh_dict.get("threshold", 0.5))
                    demo_df["threshold"] = thresh_val
                    demo_df["alert"] = (demo_df["risk_score"] >= thresh_val)
                    demo_df["calibrated_score"] = "N/A"

                # Top-K capacity calculation
                from src.metrics import compute_top_k_quota
                k_quota = compute_top_k_quota(len(demo_df), k_percent=config.TOP_K_PERCENT, method="floor")
                demo_df["batch_rank"] = demo_df["risk_score"].rank(ascending=False, method="max").astype(int)
                demo_df["in_batch_top_10_pct"] = False
                sorted_idx = demo_df["risk_score"].sort_values(ascending=False, kind="stable").index
                if k_quota is not None and k_quota > 0:
                    demo_df.loc[sorted_idx[:k_quota], "in_batch_top_10_pct"] = True

                summary = compute_batch_summary_metrics(demo_df, target_col="Bankrupt?")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("總企業家數", f"{summary['total_records']:,} 家")
                c2.metric("預警標記家數", f"{summary['alert_count']:,} 家 ({format_pct(summary['alert_proportion'])})")
                if summary["target_available"] and summary["total_positives"] is not None:
                    c3.metric("示範實際破產家數", f"{summary['total_positives']} 家")
                    if summary["top_k_enabled"] and summary["recall_at_top_k"] is not None:
                        top_k_hdr = format_top_k_header()
                        c4.metric(f"{top_k_hdr} 召回率", format_pct(summary['recall_at_top_k']))
                    else:
                        c4.metric("召回率", "未達評估門檻")
                else:
                    c3.metric("資料狀態", "無標籤資料")
                    c4.metric("Top K 評估", "未啟用")

                if summary.get("note"):
                    st.caption(f"ℹ️ {summary['note']}")

                display_df = filter_and_sort_batch(demo_df, high_risk_only=high_risk_only, sort_by_risk=sort_by_risk)
                cols_to_show = ["company_id", "risk_score", "batch_rank", "in_batch_top_10_pct", "alert", "calibrated_score", "Bankrupt?"]
                cols_present = [c for c in cols_to_show if c in display_df.columns]
                st.dataframe(display_df[cols_present], width="stretch", hide_index=True)

                csv_bytes = prepare_export_csv_bytes(display_df)
                st.download_button(
                    label="📥 匯出評分結果 (Excel UTF-8-BOM CSV)",
                    data=csv_bytes,
                    file_name=f"demo_scored_{model_selected}_{op_selected}.csv",
                    mime="text/csv",
                )

                st.markdown("---")
                st.markdown("#### 🔍 單一企業特徵歸因分析 (Local Attribution)")
                company_ids = display_df["company_id"].tolist()
                if company_ids:
                    selected_company = st.selectbox("選擇欲深入解析之企業代號", company_ids)
                    case_row = demo_df[demo_df["company_id"] == selected_company].iloc[0]

                    explain_model = "M4" if model_selected in ["E_A", "M4", "M3", "M5"] else "M1"
                    st.caption(f"歸因模型：{explain_model} (mode='demo_oof')")

                    try:
                        explain_res = explain_case(
                            case_row.to_dict(),
                            model_name=explain_model,
                            mode="demo_oof",
                            top_k=5,
                            explainer=explainer,
                        )

                        pos_df = pd.DataFrame(explain_res["top_positive_drivers"])
                        neg_df = pd.DataFrame(explain_res["top_negative_drivers"])

                        col_p, col_n = st.columns(2)
                        with col_p:
                            st.markdown("🔴 **主要風險推升特徵 (Top Risk Drivers)**")
                            if not pos_df.empty:
                                st.dataframe(pos_df[["feature", "attribution", "transformed_value"]], hide_index=True)
                            else:
                                st.caption("無顯著推升特徵")
                        with col_n:
                            st.markdown("🟢 **主要安全保護特徵 (Top Protective Drivers)**")
                            if not neg_df.empty:
                                st.dataframe(neg_df[["feature", "attribution", "transformed_value"]], hide_index=True)
                            else:
                                st.caption("無顯著保護特徵")

                        st.caption(explain_res.get("disclaimer", ""))
                    except Exception as ex:
                        st.error(f"特徵歸因分析失敗：{ex}")

        else:
            st.warning(ui_text.TAB5_UPLOAD_IN_SAMPLE_WARNING)
            uploaded_file = st.file_uploader("請上傳企業財務特徵 CSV 檔案", type=["csv"])
            allow_missing_cb = st.checkbox(ui_text.TAB5_ALLOW_MISSING_LABEL, value=False)

            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                except Exception as ex:
                    st.error(f"讀取 CSV 檔案失敗：{ex}")
                    df_upload = None

                if df_upload is not None:
                    try:
                        with st.spinner("正在執行企業財務風險評分..."):
                            scored_upload = score_mvp_batch(
                                df_upload,
                                model_name=model_selected,
                                operating_point=op_selected,
                                allow_missing=allow_missing_cb,
                                components=mvp_comp,
                            )

                        summary_u = compute_batch_summary_metrics(scored_upload, target_col="Bankrupt?")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("總上傳家數", f"{summary_u['total_records']:,} 家")
                        c2.metric("有效評分家數", f"{summary_u['valid_records']:,} 家")
                        c3.metric("預警標記家數", f"{summary_u['alert_count']:,} 家 ({format_pct(summary_u['alert_proportion'])})")
                        if summary_u["target_available"] and summary_u["total_positives"] is not None:
                            c4.metric("標籤破產家數", f"{summary_u['total_positives']} 家")
                        else:
                            c4.metric("被拒絕筆數 (缺值過高)", f"{summary_u['rejected_records']} 家")

                        if summary_u.get("note"):
                            st.caption(f"ℹ️ {summary_u['note']}")

                        disp_upload = filter_and_sort_batch(scored_upload, high_risk_only=high_risk_only, sort_by_risk=sort_by_risk)
                        st.dataframe(disp_upload, width="stretch", hide_index=True)

                        up_csv_bytes = prepare_export_csv_bytes(disp_upload)
                        st.download_button(
                            label="📥 匯出評分結果 (Excel UTF-8-BOM CSV)",
                            data=up_csv_bytes,
                            file_name=f"scored_{model_selected}_{op_selected}.csv",
                            mime="text/csv",
                        )

                        st.markdown("---")
                        st.markdown("#### 🔍 上傳資料特徵歸因分析 (Local Attribution, mode='final')")
                        row_indices = list(disp_upload.index)
                        if row_indices:
                            sel_idx = st.selectbox("選擇欲解析之樣本列號 (Index)", row_indices)
                            sel_row = disp_upload.loc[sel_idx]

                            if sel_row.get("status") == "rejected_high_missing":
                                st.warning("此樣本因缺值超過允許上限被拒絕評分，無法執行特徵歸因。")
                            else:
                                explain_model = "M4" if model_selected in ["E_A", "M4", "M3", "M5"] else "M1"
                                try:
                                    exp_res_u = explain_case(
                                        df_upload.loc[sel_idx].to_dict(),
                                        model_name=explain_model,
                                        mode="final",
                                        top_k=5,
                                        explainer=explainer,
                                    )
                                    p_df = pd.DataFrame(exp_res_u["top_positive_drivers"])
                                    n_df = pd.DataFrame(exp_res_u["top_negative_drivers"])

                                    col_p, col_n = st.columns(2)
                                    with col_p:
                                        st.markdown("🔴 **主要風險推升特徵 (Top Risk Drivers)**")
                                        if not p_df.empty:
                                            st.dataframe(p_df[["feature", "attribution", "transformed_value"]], hide_index=True)
                                        else:
                                            st.caption("無顯著推升特徵")
                                    with col_n:
                                        st.markdown("🟢 **主要安全保護特徵 (Top Protective Drivers)**")
                                        if not n_df.empty:
                                            st.dataframe(n_df[["feature", "attribution", "transformed_value"]], hide_index=True)
                                        else:
                                            st.caption("無顯著保護特徵")

                                    st.caption(exp_res_u.get("disclaimer", ""))
                                except Exception as e_attr:
                                    st.error(f"特徵歸因失敗：{e_attr}")

                    except Exception as e_score:
                        st.error(f"批次評分發生錯誤：{e_score}")

        st.markdown("---")
        st.caption("📌 **方法學與業務規範免責聲明**")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_SMALL_SAMPLE}")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_R100}")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_MARGIN_NOT_VALIDATED}")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_SHAP_VS_EA}")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_MULTICOLLINEARITY}")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_CALIBRATED_SCORE}")
        st.caption(f"• {ui_text.TAB5_DISCLAIMER_RISK_SCORE_SCALE}")


if __name__ == "__main__":
    main()
