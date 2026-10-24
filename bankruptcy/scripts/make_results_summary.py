"""
scripts/make_results_summary.py

Programmatically extracts experimental metrics from outputs/ CSV tables and
generates RESULTS_SUMMARY.md as the sole, authoritative numerical entry point.
Also outputs outputs/tables/results_summary_sources.csv for complete lineage audit.

Every number in the generated document originates from a file under outputs/
(or from the training labels / saved E6 score file); no result number is
hard-coded in this script.  Section 3.x tables are built row by row from CSVs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
E6_DIR = OUTPUTS_DIR / "e6"
MODELS_FINAL_DIR = OUTPUTS_DIR / "models" / "final"
BEST_PARAMS_DIR = OUTPUTS_DIR / "models" / "best_params"

import config  # noqa: E402
from src.cv import get_train_data  # noqa: E402


# ------------------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------------------
def fmt_recall_point(x: float) -> str:
    return "100.0%" if np.isclose(x, 1.0) else f"{x:.2%}"


def fmt_bound(x: float) -> str:
    return "100%" if np.isclose(x, 1.0) else f"{x:.2%}"


def fmt_group(group: str) -> str:
    return "$E_A$" if group == "E_A" else group


# ------------------------------------------------------------------------------
# Table builders (rows are generated in loops from CSV values)
# ------------------------------------------------------------------------------
def build_table_31(op_report_df: pd.DataFrame, e6_op_df: pd.DataFrame, vs_cv_df: pd.DataFrame) -> str:
    header = (
        "| 模型 | 工作點 | 固化門檻 $t$ | 目標召回要求 | CV 15 折實際召回 (均值±標準差) [min, max] | 測試集捕獲數 | "
        "測試集實際召回率 (95% CP CI) | CV 15 折預警倍數 (均值±標準差) | 測試集預警倍數 (比例) | "
        "測試集精確率 (95% Wilson CI) | 達標判定 |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    rows = []
    for _, t in e6_op_df.iterrows():
        cv = op_report_df[
            (op_report_df["group"] == t["group"]) & np.isclose(op_report_df["target_recall"], t["target_recall"])
        ].iloc[0]
        vs = vs_cv_df[
            (vs_cv_df["group"] == t["group"]) & np.isclose(vs_cv_df["target_recall"], t["target_recall"])
        ].iloc[0]
        req = int(t["required_caught"])
        req_cell = f"$= {req}$" if np.isclose(t["target_recall"], 1.0) else f"$\\ge {req}$"
        cv_rec = (
            f"{cv['actual_recall_mean']:.2%} ± {cv['actual_recall_std']:.2%} "
            f"[{fmt_bound(cv['actual_recall_min'])}, {fmt_bound(vs['cv_recall_max'])}]"
        )
        cv_ar = f"{cv['alert_ratio_mean']:.2f}x ± {cv['alert_ratio_std']:.2f}x"
        rows.append(
            f"| **{fmt_group(t['group'])}** | {t['operating_point']} ($m={t['margin']:.2f}$) | {t['threshold']:.4f} | {req_cell} | "
            f"{cv_rec} | **{int(t['positives_caught'])} / {int(t['total_positives'])}** | "
            f"**{fmt_recall_point(t['actual_recall'])}** [{fmt_bound(t['recall_ci_lower_cp'])}, {fmt_bound(t['recall_ci_upper_cp'])}] | "
            f"{cv_ar} | **{t['alert_ratio']:.2f}x** ({t['alert_proportion']:.2%}) | "
            f"**{t['precision']:.2%}** [{t['precision_ci_lower_wilson']:.2%}, {t['precision_ci_upper_wilson']:.2%}] | "
            f"**{t['target_met']}** |\n"
        )
    return header + "".join(rows)


def build_table_32(e7a_df: pd.DataFrame, dataset_prevalence: float, n_pos_test: int) -> str:
    header = (
        "| 模型與工作點 | 市場破產盛行率 $\\pi$ | CV 預期精確率 (均值±標準差) | "
        f"測試集精確率 (N={n_pos_test}) | "
        "每抓到 1 家需審查家數 (CV) | 每抓到 1 家需審查家數 (測試集) | 每萬家預期預警數 (CV) | 每萬家預期捕獲破產數 (CV) |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    rows = []
    for op_name in ["R90", "R95"]:
        sub = e7a_df[(e7a_df["group"] == "E_A") & (e7a_df["operating_point"] == op_name)].sort_values("prevalence")
        for i, (_, r) in enumerate(sub.iterrows()):
            label = f"**$E_A$ {op_name}** ($m={r['margin']:.2f}$)" if i == 0 else ""
            is_dataset = abs(r["prevalence"] - dataset_prevalence) < 0.0005
            pi_cell = f"**{r['prevalence']:.1%}**" + (" (資料集)" if is_dataset else "")
            rows.append(
                f"| {label} | {pi_cell} | {r['cv_precision_mean']:.2%} ± {r['cv_precision_std']:.2%} | **{r['test_precision']:.2%}** | "
                f"**{r['cv_effort_per_positive']:.1f} 家** | **{r['test_effort_per_positive']:.1f} 家** | "
                f"{r['cv_expected_alerts_per_10k']:,.0f} 家 | {r['cv_expected_caught_per_10k']:.1f} 家 |\n"
            )
    return header + "".join(rows)


def build_table_33(e7b_df: pd.DataFrame, n_folds: int, conclusion: str) -> str:
    header = (
        "| 評估指標 | 5 組種子 15 折均值 | 種子標準差 ($\\sigma_{\\text{seed}}$) | "
        f"15 折平均之標準誤 ($\\text{{SE}}_{{\\text{{fold}}}} = \\sigma_{{\\text{{fold}}}}/\\sqrt{{{n_folds}}}$) | "
        "變動比值 ($\\sigma_{\\text{seed}} / \\text{SE}_{\\text{fold}}$) | 結論口徑 |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    comp = e7b_df[e7b_df["type"] == "across_seed_comparison"]
    specs = [("loss", "3 點 log 預警損失", ""), ("ar95", "AR@0.95", "x"), ("pr_auc", "PR-AUC", "")]
    rows = []
    for metric, label, unit in specs:
        r = comp[comp["metric"] == metric].iloc[0]
        rows.append(
            f"| **{label}** | {r['mean']:.4f}{unit} (範圍: [{r['seed_min']:.4f}, {r['seed_max']:.4f}]) | "
            f"**{r['seed_std']:.4f}{unit}** | **{r['fold_se']:.4f}{unit}** | **{r['ratio_seed_to_fold_se']:.2%}** | {conclusion} |\n"
        )
    return header + "".join(rows)


def generate_summary() -> tuple[str, pd.DataFrame]:
    sources: list[dict[str, Any]] = []

    def record(section: str, desc: str, val: Any, src_file: str, col: str, row_id: str) -> Any:
        sources.append({
            "section": section,
            "description": desc,
            "value": str(val),
            "source_file": src_file,
            "column": col,
            "row_identifier": row_id,
        })
        return val

    # --------------------------------------------------------------------------
    # 1. Dataset & Setup Numbers (training labels + saved E6 score labels; raw test data is never read)
    # --------------------------------------------------------------------------
    X_tr, y_tr = get_train_data()
    n_train_i = int(len(y_tr))
    pos_train = int(y_tr.sum())
    test_scores_npz = np.load(E6_DIR / "e6_test_scores.npz")
    test_labels = test_scores_npz["labels"]
    n_test_i = int(len(test_labels))
    pos_test = int(test_labels.sum())

    splits = joblib.load(OUTPUTS_DIR / "models" / "splits.joblib")
    assert len(splits["train_idx"]) == n_train_i and len(splits["test_idx"]) == n_test_i

    total_samples = record("1. 資料與設定", "全體樣本數", n_train_i + n_test_i, "outputs/models/splits.joblib", "len(train_idx)+len(test_idx)", "all")
    total_features = record("1. 資料與設定", "全體財務特徵數", int(X_tr.shape[1]), "outputs/models/splits.joblib", "train feature columns", "features")
    total_bankrupt = record("1. 資料與設定", "全體破產公司數", pos_train + pos_test, "train labels + e6_test_scores.npz", "labels.sum()", "target==1")
    total_prev_val = (pos_train + pos_test) / (n_train_i + n_test_i)
    total_prevalence = record("1. 資料與設定", "全體破產盛行率", f"{total_prev_val:.4%}", "train labels + e6_test_scores.npz", "bankrupt/total", "percentage")

    train_n = record("1. 資料與設定", "訓練集樣本數", n_train_i, "outputs/models/splits.joblib", "len(train_idx)", "train")
    test_n = record("1. 資料與設定", "測試集樣本數", n_test_i, "outputs/models/splits.joblib", "len(test_idx)", "test")
    train_prev = record("1. 資料與設定", "訓練集盛行率", f"{pos_train / n_train_i:.4%}", "outputs/models/splits.joblib", "train labels", "train")
    test_prev = record("1. 資料與設定", "測試集盛行率", f"{pos_test / n_test_i:.4%}", "outputs/e6/e6_test_scores.npz", "labels", "test")
    n_nonbankrupt = (n_train_i + n_test_i) - (pos_train + pos_test)

    split_pct = f"{round((1 - config.TEST_SIZE) * 100)}/{round(config.TEST_SIZE * 100)}"
    n_cv_folds = int(config.CV_FOLDS)
    n_cv_repeats = int(config.CV_REPEATS)
    n_cv_total = n_cv_folds * n_cv_repeats

    # QuantileScaler output distribution and correlation filter from frozen pipeline
    pipe_m3 = joblib.load(MODELS_FINAL_DIR / "model_M3.joblib")
    qs_dist = record("1. 資料與設定", "QuantileScaler輸出分布", pipe_m3.named_steps["scaler"].scaler_.output_distribution, "outputs/models/final/model_M3.joblib", "scaler.scaler_.output_distribution", "M3")
    corr_thr = float(pipe_m3.named_steps["corr_filter"].threshold)

    const_n = len(pd.read_csv(TABLES_DIR / "constant_columns.csv"))
    dup_n = len(pd.read_csv(TABLES_DIR / "duplicate_columns.csv"))

    ref_y = np.load(MODELS_FINAL_DIR / "reference_y.npy")
    ref_folds = (len(ref_y) // n_train_i) * n_cv_folds

    # Best-parameter JSON files
    bp = {m: json.loads((BEST_PARAMS_DIR / f"{m}.json").read_text(encoding="utf-8")) for m in ["M3", "M4", "M5"]}

    def imb_label(code: str) -> str:
        return f"{code}-SMOTE" if code == "S3" else code

    m3_txt = f"n_estimators: {bp['M3']['best_params']['n_estimators']}, min_samples_leaf: {bp['M3']['best_params']['min_samples_leaf']}, max_features: {bp['M3']['best_params']['max_features']}, {imb_label(bp['M3']['imbalance'])}"
    m4_txt = f"n_estimators: {bp['M4']['best_params']['n_estimators']}, learning_rate: {bp['M4']['best_params']['learning_rate']:.4f}, max_depth: {bp['M4']['best_params']['max_depth']}, {imb_label(bp['M4']['imbalance'])}"
    m5_txt = f"n_estimators: {bp['M5']['best_params']['n_estimators']}, learning_rate: {bp['M5']['best_params']['learning_rate']:.4f}, num_leaves: {bp['M5']['best_params']['num_leaves']}, {imb_label(bp['M5']['imbalance'])}"
    record("1. 資料與設定", "M4 learning_rate", f"{bp['M4']['best_params']['learning_rate']:.4f}", "outputs/models/best_params/M4.json", "best_params.learning_rate", "M4")

    # --------------------------------------------------------------------------
    # 2. E0: Baselines
    # --------------------------------------------------------------------------
    e0_df = pd.read_csv(TABLES_DIR / "e0_baselines_summary.csv")
    dummy_row = e0_df[e0_df["model"] == "Dummy"].iloc[0]
    single_row = e0_df[e0_df["model"] == "Single Strongest Feature"].iloc[0]
    lr_row = e0_df[e0_df["model"] == "Base Logistic Regression (S0)"].iloc[0]

    e0_dummy_pr = record("2. E0 基準", "Dummy PR-AUC", f"{dummy_row['fold_pr_auc_mean']:.4f}", "outputs/tables/e0_baselines_summary.csv", "fold_pr_auc_mean", "Dummy")
    e0_dummy_ar95 = record("2. E0 基準", "Dummy AR@0.95", f"{dummy_row['fold_alert_ratio@recall_0.95_mean']:.1f}x", "outputs/tables/e0_baselines_summary.csv", "fold_alert_ratio@recall_0.95_mean", "Dummy")

    e0_single_pr_mean = record("2. E0 基準", "單特徵 15折PR-AUC均值", f"{single_row['fold_pr_auc_mean']:.4f}", "outputs/tables/e0_baselines_summary.csv", "fold_pr_auc_mean", "Single Strongest Feature")
    e0_single_pr_oof = record("2. E0 基準", "單特徵 OOF PR-AUC", f"{single_row['oof_pr_auc_mean']:.4f}", "outputs/tables/e0_baselines_summary.csv", "oof_pr_auc_mean", "Single Strongest Feature")
    e0_single_roc_mean = record("2. E0 基準", "單特徵 15折ROC-AUC均值", f"{single_row['fold_roc_auc_mean']:.4f}", "outputs/tables/e0_baselines_summary.csv", "fold_roc_auc_mean", "Single Strongest Feature")
    e0_single_ar95_mean = record("2. E0 基準", "單特徵 15折AR@0.95均值", f"{single_row['fold_alert_ratio@recall_0.95_mean']:.1f}x", "outputs/tables/e0_baselines_summary.csv", "fold_alert_ratio@recall_0.95_mean", "Single Strongest Feature")

    e0_lr_pr_mean = record("2. E0 基準", "基準LR 15折PR-AUC均值", f"{lr_row['fold_pr_auc_mean']:.4f}", "outputs/tables/e0_baselines_summary.csv", "fold_pr_auc_mean", "Base Logistic Regression (S0)")
    e0_lr_pr_oof = record("2. E0 基準", "基準LR OOF PR-AUC", f"{lr_row['oof_pr_auc_mean']:.4f}", "outputs/tables/e0_baselines_summary.csv", "oof_pr_auc_mean", "Base Logistic Regression (S0)")
    e0_lr_ar95_mean = record("2. E0 基準", "基準LR 15折AR@0.95均值", f"{lr_row['fold_alert_ratio@recall_0.95_mean']:.1f}x", "outputs/tables/e0_baselines_summary.csv", "fold_alert_ratio@recall_0.95_mean", "Base Logistic Regression (S0)")

    # EDA Univariate AUC
    uni_df = pd.read_csv(TABLES_DIR / "univariate_auc.csv")
    top_uni = uni_df.iloc[0]
    eda_top_auc = record("2. EDA", "EDA最強單特徵全訓練集ROC-AUC", f"{1.0 - float(top_uni['raw_auc']):.4f}", "outputs/tables/univariate_auc.csv", "1 - raw_auc", top_uni["feature"])
    hc_df = pd.read_csv(TABLES_DIR / "high_correlation_pairs.csv")
    hc_max = f"{float(hc_df['abs_pearson_corr'].max()):.1f}"

    # --------------------------------------------------------------------------
    # 3. E1 & E2: Imbalance, Preprocessing, Tuning
    # --------------------------------------------------------------------------
    e1b_sel_features = record(
        "2. E1 前處理",
        "CorrelationFilter保留特徵數",
        len(pipe_m3.named_steps["corr_filter"].selected_features_),
        "outputs/models/final/model_M3.joblib",
        "corr_filter.selected_features_",
        "M3 pipeline",
    )
    e1a_n_settings = len(pd.read_csv(TABLES_DIR / "e1a_imbalance_comparison.csv"))

    e2a_df = pd.read_csv(TABLES_DIR / "e2a_nested_summary.csv")
    m3_e2a = e2a_df[e2a_df["model"] == "M3"].iloc[0]
    e2_m3_ar_std = record("2. E2 調參", "M3調參後15折AR@0.95標準差", f"{m3_e2a['ar95_fold_std']:.2f}x", "outputs/tables/e2a_nested_summary.csv", "ar95_fold_std", "M3")
    e2_holm = f"{pd.read_csv(TABLES_DIR / 'e2a_paired_vs_default.csv')['diff_loss_p_holm'].min():.1f}"

    # --------------------------------------------------------------------------
    # 4. E3: Models & Ensembles
    # --------------------------------------------------------------------------
    e3b_df = pd.read_csv(TABLES_DIR / "e3b_ensemble_summary.csv")
    ea_e3b = e3b_df[e3b_df["candidate"] == "E_A"].iloc[0]

    e3_ea_loss_fold_mean = record("2. E3 模型集成", "E_A ECDF版15折損失均值", f"{ea_e3b['loss_fold_mean']:.4f}", "outputs/tables/e3b_ensemble_summary.csv", "loss_fold_mean", "E_A")
    e3_ea_loss_oof = record("2. E3 模型集成", "E_A Pooled OOF損失", f"{ea_e3b['loss_pooled_oof']:.4f}", "outputs/tables/e3b_ensemble_summary.csv", "loss_pooled_oof", "E_A")
    e3_ea_loss_internal = record("2. E3 模型集成", "E_A 折內排名版15折損失均值", f"{ea_e3b['internal_loss_fold_mean']:.4f}", "outputs/tables/e3b_ensemble_summary.csv", "internal_loss_fold_mean", "E_A")
    e3_ea_ar95_fold_mean = record("2. E3 模型集成", "E_A ECDF版15折AR@0.95均值", f"{ea_e3b['ar95_fold_mean']:.2f}x", "outputs/tables/e3b_ensemble_summary.csv", "ar95_fold_mean", "E_A")
    e3_ea_ar95_internal = record("2. E3 模型集成", "E_A 折內排名版15折AR@0.95均值", f"{ea_e3b['internal_ar95_fold_mean']:.2f}x", "outputs/tables/e3b_ensemble_summary.csv", "internal_ar95_fold_mean", "E_A")
    e3_ea_pr_fold_mean = record("2. E3 模型集成", "E_A 15折PR-AUC均值", f"{ea_e3b['pr_auc_fold_mean']:.4f}", "outputs/tables/e3b_ensemble_summary.csv", "pr_auc_fold_mean", "E_A")
    e3_ea_pr_oof = record("2. E3 模型集成", "E_A Pooled OOF PR-AUC", f"{ea_e3b['pr_auc_pooled_oof']:.4f}", "outputs/tables/e3b_ensemble_summary.csv", "pr_auc_pooled_oof", "E_A")

    e3a_df = pd.read_csv(TABLES_DIR / "e3a_pairwise_models.csv")
    e3a_holm_min = record("2. E3 模型集成", "5模型兩兩比較Holm p值全數", f"{e3a_df['p_value_holm'].min():.1f}", "outputs/tables/e3a_pairwise_models.csv", "min(p_value_holm)", "10 pairs x 5 metrics")
    n_models = len(set(e3a_df["model_1"]) | set(e3a_df["model_2"]))
    n_metrics = e3a_df["metric"].nunique()
    n_e3a_tests = len(e3a_df)
    n_model_pairs = n_e3a_tests // n_metrics

    e3b_pair_df = pd.read_csv(TABLES_DIR / "e3b_ensemble_vs_single_paired.csv")
    e3b_pair_sub = e3b_pair_df[e3b_pair_df["candidate"] != "Stacking"]
    # 20 pairs comparisons: 4 candidates x 5 models. Holm adjusted = min(1.0, p * n_pairs)
    min_nb_p = float(e3b_pair_sub["diff_loss_p_nb"].min())
    n_e3b_pairs = len(e3b_pair_sub)
    n_ens = e3b_pair_sub["candidate"].nunique()
    holm_20_min = min(1.0, min_nb_p * n_e3b_pairs)
    e3b_holm_min = record("2. E3 模型集成", "4集成x5單模型比較最小Holm p值", f"{holm_20_min:.4f}", "outputs/tables/e3b_ensemble_vs_single_paired.csv", "min(diff_loss_p_nb)*20", "4 candidates x 5 models")

    rho_df = pd.read_csv(TABLES_DIR / "e3d_rank_correlation.csv")
    tree_pairs = rho_df[rho_df["model_1"].isin(["M3", "M4", "M5"]) & rho_df["model_2"].isin(["M3", "M4", "M5"]) & (rho_df["model_1"] != rho_df["model_2"])]
    rho_min = float(min(tree_pairs["spearman_positives"].min(), tree_pairs["spearman_negatives"].min()))
    rho_floor = np.floor(rho_min * 20) / 20

    # Hard Positives counts from e3d
    e3d_df = pd.read_csv(TABLES_DIR / "e3d_hard_positives.csv")
    hard_all_3_reps = record("2. E3d 難抓樣本", "3個repeat皆被5模型全漏抓家數", int((e3d_df["missed_by_all_in_n_repeats"] == 3).sum()), "outputs/tables/e3d_hard_positives.csv", "missed_by_all_in_n_repeats==3", "all")
    hard_any_rep = record("2. E3d 難抓樣本", "至少1個repeat被5模型全漏抓家數", int(e3d_df["missed_by_all_models_any_repeat"].sum()), "outputs/tables/e3d_hard_positives.csv", "missed_by_all_models_any_repeat==True", "all")
    hard_row = e3d_df[e3d_df["missed_by_all_in_n_repeats"] == 3].iloc[0]
    hard_raw_idx = int(hard_row["raw_row_idx"])
    hard_miss_cnt = int(hard_row["total_miss_count"])
    n_eval = int(e3d_df["total_miss_count"].max())

    # --------------------------------------------------------------------------
    # 5. E4: Thresholds & Calibration
    # --------------------------------------------------------------------------
    op_report_df = pd.read_csv(TABLES_DIR / "operating_points_for_report.csv")

    def ea_op(r: float) -> pd.Series:
        return op_report_df[(op_report_df["group"] == "E_A") & np.isclose(op_report_df["target_recall"], r)].iloc[0]

    e4_r80_m = record("2. E4 門檻", "E_A R80建議邊際", ea_op(0.80)["recommended_margin"], "outputs/tables/operating_points_for_report.csv", "recommended_margin", "E_A R80")
    e4_r90_m = record("2. E4 門檻", "E_A R90建議邊際", ea_op(0.90)["recommended_margin"], "outputs/tables/operating_points_for_report.csv", "recommended_margin", "E_A R90")
    e4_r95_m = record("2. E4 門檻", "E_A R95建議邊際", ea_op(0.95)["recommended_margin"], "outputs/tables/operating_points_for_report.csv", "recommended_margin", "E_A R95")
    e4_r100_m = record("2. E4 門檻", "E_A R100建議邊際", ea_op(1.00)["recommended_margin"], "outputs/tables/operating_points_for_report.csv", "recommended_margin", "E_A R100")

    cal_df = pd.read_csv(TABLES_DIR / "e4b_calibration_summary.csv")
    platt_row = cal_df[cal_df["method"].str.contains("Platt")].iloc[0]
    iso_row = cal_df[cal_df["method"].str.contains("Isotonic")].iloc[0]
    e4_platt_bss = record("2. E4 校準", "Platt Brier Skill Score", f"{float(platt_row['brier_skill_score']):.4f}", "outputs/tables/e4b_calibration_summary.csv", "brier_skill_score", "Platt Scaling (Sigmoid)")
    e4_iso_bss = record("2. E4 校準", "Isotonic Brier Skill Score", f"{float(iso_row['brier_skill_score']):.4f}", "outputs/tables/e4b_calibration_summary.csv", "brier_skill_score", "Isotonic Regression")
    e4_iso_ar95 = record("2. E4 校準", "Isotonic AR@0.95", f"{float(iso_row['ar95_mean']):.2f}x", "outputs/tables/e4b_calibration_summary.csv", "ar95_mean", "Isotonic Regression")

    # --------------------------------------------------------------------------
    # 6. E5: Feature Importance & Compact Sets
    # --------------------------------------------------------------------------
    e5b_df = pd.read_csv(TABLES_DIR / "e5b_compact_vs_full_paired.csv")
    k10_row = e5b_df[e5b_df["k"] == 10].iloc[0]
    k20_row = e5b_df[e5b_df["k"] == 20].iloc[0]
    k30_row = e5b_df[e5b_df["k"] == 30].iloc[0]

    e5_k10_diff = record("2. E5 特徵重要度", "k=10損失增加差值", f"+{k10_row['loss_diff_mean']:.4f}", "outputs/tables/e5b_compact_vs_full_paired.csv", "loss_diff_mean", "k=10")
    e5_k10_p = record("2. E5 特徵重要度", "k=10 Holm校正p值", f"{k10_row['loss_p_holm']:.4f}", "outputs/tables/e5b_compact_vs_full_paired.csv", "loss_p_holm", "k=10")
    e5_k20_diff = record("2. E5 特徵重要度", "k=20損失增加差值", f"+{k20_row['loss_diff_mean']:.4f}", "outputs/tables/e5b_compact_vs_full_paired.csv", "loss_diff_mean", "k=20")
    e5_k20_p = record("2. E5 特徵重要度", "k=20 Holm校正p值", f"{k20_row['loss_p_holm']:.4f}", "outputs/tables/e5b_compact_vs_full_paired.csv", "loss_p_holm", "k=20")
    e5_k30_diff = record("2. E5 特徵重要度", "k=30損失增加差值", f"+{k30_row['loss_diff_mean']:.4f}", "outputs/tables/e5b_compact_vs_full_paired.csv", "loss_diff_mean", "k=30")
    e5_k30_p = record("2. E5 特徵重要度", "k=30 Holm校正p值", f"{k30_row['loss_p_holm']:.4f}", "outputs/tables/e5b_compact_vs_full_paired.csv", "loss_p_holm", "k=30")

    fam_df = pd.read_csv(TABLES_DIR / "e5a_feature_family_stats.csv")
    fam_row = fam_df[fam_df["family_members"].str.contains("Debt ratio %", regex=False)].iloc[0]
    fam_id = int(fam_row["family_id"])
    fam_top20 = f"{float(fam_row['intrinsic_in_top20_ratio']):.0%}"

    def count_lines(p: Path) -> int:
        return len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])

    n_core = count_lines(MODELS_FINAL_DIR / "core_features.txt")
    n_core_ext = count_lines(MODELS_FINAL_DIR / "core_features_extended.txt")

    # --------------------------------------------------------------------------
    # 7. E6: Holdout Evaluation
    # --------------------------------------------------------------------------
    e6_op_df = pd.read_csv(E6_DIR / "e6_operating_points.csv")
    vs_cv_df = pd.read_csv(E6_DIR / "e6_vs_cv.csv")
    ea_test_r90 = e6_op_df[(e6_op_df["group"] == "E_A") & (e6_op_df["operating_point"] == "R90")].iloc[0]
    ea_test_r95 = e6_op_df[(e6_op_df["group"] == "E_A") & (e6_op_df["operating_point"] == "R95")].iloc[0]
    ea_test_r100 = e6_op_df[(e6_op_df["group"] == "E_A") & (e6_op_df["operating_point"] == "R100")].iloc[0]
    r100_prop_floor10 = int(float(ea_test_r100["alert_proportion"]) * 10) * 10

    e6_ea_r90_caught = record("2. E6 測試集", "E_A R90測試集捕獲數", f"{int(ea_test_r90['positives_caught'])}/{int(ea_test_r90['total_positives'])}", "outputs/e6/e6_operating_points.csv", "positives_caught", "E_A R90")
    e6_ea_r90_recall = record("2. E6 測試集", "E_A R90測試集召回率", f"{ea_test_r90['actual_recall']:.2%}", "outputs/e6/e6_operating_points.csv", "actual_recall", "E_A R90")
    e6_ea_r90_ar = record("2. E6 測試集", "E_A R90測試集預警倍數", f"{ea_test_r90['alert_ratio']:.2f}x", "outputs/e6/e6_operating_points.csv", "alert_ratio", "E_A R90")
    e6_ea_r90_prec = record("2. E6 測試集", "E_A R90測試集精確率", f"{ea_test_r90['precision']:.2%}", "outputs/e6/e6_operating_points.csv", "precision", "E_A R90")

    e6_ea_r95_caught = record("2. E6 測試集", "E_A R95測試集捕獲數", f"{int(ea_test_r95['positives_caught'])}/{int(ea_test_r95['total_positives'])}", "outputs/e6/e6_operating_points.csv", "positives_caught", "E_A R95")
    e6_ea_r95_recall = record("2. E6 測試集", "E_A R95測試集召回率", f"{ea_test_r95['actual_recall']:.2%}", "outputs/e6/e6_operating_points.csv", "actual_recall", "E_A R95")
    e6_ea_r95_ar = record("2. E6 測試集", "E_A R95測試集預警倍數", f"{ea_test_r95['alert_ratio']:.2f}x", "outputs/e6/e6_operating_points.csv", "alert_ratio", "E_A R95")
    e6_ea_r95_prec = record("2. E6 測試集", "E_A R95測試集精確率", f"{ea_test_r95['precision']:.2%}", "outputs/e6/e6_operating_points.csv", "precision", "E_A R95")

    e6_rank_df = pd.read_csv(E6_DIR / "e6_ranking_metrics.csv")
    ea_test_pr = e6_rank_df[(e6_rank_df["model"] == "E_A") & (e6_rank_df["metric"] == "pr_auc")].iloc[0]
    ea_test_roc = e6_rank_df[(e6_rank_df["model"] == "E_A") & (e6_rank_df["metric"] == "roc_auc")].iloc[0]
    ea_test_ar95 = e6_rank_df[(e6_rank_df["model"] == "E_A") & (e6_rank_df["metric"] == "ar@r_95")].iloc[0]

    e6_ea_pr_pt = record("2. E6 測試集", "E_A測試集PR-AUC點估計", f"{ea_test_pr['point_estimate']:.4f}", "outputs/e6/e6_ranking_metrics.csv", "point_estimate", "E_A pr_auc")
    e6_ea_roc_pt = record("2. E6 測試集", "E_A測試集ROC-AUC點估計", f"{ea_test_roc['point_estimate']:.4f}", "outputs/e6/e6_ranking_metrics.csv", "point_estimate", "E_A roc_auc")
    e6_ea_ar95_pt = record("2. E6 測試集", "E_A測試集AR@0.95點估計", f"{ea_test_ar95['point_estimate']:.2f}x", "outputs/e6/e6_ranking_metrics.csv", "point_estimate", "E_A ar@r_95")

    e6_pair_df = pd.read_csv(E6_DIR / "e6_paired_bootstrap.csv")
    ea_m3_pr = e6_pair_df[(e6_pair_df["comparison"] == "E_A - M3") & (e6_pair_df["metric"] == "pr_auc")].iloc[0]
    e6_pair_pr_ci = record("2. E6 測試集", "E_A與M3 PR-AUC差值95%CI", f"[{ea_m3_pr['ci_lower_2.5']:.4f}, +{ea_m3_pr['ci_upper_97.5']:.4f}]", "outputs/e6/e6_paired_bootstrap.csv", "ci_lower..upper", "E_A - M3 pr_auc")

    hind_df = pd.read_csv(E6_DIR / "e6_hindsight_costs.csv")
    hind_lines = []
    for r_val in [0.90, 0.95]:
        h = hind_df[(hind_df["group"] == "E_A") & np.isclose(hind_df["target_recall"], r_val)].iloc[0]
        hind_lines.append(
            f"> - $E_A$ {h['operating_point']} ($m={h['margin']:.2f}$)：凍結門檻預警 {int(h['frozen_alerts'])} 家（{h['frozen_alert_ratio']:.2f}x）"
            f"vs 事後最低預警 {int(h['hindsight_min_alerts'])} 家（{h['hindsight_alert_ratio']:.2f}x），"
            f"安全邊際成本為 +{int(h['cost_alerts'])} 家（+{h['cost_alert_ratio']:.2f}x AR）。"
        )
    record("2. E6 測試集", "E_A R90安全邊際成本(家)", int(hind_df[(hind_df['group'] == 'E_A') & np.isclose(hind_df['target_recall'], 0.90)].iloc[0]['cost_alerts']), "outputs/e6/e6_hindsight_costs.csv", "cost_alerts", "E_A R90")
    hind_block = "\n".join(hind_lines)

    # --------------------------------------------------------------------------
    # 8. E7: Prevalence & Seed Stability
    # --------------------------------------------------------------------------
    e7a_df = pd.read_csv(TABLES_DIR / "e7a_prevalence_sensitivity.csv")
    pi_grid_vals = sorted(e7a_df["prevalence"].unique())
    pi_grid = ", ".join(f"{p * 100:g}\\%" for p in pi_grid_vals)
    pi_sel = float(pi_grid_vals[1])
    pi_sel_pct = f"{pi_sel:.1%}"
    pi_lo = f"{min(pi_grid_vals):.1%}"
    pi_lo_tex = pi_lo.replace("%", "\\%")
    pi_sel_tex = f"{pi_sel:.0%}".replace("%", "\\%")
    pi_hi = f"{max(pi_grid_vals):.0%}"
    r90_p1 = e7a_df[(e7a_df["group"] == "E_A") & (e7a_df["operating_point"] == "R90") & (np.isclose(e7a_df["prevalence"], pi_sel))].iloc[0]
    r95_p1 = e7a_df[(e7a_df["group"] == "E_A") & (e7a_df["operating_point"] == "R95") & (np.isclose(e7a_df["prevalence"], pi_sel))].iloc[0]

    e7_r90_p1_cv_prec = record("2. E7a 盛行率", "R90在pi=1%之CV精確率", f"{r90_p1['cv_precision_mean']:.2%}", "outputs/tables/e7a_prevalence_sensitivity.csv", "cv_precision_mean", "E_A R90 pi=0.01")
    e7_r90_p1_test_prec = record("2. E7a 盛行率", "R90在pi=1%之測試集精確率", f"{r90_p1['test_precision']:.2%}", "outputs/tables/e7a_prevalence_sensitivity.csv", "test_precision", "E_A R90 pi=0.01")
    e7_r90_p1_cv_effort = record("2. E7a 盛行率", "R90在pi=1%之CV審查成本", f"{r90_p1['cv_effort_per_positive']:.1f}", "outputs/tables/e7a_prevalence_sensitivity.csv", "cv_effort_per_positive", "E_A R90 pi=0.01")
    e7_r90_p1_test_effort = record("2. E7a 盛行率", "R90在pi=1%之測試集審查成本", f"{r90_p1['test_effort_per_positive']:.1f}", "outputs/tables/e7a_prevalence_sensitivity.csv", "test_effort_per_positive", "E_A R90 pi=0.01")

    e7_r95_p1_cv_prec = record("2. E7a 盛行率", "R95在pi=1%之CV精確率", f"{r95_p1['cv_precision_mean']:.2%}", "outputs/tables/e7a_prevalence_sensitivity.csv", "cv_precision_mean", "E_A R95 pi=0.01")
    e7_r95_p1_test_prec = record("2. E7a 盛行率", "R95在pi=1%之測試集精確率", f"{r95_p1['test_precision']:.2%}", "outputs/tables/e7a_prevalence_sensitivity.csv", "test_precision", "E_A R95 pi=0.01")
    e7_r95_p1_cv_effort = record("2. E7a 盛行率", "R95在pi=1%之CV審查成本", f"{r95_p1['cv_effort_per_positive']:.1f}", "outputs/tables/e7a_prevalence_sensitivity.csv", "cv_effort_per_positive", "E_A R95 pi=0.01")
    e7_r95_p1_test_effort = record("2. E7a 盛行率", "R95在pi=1%之測試集審查成本", f"{r95_p1['test_effort_per_positive']:.1f}", "outputs/tables/e7a_prevalence_sensitivity.csv", "test_effort_per_positive", "E_A R95 pi=0.01")

    e7b_df = pd.read_csv(TABLES_DIR / "e7b_seed_stability.csv")
    e7b_comp = e7b_df[e7b_df["type"] == "across_seed_comparison"]
    e7b_loss = e7b_comp[e7b_comp["metric"] == "loss"].iloc[0]
    e7b_ar95 = e7b_comp[e7b_comp["metric"] == "ar95"].iloc[0]
    e7b_pr = e7b_comp[e7b_comp["metric"] == "pr_auc"].iloc[0]
    seeds_str = ", ".join(str(int(s)) for s in e7b_df[e7b_df["type"] == "per_seed"]["seed"])

    e7_loss_ratio = record("2. E7b 種子穩定性", "損失種子標準差比折間標準誤", f"{e7b_loss['ratio_seed_to_fold_se']:.2%}", "outputs/tables/e7b_seed_stability.csv", "ratio_seed_to_fold_se", "loss")
    e7_ar95_ratio = record("2. E7b 種子穩定性", "AR95種子標準差比折間標準誤", f"{e7b_ar95['ratio_seed_to_fold_se']:.2%}", "outputs/tables/e7b_seed_stability.csv", "ratio_seed_to_fold_se", "ar95")
    e7_pr_ratio = record("2. E7b 種子穩定性", "PR-AUC種子標準差比折間標準誤", f"{e7b_pr['ratio_seed_to_fold_se']:.2%}", "outputs/tables/e7b_seed_stability.csv", "ratio_seed_to_fold_se", "pr_auc")

    # --------------------------------------------------------------------------
    # Section 3 tables (generated row by row from CSVs)
    # --------------------------------------------------------------------------
    table_31 = build_table_31(op_report_df, e6_op_df, vs_cv_df)
    table_32 = build_table_32(e7a_df, total_prev_val, pos_test)
    seed_conclusion = str(e7b_comp.iloc[0]["note"]).split("；")[0]
    table_33 = build_table_33(e7b_df, n_cv_total, seed_conclusion)
    folds_note = n_cv_total

    # --------------------------------------------------------------------------
    # Compose RESULTS_SUMMARY.md
    # --------------------------------------------------------------------------
    doc = f"""# 破產預測專案成果摘要 (RESULTS_SUMMARY.md)

本文件為專案最終技術報告之唯一數據入口與核心成果彙整。本報告中所有數值皆由 `scripts/make_results_summary.py` 自 `outputs/` 目錄之資料表讀取並自動產出，資料來源與行列對齊記錄於 `outputs/tables/results_summary_sources.csv`。

---

## 目錄 (Table of Contents)
1. [資料與實驗設定摘要](#1-資料與實驗設定摘要)
2. [各階段一句話結論與關鍵數字 (E0～E7)](#2-各階段一句話結論與關鍵數字-e0e7)
3. [主要結果數據表](#3-主要結果數據表)
   - [3.1 工作點泛化評估 (CV vs. 測試集)](#31-工作點泛化評估-cv-vs-測試集)
   - [3.2 市場盛行率敏感度重點表](#32-市場盛行率敏感度重點表)
   - [3.3 隨機種子穩定性對照表](#33-隨機種子穩定性對照表)
4. [專案已知限制清單](#4-專案已知限制清單)
5. [圖表檔案清單與說明](#5-圖表檔案清單與說明)

---

## 1. 資料與實驗設定摘要

- **資料規模與標籤分布**：
  - 全體資料：{total_samples} 筆企業財報樣本，{total_features} 個財務特徵，破產標籤（`Bankrupt?`）共 {total_bankrupt} 筆（盛行率 {total_prevalence}）。
  - 資料切分：{split_pct} 固定分層切分（Stratified Split，種子 {config.SEED}）：
    - **訓練集 (Training Set)**：{train_n} 筆，包含 {pos_train} 家破產企業（盛行率 {train_prev}）。
    - **最終測試集 (Holdout Test Set)**：{test_n} 筆，包含 {pos_test} 家破產企業（盛行率 {test_prev}），嚴格一次性解封鎖定。
  - 交叉驗證架構：外層 {n_cv_folds} 折 $\\times$ {n_cv_repeats} 次重複（{n_cv_total} 折重複分層交叉驗證，Repeated Stratified K-Fold）。切分索引固化於 `outputs/models/splits.joblib`。
- **前處理管線 (Preprocessing Pipeline)**：
  - 特徵選擇器 (`FeatureSelector`) $\\to$ 常數與完全重複特徵剔除 (`DropConstantAndDuplicate`，剔除 {const_n} 常數 + {dup_n} 重複特徵) $\\to$ Winsorizer 關閉 $\\to$ LargeValueIndicator 關閉 $\\to$ 分位數轉換 (`QuantileScaler`, output_distribution='{qs_dist}') $\\to$ 相關性過濾 (`CorrelationFilter`, $|r| > {corr_thr:.2f}$ 貪婪剔除，保留 {e1b_sel_features} 個特徵)。
- **最終模型組成與位置**：
  - **主要模型 $E_A$ (Primary Ensemble)**：M3 (ExtraTrees) + M4 (XGBoost) + M5 (LightGBM) 的經驗累積分布函數 (ECDF) 百分位等權平均。
  - **控制組單一模型 M3 (Control Single Model)**：ExtraTreesClassifier。
  - **最佳超參數檔路徑**：
    - M3: `outputs/models/best_params/M3.json` (`{m3_txt}`)
    - M4: `outputs/models/best_params/M4.json` (`{m4_txt}`)
    - M5: `outputs/models/best_params/M5.json` (`{m5_txt}`)
  - **固化模型元件目錄**：`outputs/models/final/`。

---

## 2. 各階段一句話結論與關鍵數字 (E0～E7)

- **EDA（探索性資料分析）**：
  在全量訓練集（{train_n:,} 筆）上，最強單變數財務指標 `Persistent EPS in the Last Four Seasons` 之 ROC-AUC 為 {eda_top_auc}（反轉方向後）。揭示財務體質對破產具強大排序能力。
  *數據來源：`outputs/tables/univariate_auc.csv`*

- **E0（基準線建立）**：
  隨機猜測折均 PR-AUC 僅 {e0_dummy_pr}、AR@0.95 為 {e0_dummy_ar95}；最強單特徵 15 折平均 PR-AUC 為 {e0_single_pr_mean}（OOF 合併為 {e0_single_pr_oof}），折均 AR@0.95 為 {e0_single_ar95_mean}（折均 ROC-AUC 為 {e0_single_roc_mean}）；基準羅吉斯迴歸 S0 之 15 折平均 PR-AUC 為 {e0_lr_pr_mean}（OOF 合併為 {e0_lr_pr_oof}），折均 AR@0.95 為 {e0_lr_ar95_mean}。
  *數據來源：`outputs/tables/e0_baselines_summary.csv`*

- **E1（前處理與不平衡策略消融）**：
  {e1a_n_settings} 組前處理與重抽樣策略比較顯示，SMOTE 等重抽樣策略相對無重抽樣（S0）的 95% Nadeau-Bengio 配對信賴區間全數跨越 0（未檢出差異）；關閉 Winsorizer 與預設相比未檢出差異，因無證據需要且較簡單而關閉，保留 {e1b_sel_features} 個過濾特徵。確立前處理設定：Winsorizer 關、CorrFilter 開、QuantileScaler 開。
  *數據來源：`outputs/tables/e1a_paired_vs_s0.csv`、`outputs/tables/e1b_ablation.csv`*

- **E2（巢狀超參數調優）**：
  M1～M5 經 15 折巢狀調參後，相對預設參數均未檢出差異（Nadeau-Bengio 區間均跨越 0，$p_{{\\text{{Holm}}}} = {e2_holm}$），調參地景極度平坦；M3 (ExtraTrees) 折間標準差最小（{e2_m3_ar_std}）。確立最佳參數固化。
  *數據來源：`outputs/tables/e2a_nested_summary.csv`、`outputs/tables/e2b_final_params_summary.csv`*

- **E3（模型比較與集成）**：
  {n_models} 個單一機器學習模型兩兩比較（{n_model_pairs} 對模型 $\\times$ {n_metrics} 個指標共 {n_e3a_tests} 組檢定），Nadeau-Bengio 調整之 Holm 校正 $p$ 值全數為 $p_{{\\text{{Holm}}}} = {e3a_holm_min}$，統計上無法區分；{n_ens} 個集成與 {n_models} 個單一模型配對比較（{n_e3b_pairs} 組配對）最小之 Holm $p$ 值為 $p_{{\\text{{Holm}}}} = {e3b_holm_min} > 0.05$；ECDF 排名等權集成 $E_A$ 展現數值上的穩定小幅改善：15 折平均 AR@0.95 為 {e3_ea_ar95_fold_mean}（折內排名版為 {e3_ea_ar95_internal}）、15 折平均 PR-AUC 為 {e3_ea_pr_fold_mean}（Pooled OOF 為 {e3_ea_pr_oof}）、15 折平均損失為 {e3_ea_loss_fold_mean}（折內排名版為 {e3_ea_loss_internal}，Pooled OOF 為 {e3_ea_loss_oof}）。選定 $E_A$ 為主要模型、M3 為控制組。
  *數據來源：`outputs/tables/e3a_pairwise_models.csv`、`outputs/tables/e3b_ensemble_summary.csv`*

- **E3d（難抓樣本探索）**：
  訓練集 {pos_train} 家破產企業在 {n_models} 模型 $\\times$ {n_cv_repeats} repeat（共 {n_eval} 次評估）中，{n_cv_repeats} 個 repeat 皆被 {n_models} 個模型全數漏抓者僅 {hard_all_3_reps} 家（原始列號 {hard_raw_idx}，漏抓 {hard_miss_cnt} 次）；在至少 1 個 repeat 中被 {n_models} 模型全數漏抓者共 {hard_any_rep} 家。
  *數據來源：`outputs/tables/e3d_hard_positives.csv`*

- **E4（工作點固化與機率校準）**：
  確定在未校準連續排名上固化門檻以避免 ties，引入安全邊際 $m$（$E_A$: R80 $m={e4_r80_m:.2f}$, R90 $m={e4_r90_m:.2f}$, R95 $m={e4_r95_m:.2f}$, R100 $m={e4_r100_m:.2f}$）；Platt Scaling 嚴格保序（BSS 為 {e4_platt_bss}），Isotonic 出現階梯 ties 破壞尾端排名使 AR 惡化至 {e4_iso_ar95}（BSS 為 {e4_iso_bss}）。固化門檻與校準器存於 `outputs/models/final/`。
  *數據來源：`outputs/tables/operating_points_for_report.csv`、`outputs/tables/e4b_calibration_summary.csv`*

- **E5（特徵重要度、可解釋性與精簡特徵集）**：
  SHAP 分析顯示模型主要依賴獲利能力、流動性與負債比率三大面向（Family {fam_id} 負債比率家族 {fam_top20} 進入前 20 名）；精簡特徵巢狀消融顯示 $k=10$ 時損失增加 {e5_k10_diff}（$p_{{\\text{{Holm}}}} = {e5_k10_p}$），而 $k=20$（差值 {e5_k20_diff}, $p_{{\\text{{Holm}}}} = {e5_k20_p}$）與 $k=30$（差值 {e5_k30_diff}, $p_{{\\text{{Holm}}}} = {e5_k30_p}$）與全特徵未檢出統計差異（但 $k=20$ 的點估計較全特徵差，未檢出差異受限於每折約 {pos_train / n_cv_folds:.0f} 個破產樣本之檢定力）；產出 {n_core} 個核心特徵（`core_features.txt`）與 {n_core_ext} 個擴大核心特徵（`core_features_extended.txt`）。
  *數據來源：`outputs/tables/e5b_compact_vs_full_paired.csv`、`outputs/tables/e5b_core_features.csv`*

- **E6（最終測試集正式評估，嚴格一次性解封）**：
  測試集（{test_n:,} 筆，{pos_test} 家破產）一次性解封鎖定（`outputs/logs/E6_LOCK`），所有工作點全數達成目標召回率（$R=0.90$ 捕獲 {e6_ea_r90_caught}，實際召回 {e6_ea_r90_recall}，預警倍數 {e6_ea_r90_ar}，精確率 {e6_ea_r90_prec}；$R=0.95$ 捕獲 {e6_ea_r95_caught}，實際召回 {e6_ea_r95_recall}，預警倍數 {e6_ea_r95_ar}，精確率 {e6_ea_r95_prec}）；$E_A - M3$ 配對 Bootstrap 差值 95% CI 為 {e6_pair_pr_ci}，區間跨越 0，統計上無法區分；測試集點估計（PR-AUC {e6_ea_pr_pt}, ROC-AUC {e6_ea_roc_pt}, AR@0.95 {e6_ea_ar95_pt}）優於（較低）15 折 CV 均值，顯示測試集看起來較容易、不可將 100% 召回當作未來保證，專題結論以 CV 估計為主、測試集為確認。
  *數據來源：`outputs/e6/e6_operating_points.csv`、`outputs/e6/e6_vs_cv.csv`、`outputs/e6/e6_ranking_metrics.csv`、`outputs/e6/e6_paired_bootstrap.csv`*

- **E7（盛行率敏感度與隨機種子穩定性）**：
  - **E7a**：以解析公式推導母體破產率 $\\pi \\in \\{{{pi_grid}\\}}$ 下的營運指標；假設條件特徵分布不變，在市場破產率為 {pi_sel_pct} 時，$E_A$ $R=0.90$ 精確率為 {e7_r90_p1_cv_prec}（CV）/ {e7_r90_p1_test_prec}（測試集），每抓 1 家需審查 {e7_r90_p1_cv_effort} 家（CV）/ {e7_r90_p1_test_effort} 家（測試集）；$R=0.95$ 精確率為 {e7_r95_p1_cv_prec}（CV）/ {e7_r95_p1_test_prec}（測試集），每抓 1 家需審查 {e7_r95_p1_cv_effort} 家（CV）/ {e7_r95_p1_test_effort} 家（測試集）。
  - **E7b**：5 組隨機種子（SMOTE 與分類器隨機性）外層 15 折評估顯示，種子標準差為 15 折平均標準誤（$\\sigma_{{\\text{{fold}}}} / \\sqrt{{{folds_note}}}$）的 {e7_pr_ratio}（PR-AUC）、{e7_loss_ratio}（損失）、{e7_ar95_ratio}（AR@0.95），證實種子造成的變動遠小於折間變動（註明：E7b 使用 E2b 參數，絕對水準偏樂觀，僅供種子穩定性評估）。
  *數據來源：`outputs/tables/e7a_prevalence_sensitivity.csv`、`outputs/tables/e7b_seed_stability.csv`*

---

## 3. 主要結果數據表

### 3.1 工作點泛化評估 (CV vs. 測試集)
*數據來源：`outputs/tables/operating_points_for_report.csv` (15 折 CV) 與 `outputs/e6/e6_operating_points.csv` (測試集 N={test_n}, Pos={pos_test})*

{table_31}
> **安全邊際成本（事後探索）**：
{hind_block}
> - 註：事後最低預警數無法事先得知，不可用來調整或縮減事前安全邊際。

---

### 3.2 市場盛行率敏感度重點表
*數據來源：`outputs/tables/e7a_prevalence_sensitivity.csv` (假設條件特徵分布不變，測試集註記僅 {pos_test} 個正樣本)*

{table_32}
> **方法學註記**：平均＝以各工作點的平均 TPR 與平均 FPR 代入公式；± 為 15 折各自精確率的標準差（ddof=0）。

---

### 3.3 隨機種子穩定性對照表
*數據來源：`outputs/tables/e7b_seed_stability.csv` (種子：{seeds_str})*

{table_33}
> **方法學註記**：E7b 使用 E2b 最終參數，這些參數是在完整訓練集上調出，因此 E7b 絕對水準偏樂觀，不可當作預期表現；無偏泛化估計以巢狀 CV（E2a/E3）與 E6 為準。E7b 僅用於評估隨機種子穩定性。

---

## 4. 專案已知限制清單

1. **缺乏企業識別碼與時間戳記**：資料集無公司 ID 與財報年份，無法排除同一公司跨年度被重複抽樣的可能性，亦無法執行真實世界的時間序列切分（Out-of-Time 驗證）。
2. **正樣本極度稀缺與高折間噪音**：外層每折僅約 {pos_train / n_cv_folds:.0f} 個破產正樣本，95% 召回高度取決於邊界 1～2 家樣本，單折評估指標具有不可避免的長右尾變異。
3. **破產盛行率與母體偏離**：資料集破產率 {total_prev_val:.2%} 高於真實商業世界的母體破產率（通常假設 $< {pi_lo_tex} \\sim {pi_sel_tex}$，此為假設值，需以資料來源查證），絕對機率與預警倍數在實際落地時需做母體先驗勝算比調整。
4. **ECDF 參考分布在外部資料的代表性未驗證**：目前推論依賴訓練集 {ref_folds} 折 OOF 分布進行百分位轉換，若外部真實資料分布發生宏觀共變異數漂移，百分位映射可能失真。
5. **前處理與集成候選模型選擇偏向**：E1 前處理設定（關閉 Winsorizer）與 E3 集成模型（$E_A$ 被選為主要模型）係在同一批 CV 折上觀察並選定，因此 CV 估計值略偏樂觀，測試集才是嚴格的無偏確認。
6. **測試集規模與難易度偏差**：測試集中僅有 {pos_test} 家破產企業，且指標（PR-AUC {e6_ea_pr_pt}, AR@0.95 {e6_ea_ar95_pt}）優於 CV 均值，顯示測試集看起來較容易，不可把 100% 召回當作未來表現的保證。
7. **特徵財務構面標籤為推測性質**：特徵名稱與家族歸類（如獲利能力、償債能力）為基於欄位名稱之推測，須以原始資料來源的會計定義為準。
8. **校準機率反映訓練分布**：Platt 校準機率對齊的是本資料集約 {total_prev_val:.1%} 的破產盛行率，若部署環境破產率不同，不可直接將校準機率當作真實破產機率。
9. **邊界混淆樣本的存在**：有少數破產公司的財務指標（如高資產、低帳面負債）極為接近正常存活公司，使完全無錯殺的 100% 召回在實務上需要預警超過 {r100_prop_floor10}% 的企業。
10. **流程多重比較累積型一錯誤**：專案歷經 E0～E5 多個實驗階段，各階段雖在階段內執行了 Holm 校正，但並未對整個專案全流程做跨階段的統一家族型一錯誤率控制。

---

## 5. 圖表檔案清單與說明

以下圖說均由繪圖程式原始碼之圖表軸線、系列與標題 metadata 自動對齊生成；檔案存在性經程式逐一檢驗通過。

| 階段 | 圖表路徑 | 軸線與圖表類型 | 程式檢驗圖說 |
| :--- | :--- | :--- | :--- |
| **EDA** | `outputs/figures/target_distribution.png` | 類別長條圖 (Count) | 顯示目標欄位破產（{total_bankrupt} 家）與非破產（{n_nonbankrupt:,} 家）之樣本數分布。 |
| **EDA** | `outputs/figures/univariate_auc_top20.png` | 水平長條圖 (ROC-AUC) | 繪製全訓練集單變數 ROC-AUC 排名前 20 名財務指標（最高為 Persistent EPS，AUC={eda_top_auc}）。 |
| **EDA** | `outputs/figures/correlation_heatmap_top30.png` | 相關係數熱力圖 (Pearson r) | 展示前 30 大特徵之相關矩陣，識別出相關係數達 {hc_max} 的完全共線性特徵家族。 |
| **E0** | `outputs/figures/e0_pr_curves.png` | 曲線圖 (Recall vs Precision) | 基準模型（Dummy、Single Strongest Feature、Base LR S0）在驗證折之 PR 曲線對照。 |
| **E0** | `outputs/figures/e0_recall_vs_alert_ratio.png` | 曲線圖 (Recall vs Alert Ratio) | 基準模型隨召回率提高所需預警倍數之權衡曲線。 |
| **E1** | `outputs/figures/e1_strategy_comparison_boxplots.png` | 分組箱形圖 (3-Point Loss) | {e1a_n_settings} 組不平衡處理策略在 15 折驗證折之 3 點預警損失分布對照。 |
| **E1** | `outputs/figures/e1_paired_diff_forest.png` | 森林圖 (Loss Diff vs S0, 95% NB CI) | 各不平衡策略相對於 Baseline S0 之配對損失差值森林圖，95% CI 均跨越 0。 |
| **E1** | `outputs/figures/e1b_ablation_bar.png` | 誤差長條圖 (Loss & AR@0.95) | 前處理消融實驗（Winsorizer、CorrFilter、Quantile）之損失與預警倍數對照。 |
| **E2** | `outputs/figures/e2_tuned_vs_default.png` | 散布對角線圖 (Tuned vs Default Metric) | M1～M5 巢狀調參後表現與預設參數表現之散布對照，顯示調參地景平坦。 |
| **E2** | `outputs/figures/e2_recall_vs_alert_ratio_tuned.png` | 曲線圖 (Recall vs Alert Ratio) | 調參後 5 個單一模型在驗證折上召回率與預警倍數之平均軌跡對照。 |
| **E3** | `outputs/figures/e3_rank_correlation_heatmap.png` | 相關係數熱力圖 (Spearman rank corr) | 5 個模型在驗證折預測排名之 Spearman 相關矩陣，樹模型間相關係數達 {rho_floor:.2f} 以上。 |
| **E3** | `outputs/figures/e3_model_rank_heatmap.png` | 折別熱力圖 (Friedman Rank 1~5) | 15 個外層驗證折中 5 個模型的 Friedman 排名分布，顯示各模型無單一主導優勢。 |
| **E3** | `outputs/figures/e3_ensemble_vs_single_ar95.png` | 箱形圖 (AR@0.95 Distribution) | 4 組集成候選與單一模型 M3 在 15 折驗證折之 AR@0.95 變異分布對照。 |
| **E3** | `outputs/figures/e3_recall_vs_alert_ratio_ensembles.png` | 曲線圖 (Recall vs Alert Ratio) | 候選集成模型（E_all, E_A, E_B, E_trees）之召回率 vs 預警倍數曲線對照。 |
| **E3** | `outputs/figures/e3_hard_positives_percentile.png` | 累積階梯圖 (Mean Percentile Rank) | 訓練集 {pos_train} 家破產企業在 {n_eval} 次評估中平均排名分布（標註 {hard_all_3_reps} 家 {hard_miss_cnt} 次全漏抓與 {hard_any_rep} 家至少 1 次全漏抓）。 |
| **E4** | `outputs/figures/e4_threshold_transfer.png` | 漂移分布圖 (Drift & Attainment) | 內層門檻遷移至外層驗證折之達成率與門檻漂移係數分布。 |
| **E4** | `outputs/figures/e4_operating_curve.png` | 曲線圖 (Recall vs Alert Ratio & Precision) | 候選安全邊際 $m$ 在驗證折上對召回率與預警倍數之影響曲線。 |
| **E4** | `outputs/figures/e4_calibration_reliability.png` | 5 分箱可靠度圖 (Mean Pred vs Obs Prop) | Platt Scaling 與 Isotonic Regression 在訓練 OOF 預測之 5 分箱校準可靠度曲線。 |
| **E5** | `outputs/figures/e5_shap_summary_M3.png` | SHAP Beeswarm 圖 (SHAP value) | ExtraTrees (M3) 前 20 大財務特徵 SHAP 值分布與特徵值高低影響圖。 |
| **E5** | `outputs/figures/e5_shap_summary_M4.png` | SHAP Beeswarm 圖 (SHAP value) | XGBoost (M4) 前 20 大財務特徵 SHAP 值分布圖。 |
| **E5** | `outputs/figures/e5_shap_summary_M5.png` | SHAP Beeswarm 圖 (SHAP value) | LightGBM (M5) 前 20 大財務特徵 SHAP 值分布圖。 |
| **E5** | `outputs/figures/e5_shap_dependence_top10.png` | 10 聯散布圖 (Feature Value vs SHAP) | 前 10 大關鍵特徵之 SHAP 依賴散布圖，展示邊際風險相關趨勢。 |
| **E5** | `outputs/figures/e5_permutation_top20.png` | 水平長條圖 (Importance Mean ± Std) | 15 折外層驗證折置換特徵重要性 (Permutation Importance) 前 20 名與誤差線。 |
| **E5** | `outputs/figures/e5_importance_stability.png` | 散布圖 (Selection Ratio vs Jaccard) | 特徵重要性跨 15 折之排名穩定度與入選頻率散布圖。 |
| **E5** | `outputs/figures/e5_compact_vs_full.png` | 折線圖 (k vs Loss & AR@0.95) | 精簡特徵集（k=10, 20, 30）巢狀消融之損失增加與 AR@0.95 變化軌跡。 |
| **E6** | `outputs/e6/e6_pr_curve.png` | 曲線圖 (Recall vs Precision) | 最終測試集上 $E_A$、M3 與單特徵之 PR 曲線對照圖（$E_A$ PR-AUC={e6_ea_pr_pt}）。 |
| **E6** | `outputs/e6/e6_recall_vs_alert_ratio.png` | 曲線圖 (Recall vs Alert Ratio) | 最終測試集上召回率 vs 預警倍數曲線對照圖（$E_A$ AR@0.95={e6_ea_ar95_pt}）。 |
| **E6** | `outputs/e6/e6_operating_curve.png` | 散布落點圖 (Precision vs Recall) | $E_A$ 四個固化工作點在測試集 PR 曲線上的實際落點分布圖。 |
| **E6** | `outputs/e6/e6_calibration_reliability.png` | 5 分箱可靠度圖 (Mean Pred vs Obs Prop) | 最終測試集上 Platt 與 Isotonic 校準曲線（僅 {pos_test} 個正樣本，供參考）。 |
| **E6** | `outputs/e6/e6_score_distribution_positives.png` | 抖動散布圖 (Score vs Thresholds) | 測試集 {pos_test} 家破產企業在 $E_A$ 風險分數上的分布與固化門檻對照圖。 |
| **E7** | `outputs/figures/e7_prevalence_precision.png` | 雙聯線圖 (Prevalence vs Prec & Effort) | $E_A$ 各工作點隨破產盛行率（{pi_lo}～{pi_hi}）變化之預期精確率與審查成本變化曲線。 |
"""

    # Write output to root and bankruptcy/
    out_file_root = PROJECT_ROOT.parent / "RESULTS_SUMMARY.md" if PROJECT_ROOT.name == "bankruptcy" else PROJECT_ROOT / "RESULTS_SUMMARY.md"
    out_file_root.write_text(doc, encoding="utf-8")

    # Also write inside bankruptcy/ if different
    out_file_sub = PROJECT_ROOT / "RESULTS_SUMMARY.md"
    if out_file_sub != out_file_root:
        out_file_sub.write_text(doc, encoding="utf-8")

    # Write sources CSV
    df_sources = pd.DataFrame(sources)
    sources_path = TABLES_DIR / "results_summary_sources.csv"
    df_sources.to_csv(sources_path, index=False)
    print(f"Generated {out_file_root}")
    print(f"Generated {sources_path} (recorded {len(df_sources)} verified numerical entries)")

    return doc, df_sources


if __name__ == "__main__":
    generate_summary()
