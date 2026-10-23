"""
tests/test_results_summary.py

Verifies that RESULTS_SUMMARY.md and results_summary_sources.csv are correctly
generated from experimental output CSVs, and that >= 15 key experimental metrics
match the source CSV data exactly.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest

from scripts.make_results_summary import generate_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
E6_DIR = PROJECT_ROOT / "outputs" / "e6"
SOURCES_CSV = TABLES_DIR / "results_summary_sources.csv"
SUMMARY_MD = PROJECT_ROOT / "RESULTS_SUMMARY.md"


def test_results_summary_generation():
    """Verify that generate_summary produces valid markdown and sources table."""
    doc, df_sources = generate_summary()
    assert len(doc) > 1000
    assert len(df_sources) >= 15
    assert SOURCES_CSV.exists()
    assert SUMMARY_MD.exists()


def test_sources_table_structure():
    """Verify required schema and non-emptiness of results_summary_sources.csv."""
    assert SOURCES_CSV.exists(), "results_summary_sources.csv must exist"
    df = pd.read_csv(SOURCES_CSV)
    expected_cols = {"section", "description", "value", "source_file", "column", "row_identifier"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 15, f"Expected >= 15 recorded sources, found {len(df)}"
    assert not df["value"].isna().any(), "No recorded value should be NaN"


def test_fifteen_key_metrics_against_sources():
    """
    Directly verify >= 15 key numerical entries from results_summary_sources.csv
    against the raw underlying output files.
    """
    sources_df = pd.read_csv(SOURCES_CSV).set_index("description")

    # 1. Dummy PR-AUC
    e0_df = pd.read_csv(TABLES_DIR / "e0_baselines_summary.csv").set_index("model")
    assert sources_df.loc["Dummy PR-AUC", "value"] == f"{e0_df.loc['Dummy', 'fold_pr_auc_mean']:.4f}"

    # 2. Single Strongest Feature 15-fold PR-AUC mean
    assert sources_df.loc["單特徵 15折PR-AUC均值", "value"] == f"{e0_df.loc['Single Strongest Feature', 'fold_pr_auc_mean']:.4f}"

    # 3. Single Strongest Feature OOF PR-AUC
    assert sources_df.loc["單特徵 OOF PR-AUC", "value"] == f"{e0_df.loc['Single Strongest Feature', 'oof_pr_auc_mean']:.4f}"

    # 4. Base LR 15-fold PR-AUC mean
    assert sources_df.loc["基準LR 15折PR-AUC均值", "value"] == f"{e0_df.loc['Base Logistic Regression (S0)', 'fold_pr_auc_mean']:.4f}"

    # 5. Base LR OOF PR-AUC
    assert sources_df.loc["基準LR OOF PR-AUC", "value"] == f"{e0_df.loc['Base Logistic Regression (S0)', 'oof_pr_auc_mean']:.4f}"

    # 6. EDA Top Univariate AUC (1 - raw_auc)
    uni_df = pd.read_csv(TABLES_DIR / "univariate_auc.csv")
    expected_eda_auc = f"{1.0 - float(uni_df.iloc[0]['raw_auc']):.4f}"
    assert sources_df.loc["EDA最強單特徵全訓練集ROC-AUC", "value"] == expected_eda_auc

    # 7. E2 M3 fold AR@0.95 std
    e2_df = pd.read_csv(TABLES_DIR / "e2a_nested_summary.csv").set_index("model")
    assert sources_df.loc["M3調參後15折AR@0.95標準差", "value"] == f"{e2_df.loc['M3', 'ar95_fold_std']:.2f}x"

    # 8. E3 E_A ECDF fold loss mean
    e3b_df = pd.read_csv(TABLES_DIR / "e3b_ensemble_summary.csv").set_index("candidate")
    assert sources_df.loc["E_A ECDF版15折損失均值", "value"] == f"{e3b_df.loc['E_A', 'loss_fold_mean']:.4f}"

    # 9. E3 E_A Pooled OOF loss
    assert sources_df.loc["E_A Pooled OOF損失", "value"] == f"{e3b_df.loc['E_A', 'loss_pooled_oof']:.4f}"

    # 10. E3 E_A internal rank fold loss mean
    assert sources_df.loc["E_A 折內排名版15折損失均值", "value"] == f"{e3b_df.loc['E_A', 'internal_loss_fold_mean']:.4f}"

    # 11. E3 E_A fold PR-AUC mean
    assert sources_df.loc["E_A 15折PR-AUC均值", "value"] == f"{e3b_df.loc['E_A', 'pr_auc_fold_mean']:.4f}"

    # 12. E3 5-model pairwise Holm p-value all 1.0
    e3a_df = pd.read_csv(TABLES_DIR / "e3a_pairwise_models.csv")
    assert sources_df.loc["5模型兩兩比較Holm p值全數", "value"] == f"{e3a_df['p_value_holm'].min():.1f}"

    # 13. E3 4 ensemble x 5 single models minimum Holm p-value
    e3b_pair_df = pd.read_csv(TABLES_DIR / "e3b_ensemble_vs_single_paired.csv")
    e3b_pair_sub = e3b_pair_df[e3b_pair_df["candidate"] != "Stacking"]
    expected_e3_holm = f"{min(1.0, float(e3b_pair_sub['diff_loss_p_nb'].min()) * len(e3b_pair_sub)):.4f}"
    assert sources_df.loc["4集成x5單模型比較最小Holm p值", "value"] == expected_e3_holm

    # 14. E3d hard positives missed in all 3 repeats
    e3d_df = pd.read_csv(TABLES_DIR / "e3d_hard_positives.csv")
    expected_hard_3 = str(int((e3d_df["missed_by_all_in_n_repeats"] == 3).sum()))
    assert sources_df.loc["3個repeat皆被5模型全漏抓家數", "value"] == expected_hard_3

    # 15. E4 Platt BSS
    cal_df = pd.read_csv(TABLES_DIR / "e4b_calibration_summary.csv")
    platt_row = cal_df[cal_df["method"].str.contains("Platt")].iloc[0]
    assert sources_df.loc["Platt Brier Skill Score", "value"] == f"{float(platt_row['brier_skill_score']):.4f}"

    # 16. E4 Isotonic BSS
    iso_row = cal_df[cal_df["method"].str.contains("Isotonic")].iloc[0]
    assert sources_df.loc["Isotonic Brier Skill Score", "value"] == f"{float(iso_row['brier_skill_score']):.4f}"

    # 17. E5 Compact k=10 loss diff
    e5b_df = pd.read_csv(TABLES_DIR / "e5b_compact_vs_full_paired.csv").set_index("k")
    assert sources_df.loc["k=10損失增加差值", "value"] == f"+{e5b_df.loc[10, 'loss_diff_mean']:.4f}"

    # 18. E5 Compact k=30 loss diff
    assert sources_df.loc["k=30損失增加差值", "value"] == f"+{e5b_df.loc[30, 'loss_diff_mean']:.4f}"

    # 19. E6 E_A R90 test caught
    e6_op_df = pd.read_csv(E6_DIR / "e6_operating_points.csv")
    ea_r90 = e6_op_df[(e6_op_df["group"] == "E_A") & (e6_op_df["operating_point"] == "R90")].iloc[0]
    assert sources_df.loc["E_A R90測試集捕獲數", "value"] == f"{int(ea_r90['positives_caught'])}/{int(ea_r90['total_positives'])}"

    # 20. E6 E_A R95 test caught
    ea_r95 = e6_op_df[(e6_op_df["group"] == "E_A") & (e6_op_df["operating_point"] == "R95")].iloc[0]
    assert sources_df.loc["E_A R95測試集捕獲數", "value"] == f"{int(ea_r95['positives_caught'])}/{int(ea_r95['total_positives'])}"

    # 21. E6 E_A PR-AUC point estimate
    e6_rank_df = pd.read_csv(E6_DIR / "e6_ranking_metrics.csv")
    ea_pr = e6_rank_df[(e6_rank_df["model"] == "E_A") & (e6_rank_df["metric"] == "pr_auc")].iloc[0]
    assert sources_df.loc["E_A測試集PR-AUC點估計", "value"] == f"{ea_pr['point_estimate']:.4f}"

    # 22. E7b Seed stability loss ratio
    e7b_df = pd.read_csv(TABLES_DIR / "e7b_seed_stability.csv")
    e7b_loss = e7b_df[(e7b_df["type"] == "across_seed_comparison") & (e7b_df["metric"] == "loss")].iloc[0]
    assert sources_df.loc["損失種子標準差比折間標準誤", "value"] == f"{e7b_loss['ratio_seed_to_fold_se']:.2%}"


def test_forbidden_words_in_summary():
    """Verify that RESULTS_SUMMARY.md does not contain subjective or banned terms."""
    content = SUMMARY_MD.read_text(encoding="utf-8")
    banned_phrases = [
        "顯著降低負擔",
        "明確優勢",
        "依法規審計標準",
        "法規審計標準",
        "可靠度圖顯示對齊優異",
    ]
    for phrase in banned_phrases:
        assert phrase not in content, f"Found banned phrase '{phrase}' in RESULTS_SUMMARY.md"

    # Verify that market prevalence < 0.5% ~ 1% is annotated as hypothetical assumption
    assert "此為假設值" in content, "Prevalence assumption must be explicitly annotated as hypothetical"
