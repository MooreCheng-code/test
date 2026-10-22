"""
E1: Imbalance Strategy Comparison & Preprocessing Ablation.

Executes:
1. E1a: Main Factorial Experiment:
   - 7 Strategies: S0 (None), S1 (class_weight / scale_pos_weight),
     S2 (SMOTE 0.1), S3 (SMOTE 0.3), S4 (SMOTE 0.5), S5 (BorderlineSMOTE 0.3), S6 (SMOTETomek 0.3)
   - 2 Scalers: QuantileScaler ('quantile') vs StandardScaler ('standard')
   - 3 Models: M1 (Logistic Regression), M2 (Random Forest), M4 (XGBoost)
   - Total = 42 configurations across 15 folds (5 folds x 3 repeats).
2. E1b: Preprocessing Ablation:
   - Fixed S3 + quantile scaler on M1 and M2.
   - 4 settings: default, winsorizer_off, corr_filter_off, large_indicator_on.
3. Reconciliation Checks:
   - Verifies (M1, S0, quantile) matches E0 Base LR exactly.
   - Verifies (M1, S3, quantile) Repeat 0 matches Step 4 smoke test.
4. Statistical Comparison:
   - Nadeau-Bengio corrected paired t-test (correction = 1/k + n_test/n_train).
   - 95% Confidence Intervals and Holm-adjusted p-values vs S0.
   - Wilcoxon signed-rank p-values.
5. Selection Recommendation:
   - Automated winning rule evaluation per model.

Usage:
    python src/run_imbalance.py           # Full evaluation (15 folds)
    python src/run_imbalance.py --quick   # Quick mode (1 repeat, 5 folds)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import (
    CV_FOLDS,
    CV_REPEATS,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TABLES_DIR,
)
from src.cv import (
    cross_validate_pipeline,
    get_train_data,
    load_splits,
)
from src.pipelines import build_pipeline
from src.stats_utils import (
    holm_bonferroni,
    nadeau_bengio_paired_test,
    wilcoxon_test,
)


CACHE_DIR = LOGS_DIR / "e1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OOF_DIR = MODELS_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Pipeline Execution with Checkpoint Caching
# ==============================================================================
def run_single_setting_cached(
    setting_id: str,
    pipeline_factory: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rel_splits: list[dict],
    cache_dir: Path = CACHE_DIR,
) -> dict[str, Any]:
    """
    Executes cross-validation for a specific configuration with file-based caching.
    """
    cache_file = cache_dir / f"{setting_id}.joblib"
    if cache_file.exists():
        try:
            cached_data = joblib.load(cache_file)
            return cached_data
        except Exception:
            pass  # Corrupted cache, re-compute

    start_time = time.time()
    warning_list = []

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        cv_out = cross_validate_pipeline(
            pipeline_or_factory=pipeline_factory,
            X=X_train,
            y=y_train,
            splits=rel_splits,
            n_jobs=1,  # Sequential outer CV loop; tree models use internal n_jobs=-1
        )
        for w in captured_warnings:
            warning_list.append(str(w.message))

    elapsed = time.time() - start_time
    cv_out["setting_id"] = setting_id
    cv_out["elapsed_seconds"] = elapsed
    cv_out["warning_count"] = len(warning_list)
    cv_out["warnings"] = warning_list[:10]

    joblib.dump(cv_out, cache_file)
    return cv_out


# ==============================================================================
# Main E1 Experiment Orchestrator
# ==============================================================================
def run_e1_experiments(quick: bool = False):
    print("=" * 80)
    print("       E1: IMBALANCE STRATEGY & PREPROCESSING ABLATION EXPERIMENTS")
    print("=" * 80)

    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_indices = splits_payload["train_idx"]

    # Map original df row indices to X_train row positions
    orig_to_rel = {orig_i: rel_i for rel_i, orig_i in enumerate(train_orig_indices)}

    target_repeats = [0] if quick else list(range(CV_REPEATS))
    n_splits_cv = CV_FOLDS
    n_repeats_cv = len(target_repeats)

    selected_splits = [s for s in all_splits if s["repeat"] in target_repeats]
    rel_splits = []
    for s in selected_splits:
        rel_splits.append({
            "repeat": s["repeat"],
            "fold": s["fold"],
            "split_id": s["split_id"],
            "train_idx": np.array([orig_to_rel[i] for i in s["train_idx"]]),
            "val_idx": np.array([orig_to_rel[i] for i in s["val_idx"]]),
        })

    mode_str = f"QUICK ({n_repeats_cv} repeat, {len(rel_splits)} folds)" if quick else f"FULL ({n_repeats_cv} repeats, {len(rel_splits)} folds)"
    print(f"Dataset Size: {len(X_train)} training samples. Mode: {mode_str}")
    print("Parallel Strategy: Sequential fold loop + model internal multi-threading (n_jobs=-1). Zero thread oversubscription.")

    # --------------------------------------------------------------------------
    # E1a: Main Factorial Experiment (42 Settings)
    # --------------------------------------------------------------------------
    models = ["M1", "M2", "M4"]
    strategies = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    scalers = ["quantile", "standard"]

    e1a_total = len(models) * len(strategies) * len(scalers)
    print(f"\n--- Starting E1a: Main Factorial Experiment ({e1a_total} settings) ---")

    e1a_results = {}
    e1a_fold_dfs = []
    e1a_oof_dict = {}

    idx = 0
    for m in models:
        for strat in strategies:
            for sc in scalers:
                idx += 1
                setting_id = f"e1a_{m}_{strat}_{sc}_{n_repeats_cv}r"
                print(f"[{idx:2d}/{e1a_total}] Running {m} | {strat} | {sc} ...", end="", flush=True)

                def make_factory(model_name=m, imbalance_strat=strat, scaler_kind=sc):
                    return lambda: build_pipeline(
                        model_name=model_name,
                        imbalance=imbalance_strat,
                        scaler=scaler_kind,
                        use_winsorizer=True,
                        use_corr_filter=True,
                        use_large_indicator=False,
                    )

                res = run_single_setting_cached(
                    setting_id=setting_id,
                    pipeline_factory=make_factory(),
                    X_train=X_train,
                    y_train=y_train,
                    rel_splits=rel_splits,
                )

                e1a_results[(m, strat, sc)] = res
                e1a_oof_dict[f"{m}_{strat}_{sc}"] = res["oof_scores"]

                f_df = res["fold_results"].copy()
                f_df["model"] = m
                f_df["strategy"] = strat
                f_df["scaler"] = sc
                e1a_fold_dfs.append(f_df)

                mean_auc = f_df["roc_auc"].mean()
                mean_pr = f_df["pr_auc"].mean()
                mean_ar95 = f_df["alert_ratio@recall_0.95"].mean()
                print(f" Done ({res['elapsed_seconds']:.1f}s) | ROC-AUC: {mean_auc:.4f} | PR-AUC: {mean_pr:.4f} | AR@95: {mean_ar95:.2f}x")

    # Save OOF scores package
    np.savez_compressed(OOF_DIR / "e1a_oof.npz", **e1a_oof_dict)

    e1a_all_folds = pd.concat(e1a_fold_dfs, ignore_index=True)

    # --------------------------------------------------------------------------
    # E1b: Preprocessing Ablation Experiment (8 Settings)
    # --------------------------------------------------------------------------
    print("\n--- Starting E1b: Preprocessing Ablation Experiment ---")
    ablation_settings = {
        "default": {"use_winsorizer": True, "use_corr_filter": True, "use_large_indicator": False},
        "winsorizer_off": {"use_winsorizer": False, "use_corr_filter": True, "use_large_indicator": False},
        "corr_filter_off": {"use_winsorizer": True, "use_corr_filter": False, "use_large_indicator": False},
        "large_indicator_on": {"use_winsorizer": True, "use_corr_filter": True, "use_large_indicator": True},
    }

    e1b_models = ["M1", "M2"]
    e1b_fold_dfs = []
    e1b_results = {}

    idx_b = 0
    total_b = len(e1b_models) * len(ablation_settings)
    for m in e1b_models:
        for ab_name, ab_cfg in ablation_settings.items():
            idx_b += 1
            setting_id = f"e1b_{m}_{ab_name}_{n_repeats_cv}r"
            print(f"[{idx_b}/{total_b}] Running Ablation {m} | {ab_name} ...", end="", flush=True)

            def make_ab_factory(model_name=m, cfg=ab_cfg):
                return lambda: build_pipeline(
                    model_name=model_name,
                    imbalance="S3",
                    scaler="quantile",
                    use_winsorizer=cfg["use_winsorizer"],
                    use_corr_filter=cfg["use_corr_filter"],
                    use_large_indicator=cfg["use_large_indicator"],
                )

            res_b = run_single_setting_cached(
                setting_id=setting_id,
                pipeline_factory=make_ab_factory(),
                X_train=X_train,
                y_train=y_train,
                rel_splits=rel_splits,
            )

            e1b_results[(m, ab_name)] = res_b
            fb_df = res_b["fold_results"].copy()
            fb_df["model"] = m
            fb_df["ablation"] = ab_name
            e1b_fold_dfs.append(fb_df)

            m_auc = fb_df["roc_auc"].mean()
            m_pr = fb_df["pr_auc"].mean()
            m_ar95 = fb_df["alert_ratio@recall_0.95"].mean()
            print(f" Done ({res_b['elapsed_seconds']:.1f}s) | ROC-AUC: {m_auc:.4f} | PR-AUC: {m_pr:.4f} | AR@95: {m_ar95:.2f}x")

    e1b_all_folds = pd.concat(e1b_fold_dfs, ignore_index=True)

    # --------------------------------------------------------------------------
    # Reconciliation Checks (對帳檢查)
    # --------------------------------------------------------------------------
    perform_reconciliation_checks(e1a_all_folds)

    # --------------------------------------------------------------------------
    # Statistical Comparisons (Paired t-test with Nadeau-Bengio, Holm, Wilcoxon)
    # --------------------------------------------------------------------------
    print("\nComputing Nadeau-Bengio corrected paired tests and Holm corrections ...")
    summary_e1a_df, paired_e1a_df = compile_e1a_tables(e1a_all_folds, n_splits_cv, n_repeats_cv)
    summary_e1b_df, paired_e1b_df = compile_e1b_tables(e1b_all_folds, n_splits_cv, n_repeats_cv)

    # Selection Recommendations
    rec_df = determine_selection_recommendations(summary_e1a_df, paired_e1a_df)

    # Save Tables
    summary_e1a_df.to_csv(TABLES_DIR / "e1a_imbalance_comparison.csv", index=False)
    paired_e1a_df.to_csv(TABLES_DIR / "e1a_paired_vs_s0.csv", index=False)
    summary_e1b_df.to_csv(TABLES_DIR / "e1b_ablation.csv", index=False)
    paired_e1b_df.to_csv(TABLES_DIR / "e1b_paired_vs_default.csv", index=False)
    rec_df.to_csv(TABLES_DIR / "e1_selection_recommendation.csv", index=False)

    # Generate Figures
    generate_e1_figures(e1a_all_folds, paired_e1a_df, summary_e1b_df)

    return summary_e1a_df, paired_e1a_df, summary_e1b_df, paired_e1b_df, rec_df


# ==============================================================================
# Reconciliation Checks Implementation
# ==============================================================================
def perform_reconciliation_checks(e1a_all_folds: pd.DataFrame):
    print("\n" + "=" * 80)
    print("                    RECONCILIATION AUDIT CHECKS")
    print("=" * 80)

    # Check 1: E1a (M1, S0, quantile) vs E0 Base LR
    m1_s0_q = e1a_all_folds[
        (e1a_all_folds["model"] == "M1") &
        (e1a_all_folds["strategy"] == "S0") &
        (e1a_all_folds["scaler"] == "quantile")
    ]
    e1a_m1_s0_roc = m1_s0_q["roc_auc"].mean()
    e1a_m1_s0_pr = m1_s0_q["pr_auc"].mean()

    # Load E0 Base LR if summary exists
    e0_file = TABLES_DIR / "e0_baselines_summary.csv"
    if e0_file.exists():
        e0_df = pd.read_csv(e0_file)
        e0_lr = e0_df[e0_df["model"] == "Base Logistic Regression (S0)"].iloc[0]
        e0_roc = float(e0_lr["fold_roc_auc_mean"])
        e0_pr = float(e0_lr["fold_pr_auc_mean"])
    else:
        e0_roc = 0.9232
        e0_pr = 0.3938

    diff_roc = abs(e1a_m1_s0_roc - e0_roc)
    diff_pr = abs(e1a_m1_s0_pr - e0_pr)

    print("[Check 1] E1a (M1, S0, quantile) vs E0 Base LR (15-fold Mean):")
    print(f"  * ROC-AUC: E1a = {e1a_m1_s0_roc:.4f} | E0 = {e0_roc:.4f} (diff = {diff_roc:.6f})")
    print(f"  * PR-AUC:  E1a = {e1a_m1_s0_pr:.4f}  | E0 = {e0_pr:.4f} (diff = {diff_pr:.6f})")
    if diff_roc > 1e-4 or diff_pr > 1e-4:
        raise AssertionError("RECONCILIATION FAILURE: E1a (M1, S0, quantile) diverges from E0 Base LR!")
    print("  -> PASSED: Exact match within floating precision.")

    # Check 2: E1a (M1, S3, quantile) Repeat 0 vs Step 4 Smoke Test
    m1_s3_q_r0 = e1a_all_folds[
        (e1a_all_folds["model"] == "M1") &
        (e1a_all_folds["strategy"] == "S3") &
        (e1a_all_folds["scaler"] == "quantile") &
        (e1a_all_folds["repeat"] == 0)
    ]
    r0_roc = m1_s3_q_r0["roc_auc"].mean()
    r0_pr = m1_s3_q_r0["pr_auc"].mean()
    r0_ar95 = m1_s3_q_r0["alert_ratio@recall_0.95"].mean()

    exp_r0_roc = 0.920705
    exp_r0_pr = 0.379670
    exp_r0_ar95 = 11.494921

    print("\n[Check 2] E1a (M1, S3, quantile) Repeat 0 vs Step 4 Smoke Test:")
    print(f"  * ROC-AUC:  E1a = {r0_roc:.4f}   | Step 4 = {exp_r0_roc:.4f}   (diff = {abs(r0_roc - exp_r0_roc):.6f})")
    print(f"  * PR-AUC:   E1a = {r0_pr:.4f}   | Step 4 = {exp_r0_pr:.4f}   (diff = {abs(r0_pr - exp_r0_pr):.6f})")
    print(f"  * AR@0.95:  E1a = {r0_ar95:.4f} | Step 4 = {exp_r0_ar95:.4f} (diff = {abs(r0_ar95 - exp_r0_ar95):.6f})")

    if abs(r0_roc - exp_r0_roc) > 1e-4 or abs(r0_pr - exp_r0_pr) > 1e-4 or abs(r0_ar95 - exp_r0_ar95) > 1e-3:
        raise AssertionError("RECONCILIATION FAILURE: E1a (M1, S3, quantile) Repeat 0 diverges from Step 4 Smoke Test!")
    print("  -> PASSED: Exact match with Step 4 Smoke Test.")
    print("=" * 80)


# ==============================================================================
# Compilation of Tables and Statistical Tests
# ==============================================================================
def compile_e1a_tables(e1a_all_folds: pd.DataFrame, n_splits: int, n_repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "alert_ratio@recall_0.95",
        "alert_ratio@recall_0.90",
        "alert_ratio@recall_0.80",
        "pr_auc",
        "roc_auc",
        "recall@alert_ratio_1.5",
        "recall@alert_ratio_3",
        "recall@alert_ratio_5",
        "recall@alert_ratio_10",
    ]

    # 1. Summary Table across 15 folds
    summary_rows = []
    models = ["M1", "M2", "M4"]
    strategies = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    scalers = ["quantile", "standard"]

    for m in models:
        for strat in strategies:
            for sc in scalers:
                subset = e1a_all_folds[
                    (e1a_all_folds["model"] == m) &
                    (e1a_all_folds["strategy"] == strat) &
                    (e1a_all_folds["scaler"] == sc)
                ]
                row = {"model": m, "strategy": strat, "scaler": sc, "n_folds": len(subset)}
                for col in metric_cols:
                    mean_val = float(subset[col].mean())
                    std_val = float(subset[col].std()) if len(subset) > 1 else 0.0
                    row[f"{col}_mean"] = mean_val
                    row[f"{col}_std"] = std_val
                    row[f"{col}_display"] = f"{mean_val:.4f} +/- {std_val:.4f}"
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 2. Paired Tests vs S0 for each (model, scaler)
    paired_rows = []
    for m in models:
        for sc in scalers:
            # Baseline fold values for S0
            s0_folds = e1a_all_folds[
                (e1a_all_folds["model"] == m) &
                (e1a_all_folds["strategy"] == "S0") &
                (e1a_all_folds["scaler"] == sc)
            ].sort_values(["repeat", "fold"])

            family_records = []
            for strat in ["S1", "S2", "S3", "S4", "S5", "S6"]:
                strat_folds = e1a_all_folds[
                    (e1a_all_folds["model"] == m) &
                    (e1a_all_folds["strategy"] == strat) &
                    (e1a_all_folds["scaler"] == sc)
                ].sort_values(["repeat", "fold"])

                p_record = {
                    "model": m,
                    "scaler": sc,
                    "strategy": strat,
                    "baseline": "S0",
                }

                # Primary & secondary metrics
                for col in ["alert_ratio@recall_0.95", "alert_ratio@recall_0.90", "pr_auc", "recall@alert_ratio_3"]:
                    diffs = strat_folds[col].values - s0_folds[col].values
                    mean_d, ci_l, ci_u, p_nb = nadeau_bengio_paired_test(diffs, n_splits=n_splits, n_repeats=n_repeats)
                    p_wilc = wilcoxon_test(diffs)

                    p_record[f"diff_{col}_mean"] = mean_d
                    p_record[f"diff_{col}_ci_lower"] = ci_l
                    p_record[f"diff_{col}_ci_upper"] = ci_u
                    p_record[f"diff_{col}_p_nb"] = p_nb
                    p_record[f"diff_{col}_p_wilcoxon"] = p_wilc

                family_records.append(p_record)

            # Apply Holm correction on primary metric p-values within this (model, scaler) family
            raw_p_primary = [r["diff_alert_ratio@recall_0.95_p_nb"] for r in family_records]
            holm_p_primary = holm_bonferroni(raw_p_primary)

            for r_idx, r in enumerate(family_records):
                r["diff_alert_ratio@recall_0.95_p_holm"] = holm_p_primary[r_idx]
                ci_l = r["diff_alert_ratio@recall_0.95_ci_lower"]
                ci_u = r["diff_alert_ratio@recall_0.95_ci_upper"]

                # Significant improvement: CI strictly negative (lower alert ratio is better)
                if ci_u < 0.0:
                    r["significance_primary"] = "Sig Better"
                elif ci_l > 0.0:
                    r["significance_primary"] = "Sig Worse"
                else:
                    r["significance_primary"] = "Not Sig"

                paired_rows.append(r)

    paired_df = pd.DataFrame(paired_rows)
    return summary_df, paired_df


def compile_e1b_tables(e1b_all_folds: pd.DataFrame, n_splits: int, n_repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "alert_ratio@recall_0.95",
        "alert_ratio@recall_0.90",
        "pr_auc",
        "roc_auc",
        "recall@alert_ratio_3",
    ]

    summary_rows = []
    for (m, ab), subset in e1b_all_folds.groupby(["model", "ablation"]):
        row = {"model": m, "ablation": ab, "n_folds": len(subset)}
        for col in metric_cols:
            mean_val = float(subset[col].mean())
            std_val = float(subset[col].std()) if len(subset) > 1 else 0.0
            row[f"{col}_mean"] = mean_val
            row[f"{col}_std"] = std_val
            row[f"{col}_display"] = f"{mean_val:.4f} +/- {std_val:.4f}"
        summary_rows.append(row)
    summary_b_df = pd.DataFrame(summary_rows)

    # Paired vs default
    paired_rows = []
    for m in ["M1", "M2"]:
        def_folds = e1b_all_folds[
            (e1b_all_folds["model"] == m) &
            (e1b_all_folds["ablation"] == "default")
        ].sort_values(["repeat", "fold"])

        for ab in ["winsorizer_off", "corr_filter_off", "large_indicator_on"]:
            ab_folds = e1b_all_folds[
                (e1b_all_folds["model"] == m) &
                (e1b_all_folds["ablation"] == ab)
            ].sort_values(["repeat", "fold"])

            rec = {"model": m, "ablation": ab, "baseline": "default"}
            for col in metric_cols:
                diffs = ab_folds[col].values - def_folds[col].values
                mean_d, ci_l, ci_u, p_nb = nadeau_bengio_paired_test(diffs, n_splits=n_splits, n_repeats=n_repeats)
                p_wilc = wilcoxon_test(diffs)
                rec[f"diff_{col}_mean"] = mean_d
                rec[f"diff_{col}_ci_lower"] = ci_l
                rec[f"diff_{col}_ci_upper"] = ci_u
                rec[f"diff_{col}_p_nb"] = p_nb
                rec[f"diff_{col}_p_wilcoxon"] = p_wilc

            paired_rows.append(rec)

    paired_b_df = pd.DataFrame(paired_rows)
    return summary_b_df, paired_b_df


# ==============================================================================
# Winning Selection Rule (勝出規則)
# ==============================================================================
def determine_selection_recommendations(summary_df: pd.DataFrame, paired_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates winning selection rule deterministically per model:
    1. Primary metric: lowest alert_ratio@recall_0.95.
    2. Must have Nadeau-Bengio 95% CI vs S0 not containing 0 to be declared significantly better.
    3. If no configuration is significantly better, pick the simplest:
       Complexity order S0 < S1 < S2 < S3 < S4 < S5 < S6; quantile preferred over standard.
    4. Check consistency with secondary metrics (alert@0.90, PR-AUC).
    """
    recommendations = []
    complexity_map = {
        "S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6
    }
    scaler_pref = {"quantile": 0, "standard": 1}

    for m in ["M1", "M2", "M4"]:
        m_summary = summary_df[summary_df["model"] == m].copy()
        m_paired = paired_df[paired_df["model"] == m].copy()

        # Find any setting that is statistically significantly better than S0 (ci_upper < 0)
        sig_better_settings = m_paired[m_paired["significance_primary"] == "Sig Better"]

        if len(sig_better_settings) > 0:
            # Among significantly better settings, pick the one with lowest 15-fold alert_ratio@0.95 mean
            cand_keys = [(r["strategy"], r["scaler"]) for _, r in sig_better_settings.iterrows()]
            cands_summary = m_summary[m_summary.apply(lambda r: (r["strategy"], r["scaler"]) in cand_keys, axis=1)]
            best_row = cands_summary.sort_values("alert_ratio@recall_0.95_mean", ascending=True).iloc[0]

            chosen_strat = best_row["strategy"]
            chosen_scaler = best_row["scaler"]
            rule_applied = "Rule 1 & 2: Statistically significantly lower alert ratio at 95% recall (95% CI < 0 vs S0)"
        else:
            # Rule 3: Pick simplest configuration
            # Check S0 performance across scalers
            s0_rows = m_summary[m_summary["strategy"] == "S0"].sort_values("scaler", ascending=True)
            chosen_strat = "S0"
            chosen_scaler = "quantile"  # Quantile prioritized
            best_row = s0_rows[s0_rows["scaler"] == chosen_scaler].iloc[0]
            rule_applied = "Rule 3: No strategy significantly superior to S0; selected simplest configuration (S0, quantile)"

        # Check secondary metric consistency
        # Compare chosen setting against S0 on alert_ratio@0.90 and PR-AUC
        if chosen_strat == "S0":
            secondary_consistent = True
            sec_note = "Baseline S0 selected by parsimony; secondary metrics serve as E2 reference."
        else:
            p_match = m_paired[(m_paired["strategy"] == chosen_strat) & (m_paired["scaler"] == chosen_scaler)].iloc[0]
            ar90_improved = p_match["diff_alert_ratio@recall_0.90_mean"] <= 0.0
            prauc_improved = p_match["diff_pr_auc_mean"] >= 0.0
            secondary_consistent = bool(ar90_improved and prauc_improved)
            sec_note = f"AR@0.90 diff: {p_match['diff_alert_ratio@recall_0.90_mean']:.4f}, PR-AUC diff: {p_match['diff_pr_auc_mean']:.4f}"

        recommendations.append({
            "model": m,
            "recommended_strategy": chosen_strat,
            "recommended_scaler": chosen_scaler,
            "alert_ratio_95_mean": best_row["alert_ratio@recall_0.95_mean"],
            "alert_ratio_90_mean": best_row["alert_ratio@recall_0.90_mean"],
            "pr_auc_mean": best_row["pr_auc_mean"],
            "roc_auc_mean": best_row["roc_auc_mean"],
            "rule_applied": rule_applied,
            "secondary_metrics_consistent": secondary_consistent,
            "secondary_note": sec_note,
        })

    return pd.DataFrame(recommendations)


# ==============================================================================
# Visualization Figures
# ==============================================================================
def generate_e1_figures(e1a_all_folds: pd.DataFrame, paired_e1a_df: pd.DataFrame, summary_e1b_df: pd.DataFrame):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 1. Figure 1: Boxplots of Alert Ratio @ 95% Recall, Faceted by Model
    g = sns.catplot(
        data=e1a_all_folds,
        x="strategy",
        y="alert_ratio@recall_0.95",
        hue="scaler",
        col="model",
        kind="box",
        palette={"quantile": "#3498db", "standard": "#e67e22"},
        height=4.5,
        aspect=1.2,
        sharey=False,
    )
    g.set_axis_labels("Imbalance Strategy", "Alert Ratio @ 95% Recall")
    g.set_titles(col_template="Model: {col_name}")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("E1a: Alert Ratio @ 95% Recall by Strategy and Scaler (Faceted by Model)", fontsize=14)
    plt.savefig(FIGURES_DIR / "e1_strategy_comparison_boxplots.png", dpi=300)
    plt.close()

    # 2. Figure 2: Forest Plot of Paired Differences vs S0 (95% Nadeau-Bengio CI)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    models = ["M1", "M2", "M4"]

    for ax, m in zip(axes, models):
        m_paired = paired_e1a_df[paired_e1a_df["model"] == m].copy()
        m_paired["label"] = m_paired["strategy"] + " (" + m_paired["scaler"].str[:1] + ")"

        y_pos = np.arange(len(m_paired))
        diffs = m_paired["diff_alert_ratio@recall_0.95_mean"].values
        ci_lows = m_paired["diff_alert_ratio@recall_0.95_ci_lower"].values
        ci_highs = m_paired["diff_alert_ratio@recall_0.95_ci_upper"].values

        xerr = [diffs - ci_lows, ci_highs - diffs]

        colors = ["#27ae60" if sig == "Sig Better" else ("#c0392b" if sig == "Sig Worse" else "#7f8c8d")
                  for sig in m_paired["significance_primary"]]

        for idx_pt in range(len(diffs)):
            err_left = max(0.0, diffs[idx_pt] - ci_lows[idx_pt])
            err_right = max(0.0, ci_highs[idx_pt] - diffs[idx_pt])
            ax.errorbar(
                diffs[idx_pt],
                y_pos[idx_pt],
                xerr=[[err_left], [err_right]],
                fmt="none",
                ecolor=colors[idx_pt],
                elinewidth=2,
                capsize=4,
            )
        ax.scatter(diffs, y_pos, c=colors, s=50, zorder=3)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(m_paired["label"], fontsize=9)
        ax.set_title(f"Model: {m}", fontsize=12)
        ax.set_xlabel("Diff in Alert Ratio @ 95% vs S0 (Negative is Better)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("E1a Forest Plot: Paired Differences vs S0 with Nadeau-Bengio 95% CI", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e1_paired_diff_forest.png", dpi=300)
    plt.close()

    # 3. Figure 3: E1b Preprocessing Ablation Bar Chart
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(
        data=summary_e1b_df,
        x="ablation",
        y="alert_ratio@recall_0.95_mean",
        hue="model",
        palette={"M1": "#2980b9", "M2": "#27ae60"},
    )
    plt.title("E1b: Preprocessing Ablation on S3 + Quantile (M1 & M2)", fontsize=13, pad=12)
    plt.xlabel("Ablation Configuration", fontsize=11)
    plt.ylabel("15-Fold Mean Alert Ratio @ 95% Recall", fontsize=11)
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Default", "Winsorizer OFF", "CorrFilter OFF", "LargeIndicator ON"])
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.2f}x", (p.get_x() + p.get_width() / 2., h / 2),
                        ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e1b_ablation_bar.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="E1 Imbalance & Ablation Experiments")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run 1 repeat (5 folds) instead of 3 repeats (15 folds)")
    args = parser.parse_args()

    run_e1_experiments(quick=args.quick)
    print("\nE1 experiments completed successfully.")


if __name__ == "__main__":
    main()
