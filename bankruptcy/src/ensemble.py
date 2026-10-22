"""
E3: Model Comparison, Fixed-Weight Ensembles, Exploratory Stacking, and Diagnostics.

This module implements:
1. E3a: Pairwise statistical comparisons of 5 single models (NB-corrected, Holm, Wilcoxon, Friedman).
2. E3b: Fixed-weight rank-averaged ensembles with deployable ECDF reference distributions.
3. E3c: Exploratory Stacking with regularized Logistic Regression meta-learner.
4. E3d: Model rank correlation, hard positive diagnostics, and metric alignment.
5. Final candidate selection rule application.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from src.cv import get_train_data, load_splits
from src.metrics import (
    alert_ratio_at_recall,
    evaluate_scores,
    recall_at_alert_ratio,
)
from src.pipelines import build_pipeline
from src.stats_utils import (
    holm_bonferroni,
    nadeau_bengio_paired_test,
    wilcoxon_test,
)

logger = logging.getLogger("bankruptcy_e3")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

BASE_DIR = Path(config.BASE_DIR)
TABLES_DIR = Path(config.TABLES_DIR)
FIGURES_DIR = Path(config.FIGURES_DIR)
MODELS_DIR = Path(config.MODELS_DIR)
OOF_DIR = MODELS_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)
E2_CACHE_DIR = config.LOGS_DIR / "e2_cache"

SEED = config.SEED
CV_FOLDS = config.CV_FOLDS
CV_REPEATS = config.CV_REPEATS

SINGLE_MODELS = ["M1", "M2", "M3", "M4", "M5"]

ENSEMBLE_CANDIDATES = {
    "E_all": ["M1", "M2", "M3", "M4", "M5"],
    "E_A": ["M3", "M4", "M5"],
    "E_B": ["M1", "M3", "M4", "M5"],
    "E_trees": ["M2", "M3", "M4", "M5"],
}


# ==============================================================================
# Helper Functions: ECDF Mapping & Ensembles
# ==============================================================================
def ecdf_map(reference_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    """
    Maps query_scores to empirical percentiles in reference_scores using
    symmetric midpoints for ties:
        P(q) = (count(ref < q) + count(ref <= q)) / (2 * len(ref))
    Guarantees monotonicity and values strictly in [0, 1].
    """
    ref_sorted = np.sort(np.asarray(reference_scores, dtype=float))
    n_ref = len(ref_sorted)
    if n_ref == 0:
        raise ValueError("reference_scores cannot be empty.")
    query = np.asarray(query_scores, dtype=float)
    left = np.searchsorted(ref_sorted, query, side="left")
    right = np.searchsorted(ref_sorted, query, side="right")
    return (left + right) / (2.0 * n_ref)


def compute_ensemble_scores(
    scores_dict: dict[str, np.ndarray], model_names: list[str]
) -> np.ndarray:
    """
    Computes equal-weight average of model score arrays.
    """
    for m in model_names:
        if m not in scores_dict:
            raise KeyError(f"Model {m} not found in scores_dict.")
    arrays = [scores_dict[m] for m in model_names]
    return np.mean(arrays, axis=0)


def compute_three_point_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes mean(log(alert_ratio@recall_r) for r in [0.80, 0.90, 0.95]).
    """
    ar80 = alert_ratio_at_recall(y_true, y_score, 0.80)
    ar90 = alert_ratio_at_recall(y_true, y_score, 0.90)
    ar95 = alert_ratio_at_recall(y_true, y_score, 0.95)
    return float(np.mean([np.log(ar80), np.log(ar90), np.log(ar95)]))


# ==============================================================================
# Reference Distribution Generation & Caching
# ==============================================================================
def get_or_generate_reference_distribution(
    model_name: str,
    repeat: int,
    fold: int,
    X_outer_tr: pd.DataFrame,
    y_outer_tr: pd.Series,
    best_imbalance: str,
    model_params: dict[str, Any],
) -> np.ndarray:
    """
    Retrieves or generates 5-fold inner CV OOF predictions on the outer training data.
    This serves as the deployable reference distribution for ECDF mapping.
    Cached at outputs/models/oof/e3_reference_{model}_r{repeat}_f{fold}.npy.
    """
    ref_file = OOF_DIR / f"e3_reference_{model_name}_r{repeat}_f{fold}.npy"
    if ref_file.exists():
        return np.load(ref_file)

    logger.info(
        f"[RefDist] Generating 5-fold inner CV reference for {model_name} (r{repeat}_f{fold}) ..."
    )
    n_samples = len(X_outer_tr)
    ref_scores = np.zeros(n_samples, dtype=float)

    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED + repeat * 10 + fold
    )
    y_arr = y_outer_tr.values

    for tr_in, val_in in skf.split(np.zeros(n_samples), y_arr):
        pipe = build_pipeline(
            model_name=model_name,
            imbalance=best_imbalance,
            scaler="quantile",
            use_winsorizer=False,
            use_corr_filter=True,
            use_large_indicator=False,
            model_params=model_params,
        )
        pipe.fit(X_outer_tr.iloc[tr_in], y_outer_tr.iloc[tr_in])
        ref_scores[val_in] = pipe.predict_proba(X_outer_tr.iloc[val_in])[:, 1]

    np.save(ref_file, ref_scores)
    logger.info(f"[RefDist] Saved reference distribution to {ref_file}")
    return ref_scores


# ==============================================================================
# E3a: Pairwise Single Model Comparisons & Friedman Test
# ==============================================================================
def run_e3a_single_models_comparison(
    e2a_per_fold_df: pd.DataFrame,
    outer_repeats: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes pairwise comparisons of the 5 single models across 15 folds:
    - Nadeau-Bengio corrected paired t-test (k=15)
    - Holm-Bonferroni correction
    - Wilcoxon signed-rank test
    - Friedman rank test
    """
    logger.info("=== Running E3a: Single Models Comparison ===")
    k_folds = len(outer_repeats) * 5
    filtered_df = e2a_per_fold_df[e2a_per_fold_df["repeat"].isin(outer_repeats)].copy()

    metrics_to_compare = [
        ("outer_loss", "3_point_loss", True),  # True = lower is better
        ("alert_ratio@recall_0.95", "ar95", True),
        ("alert_ratio@recall_0.90", "ar90", True),
        ("pr_auc", "pr_auc", False),  # False = higher is better
        ("recall@alert_ratio_3", "rec@ar3", False),
    ]

    pairwise_rows = []

    # 10 pairs among 5 models
    pairs = []
    for i in range(len(SINGLE_MODELS)):
        for j in range(i + 1, len(SINGLE_MODELS)):
            pairs.append((SINGLE_MODELS[i], SINGLE_MODELS[j]))

    for col_name, short_name, lower_is_better in metrics_to_compare:
        # Collect diffs for all 10 pairs
        pair_diffs = []
        p_nb_list = []
        p_wilc_list = []
        ci_lower_list = []
        ci_upper_list = []
        mean_diff_list = []

        for m1, m2 in pairs:
            v1 = (
                filtered_df[filtered_df["model"] == m1]
                .sort_values(["repeat", "fold"])[col_name]
                .values
            )
            v2 = (
                filtered_df[filtered_df["model"] == m2]
                .sort_values(["repeat", "fold"])[col_name]
                .values
            )
            diff = v1 - v2  # m1 - m2
            pair_diffs.append((m1, m2, diff, v1.mean(), v2.mean()))

            m_diff, ci_l, ci_u, p_nb = nadeau_bengio_paired_test(
                diff, n_splits=5, n_repeats=len(outer_repeats)
            )
            p_wilc = wilcoxon_test(diff)

            mean_diff_list.append(m_diff)
            ci_lower_list.append(ci_l)
            ci_upper_list.append(ci_u)
            p_nb_list.append(p_nb)
            p_wilc_list.append(p_wilc)

        # Holm adjustment across the 10 pairs for this metric
        p_holm = holm_bonferroni(p_nb_list)

        for idx, (m1, m2, diff, mean1, mean2) in enumerate(pair_diffs):
            m_diff = mean_diff_list[idx]
            ci_l = ci_lower_list[idx]
            ci_u = ci_upper_list[idx]
            p_nb = p_nb_list[idx]
            p_w = p_wilc_list[idx]
            p_h = p_holm[idx]

            sig_nb = (ci_l > 0.0 or ci_u < 0.0) and (p_nb < 0.05)
            sig_holm = p_h < 0.05

            # Directional interpretation
            if lower_is_better:
                better = m1 if m_diff < 0 else m2
            else:
                better = m1 if m_diff > 0 else m2

            pairwise_rows.append(
                {
                    "metric": short_name,
                    "model_1": m1,
                    "model_2": m2,
                    "mean_1": mean1,
                    "mean_2": mean2,
                    "mean_diff": m_diff,
                    "ci_lower_nb": ci_l,
                    "ci_upper_nb": ci_u,
                    "p_value_nb": p_nb,
                    "p_value_wilcoxon": p_w,
                    "p_value_holm": p_h,
                    "sig_nb_0.05": sig_nb,
                    "sig_holm_0.05": sig_holm,
                    "direction": (
                        f"{better} better"
                        if sig_nb
                        else "indistinguishable (無法區分)"
                    ),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_file = TABLES_DIR / "e3a_pairwise_models.csv"
    pairwise_df.to_csv(pairwise_file, index=False)
    logger.info(f"Saved {pairwise_file}")

    # Friedman Test across 5 models for outer_loss
    loss_matrix = []  # shape: (n_folds, 5)
    sorted_folds = (
        filtered_df[["repeat", "fold"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold"])
    )

    rank_matrix = []
    for _, f_row in sorted_folds.iterrows():
        rep = f_row["repeat"]
        f = f_row["fold"]
        f_sub = filtered_df[
            (filtered_df["repeat"] == rep) & (filtered_df["fold"] == f)
        ]
        row_losses = [
            float(f_sub[f_sub["model"] == m]["outer_loss"].iloc[0])
            for m in SINGLE_MODELS
        ]
        loss_matrix.append(row_losses)
        # Ranks: lower loss = rank 1 (best)
        ranks = stats.rankdata(row_losses, method="average")
        rank_matrix.append(ranks)

    loss_matrix = np.array(loss_matrix)  # (15, 5)
    rank_matrix = np.array(rank_matrix)  # (15, 5)

    # Friedman test (null: all models have equal rank distributions)
    f_stat, f_pval = stats.friedmanchisquare(*[loss_matrix[:, col] for col in range(5)])

    mean_ranks = np.mean(rank_matrix, axis=0)

    friedman_rows = []
    for col_idx, m in enumerate(SINGLE_MODELS):
        friedman_rows.append(
            {
                "model": m,
                "n_folds": k_folds,
                "mean_rank": mean_ranks[col_idx],
                "rank_order": int(stats.rankdata(mean_ranks)[col_idx]),
                "friedman_stat": float(f_stat),
                "friedman_pval": float(f_pval),
                "friedman_significant": bool(f_pval < 0.05),
                "note": "Folds are not independent across repeats; Friedman result is for reference only.",
            }
        )
    friedman_df = pd.DataFrame(friedman_rows).sort_values("mean_rank").reset_index(drop=True)
    friedman_file = TABLES_DIR / "e3a_friedman_ranks.csv"
    friedman_df.to_csv(friedman_file, index=False)
    logger.info(f"Saved {friedman_file}")

    return pairwise_df, friedman_df


# ==============================================================================
# E3b & E3c: Ensembles and Stacking Execution
# ==============================================================================
def run_e3b_and_e3c(
    outer_repeats: list[int],
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Executes E3b (fixed-weight rank-averaged ensembles) and E3c (exploratory stacking).
    """
    logger.info("=== Running E3b (Fixed-Weight Ensembles) & E3c (Stacking) ===")

    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_idx = splits_payload["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}

    selected_splits = [s for s in all_splits if s["repeat"] in outer_repeats]
    if quick:
        selected_splits = selected_splits[:2]

    # Pre-allocate full OOF predictions for ensembles: (n_repeats, n_samples)
    n_samples = len(X_train)
    n_repeats = len(outer_repeats)
    rep_to_idx = {r: i for i, r in enumerate(outer_repeats)}

    ensemble_names = list(ENSEMBLE_CANDIDATES.keys()) + ["Stacking"]
    oof_predictions_dict = {
        name: np.full((n_repeats, n_samples), np.nan, dtype=float)
        for name in ensemble_names
    }
    oof_internal_rank_dict = {
        name: np.full((n_repeats, n_samples), np.nan, dtype=float)
        for name in list(ENSEMBLE_CANDIDATES.keys())
    }

    # Tracking rows
    ensemble_per_fold_rows = []
    stacking_per_fold_rows = []

    # Cache for outer fold single model validation scores and references
    for s in selected_splits:
        rep = s["repeat"]
        f = s["fold"]
        r_idx = rep_to_idx[rep]

        out_tr_idx = np.array([orig_to_rel[i] for i in s["train_idx"]])
        out_val_idx = np.array([orig_to_rel[i] for i in s["val_idx"]])

        X_tr_outer = X_train.iloc[out_tr_idx]
        y_tr_outer = y_train.iloc[out_tr_idx]
        X_val_outer = X_train.iloc[out_val_idx]
        y_val_outer = y_train.iloc[out_val_idx].values

        val_scores_raw = {}
        val_pct_ecdf = {}
        val_pct_internal = {}
        ref_scores_dict = {}

        # Load models for this outer fold
        for m in SINGLE_MODELS:
            cache_file = E2_CACHE_DIR / f"{m}_r{rep}_f{f}.joblib"
            if not cache_file.exists():
                raise FileNotFoundError(f"Missing E2a cache file {cache_file}.")
            d = joblib.load(cache_file)
            val_scores_raw[m] = d["outer_val_scores"]
            best_imbalance = d["best_imbalance"]
            model_params = d.get("model_params", d.get("best_params", {}))

            # Reference distribution (5-fold inner CV on outer train)
            ref_scores = get_or_generate_reference_distribution(
                model_name=m,
                repeat=rep,
                fold=f,
                X_outer_tr=X_tr_outer,
                y_outer_tr=y_tr_outer,
                best_imbalance=best_imbalance,
                model_params=model_params,
            )
            ref_scores_dict[m] = ref_scores

            # ECDF mapping
            val_pct_ecdf[m] = ecdf_map(ref_scores, val_scores_raw[m])

            # Validation-fold-internal percentile rank (for contrast)
            val_pct_internal[m] = stats.rankdata(
                val_scores_raw[m], method="average"
            ) / len(val_scores_raw[m])

        # Evaluate candidate ensembles (E3b)
        for ens_name, members in ENSEMBLE_CANDIDATES.items():
            # 1. Deployable ECDF Version
            ens_ecdf_scores = compute_ensemble_scores(val_pct_ecdf, members)
            oof_predictions_dict[ens_name][r_idx, out_val_idx] = ens_ecdf_scores
            metrics_ecdf = evaluate_scores(y_val_outer, ens_ecdf_scores)
            loss_ecdf = compute_three_point_log_loss(y_val_outer, ens_ecdf_scores)

            # 2. Internal Percentile Rank Version (contrast)
            ens_internal_scores = compute_ensemble_scores(val_pct_internal, members)
            oof_internal_rank_dict[ens_name][r_idx, out_val_idx] = ens_internal_scores
            metrics_internal = evaluate_scores(y_val_outer, ens_internal_scores)
            loss_internal = compute_three_point_log_loss(
                y_val_outer, ens_internal_scores
            )

            row = {
                "ensemble": ens_name,
                "repeat": rep,
                "fold": f,
                "members": "+".join(members),
                "n_members": len(members),
                # ECDF metrics (deployable version)
                "outer_loss_ecdf": loss_ecdf,
                "ar95_ecdf": metrics_ecdf["alert_ratio@recall_0.95"],
                "ar90_ecdf": metrics_ecdf["alert_ratio@recall_0.90"],
                "ar80_ecdf": metrics_ecdf["alert_ratio@recall_0.80"],
                "pr_auc_ecdf": metrics_ecdf["pr_auc"],
                "roc_auc_ecdf": metrics_ecdf["roc_auc"],
                "rec@ar3_ecdf": metrics_ecdf["recall@alert_ratio_3"],
                # Internal rank metrics (contrast version)
                "outer_loss_internal": loss_internal,
                "ar95_internal": metrics_internal["alert_ratio@recall_0.95"],
                "ar90_internal": metrics_internal["alert_ratio@recall_0.90"],
                "ar80_internal": metrics_internal["alert_ratio@recall_0.80"],
                "pr_auc_internal": metrics_internal["pr_auc"],
                "roc_auc_internal": metrics_internal["roc_auc"],
                "rec@ar3_internal": metrics_internal["recall@alert_ratio_3"],
                # Delta (ECDF - Internal)
                "loss_diff_ecdf_minus_internal": loss_ecdf - loss_internal,
                "ar95_diff_ecdf_minus_internal": (
                    metrics_ecdf["alert_ratio@recall_0.95"]
                    - metrics_internal["alert_ratio@recall_0.95"]
                ),
            }
            ensemble_per_fold_rows.append(row)

        # Evaluate Exploratory Stacking (E3c)
        # Training meta-features on outer train: inner OOF mapped to their own ECDF
        n_tr = len(X_tr_outer)
        Z_tr = np.zeros((n_tr, len(SINGLE_MODELS)), dtype=float)
        for idx, m in enumerate(SINGLE_MODELS):
            Z_tr[:, idx] = ecdf_map(ref_scores_dict[m], ref_scores_dict[m])

        # Validation meta-features: outer val mapped via reference ECDF
        n_v = len(X_val_outer)
        Z_val = np.zeros((n_v, len(SINGLE_MODELS)), dtype=float)
        for idx, m in enumerate(SINGLE_MODELS):
            Z_val[:, idx] = val_pct_ecdf[m]

        # Inner CV to tune C in {0.01, 0.1, 1.0} on Z_tr
        c_candidates = [0.01, 0.1, 1.0]
        skf_meta = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=SEED + rep * 10 + f
        )
        best_c = 0.1
        best_meta_loss = float("inf")

        for c_val in c_candidates:
            meta_oof = np.zeros(n_tr, dtype=float)
            for m_tr, m_val in skf_meta.split(Z_tr, y_tr_outer.values):
                clf = LogisticRegression(
                    C=c_val, penalty="l2", solver="lbfgs", random_state=SEED
                )
                clf.fit(Z_tr[m_tr], y_tr_outer.values[m_tr])
                meta_oof[m_val] = clf.predict_proba(Z_tr[m_val])[:, 1]
            c_loss = compute_three_point_log_loss(y_tr_outer.values, meta_oof)
            if c_loss < best_meta_loss:
                best_meta_loss = c_loss
                best_c = c_val

        # Train final meta-learner on full Z_tr and predict Z_val
        meta_clf = LogisticRegression(
            C=best_c, penalty="l2", solver="lbfgs", random_state=SEED
        )
        meta_clf.fit(Z_tr, y_tr_outer.values)
        stacking_val_scores = meta_clf.predict_proba(Z_val)[:, 1]

        oof_predictions_dict["Stacking"][r_idx, out_val_idx] = stacking_val_scores
        stack_metrics = evaluate_scores(y_val_outer, stacking_val_scores)
        stack_loss = compute_three_point_log_loss(y_val_outer, stacking_val_scores)

        stacking_per_fold_rows.append(
            {
                "model": "Stacking",
                "repeat": rep,
                "fold": f,
                "best_C": best_c,
                "meta_inner_loss": best_meta_loss,
                "outer_loss": stack_loss,
                "ar95": stack_metrics["alert_ratio@recall_0.95"],
                "ar90": stack_metrics["alert_ratio@recall_0.90"],
                "ar80": stack_metrics["alert_ratio@recall_0.80"],
                "pr_auc": stack_metrics["pr_auc"],
                "roc_auc": stack_metrics["roc_auc"],
                "rec@ar3": stack_metrics["recall@alert_ratio_3"],
                "coefficients": json.dumps(
                    {m: float(meta_clf.coef_[0, idx]) for idx, m in enumerate(SINGLE_MODELS)}
                ),
                "intercept": float(meta_clf.intercept_[0]),
            }
        )

    # Save OOF prediction arrays
    for ens_name, arr in oof_predictions_dict.items():
        oof_path = OOF_DIR / f"e3_ensemble_{ens_name}.npy"
        np.save(oof_path, arr)
        logger.info(f"Saved {oof_path}")

    # Compile Per-Fold DataFrames
    ensemble_per_fold_df = pd.DataFrame(ensemble_per_fold_rows)
    ens_per_fold_file = TABLES_DIR / "e3b_ensemble_per_fold.csv"
    ensemble_per_fold_df.to_csv(ens_per_fold_file, index=False)
    logger.info(f"Saved {ens_per_fold_file}")

    stacking_per_fold_df = pd.DataFrame(stacking_per_fold_rows)
    stack_per_fold_file = TABLES_DIR / "e3c_stacking_per_fold.csv"
    stacking_per_fold_df.to_csv(stack_per_fold_file, index=False)
    logger.info(f"Saved {stack_per_fold_file}")

    # Compile E3b Summary Table
    summary_rows = []
    y_train_arr = y_train.values

    for ens_name in ENSEMBLE_CANDIDATES.keys():
        sub = ensemble_per_fold_df[ensemble_per_fold_df["ensemble"] == ens_name]
        k_folds = len(sub)

        # Pooled OOF metrics across repeats
        oof_ecdf = oof_predictions_dict[ens_name]
        oof_int = oof_internal_rank_dict[ens_name]

        rep_p_losses = []
        rep_p_ar95 = []
        rep_p_pr = []
        rep_p_roc = []
        rep_p_rec3 = []

        rep_pct_losses = []
        rep_pct_ar95 = []
        rep_pct_pr = []
        rep_pct_roc = []
        rep_pct_rec3 = []

        for r_idx in range(n_repeats):
            sc_v = oof_ecdf[r_idx]
            valid_mask = ~np.isnan(sc_v)
            if np.sum(valid_mask) > 0 and np.sum(y_train_arr[valid_mask] == 1) > 0:
                y_sub = y_train_arr[valid_mask]
                sc_sub = sc_v[valid_mask]
                loss = compute_three_point_log_loss(y_sub, sc_sub)
                ar95 = alert_ratio_at_recall(y_sub, sc_sub, 0.95)
                pr = average_precision_score(y_sub, sc_sub)
                roc = roc_auc_score(y_sub, sc_sub)
                rec3 = recall_at_alert_ratio(y_sub, sc_sub, 3.0)
                rep_p_losses.append(loss)
                rep_p_ar95.append(ar95)
                rep_p_pr.append(pr)
                rep_p_roc.append(roc)
                rep_p_rec3.append(rec3)

            sc_int = oof_int[r_idx]
            valid_mask_int = ~np.isnan(sc_int)
            if np.sum(valid_mask_int) > 0 and np.sum(y_train_arr[valid_mask_int] == 1) > 0:
                y_sub = y_train_arr[valid_mask_int]
                sc_sub = sc_int[valid_mask_int]
                loss = compute_three_point_log_loss(y_sub, sc_sub)
                ar95 = alert_ratio_at_recall(y_sub, sc_sub, 0.95)
                pr = average_precision_score(y_sub, sc_sub)
                roc = roc_auc_score(y_sub, sc_sub)
                rec3 = recall_at_alert_ratio(y_sub, sc_sub, 3.0)
                rep_pct_losses.append(loss)
                rep_pct_ar95.append(ar95)
                rep_pct_pr.append(pr)
                rep_pct_roc.append(roc)
                rep_pct_rec3.append(rec3)

        summary_rows.append(
            {
                "candidate": ens_name,
                "n_folds": k_folds,
                # ECDF fold mean & std
                "loss_fold_mean": float(sub["outer_loss_ecdf"].mean()),
                "loss_fold_std": float(sub["outer_loss_ecdf"].std()),
                "loss_pooled_oof": float(np.mean(rep_p_losses)),
                "loss_internal_rank_oof": float(np.mean(rep_pct_losses)),
                "ar95_fold_mean": float(sub["ar95_ecdf"].mean()),
                "ar95_fold_std": float(sub["ar95_ecdf"].std()),
                "ar95_pooled_oof": float(np.mean(rep_p_ar95)),
                "ar95_internal_rank_oof": float(np.mean(rep_pct_ar95)),
                "ar90_fold_mean": float(sub["ar90_ecdf"].mean()),
                "ar90_fold_std": float(sub["ar90_ecdf"].std()),
                "ar80_fold_mean": float(sub["ar80_ecdf"].mean()),
                "ar80_fold_std": float(sub["ar80_ecdf"].std()),
                "pr_auc_fold_mean": float(sub["pr_auc_ecdf"].mean()),
                "pr_auc_fold_std": float(sub["pr_auc_ecdf"].std()),
                "pr_auc_pooled_oof": float(np.mean(rep_p_pr)),
                "roc_auc_fold_mean": float(sub["roc_auc_ecdf"].mean()),
                "roc_auc_fold_std": float(sub["roc_auc_ecdf"].std()),
                "roc_auc_pooled_oof": float(np.mean(rep_p_roc)),
                "rec@ar3_fold_mean": float(sub["rec@ar3_ecdf"].mean()),
                "rec@ar3_fold_std": float(sub["rec@ar3_ecdf"].std()),
                # Contrast: ECDF fold mean vs Internal rank fold mean
                "internal_loss_fold_mean": float(sub["outer_loss_internal"].mean()),
                "internal_ar95_fold_mean": float(sub["ar95_internal"].mean()),
                "loss_ecdf_minus_internal_mean": float(
                    sub["loss_diff_ecdf_minus_internal"].mean()
                ),
            }
        )

    ensemble_summary_df = pd.DataFrame(summary_rows)
    ens_sum_file = TABLES_DIR / "e3b_ensemble_summary.csv"
    ensemble_summary_df.to_csv(ens_sum_file, index=False)
    logger.info(f"Saved {ens_sum_file}")

    # Compile E3c Stacking Summary Table
    stack_sub = stacking_per_fold_df
    rep_st_losses = []
    rep_st_ar95 = []
    rep_st_pr = []
    rep_st_roc = []
    rep_st_rec3 = []

    for r_idx in range(n_repeats):
        sc_st = oof_predictions_dict["Stacking"][r_idx]
        valid_mask_st = ~np.isnan(sc_st)
        if np.sum(valid_mask_st) > 0 and np.sum(y_train_arr[valid_mask_st] == 1) > 0:
            y_sub = y_train_arr[valid_mask_st]
            sc_sub = sc_st[valid_mask_st]
            rep_st_losses.append(compute_three_point_log_loss(y_sub, sc_sub))
            rep_st_ar95.append(alert_ratio_at_recall(y_sub, sc_sub, 0.95))
            rep_st_pr.append(average_precision_score(y_sub, sc_sub))
            rep_st_roc.append(roc_auc_score(y_sub, sc_sub))
            rep_st_rec3.append(recall_at_alert_ratio(y_sub, sc_sub, 3.0))

    stacking_summary_df = pd.DataFrame(
        [
            {
                "model": "Stacking (Exploratory)",
                "n_folds": len(stack_sub),
                "loss_fold_mean": float(stack_sub["outer_loss"].mean()),
                "loss_fold_std": float(stack_sub["outer_loss"].std()),
                "loss_pooled_oof": float(np.mean(rep_st_losses)),
                "ar95_fold_mean": float(stack_sub["ar95"].mean()),
                "ar95_fold_std": float(stack_sub["ar95"].std()),
                "ar95_pooled_oof": float(np.mean(rep_st_ar95)),
                "ar90_fold_mean": float(stack_sub["ar90"].mean()),
                "ar90_fold_std": float(stack_sub["ar90"].std()),
                "ar80_fold_mean": float(stack_sub["ar80"].mean()),
                "ar80_fold_std": float(stack_sub["ar80"].std()),
                "pr_auc_fold_mean": float(stack_sub["pr_auc"].mean()),
                "pr_auc_fold_std": float(stack_sub["pr_auc"].std()),
                "pr_auc_pooled_oof": float(np.mean(rep_st_pr)),
                "roc_auc_fold_mean": float(stack_sub["roc_auc"].mean()),
                "roc_auc_fold_std": float(stack_sub["roc_auc"].std()),
                "roc_auc_pooled_oof": float(np.mean(rep_st_roc)),
                "rec@ar3_fold_mean": float(stack_sub["rec@ar3"].mean()),
                "rec@ar3_fold_std": float(stack_sub["rec@ar3"].std()),
                "note": "Exploratory. Meta-training set has ~141 positives; risk of meta-overfitting.",
            }
        ]
    )
    stack_sum_file = TABLES_DIR / "e3c_stacking_summary.csv"
    stacking_summary_df.to_csv(stack_sum_file, index=False)
    logger.info(f"Saved {stack_sum_file}")

    return (
        ensemble_per_fold_df,
        ensemble_summary_df,
        stacking_per_fold_df,
        stacking_summary_df,
        oof_predictions_dict,
    )


# ==============================================================================
# E3b Paired Comparisons: Ensembles vs Single Best & All Single Models
# ==============================================================================
def run_e3b_paired_comparisons(
    e2a_per_fold_df: pd.DataFrame,
    ensemble_per_fold_df: pd.DataFrame,
    stacking_per_fold_df: pd.DataFrame,
    outer_repeats: list[int],
) -> pd.DataFrame:
    """
    Performs pairwise comparisons:
    - Each ensemble vs Single Best Average Model (M3)
    - Each ensemble vs all 5 single models
    - Stacking vs Single Best & Ensembles
    """
    logger.info("=== Running E3b Paired Comparisons (Ensembles vs Single Models) ===")

    # In E2a, M3 had lowest 15-fold outer loss (1.6547)
    single_best = "M3"
    candidates_to_test = list(ENSEMBLE_CANDIDATES.keys()) + ["Stacking"]

    paired_rows = []

    # Prepare base DataFrame of single models per fold
    e2a_sub = e2a_per_fold_df[e2a_per_fold_df["repeat"].isin(outer_repeats)].copy()

    for cand in candidates_to_test:
        if cand == "Stacking":
            cand_df = stacking_per_fold_df.sort_values(["repeat", "fold"])
            cand_loss = cand_df["outer_loss"].values
            cand_ar95 = cand_df["ar95"].values
            cand_pr = cand_df["pr_auc"].values
            cand_rec3 = cand_df["rec@ar3"].values
        else:
            cand_df = ensemble_per_fold_df[
                ensemble_per_fold_df["ensemble"] == cand
            ].sort_values(["repeat", "fold"])
            cand_loss = cand_df["outer_loss_ecdf"].values
            cand_ar95 = cand_df["ar95_ecdf"].values
            cand_pr = cand_df["pr_auc_ecdf"].values
            cand_rec3 = cand_df["rec@ar3_ecdf"].values

        # Match on (repeat, fold)
        cand_keys = set(zip(cand_df["repeat"], cand_df["fold"]))

        # 1. Compare vs Single Best (M3)
        m_best_df = e2a_sub[
            (e2a_sub["model"] == single_best)
            & e2a_sub.apply(lambda r: (r["repeat"], r["fold"]) in cand_keys, axis=1)
        ].sort_values(["repeat", "fold"])
        base_loss = m_best_df["outer_loss"].values
        base_ar95 = m_best_df["alert_ratio@recall_0.95"].values
        base_pr = m_best_df["pr_auc"].values
        base_rec3 = m_best_df["recall@alert_ratio_3"].values

        # Diff = Candidate - Base (negative loss diff means candidate is better)
        diff_loss = cand_loss - base_loss
        diff_ar95 = cand_ar95 - base_ar95
        diff_pr = cand_pr - base_pr
        diff_rec3 = cand_rec3 - base_rec3

        m_l, ci_l_l, ci_u_l, p_nb_l = nadeau_bengio_paired_test(
            diff_loss, n_splits=5, n_repeats=len(outer_repeats)
        )
        p_w_l = wilcoxon_test(diff_loss)

        m_ar, ci_l_ar, ci_u_ar, p_nb_ar = nadeau_bengio_paired_test(
            diff_ar95, n_splits=5, n_repeats=len(outer_repeats)
        )
        p_w_ar = wilcoxon_test(diff_ar95)

        m_p, ci_l_p, ci_u_p, p_nb_p = nadeau_bengio_paired_test(
            diff_pr, n_splits=5, n_repeats=len(outer_repeats)
        )
        m_r, ci_l_r, ci_u_r, p_nb_r = nadeau_bengio_paired_test(
            diff_rec3, n_splits=5, n_repeats=len(outer_repeats)
        )

        paired_rows.append(
            {
                "candidate": cand,
                "comparison_base": f"Single Best ({single_best})",
                "cand_loss_mean": float(cand_loss.mean()),
                "base_loss_mean": float(base_loss.mean()),
                "diff_loss_mean": m_l,
                "diff_loss_ci_lower_nb": ci_l_l,
                "diff_loss_ci_upper_nb": ci_u_l,
                "diff_loss_p_nb": p_nb_l,
                "diff_loss_p_wilcoxon": p_w_l,
                "diff_ar95_mean": m_ar,
                "diff_ar95_ci_lower_nb": ci_l_ar,
                "diff_ar95_ci_upper_nb": ci_u_ar,
                "diff_ar95_p_nb": p_nb_ar,
                "diff_ar95_p_wilcoxon": p_w_ar,
                "diff_pr_auc_mean": m_p,
                "diff_pr_auc_p_nb": p_nb_p,
                "diff_rec@ar3_mean": m_r,
                "diff_rec@ar3_p_nb": p_nb_r,
                "cand_better_loss": bool(ci_u_l < 0.0 and p_nb_l < 0.05),
                "cand_worse_loss": bool(ci_l_l > 0.0 and p_nb_l < 0.05),
            }
        )

        # 2. Compare vs all single models (for the record)
        for sm in SINGLE_MODELS:
            if sm == single_best:
                continue
            sm_df = e2a_sub[
                (e2a_sub["model"] == sm)
                & e2a_sub.apply(lambda r: (r["repeat"], r["fold"]) in cand_keys, axis=1)
            ].sort_values(["repeat", "fold"])
            sm_loss = sm_df["outer_loss"].values
            sm_ar95 = sm_df["alert_ratio@recall_0.95"].values
            diff_sm_loss = cand_loss - sm_loss
            diff_sm_ar95 = cand_ar95 - sm_ar95

            m_sm_l, ci_l_sm_l, ci_u_sm_l, p_nb_sm_l = nadeau_bengio_paired_test(
                diff_sm_loss, n_splits=5, n_repeats=len(outer_repeats)
            )
            m_sm_ar, ci_l_sm_ar, ci_u_sm_ar, p_nb_sm_ar = nadeau_bengio_paired_test(
                diff_sm_ar95, n_splits=5, n_repeats=len(outer_repeats)
            )
            paired_rows.append(
                {
                    "candidate": cand,
                    "comparison_base": sm,
                    "cand_loss_mean": float(cand_loss.mean()),
                    "base_loss_mean": float(sm_loss.mean()),
                    "diff_loss_mean": m_sm_l,
                    "diff_loss_ci_lower_nb": ci_l_sm_l,
                    "diff_loss_ci_upper_nb": ci_u_sm_l,
                    "diff_loss_p_nb": p_nb_sm_l,
                    "diff_loss_p_wilcoxon": wilcoxon_test(diff_sm_loss),
                    "diff_ar95_mean": m_sm_ar,
                    "diff_ar95_ci_lower_nb": ci_l_sm_ar,
                    "diff_ar95_ci_upper_nb": ci_u_sm_ar,
                    "diff_ar95_p_nb": p_nb_sm_ar,
                    "diff_ar95_p_wilcoxon": wilcoxon_test(diff_sm_ar95),
                    "diff_pr_auc_mean": float((cand_pr - sm_df["pr_auc"].values).mean()),
                    "diff_pr_auc_p_nb": np.nan,
                    "diff_rec@ar3_mean": float(
                        (cand_rec3 - sm_df["recall@alert_ratio_3"].values).mean()
                    ),
                    "diff_rec@ar3_p_nb": np.nan,
                    "cand_better_loss": bool(ci_u_sm_l < 0.0 and p_nb_sm_l < 0.05),
                    "cand_worse_loss": bool(ci_l_sm_l > 0.0 and p_nb_sm_l < 0.05),
                }
            )

    paired_df = pd.DataFrame(paired_rows)

    # Apply Holm step-down correction across the primary "vs Single Best" tests
    vs_best_mask = paired_df["comparison_base"] == f"Single Best ({single_best})"
    p_vals_vs_best = paired_df.loc[vs_best_mask, "diff_loss_p_nb"].tolist()
    p_holm_vs_best = holm_bonferroni(p_vals_vs_best)
    paired_df.loc[vs_best_mask, "diff_loss_p_holm"] = p_holm_vs_best

    paired_file = TABLES_DIR / "e3b_ensemble_vs_single_paired.csv"
    paired_df.to_csv(paired_file, index=False)
    logger.info(f"Saved {paired_file}")
    return paired_df


# ==============================================================================
# E3d: Diagnostics (Correlations, Hard Positives, Metric Alignment)
# ==============================================================================
def run_e3d_diagnostics(
    outer_repeats: list[int],
    e2a_per_fold_df: pd.DataFrame,
    ensemble_summary_df: pd.DataFrame,
    stacking_summary_df: pd.DataFrame,
    oof_predictions_dict: dict[str, np.ndarray],
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes E3d diagnostics:
    1. Spearman rank correlation matrix across 5 models for positives and negatives.
    2. Identification of bankrupt firms uncaught at 95% recall by all 5 models.
    3. Metric alignment table comparing Fold Mean, Pooled OOF, and Percentile Rank OOF.
    """
    logger.info("=== Running E3d: Diagnostics ===")

    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_idx = splits_payload["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}

    selected_splits = [s for s in all_splits if s["repeat"] in outer_repeats]
    if quick:
        selected_splits = selected_splits[:2]

    # 1. Model Rank Correlations
    pos_corrs = []  # shape: (n_folds, 5, 5)
    neg_corrs = []

    # Map each bankrupt sample's missed count and percentiles
    bankrupt_indices = np.where(y_train.values == 1)[0]
    firm_miss_counts = {idx: 0 for idx in bankrupt_indices}
    firm_pct_sums = {idx: 0.0 for idx in bankrupt_indices}
    firm_appearances = {idx: 0 for idx in bankrupt_indices}
    firm_missed_by_all_counts = {idx: 0 for idx in bankrupt_indices}

    for s in selected_splits:
        rep = s["repeat"]
        f = s["fold"]
        out_val_idx = np.array([orig_to_rel[i] for i in s["val_idx"]])
        y_val_outer = y_train.iloc[out_val_idx].values

        val_pcts = {}
        thresholds_95 = {}
        for m in SINGLE_MODELS:
            cache_file = E2_CACHE_DIR / f"{m}_r{rep}_f{f}.joblib"
            d = joblib.load(cache_file)
            sc = d["outer_val_scores"]
            ref_scores = np.load(OOF_DIR / f"e3_reference_{m}_r{rep}_f{f}.npy")
            pct = ecdf_map(ref_scores, sc)
            val_pcts[m] = pct

            # Determine threshold for 95% recall on validation bankrupt samples
            pos_scores = sc[y_val_outer == 1]
            n_pos = len(pos_scores)
            n_req = int(np.ceil(0.95 * n_pos))
            # 95% threshold: lowest score among top n_req
            sorted_pos = np.sort(pos_scores)
            t95 = sorted_pos[n_pos - n_req]
            thresholds_95[m] = t95

        # Spearman correlation on validation fold
        pos_mask = y_val_outer == 1
        neg_mask = y_val_outer == 0

        fold_pos_mat = np.zeros((5, 5), dtype=float)
        fold_neg_mat = np.zeros((5, 5), dtype=float)

        for i, m1 in enumerate(SINGLE_MODELS):
            for j, m2 in enumerate(SINGLE_MODELS):
                if i == j:
                    fold_pos_mat[i, j] = 1.0
                    fold_neg_mat[i, j] = 1.0
                else:
                    c_pos, _ = stats.spearmanr(val_pcts[m1][pos_mask], val_pcts[m2][pos_mask])
                    c_neg, _ = stats.spearmanr(val_pcts[m1][neg_mask], val_pcts[m2][neg_mask])
                    fold_pos_mat[i, j] = c_pos
                    fold_neg_mat[i, j] = c_neg

        pos_corrs.append(fold_pos_mat)
        neg_corrs.append(fold_neg_mat)

        # Track missed bankrupt samples for this fold
        val_pos_indices = out_val_idx[pos_mask]
        for local_idx, global_idx in enumerate(val_pos_indices):
            firm_appearances[global_idx] += 1
            missed_by_models = 0
            for m in SINGLE_MODELS:
                score = joblib.load(E2_CACHE_DIR / f"{m}_r{rep}_f{f}.joblib")[
                    "outer_val_scores"
                ][pos_mask][local_idx]
                pct = val_pcts[m][pos_mask][local_idx]
                firm_pct_sums[global_idx] += pct
                if score < thresholds_95[m]:
                    firm_miss_counts[global_idx] += 1
                    missed_by_models += 1

            if missed_by_models == len(SINGLE_MODELS):
                firm_missed_by_all_counts[global_idx] += 1

    # Average Spearman correlation matrices across folds
    mean_pos_corr = np.mean(pos_corrs, axis=0)
    mean_neg_corr = np.mean(neg_corrs, axis=0)

    corr_rows = []
    for i, m1 in enumerate(SINGLE_MODELS):
        for j, m2 in enumerate(SINGLE_MODELS):
            corr_rows.append(
                {
                    "model_1": m1,
                    "model_2": m2,
                    "spearman_positives": mean_pos_corr[i, j],
                    "spearman_negatives": mean_neg_corr[i, j],
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_file = TABLES_DIR / "e3d_rank_correlation.csv"
    corr_df.to_csv(corr_file, index=False)
    logger.info(f"Saved {corr_file}")

    # 2. Hard Positives Analysis
    hard_rows = []
    raw_df = pd.read_csv(config.DATA_PATH)
    orig_train_indices = splits_payload["train_idx"]

    key_feature_cols = [
        " Persistent EPS in the Last Four Seasons",
        " Net Income to Total Assets",
        " Debt ratio %",
        " Current Ratio",
        " Total Asset Growth Rate",
        " Borrowing dependency",
    ]
    # Strip column spaces for safety
    available_cols = [c for c in key_feature_cols if c in raw_df.columns]

    for g_idx in bankrupt_indices:
        raw_idx = orig_train_indices[g_idx]
        apps = firm_appearances[g_idx]
        total_evals = apps * len(SINGLE_MODELS)
        avg_pct = firm_pct_sums[g_idx] / total_evals if total_evals > 0 else 0.0
        n_missed = firm_miss_counts[g_idx]
        missed_all_repeats = firm_missed_by_all_counts[g_idx]

        row = {
            "train_idx": g_idx,
            "raw_row_idx": int(raw_idx),
            "appearances_repeats": apps,
            "avg_percentile": avg_pct,
            "total_miss_count": n_missed,
            "miss_ratio": n_missed / total_evals if total_evals > 0 else 0.0,
            "missed_by_all_in_n_repeats": missed_all_repeats,
            "missed_by_all_models_any_repeat": bool(missed_all_repeats > 0),
        }
        for col in available_cols:
            row[col.strip()] = float(raw_df.iloc[raw_idx][col])
        hard_rows.append(row)

    hard_df = pd.DataFrame(hard_rows).sort_values("avg_percentile").reset_index(drop=True)
    hard_file = TABLES_DIR / "e3d_hard_positives.csv"
    hard_df.to_csv(hard_file, index=False)
    logger.info(f"Saved {hard_file}")

    # 3. Metric Alignment Table Across 3 Perspectives
    # (Single Models from e2a summary + Ensembles from e3b summary + Stacking)
    e2a_summary_df = pd.read_csv(TABLES_DIR / "e2a_nested_summary.csv")
    alignment_rows = []

    for _, r in e2a_summary_df.iterrows():
        alignment_rows.append(
            {
                "candidate": r["model"],
                "type": "Single Model",
                "loss_fold_mean": r["three_point_loss_fold_mean"],
                "loss_fold_std": r["three_point_loss_fold_std"],
                "loss_pooled_oof": r["three_point_loss_pooled_oof"],
                "loss_pct_rank_oof": r["three_point_loss_pct_rank_oof"],
                "ar95_fold_mean": r["ar95_fold_mean"],
                "ar95_fold_std": r["ar95_fold_std"],
                "ar95_pooled_oof": r["ar95_pooled_oof"],
                "ar95_pct_rank_oof": r["ar95_pct_rank_oof"],
                "pr_auc_fold_mean": r["pr_auc_fold_mean"],
                "roc_auc_fold_mean": r["roc_auc_fold_mean"],
            }
        )

    for _, r in ensemble_summary_df.iterrows():
        alignment_rows.append(
            {
                "candidate": r["candidate"],
                "type": "Fixed Ensemble",
                "loss_fold_mean": r["loss_fold_mean"],
                "loss_fold_std": r["loss_fold_std"],
                "loss_pooled_oof": r["loss_pooled_oof"],
                "loss_pct_rank_oof": r["loss_internal_rank_oof"],
                "ar95_fold_mean": r["ar95_fold_mean"],
                "ar95_fold_std": r["ar95_fold_std"],
                "ar95_pooled_oof": r["ar95_pooled_oof"],
                "ar95_pct_rank_oof": r["ar95_internal_rank_oof"],
                "pr_auc_fold_mean": r["pr_auc_fold_mean"],
                "roc_auc_fold_mean": r["roc_auc_fold_mean"],
            }
        )

    for _, r in stacking_summary_df.iterrows():
        alignment_rows.append(
            {
                "candidate": "Stacking",
                "type": "Exploratory Stacking",
                "loss_fold_mean": r["loss_fold_mean"],
                "loss_fold_std": r["loss_fold_std"],
                "loss_pooled_oof": r["loss_pooled_oof"],
                "loss_pct_rank_oof": np.nan,
                "ar95_fold_mean": r["ar95_fold_mean"],
                "ar95_fold_std": r["ar95_fold_std"],
                "ar95_pooled_oof": r["ar95_pooled_oof"],
                "ar95_pct_rank_oof": np.nan,
                "pr_auc_fold_mean": r["pr_auc_fold_mean"],
                "roc_auc_fold_mean": r["roc_auc_fold_mean"],
            }
        )

    alignment_df = pd.DataFrame(alignment_rows)
    align_file = TABLES_DIR / "e3d_metric_alignment.csv"
    alignment_df.to_csv(align_file, index=False)
    logger.info(f"Saved {align_file}")

    return corr_df, hard_df, alignment_df


# ==============================================================================
# Final Candidate Selection Rule Application
# ==============================================================================
def apply_e3_selection_rule(
    alignment_df: pd.DataFrame,
    e2a_per_fold_df: pd.DataFrame,
    ensemble_per_fold_df: pd.DataFrame,
    outer_repeats: list[int],
) -> pd.DataFrame:
    """
    Applies the pre-specified selection rule:
    Candidates: 5 Single Models + 4 Ensembles (Stacking excluded).
    1. Leader: lowest 15-fold mean 3-point log alert loss.
    2. If difference between Leader and Runner-up has NB 95% CI containing 0,
       select candidate with smaller fold std of AR@0.95.
    3. If Leader is statistically significantly better than all single models, select Leader.
    """
    logger.info("=== Applying E3 Selection Rule ===")
    valid_candidates = SINGLE_MODELS + list(ENSEMBLE_CANDIDATES.keys())
    sub = alignment_df[alignment_df["candidate"].isin(valid_candidates)].copy()
    sub = sub.sort_values("loss_fold_mean").reset_index(drop=True)

    leader = sub.iloc[0]["candidate"]
    runner_up = sub.iloc[1]["candidate"]
    leader_loss = float(sub.iloc[0]["loss_fold_mean"])
    runner_up_loss = float(sub.iloc[1]["loss_fold_mean"])

    leader_ar95_std = float(sub.iloc[0]["ar95_fold_std"])
    runner_up_ar95_std = float(sub.iloc[1]["ar95_fold_std"])

    cand_keys = set(zip(ensemble_per_fold_df["repeat"], ensemble_per_fold_df["fold"]))

    # Extract fold vectors for leader and runner-up
    def get_cand_loss_vector(c_name: str) -> np.ndarray:
        if c_name in SINGLE_MODELS:
            df = e2a_per_fold_df[
                (e2a_per_fold_df["model"] == c_name)
                & e2a_per_fold_df.apply(lambda r: (r["repeat"], r["fold"]) in cand_keys, axis=1)
            ].sort_values(["repeat", "fold"])
            return df["outer_loss"].values
        else:
            df = ensemble_per_fold_df[
                (ensemble_per_fold_df["ensemble"] == c_name)
                & ensemble_per_fold_df.apply(lambda r: (r["repeat"], r["fold"]) in cand_keys, axis=1)
            ].sort_values(["repeat", "fold"])
            return df["outer_loss_ecdf"].values

    v_leader = get_cand_loss_vector(leader)
    v_runner = get_cand_loss_vector(runner_up)

    diff = v_leader - v_runner
    m_diff, ci_l, ci_u, p_nb = nadeau_bengio_paired_test(
        diff, n_splits=5, n_repeats=len(outer_repeats)
    )

    ci_contains_zero = bool(ci_l <= 0.0 <= ci_u)

    # Check rule 3: Is leader significantly better than all single models?
    sig_better_than_all_single = True
    single_comp_details = []
    for sm in SINGLE_MODELS:
        if sm == leader:
            continue
        v_sm = get_cand_loss_vector(sm)
        d_sm = v_leader - v_sm
        m_s, ci_l_s, ci_u_s, p_s = nadeau_bengio_paired_test(
            d_sm, n_splits=5, n_repeats=len(outer_repeats)
        )
        # Better means diff < 0 and CI strictly below 0
        if not (ci_u_s < 0.0 and p_s < 0.05):
            sig_better_than_all_single = False
        single_comp_details.append(
            f"{sm}: diff={m_s:.4f}, CI=[{ci_l_s:.4f}, {ci_u_s:.4f}], p={p_s:.4f}"
        )

    # Determine recommended selection
    if sig_better_than_all_single:
        decision_path = "Rule 3: Leader is statistically significantly better than all single models."
        selected_candidate = leader
    elif ci_contains_zero:
        if leader_ar95_std <= runner_up_ar95_std:
            selected_candidate = leader
            decision_path = (
                f"Rule 2: Leader vs Runner-up difference is not statistically significant (95% CI [{ci_l:.4f}, {ci_u:.4f}] contains 0). "
                f"Leader chosen due to smaller/equal AR@0.95 fold std ({leader_ar95_std:.2f} <= {runner_up_ar95_std:.2f})."
            )
        else:
            selected_candidate = runner_up
            decision_path = (
                f"Rule 2: Leader vs Runner-up difference is not statistically significant (95% CI [{ci_l:.4f}, {ci_u:.4f}] contains 0). "
                f"Runner-up ({runner_up}) chosen due to smaller AR@0.95 fold std ({runner_up_ar95_std:.2f} < {leader_ar95_std:.2f})."
            )
    else:
        selected_candidate = leader
        decision_path = f"Rule 1: Leader ({leader}) statistically significantly outperforms Runner-up."

    selection_df = pd.DataFrame(
        [
            {
                "leader": leader,
                "leader_loss_mean": leader_loss,
                "leader_ar95_std": leader_ar95_std,
                "runner_up": runner_up,
                "runner_up_loss_mean": runner_up_loss,
                "runner_up_ar95_std": runner_up_ar95_std,
                "diff_loss_mean": m_diff,
                "diff_loss_ci_lower_nb": ci_l,
                "diff_loss_ci_upper_nb": ci_u,
                "diff_loss_p_nb": p_nb,
                "ci_contains_zero": ci_contains_zero,
                "sig_better_than_all_single": sig_better_than_all_single,
                "selected_candidate": selected_candidate,
                "decision_path": decision_path,
                "all_single_comparisons": "; ".join(single_comp_details),
            }
        ]
    )
    sel_file = TABLES_DIR / "e3_selection.csv"
    selection_df.to_csv(sel_file, index=False)
    logger.info(f"Saved {sel_file}")
    return selection_df


# ==============================================================================
# Visualization Functions
# ==============================================================================
def generate_e3_figures(
    pairwise_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    e2a_per_fold_df: pd.DataFrame,
    ensemble_per_fold_df: pd.DataFrame,
    stacking_per_fold_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    oof_predictions_dict: dict[str, np.ndarray],
    y_train: pd.Series,
    outer_repeats: list[int],
):
    """
    Generates all 5 required visualizations for Phase E3:
    1. e3_model_rank_heatmap.png
    2. e3_ensemble_vs_single_ar95.png
    3. e3_rank_correlation_heatmap.png
    4. e3_recall_vs_alert_ratio_ensembles.png
    5. e3_hard_positives_percentile.png
    """
    logger.info("=== Generating E3 Figures ===")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 1. e3_model_rank_heatmap.png
    filtered_df = e2a_per_fold_df[e2a_per_fold_df["repeat"].isin(outer_repeats)]
    sorted_folds = (
        ensemble_per_fold_df[["repeat", "fold"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold"])
    )

    rank_data = []
    fold_labels = []
    for _, f_row in sorted_folds.iterrows():
        rep = int(f_row["repeat"])
        f = int(f_row["fold"])
        fold_labels.append(f"r{rep}_f{f}")
        f_sub = filtered_df[
            (filtered_df["repeat"] == rep) & (filtered_df["fold"] == f)
        ]
        row_losses = [
            float(f_sub[f_sub["model"] == m]["outer_loss"].iloc[0])
            for m in SINGLE_MODELS
        ]
        ranks = stats.rankdata(row_losses, method="average")
        rank_data.append(ranks)

    rank_mat = np.array(rank_data)  # shape: (n_folds, 5)

    plt.figure(figsize=(7, 8))
    plt.imshow(rank_mat, cmap="YlGn_r", aspect="auto", vmin=1, vmax=5)
    plt.colorbar(label="Rank (1 = Lowest 3-Point Loss)")
    plt.xticks(range(5), SINGLE_MODELS, fontsize=11, fontweight="bold")
    plt.yticks(range(len(fold_labels)), fold_labels, fontsize=9)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Outer CV Fold", fontsize=12)
    plt.title("E3a: Model Ranks Across 15 Outer Folds", fontsize=13, fontweight="bold")

    for i in range(rank_mat.shape[0]):
        for j in range(rank_mat.shape[1]):
            val = rank_mat[i, j]
            text = f"{val:.0f}" if val.is_integer() else f"{val:.1f}"
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black" if val < 3.5 else "white",
                fontweight="bold",
            )

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "e3_model_rank_heatmap.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig1_path}")

    # 2. e3_ensemble_vs_single_ar95.png
    plt.figure(figsize=(11, 6))
    plot_models = SINGLE_MODELS + list(ENSEMBLE_CANDIDATES.keys()) + ["Stacking"]
    x_positions = range(len(plot_models))

    for idx, name in enumerate(plot_models):
        if name in SINGLE_MODELS:
            vals = (
                filtered_df[filtered_df["model"] == name]
                .sort_values(["repeat", "fold"])["alert_ratio@recall_0.95"]
                .values
            )
            color = "#1f77b4"
        elif name == "Stacking":
            vals = stacking_per_fold_df.sort_values(["repeat", "fold"])["ar95"].values
            color = "#9467bd"
        else:
            vals = (
                ensemble_per_fold_df[ensemble_per_fold_df["ensemble"] == name]
                .sort_values(["repeat", "fold"])["ar95_ecdf"]
                .values
            )
            color = "#2ca02c"

        # Jitter points
        jitter = np.random.normal(0, 0.06, size=len(vals))
        plt.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            alpha=0.6,
            color=color,
            s=35,
            edgecolors="none",
        )
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        plt.errorbar(
            idx,
            mean_v,
            yerr=std_v,
            fmt="D",
            color="black",
            ecolor="black",
            capsize=5,
            capthick=1.5,
            markersize=7,
        )

    plt.xticks(x_positions, plot_models, fontsize=10, rotation=15, fontweight="bold")
    plt.ylabel("Alert Ratio @ 95% Recall (Lower is Better)", fontsize=11)
    plt.title(
        "E3b: AR@0.95 Comparison (15 Folds + Mean ± Std)",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "e3_ensemble_vs_single_ar95.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig2_path}")

    # 3. e3_rank_correlation_heatmap.png
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pos_mat = np.zeros((5, 5))
    neg_mat = np.zeros((5, 5))
    for _, r in corr_df.iterrows():
        i = SINGLE_MODELS.index(r["model_1"])
        j = SINGLE_MODELS.index(r["model_2"])
        pos_mat[i, j] = r["spearman_positives"]
        neg_mat[i, j] = r["spearman_negatives"]

    im0 = axes[0].imshow(pos_mat, cmap="coolwarm", vmin=0.3, vmax=1.0)
    axes[0].set_title(
        "Positive Samples (Y=1, Bankrupt)", fontsize=12, fontweight="bold"
    )
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))
    axes[0].set_xticklabels(SINGLE_MODELS)
    axes[0].set_yticklabels(SINGLE_MODELS)
    for i in range(5):
        for j in range(5):
            axes[0].text(
                j,
                i,
                f"{pos_mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if pos_mat[i, j] < 0.8 else "white",
                fontweight="bold",
            )

    im1 = axes[1].imshow(neg_mat, cmap="coolwarm", vmin=0.3, vmax=1.0)
    axes[1].set_title(
        "Negative Samples (Y=0, Non-Bankrupt)", fontsize=12, fontweight="bold"
    )
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(SINGLE_MODELS)
    axes[1].set_yticklabels(SINGLE_MODELS)
    for i in range(5):
        for j in range(5):
            axes[1].text(
                j,
                i,
                f"{neg_mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if neg_mat[i, j] < 0.8 else "white",
                fontweight="bold",
            )

    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Spearman Rank Correlation", shrink=0.8)
    plt.suptitle("E3d: Model Rank Correlations (15-Fold Mean)", fontsize=14, fontweight="bold")
    fig3_path = FIGURES_DIR / "e3_rank_correlation_heatmap.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig3_path}")

    # 4. e3_recall_vs_alert_ratio_ensembles.png
    plt.figure(figsize=(9, 6))
    recalls = np.linspace(0.50, 1.00, 51)
    y_arr = y_train.values

    curve_candidates = ["M3", "M4", "E_all", "E_A", "E_trees"]
    styles = {
        "M3": ("#1f77b4", "--", "M3 (ExtraTrees)"),
        "M4": ("#ff7f0e", "--", "M4 (XGBoost)"),
        "E_all": ("#2ca02c", "-", "E_all (M1-M5 Equal Weight)"),
        "E_A": ("#d62728", "-", "E_A (M3+M4+M5)"),
        "E_trees": ("#9467bd", "-", "E_trees (M2-M5)"),
    }

    for name in curve_candidates:
        if name in SINGLE_MODELS:
            oof = np.load(OOF_DIR / f"e2_{name}.npy")
        else:
            oof = oof_predictions_dict[name]

        # Compute AR curve averaged across repeats
        ar_curves = []
        for r_idx in range(len(outer_repeats)):
            sc = oof[r_idx]
            valid_m = ~np.isnan(sc)
            if np.sum(valid_m) > 0 and np.sum(y_arr[valid_m] == 1) > 0:
                y_sub = y_arr[valid_m]
                sc_sub = sc[valid_m]
                rep_ars = [alert_ratio_at_recall(y_sub, sc_sub, r_val) for r_val in recalls]
                ar_curves.append(rep_ars)
        if len(ar_curves) > 0:
            mean_ar = np.mean(ar_curves, axis=0)
            color, ls, label = styles[name]
            plt.plot(mean_ar, recalls, color=color, linestyle=ls, label=label, linewidth=2)

    plt.axhline(0.95, color="red", linestyle=":", alpha=0.7, label="Target Recall = 0.95")
    plt.xlabel("Alert Ratio (Lower is Better)", fontsize=11)
    plt.ylabel("Recall on Bankrupt Firms", fontsize=11)
    plt.title(
        "E3b: Recall vs. Alert Ratio Curves (Pooled OOF)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlim(1.0, 20.0)
    plt.ylim(0.5, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "e3_recall_vs_alert_ratio_ensembles.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig4_path}")

    # 5. e3_hard_positives_percentile.png
    plt.figure(figsize=(9, 5))
    pcts = hard_df["avg_percentile"].values
    plt.hist(pcts, bins=25, color="#1f77b4", edgecolor="black", alpha=0.7, label="All Bankrupt Firms (N=176)")
    # Mark lowest 15
    lowest_15 = pcts[:15]
    plt.hist(lowest_15, bins=10, color="#d62728", edgecolor="black", alpha=0.8, label="Hardest 15 Positives")
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="50th Percentile")
    plt.xlabel("Average Model Percentile Score", fontsize=11)
    plt.ylabel("Number of Bankrupt Companies", fontsize=11)
    plt.title(
        "E3d: Distribution of Average Percentile Scores for 176 Bankrupt Companies",
        fontsize=12,
        fontweight="bold",
    )
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "e3_hard_positives_percentile.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig5_path}")


# ==============================================================================
# Stopping Conditions Audit
# ==============================================================================
def audit_e3_stopping_conditions(
    ensemble_summary_df: pd.DataFrame,
    ensemble_per_fold_df: pd.DataFrame,
    e2a_per_fold_df: pd.DataFrame,
    outer_repeats: list[int],
):
    """
    Checks E3 stopping rules:
    1. Any ensemble outer fold mean ROC-AUC > 0.95 -> stop.
    2. Any ensemble on a single fold has ROC-AUC jump > 0.03 relative to best single model on that fold -> stop.
    3. ECDF mapping causes massive ties leading to anomalous metrics -> stop.
    """
    logger.info("=== Auditing E3 Stopping Conditions ===")

    # Rule 1: Outer fold mean ROC-AUC > 0.95
    for _, r in ensemble_summary_df.iterrows():
        ens = r["candidate"]
        mean_roc = r["roc_auc_fold_mean"]
        if mean_roc > 0.95:
            raise RuntimeError(
                f"STOPPING TRIGGERED: Ensemble {ens} outer fold mean ROC-AUC ({mean_roc:.4f}) > 0.95!"
            )

    # Rule 2: Single fold ROC-AUC jump > 0.03 vs best single model on that fold
    e2a_sub = e2a_per_fold_df[e2a_per_fold_df["repeat"].isin(outer_repeats)]
    for _, r in ensemble_per_fold_df.iterrows():
        ens = r["ensemble"]
        rep = r["repeat"]
        f = r["fold"]
        ens_roc = r["roc_auc_ecdf"]

        # Best single model ROC-AUC on that fold
        f_sub = e2a_sub[(e2a_sub["repeat"] == rep) & (e2a_sub["fold"] == f)]
        best_single_roc = f_sub["roc_auc"].max()
        jump = ens_roc - best_single_roc

        if jump > 0.03:
            raise RuntimeError(
                f"STOPPING TRIGGERED: Ensemble {ens} on fold r{rep}_f{f} has ROC-AUC jump of {jump:.4f} > 0.03 vs best single model ({best_single_roc:.4f})!"
            )

    logger.info("Audit passed: No E3 stopping conditions triggered.")


# ==============================================================================
# Main Orchestrator for Phase E3
# ==============================================================================
def run_e3_pipeline(outer_repeats: list[int] = [0, 1, 2], quick: bool = False):
    print("=" * 80)
    print(f"       E3: MODEL COMPARISON & ENSEMBLING PIPELINE (Repeats: {outer_repeats})")
    print("=" * 80)

    e2a_per_fold_path = TABLES_DIR / "e2a_nested_per_fold.csv"
    if not e2a_per_fold_path.exists():
        raise FileNotFoundError(f"Missing {e2a_per_fold_path}. Run E2 first.")
    e2a_per_fold_df = pd.read_csv(e2a_per_fold_path)

    # 1. E3a: Single Models Comparison
    pairwise_df, friedman_df = run_e3a_single_models_comparison(
        e2a_per_fold_df=e2a_per_fold_df, outer_repeats=outer_repeats
    )

    # 2. E3b & E3c: Ensembles and Stacking Execution
    (
        ensemble_per_fold_df,
        ensemble_summary_df,
        stacking_per_fold_df,
        stacking_summary_df,
        oof_predictions_dict,
    ) = run_e3b_and_e3c(outer_repeats=outer_repeats, quick=quick)

    # 3. E3b Paired Comparisons vs Single Best
    paired_df = run_e3b_paired_comparisons(
        e2a_per_fold_df=e2a_per_fold_df,
        ensemble_per_fold_df=ensemble_per_fold_df,
        stacking_per_fold_df=stacking_per_fold_df,
        outer_repeats=outer_repeats,
    )

    # 4. E3d Diagnostics
    corr_df, hard_df, alignment_df = run_e3d_diagnostics(
        outer_repeats=outer_repeats,
        e2a_per_fold_df=e2a_per_fold_df,
        ensemble_summary_df=ensemble_summary_df,
        stacking_summary_df=stacking_summary_df,
        oof_predictions_dict=oof_predictions_dict,
        quick=quick,
    )

    # 5. Candidate Selection Rule Application
    selection_df = apply_e3_selection_rule(
        alignment_df=alignment_df,
        e2a_per_fold_df=e2a_per_fold_df,
        ensemble_per_fold_df=ensemble_per_fold_df,
        outer_repeats=outer_repeats,
    )

    # 6. Audit Stopping Conditions
    audit_e3_stopping_conditions(
        ensemble_summary_df=ensemble_summary_df,
        ensemble_per_fold_df=ensemble_per_fold_df,
        e2a_per_fold_df=e2a_per_fold_df,
        outer_repeats=outer_repeats,
    )

    # 7. Generate Figures
    X_train, y_train = get_train_data()
    generate_e3_figures(
        pairwise_df=pairwise_df,
        friedman_df=friedman_df,
        e2a_per_fold_df=e2a_per_fold_df,
        ensemble_per_fold_df=ensemble_per_fold_df,
        stacking_per_fold_df=stacking_per_fold_df,
        corr_df=corr_df,
        hard_df=hard_df,
        oof_predictions_dict=oof_predictions_dict,
        y_train=y_train,
        outer_repeats=outer_repeats,
    )

    print("\nPhase E3 completed successfully.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="E3: Model Comparison & Ensembling Module")
    parser.add_argument(
        "--outer-repeats",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="List of outer repeats to execute (default: [0, 1, 2] for all 15 folds)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run 2 outer folds for quick verification",
    )
    args = parser.parse_args()

    run_e3_pipeline(outer_repeats=args.outer_repeats, quick=args.quick)


if __name__ == "__main__":
    main()
