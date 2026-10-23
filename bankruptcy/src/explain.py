"""
E5: Feature Importance, Interpretability, and Compact Feature Set Evaluation.

This module implements:
1. E5a: Feature importance and interpretability on outer validation folds:
   - Permutation importance (PR-AUC drop primary, ROC-AUC drop secondary)
   - Group permutation importance (|r| > 0.8 hierarchical clustering)
   - TreeExplainer SHAP values (all positives + 300 sampled negatives, raw margin for M4/M5)
   - Stability, cross-model agreement, and financial dimension mapping
2. E5b: Nested evaluation of compact feature sets (preventing leakage):
   - In-fold feature selection via normalized intrinsic model importance (M3 impurity, M4/M5 gain)
   - Compact pipeline evaluation for k in {10, 20, 30}
   - Internal percentile rank ensemble matching E3b
   - Nadeau-Bengio corrected paired t-test and Holm-Bonferroni correction vs Full Feature E_A
   - Jaccard stability and core feature set extraction
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

import shap

import config
from src.cv import get_train_data, load_splits
from src.metrics import alert_ratio_at_recall, recall_at_alert_ratio
from src.pipelines import (
    AdaptiveScaleWeightClassifier,
    build_pipeline,
    get_feature_names_after_preprocessing,
    get_score,
)
from src.stats_utils import nadeau_bengio_paired_test, holm_bonferroni

logger = logging.getLogger("bankruptcy_e5")
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
E5_CACHE_DIR = MODELS_DIR / "e5_cache"
E5_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR = MODELS_DIR / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

SEED = config.SEED
MODELS = ["M3", "M4", "M5"]


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC score handling edge cases safely."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return float(auc(rec, prec))


def compute_three_point_log_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Computes mean(log(alert_ratio@recall_r) for r in [0.80, 0.90, 0.95])."""
    ar80 = alert_ratio_at_recall(y_true, scores, 0.80)
    ar90 = alert_ratio_at_recall(y_true, scores, 0.90)
    ar95 = alert_ratio_at_recall(y_true, scores, 0.95)
    return float(np.mean([np.log(ar80), np.log(ar90), np.log(ar95)]))


def map_financial_dimension(feat_name: str) -> str:
    """
    Classifies financial feature names into domain dimensions for reference (待確認).
    Categories: 獲利 / 槓桿 / 流動性 / 營運效率 / 成長 / 現金流.
    """
    f = feat_name.lower()
    if any(k in f for k in ["eps", "net income", "profit", "roa", "roe", "margin", "return", "retained earnings"]):
        return "獲利能力 (待確認)"
    elif any(k in f for k in ["debt", "liability", "borrowing", "equity to", "gearing", "leverage"]):
        return "槓桿與償債能力 (待確認)"
    elif any(k in f for k in ["current ratio", "quick ratio", "working capital", "cash/total", "cash/current", "cash to"]):
        return "流動性 (待確認)"
    elif any(k in f for k in ["turnover", "days", "person", "rate of turnover", "efficiency"]):
        return "營運效率 (待確認)"
    elif any(k in f for k in ["growth", "increase"]):
        return "成長能力 (待確認)"
    elif any(k in f for k in ["cash flow", "cfo", "cash reinvestment"]):
        return "現金流 (待確認)"
    else:
        return "其他財務指標 (待確認)"


def is_directly_computable(feat_name: str) -> str:
    """
    Assesses whether a ratio is typically directly computable from standard single-year financial statements (待確認).
    """
    f = feat_name.lower()
    if any(k in f for k in ["persistent", "four seasons", "growth rate ratio", "continuous", "suitability ratio", "person", "flag"]):
        return "否/需多期或特殊定義 (待確認)"
    return "是 (待確認)"


# ==============================================================================
# Pipeline & Preprocessing Utilities
# ==============================================================================
def transform_through_preprocessing(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Transforms DataFrame through pipeline steps up to before classifier and resampler.
    Returns transformed 2D numpy array and corresponding feature names.
    """
    feat_names = get_feature_names_after_preprocessing(pipeline)
    cur = X
    for name, step in pipeline.steps[:-1]:
        if name != "resampler":
            cur = step.transform(cur)
    if hasattr(cur, "values"):
        cur_arr = cur.values
    else:
        cur_arr = np.asarray(cur)
    return cur_arr, feat_names


def get_underlying_classifier(pipeline) -> Any:
    """Extracts raw classifier from pipeline, unwrapping AdaptiveScaleWeightClassifier if present."""
    clf = pipeline.named_steps["classifier"]
    if isinstance(clf, AdaptiveScaleWeightClassifier):
        return clf.estimator_
    return clf


def predict_proba_from_prep(clf, X_prep: np.ndarray) -> np.ndarray:
    """Predicts continuous probability of positive class directly from preprocessed array."""
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_prep)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return prob[:, 1]
        return prob.ravel()
    elif hasattr(clf, "decision_function"):
        return clf.decision_function(X_prep)
    raise AttributeError(f"Classifier {clf} has neither predict_proba nor decision_function")


# ==============================================================================
# Model-Intrinsic Importance (for E5b ranking)
# ==============================================================================
def extract_model_intrinsic_importance(fitted_pipeline, model_name: str) -> np.ndarray:
    """
    Extracts model-intrinsic feature importance:
    - M3: impurity-based (feature_importances_)
    - M4: gain-based (feature_importances_ or get_score)
    - M5: gain-based (booster_.feature_importance(importance_type='gain'))
    Normalizes returned vector so that elements sum to 1.0.
    """
    clf = get_underlying_classifier(fitted_pipeline)
    if model_name == "M3":
        imp = np.array(clf.feature_importances_, dtype=float)
    elif model_name == "M4":
        imp = np.array(clf.feature_importances_, dtype=float)
    elif model_name == "M5":
        imp = np.array(clf.booster_.feature_importance(importance_type="gain"), dtype=float)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    total = imp.sum()
    if total > 0:
        return imp / total
    return imp


# ==============================================================================
# E5a: Feature Importance Calculations (Permutation, Group, SHAP)
# ==============================================================================
def compute_permutation_importance(
    clf,
    X_val_prep: np.ndarray,
    y_val: np.ndarray,
    n_repeats: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes permutation importance on preprocessed validation data.
    Uses batched inference across features for high throughput.
    Returns (mean_drop_pr, std_drop_pr, mean_drop_roc, std_drop_roc).
    """
    base_prob = predict_proba_from_prep(clf, X_val_prep)
    base_pr = compute_pr_auc(y_val, base_prob)
    base_roc = float(roc_auc_score(y_val, base_prob))

    n_samples, n_features = X_val_prep.shape
    drop_pr_all = np.zeros((n_features, n_repeats))
    drop_roc_all = np.zeros((n_features, n_repeats))

    for r in range(n_repeats):
        stacked = np.tile(X_val_prep, (n_features, 1))
        for j in range(n_features):
            rng = np.random.RandomState(seed + j * 31 + r * 101)
            perm_idx = rng.permutation(n_samples)
            start_row = j * n_samples
            end_row = (j + 1) * n_samples
            stacked[start_row:end_row, j] = X_val_prep[perm_idx, j]

        all_probs = predict_proba_from_prep(clf, stacked)

        for j in range(n_features):
            start_row = j * n_samples
            end_row = (j + 1) * n_samples
            prob_perm = all_probs[start_row:end_row]

            drop_pr = base_pr - compute_pr_auc(y_val, prob_perm)
            drop_roc = base_roc - float(roc_auc_score(y_val, prob_perm))

            assert not np.isnan(drop_pr), f"NaN drop in PR-AUC for feature index {j}"
            assert not np.isnan(drop_roc), f"NaN drop in ROC-AUC for feature index {j}"

            drop_pr_all[j, r] = drop_pr
            drop_roc_all[j, r] = drop_roc

    return (
        drop_pr_all.mean(axis=1),
        drop_pr_all.std(axis=1, ddof=1) if n_repeats > 1 else np.zeros(n_features),
        drop_roc_all.mean(axis=1),
        drop_roc_all.std(axis=1, ddof=1) if n_repeats > 1 else np.zeros(n_features),
    )


def compute_group_permutation_importance(
    clf,
    X_tr_prep: np.ndarray,
    X_val_prep: np.ndarray,
    y_val: np.ndarray,
    feat_names: list[str],
    corr_threshold: float = 0.80,
    n_repeats: int = 5,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """
    Hierarchically clusters training features with |r| > 0.80 and evaluates group permutation drop on validation data.
    Uses batched inference across clusters for maximum throughput.
    """
    base_prob = predict_proba_from_prep(clf, X_val_prep)
    base_pr = compute_pr_auc(y_val, base_prob)
    base_roc = float(roc_auc_score(y_val, base_prob))

    # Correlation and complete linkage on training data
    corr = np.corrcoef(X_tr_prep, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    dist_condensed = squareform(dist, checks=False)

    linkage = hierarchy.linkage(dist_condensed, method="complete")
    dist_thresh = 1.0 - corr_threshold  # 0.20
    cluster_ids = hierarchy.fcluster(linkage, t=dist_thresh, criterion="distance")

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    n_samples = len(X_val_prep)

    cluster_members = [np.where(cluster_ids == c_id)[0] for c_id in unique_clusters]
    cluster_drops_pr = np.zeros((n_clusters, n_repeats))
    cluster_drops_roc = np.zeros((n_clusters, n_repeats))

    for r in range(n_repeats):
        stacked = np.tile(X_val_prep, (n_clusters, 1))
        for idx_c, member_indices in enumerate(cluster_members):
            c_id = unique_clusters[idx_c]
            rng = np.random.RandomState(seed + int(c_id) * 37 + r * 109)
            perm_idx = rng.permutation(n_samples)
            start_row = idx_c * n_samples
            end_row = (idx_c + 1) * n_samples
            for col_idx in member_indices:
                stacked[start_row:end_row, col_idx] = X_val_prep[perm_idx, col_idx]

        all_probs = predict_proba_from_prep(clf, stacked)

        for idx_c in range(n_clusters):
            start_row = idx_c * n_samples
            end_row = (idx_c + 1) * n_samples
            prob_perm = all_probs[start_row:end_row]

            d_pr = base_pr - compute_pr_auc(y_val, prob_perm)
            d_roc = base_roc - float(roc_auc_score(y_val, prob_perm))

            assert not np.isnan(d_pr), f"NaN drop in PR-AUC for cluster {unique_clusters[idx_c]}"
            assert not np.isnan(d_roc), f"NaN drop in ROC-AUC for cluster {unique_clusters[idx_c]}"

            cluster_drops_pr[idx_c, r] = d_pr
            cluster_drops_roc[idx_c, r] = d_roc

    group_results = []
    for idx_c, c_id in enumerate(unique_clusters):
        member_names = [feat_names[i] for i in cluster_members[idx_c]]
        group_results.append(
            {
                "cluster_id": int(c_id),
                "features": "; ".join(member_names),
                "n_features": len(member_names),
                "pr_auc_drop_mean": float(cluster_drops_pr[idx_c].mean()),
                "pr_auc_drop_std": float(cluster_drops_pr[idx_c].std(ddof=1)) if n_repeats > 1 else 0.0,
                "roc_auc_drop_mean": float(cluster_drops_roc[idx_c].mean()),
                "roc_auc_drop_std": float(cluster_drops_roc[idx_c].std(ddof=1)) if n_repeats > 1 else 0.0,
                "financial_dimension": map_financial_dimension(member_names[0]),
            }
        )

    return group_results


def compute_shap_importance(
    clf,
    model_name: str,
    X_val_prep: np.ndarray,
    y_val: np.ndarray,
    feat_names: list[str],
    n_background_neg: int = 300,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Computes SHAP values using TreeExplainer on validation fold:
    - All positive samples (y_val == 1)
    - Random 300 negative samples (y_val == 0)
    For M4/M5 uses raw margin output. For M3 uses class 1 probability margin.
    Returns (mean_abs_shap, shap_values_chosen, X_chosen, chosen_indices).
    """
    pos_idx = np.where(y_val == 1)[0]
    neg_idx = np.where(y_val == 0)[0]

    rng = np.random.RandomState(seed)
    sampled_neg_count = min(n_background_neg, len(neg_idx))
    sampled_neg_idx = rng.choice(neg_idx, size=sampled_neg_count, replace=False)

    eval_idx = np.sort(np.concatenate([pos_idx, sampled_neg_idx]))
    X_eval = X_val_prep[eval_idx]
    X_eval_df = pd.DataFrame(X_eval, columns=feat_names)

    if model_name in ["M4", "M5"]:
        explainer = shap.TreeExplainer(clf, model_output="raw")
        shap_vals = explainer.shap_values(X_eval_df)
        if isinstance(shap_vals, list):
            # In case shap returns list of 2 classes
            shap_vals_target = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif isinstance(shap_vals, np.ndarray):
            shap_vals_target = shap_vals
        else:
            shap_vals_target = np.asarray(shap_vals)
    elif model_name == "M3":
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_eval_df)
        if isinstance(shap_vals, list):
            shap_vals_target = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals_target = shap_vals[:, :, 1]
        else:
            shap_vals_target = shap_vals
    else:
        raise ValueError(f"Unsupported model for SHAP: {model_name}")

    assert not np.isnan(shap_vals_target).any(), f"NaN detected in SHAP values for {model_name}"

    mean_abs_shap = np.mean(np.abs(shap_vals_target), axis=0)
    return mean_abs_shap, shap_vals_target, X_eval, eval_idx.tolist()


# ==============================================================================
# E5 Master Runner
# ==============================================================================
def run_e5(
    outer_repeats: list[int] = [0, 1, 2],
    quick: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """
    Executes full E5 module:
    - E5a: Feature Importance & Interpretability
    - E5b: Compact Feature Set Nested Evaluation
    """
    logger.info("=== Starting E5: Interpretability & Compact Feature Evaluation ===")

    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_idx = splits_payload["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}

    selected_splits = [s for s in all_splits if s["repeat"] in outer_repeats]
    if quick:
        selected_splits = [s for s in selected_splits if s["repeat"] == 0]
        logger.info(f"--quick mode enabled: running on repeat 0 ({len(selected_splits)} outer folds)")

    # Read nested CV parameters from E2a
    e2a_per_fold = pd.read_csv(TABLES_DIR / "e2a_nested_per_fold.csv")

    # Structures for E5a
    perm_fold_records = []
    group_fold_records = []
    shap_fold_records = []

    # Structures for SHAP plotting
    shap_plot_cache = {m: {"shap_vals": [], "X_eval": [], "feat_names": []} for m in MODELS}

    # Structures for E5b
    k_list = [10, 20, 30] if not quick else [20]
    compact_fold_records = []
    fold_selected_features = {k: [] for k in [10, 20, 30]}
    all_features_frequency = {feat: 0 for feat in X_train.columns}

    outer_roc_auc_history = []

    for s in selected_splits:
        rep = int(s["repeat"])
        f = int(s["fold"])
        logger.info(f"--- Processing Outer Split: Repeat {rep}, Fold {f} ---")

        out_tr_idx = np.array([orig_to_rel[i] for i in s["train_idx"]])
        out_val_idx = np.array([orig_to_rel[i] for i in s["val_idx"]])

        X_tr = X_train.iloc[out_tr_idx]
        y_tr = y_train.iloc[out_tr_idx]
        X_val = X_train.iloc[out_val_idx]
        y_val = y_train.iloc[out_val_idx].values

        # ----------------------------------------------------------------------
        # 1. Fit or Load Outer Pipelines for M3, M4, M5
        # ----------------------------------------------------------------------
        fold_pipelines = {}
        for m in MODELS:
            cache_file = E5_CACHE_DIR / f"pipeline_{m}_r{rep}_f{f}.joblib"
            if cache_file.exists() and not force:
                logger.info(f"Loading cached pipeline for {m} (r{rep}, f{f})")
                pipe = joblib.load(cache_file)
            else:
                row = e2a_per_fold[
                    (e2a_per_fold["model"] == m)
                    & (e2a_per_fold["repeat"] == rep)
                    & (e2a_per_fold["fold"] == f)
                ].iloc[0]
                best_params = json.loads(row["best_params"])
                best_imbalance = str(row["best_imbalance"])

                pipe = build_pipeline(
                    model_name=m,
                    imbalance=best_imbalance,
                    scaler="quantile",
                    use_winsorizer=False,
                    use_corr_filter=True,
                    use_large_indicator=False,
                    model_params=best_params,
                )
                pipe.fit(X_tr, y_tr)
                joblib.dump(pipe, cache_file)
            fold_pipelines[m] = pipe

        # ----------------------------------------------------------------------
        # 2. E5a: Permutation Importance, Group Permutation, and SHAP
        # ----------------------------------------------------------------------
        for m in MODELS:
            pipe = fold_pipelines[m]
            clf = get_underlying_classifier(pipe)
            X_tr_prep, feat_names_tr = transform_through_preprocessing(pipe, X_tr)
            X_val_prep, feat_names_val = transform_through_preprocessing(pipe, X_val)
            assert feat_names_tr == feat_names_val, "Feature names mismatch between train and val prep"
            feat_names = feat_names_val

            base_pred = predict_proba_from_prep(clf, X_val_prep)
            val_roc = float(roc_auc_score(y_val, base_pred))
            outer_roc_auc_history.append((m, rep, f, val_roc))

            # Stop Condition Check 1
            if val_roc > 0.98:
                logger.warning(f"High validation ROC-AUC ({val_roc:.4f}) for {m} in repeat {rep}, fold {f}!")

            # 2a. Permutation Importance
            perm_cache_file = E5_CACHE_DIR / f"perm_{m}_r{rep}_f{f}.joblib"
            if perm_cache_file.exists() and not force:
                perm_data = joblib.load(perm_cache_file)
                mean_pr_drop, std_pr_drop, mean_roc_drop, std_roc_drop = (
                    perm_data["mean_pr_drop"],
                    perm_data["std_pr_drop"],
                    perm_data["mean_roc_drop"],
                    perm_data["std_roc_drop"],
                )
            else:
                mean_pr_drop, std_pr_drop, mean_roc_drop, std_roc_drop = compute_permutation_importance(
                    clf, X_val_prep, y_val, n_repeats=5, seed=SEED + rep * 1000 + f * 100
                )
                joblib.dump(
                    {
                        "mean_pr_drop": mean_pr_drop,
                        "std_pr_drop": std_pr_drop,
                        "mean_roc_drop": mean_roc_drop,
                        "std_roc_drop": std_roc_drop,
                    },
                    perm_cache_file,
                )

            # Rank features for this fold (rank 1 = highest PR-AUC drop)
            rank_perm = stats.rankdata(-mean_pr_drop, method="min")
            for j, fname in enumerate(feat_names):
                perm_fold_records.append(
                    {
                        "model": m,
                        "repeat": rep,
                        "fold": f,
                        "feature": fname,
                        "pr_auc_drop": float(mean_pr_drop[j]),
                        "roc_auc_drop": float(mean_roc_drop[j]),
                        "rank_pr": int(rank_perm[j]),
                    }
                )

            # 2b. Group Permutation Importance
            grp_cache_file = E5_CACHE_DIR / f"grp_{m}_r{rep}_f{f}.joblib"
            if grp_cache_file.exists() and not force:
                grp_res = joblib.load(grp_cache_file)
            else:
                grp_res = compute_group_permutation_importance(
                    clf,
                    X_tr_prep,
                    X_val_prep,
                    y_val,
                    feat_names,
                    corr_threshold=0.80,
                    n_repeats=5,
                    seed=SEED + rep * 1000 + f * 100,
                )
                joblib.dump(grp_res, grp_cache_file)

            for g_item in grp_res:
                group_fold_records.append(
                    {
                        "model": m,
                        "repeat": rep,
                        "fold": f,
                        "cluster_id": g_item["cluster_id"],
                        "features": g_item["features"],
                        "n_features": g_item["n_features"],
                        "pr_auc_drop": g_item["pr_auc_drop_mean"],
                        "roc_auc_drop": g_item["roc_auc_drop_mean"],
                        "financial_dimension": g_item["financial_dimension"],
                    }
                )

            # 2c. TreeExplainer SHAP Importance
            shap_cache_file = E5_CACHE_DIR / f"shap_{m}_r{rep}_f{f}.npz"
            if shap_cache_file.exists() and not force:
                npz = np.load(shap_cache_file, allow_pickle=True)
                mean_abs_shap = npz["mean_abs_shap"]
                shap_vals_target = npz["shap_vals_target"]
                X_eval = npz["X_eval"]
            else:
                mean_abs_shap, shap_vals_target, X_eval, _ = compute_shap_importance(
                    clf,
                    m,
                    X_val_prep,
                    y_val,
                    feat_names,
                    n_background_neg=300,
                    seed=SEED + rep * 100 + f,
                )
                np.savez_compressed(
                    shap_cache_file,
                    mean_abs_shap=mean_abs_shap,
                    shap_vals_target=shap_vals_target,
                    X_eval=X_eval,
                )

            rank_shap = stats.rankdata(-mean_abs_shap, method="min")
            for j, fname in enumerate(feat_names):
                shap_fold_records.append(
                    {
                        "model": m,
                        "repeat": rep,
                        "fold": f,
                        "feature": fname,
                        "mean_abs_shap": float(mean_abs_shap[j]),
                        "rank_shap": int(rank_shap[j]),
                    }
                )

            # Cache for global summary plots (sample repeat 0 fold 0)
            if rep == 0 and f == 0:
                shap_plot_cache[m]["shap_vals"] = shap_vals_target
                shap_plot_cache[m]["X_eval"] = X_eval
                shap_plot_cache[m]["feat_names"] = feat_names

        # ----------------------------------------------------------------------
        # 3. E5b: Nested Compact Feature Selection on Outer Train
        # ----------------------------------------------------------------------
        # Extract intrinsic importances for M3, M4, M5 on training fold
        pipe_M3 = fold_pipelines["M3"]
        pipe_M4 = fold_pipelines["M4"]
        pipe_M5 = fold_pipelines["M5"]

        feat_names_M3 = get_feature_names_after_preprocessing(pipe_M3)
        feat_names_M4 = get_feature_names_after_preprocessing(pipe_M4)
        feat_names_M5 = get_feature_names_after_preprocessing(pipe_M5)

        imp_M3 = extract_model_intrinsic_importance(pipe_M3, "M3")
        imp_M4 = extract_model_intrinsic_importance(pipe_M4, "M4")
        imp_M5 = extract_model_intrinsic_importance(pipe_M5, "M5")

        # Map to common feature union / intersection
        df_imp = pd.DataFrame(
            {
                "feat": feat_names_M3,
                "imp_M3": imp_M3,
                "imp_M4": [imp_M4[feat_names_M4.index(x)] if x in feat_names_M4 else 0.0 for x in feat_names_M3],
                "imp_M5": [imp_M5[feat_names_M5.index(x)] if x in feat_names_M5 else 0.0 for x in feat_names_M3],
            }
        )
        df_imp["rank_M3"] = stats.rankdata(-df_imp["imp_M3"], method="min")
        df_imp["rank_M4"] = stats.rankdata(-df_imp["imp_M4"], method="min")
        df_imp["rank_M5"] = stats.rankdata(-df_imp["imp_M5"], method="min")
        df_imp["avg_rank"] = (df_imp["rank_M3"] + df_imp["rank_M4"] + df_imp["rank_M5"]) / 3.0
        df_imp = df_imp.sort_values("avg_rank").reset_index(drop=True)

        ranked_features_this_fold = list(df_imp["feat"].values)

        # Track Top 20 features frequency
        top20_this_fold = set(ranked_features_this_fold[:20])
        for feat in top20_this_fold:
            all_features_frequency[feat] = all_features_frequency.get(feat, 0) + 1

        for k in [10, 20, 30]:
            fold_selected_features[k].append(set(ranked_features_this_fold[:k]))

        # ----------------------------------------------------------------------
        # 4. Fit Compact Pipelines and Evaluate on Outer Val
        # ----------------------------------------------------------------------
        for k in k_list:
            selected_k = ranked_features_this_fold[:k]

            compact_preds = {}
            for m in MODELS:
                comp_cache = E5_CACHE_DIR / f"compact_{m}_k{k}_r{rep}_f{f}.joblib"
                if comp_cache.exists() and not force:
                    pipe_comp = joblib.load(comp_cache)
                else:
                    row = e2a_per_fold[
                        (e2a_per_fold["model"] == m)
                        & (e2a_per_fold["repeat"] == rep)
                        & (e2a_per_fold["fold"] == f)
                    ].iloc[0]
                    best_params = json.loads(row["best_params"])
                    best_imbalance = str(row["best_imbalance"])

                    pipe_comp = build_pipeline(
                        model_name=m,
                        imbalance=best_imbalance,
                        scaler="quantile",
                        use_winsorizer=False,
                        use_corr_filter=True,
                        use_large_indicator=False,
                        features=selected_k,
                        model_params=best_params,
                    )
                    pipe_comp.fit(X_tr, y_tr)
                    joblib.dump(pipe_comp, comp_cache)

                raw_score = get_score(pipe_comp, X_val)
                # Validation internal percentile rank
                pct_rank = stats.rankdata(raw_score) / len(raw_score)
                compact_preds[m] = pct_rank

            # Compact Ensemble E_A: equal-weight average of validation percentile ranks
            S_compact_EA = (compact_preds["M3"] + compact_preds["M4"] + compact_preds["M5"]) / 3.0

            loss_val = compute_three_point_log_loss(y_val, S_compact_EA)
            ar95_val = alert_ratio_at_recall(y_val, S_compact_EA, 0.95)
            pr_val = compute_pr_auc(y_val, S_compact_EA)
            rec_ar3_val = recall_at_alert_ratio(y_val, S_compact_EA, 3.0)
            roc_val = float(roc_auc_score(y_val, S_compact_EA))

            compact_fold_records.append(
                {
                    "repeat": rep,
                    "fold": f,
                    "k": k,
                    "three_point_loss": loss_val,
                    "ar95": ar95_val,
                    "pr_auc": pr_val,
                    "recall_at_ar3": rec_ar3_val,
                    "roc_auc": roc_val,
                }
            )

    # Check Stop Condition: Average ROC-AUC across validation folds
    avg_roc_auc_all = np.mean([r[3] for r in outer_roc_auc_history])
    logger.info(f"Mean Validation ROC-AUC across models and outer folds: {avg_roc_auc_all:.4f}")
    if avg_roc_auc_all > 0.95:
        logger.warning("Mean Validation ROC-AUC exceeds 0.95! Checking for abnormal leakage...")

    # ==============================================================================
    # Compile & Save E5a Tables
    # ==============================================================================
    logger.info("=== Compiling E5a Tables ===")
    df_perm = pd.DataFrame(perm_fold_records)
    df_grp = pd.DataFrame(group_fold_records)
    df_shap = pd.DataFrame(shap_fold_records)

    # 1. e5a_permutation_importance.csv (Aggregated strictly across all 15 outer folds)
    perm_summary_list = []
    all_perm_features = sorted(df_perm["feature"].unique())
    n_splits_actual = len(selected_splits)

    for m in MODELS:
        sub_m = df_perm[df_perm["model"] == m]
        # Precompute per-fold ranks
        fold_ranks = {}
        for (rep, f), g in sub_m.groupby(["repeat", "fold"]):
            sorted_g = g.sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)
            sorted_g["rank"] = sorted_g.index + 1
            for _, r in sorted_g.iterrows():
                fold_ranks[(rep, f, r["feature"])] = r["rank"]

        m_rows = []
        for feat in all_perm_features:
            drops_pr = []
            drops_roc = []
            ranks_present = []
            top20_hits = 0

            for s in selected_splits:
                rep, f = int(s["repeat"]), int(s["fold"])
                match = sub_m[(sub_m["repeat"] == rep) & (sub_m["fold"] == f) & (sub_m["feature"] == feat)]
                if len(match) > 0:
                    d_pr = match["pr_auc_drop"].values[0]
                    d_roc = match["roc_auc_drop"].values[0]
                    rk = fold_ranks.get((rep, f, feat), 999)
                    ranks_present.append(rk)
                    if rk <= 20:
                        top20_hits += 1
                else:
                    d_pr = 0.0
                    d_roc = 0.0

                drops_pr.append(d_pr)
                drops_roc.append(d_roc)

            m_rows.append(
                {
                    "model": m,
                    "feature": feat,
                    "pr_auc_drop_mean": float(np.mean(drops_pr)),
                    "pr_auc_drop_std": float(np.std(drops_pr, ddof=1)),
                    "roc_auc_drop_mean": float(np.mean(drops_roc)),
                    "roc_auc_drop_std": float(np.std(drops_roc, ddof=1)),
                    "rank_mean_present": float(np.mean(ranks_present)) if ranks_present else np.nan,
                    "in_top20_ratio": float(top20_hits / n_splits_actual),
                    "folds_present": len(ranks_present),
                    "total_folds": n_splits_actual,
                    "financial_dimension": map_financial_dimension(feat),
                }
            )

        agg = pd.DataFrame(m_rows).sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)
        agg["overall_rank"] = agg.index + 1
        perm_summary_list.append(agg)

    df_perm_summary = pd.concat(perm_summary_list, ignore_index=True)
    df_perm_summary.to_csv(TABLES_DIR / "e5a_permutation_importance.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5a_permutation_importance.csv'}")

    # 2. e5a_group_importance.csv
    grp_summary = (
        df_grp.groupby(["model", "cluster_id", "features", "n_features", "financial_dimension"])
        .agg(
            pr_auc_drop_mean=("pr_auc_drop", "mean"),
            pr_auc_drop_std=("pr_auc_drop", "std"),
            roc_auc_drop_mean=("roc_auc_drop", "mean"),
            roc_auc_drop_std=("roc_auc_drop", "std"),
        )
        .reset_index()
        .sort_values(["model", "pr_auc_drop_mean"], ascending=[True, False])
    )
    grp_summary.to_csv(TABLES_DIR / "e5a_group_importance.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5a_group_importance.csv'}")

    # 3. e5a_shap_importance.csv (Aggregated strictly across all 15 outer folds)
    shap_summary_list = []
    all_shap_features = sorted(df_shap["feature"].unique())

    for m in MODELS:
        sub_m = df_shap[df_shap["model"] == m]
        fold_ranks_shap = {}
        for (rep, f), g in sub_m.groupby(["repeat", "fold"]):
            sorted_g = g.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            sorted_g["rank"] = sorted_g.index + 1
            for _, r in sorted_g.iterrows():
                fold_ranks_shap[(rep, f, r["feature"])] = r["rank"]

        m_shap_rows = []
        for feat in all_shap_features:
            shap_vals_list = []
            ranks_present = []
            top20_hits = 0

            for s in selected_splits:
                rep, f = int(s["repeat"]), int(s["fold"])
                match = sub_m[(sub_m["repeat"] == rep) & (sub_m["fold"] == f) & (sub_m["feature"] == feat)]
                if len(match) > 0:
                    val = match["mean_abs_shap"].values[0]
                    rk = fold_ranks_shap.get((rep, f, feat), 999)
                    ranks_present.append(rk)
                    if rk <= 20:
                        top20_hits += 1
                else:
                    val = 0.0

                shap_vals_list.append(val)

            m_shap_rows.append(
                {
                    "model": m,
                    "feature": feat,
                    "mean_abs_shap_mean": float(np.mean(shap_vals_list)),
                    "mean_abs_shap_std": float(np.std(shap_vals_list, ddof=1)),
                    "rank_mean_present": float(np.mean(ranks_present)) if ranks_present else np.nan,
                    "in_top20_ratio": float(top20_hits / n_splits_actual),
                    "folds_present": len(ranks_present),
                    "total_folds": n_splits_actual,
                    "financial_dimension": map_financial_dimension(feat),
                }
            )

        agg_shap = pd.DataFrame(m_shap_rows).sort_values("mean_abs_shap_mean", ascending=False).reset_index(drop=True)
        agg_shap["overall_rank"] = agg_shap.index + 1
        shap_summary_list.append(agg_shap)

    df_shap_summary = pd.concat(shap_summary_list, ignore_index=True)
    df_shap_summary.to_csv(TABLES_DIR / "e5a_shap_importance.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5a_shap_importance.csv'}")

    # 4. e5a_importance_stability.csv
    stability_records = []
    for m in MODELS:
        sub_summary = df_perm_summary[df_perm_summary["model"] == m]
        for _, r in sub_summary.iterrows():
            stability_records.append(
                {
                    "model": m,
                    "feature": r["feature"],
                    "pr_auc_drop_mean": r["pr_auc_drop_mean"],
                    "in_top20_ratio": r["in_top20_ratio"],
                    "folds_present": r["folds_present"],
                    "total_folds": n_splits_actual,
                    "financial_dimension": r["financial_dimension"],
                }
            )
    df_stability = pd.DataFrame(stability_records).sort_values(["model", "pr_auc_drop_mean"], ascending=[True, False]).reset_index(drop=True)
    df_stability.to_csv(TABLES_DIR / "e5a_importance_stability.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5a_importance_stability.csv'}")

    # 5. e5a_cross_model_agreement.csv
    # Calculate Spearman correlation and Top 20 overlap across M3, M4, M5
    piv_perm = df_perm_summary.pivot(index="feature", columns="model", values="pr_auc_drop_mean").dropna()
    piv_shap = df_shap_summary.pivot(index="feature", columns="model", values="mean_abs_shap_mean").dropna()

    agreement_rows = []
    pairs = [("M3", "M4"), ("M3", "M5"), ("M4", "M5")]
    for m_a, m_b in pairs:
        sp_perm, p_perm = stats.spearmanr(piv_perm[m_a], piv_perm[m_b])
        sp_shap, p_shap = stats.spearmanr(piv_shap[m_a], piv_shap[m_b])

        top20_perm_a = set(df_perm_summary[df_perm_summary["model"] == m_a].head(20)["feature"])
        top20_perm_b = set(df_perm_summary[df_perm_summary["model"] == m_b].head(20)["feature"])
        overlap_perm = len(top20_perm_a.intersection(top20_perm_b))

        top20_shap_a = set(df_shap_summary[df_shap_summary["model"] == m_a].head(20)["feature"])
        top20_shap_b = set(df_shap_summary[df_shap_summary["model"] == m_b].head(20)["feature"])
        overlap_shap = len(top20_shap_a.intersection(top20_shap_b))

        agreement_rows.append(
            {
                "pair": f"{m_a} vs {m_b}",
                "spearman_perm_importance": float(sp_perm),
                "p_value_perm": float(p_perm),
                "overlap_top20_perm": overlap_perm,
                "spearman_shap_importance": float(sp_shap),
                "p_value_shap": float(p_shap),
                "overlap_top20_shap": overlap_shap,
            }
        )

    # 3-model intersection
    top20_all_perm = set.intersection(
        set(df_perm_summary[df_perm_summary["model"] == "M3"].head(20)["feature"]),
        set(df_perm_summary[df_perm_summary["model"] == "M4"].head(20)["feature"]),
        set(df_perm_summary[df_perm_summary["model"] == "M5"].head(20)["feature"]),
    )
    top20_all_shap = set.intersection(
        set(df_shap_summary[df_shap_summary["model"] == "M3"].head(20)["feature"]),
        set(df_shap_summary[df_shap_summary["model"] == "M4"].head(20)["feature"]),
        set(df_shap_summary[df_shap_summary["model"] == "M5"].head(20)["feature"]),
    )
    agreement_rows.append(
        {
            "pair": "M3 & M4 & M5 (All Three)",
            "spearman_perm_importance": float(np.mean([r["spearman_perm_importance"] for r in agreement_rows[:3]])),
            "p_value_perm": np.nan,
            "overlap_top20_perm": len(top20_all_perm),
            "spearman_shap_importance": float(np.mean([r["spearman_shap_importance"] for r in agreement_rows[:3]])),
            "p_value_shap": np.nan,
            "overlap_top20_shap": len(top20_all_shap),
        }
    )
    df_agreement = pd.DataFrame(agreement_rows)
    df_agreement.to_csv(TABLES_DIR / "e5a_cross_model_agreement.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5a_cross_model_agreement.csv'}")

    # ==============================================================================
    # Compile & Save E5b Tables
    # ==============================================================================
    logger.info("=== Compiling E5b Tables ===")
    df_compact_fold = pd.DataFrame(compact_fold_records)
    df_compact_fold.to_csv(TABLES_DIR / "e5b_compact_per_fold.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5b_compact_per_fold.csv'}")

    # Load Full Feature E_A reference metrics from e3b_ensemble_per_fold.csv
    e3b_per_fold = pd.read_csv(TABLES_DIR / "e3b_ensemble_per_fold.csv")
    full_EA_folds = e3b_per_fold[e3b_per_fold["ensemble"] == "E_A"].copy()

    # Merge compact and full for paired statistical testing
    paired_rows = []
    compact_summary_rows = []

    # Include full feature summary row
    compact_summary_rows.append(
        {
            "feature_set": "Full (All Preprocessed)",
            "k": len(X_train.columns),
            "loss_mean": float(full_EA_folds["outer_loss_internal"].mean()),
            "loss_std": float(full_EA_folds["outer_loss_internal"].std()),
            "ar95_mean": float(full_EA_folds["ar95_internal"].mean()),
            "ar95_std": float(full_EA_folds["ar95_internal"].std()),
            "pr_auc_mean": float(full_EA_folds["pr_auc_internal"].mean()),
            "pr_auc_std": float(full_EA_folds["pr_auc_internal"].std()),
            "recall_ar3_mean": float(full_EA_folds["rec@ar3_internal"].mean()),
            "recall_ar3_std": float(full_EA_folds["rec@ar3_internal"].std()),
            "roc_auc_mean": float(full_EA_folds["roc_auc_internal"].mean()),
            "roc_auc_std": float(full_EA_folds["roc_auc_internal"].std()),
        }
    )

    for k in k_list:
        sub_k = df_compact_fold[df_compact_fold["k"] == k].sort_values(["repeat", "fold"]).reset_index(drop=True)
        sub_full = full_EA_folds.sort_values(["repeat", "fold"]).reset_index(drop=True)

        # Ensure split alignment
        if quick:
            sub_full = sub_full[sub_full["repeat"] == 0].reset_index(drop=True)

        assert len(sub_k) == len(sub_full), f"Fold count mismatch: {len(sub_k)} vs {len(sub_full)}"

        compact_summary_rows.append(
            {
                "feature_set": f"Top {k} Features",
                "k": k,
                "loss_mean": float(sub_k["three_point_loss"].mean()),
                "loss_std": float(sub_k["three_point_loss"].std()),
                "ar95_mean": float(sub_k["ar95"].mean()),
                "ar95_std": float(sub_k["ar95"].std()),
                "pr_auc_mean": float(sub_k["pr_auc"].mean()),
                "pr_auc_std": float(sub_k["pr_auc"].std()),
                "recall_ar3_mean": float(sub_k["recall_at_ar3"].mean()),
                "recall_ar3_std": float(sub_k["recall_at_ar3"].std()),
                "roc_auc_mean": float(sub_k["roc_auc"].mean()),
                "roc_auc_std": float(sub_k["roc_auc"].std()),
            }
        )

        # Check Stop Condition 2: Ensure compact results are NOT identical to full to 4 decimals
        diff_loss = sub_k["three_point_loss"].values - sub_full["outer_loss_internal"].values
        max_abs_diff = np.max(np.abs(diff_loss))
        logger.info(f"k={k}: Max absolute loss diff vs Full: {max_abs_diff:.6f}")
        assert max_abs_diff > 1e-4, f"Compact k={k} results identical to full feature set! Check FeatureSelector."

        # Compute Paired Tests for Primary and Secondary Metrics
        diff_ar95 = sub_k["ar95"].values - sub_full["ar95_internal"].values
        diff_pr = sub_k["pr_auc"].values - sub_full["pr_auc_internal"].values
        diff_rec3 = sub_k["recall_at_ar3"].values - sub_full["rec@ar3_internal"].values

        mean_d_loss, ci_low_loss, ci_high_loss, p_loss = nadeau_bengio_paired_test(diff_loss, n_splits=5)
        mean_d_ar, ci_low_ar, ci_high_ar, p_ar = nadeau_bengio_paired_test(diff_ar95, n_splits=5)
        mean_d_pr, ci_low_pr, ci_high_pr, p_pr = nadeau_bengio_paired_test(diff_pr, n_splits=5)
        mean_d_rec3, ci_low_rec3, ci_high_rec3, p_rec3 = nadeau_bengio_paired_test(diff_rec3, n_splits=5)

        paired_rows.append(
            {
                "k": k,
                "comparison": f"Compact(k={k}) vs Full E_A",
                "loss_diff_mean": mean_d_loss,
                "loss_p_value": p_loss,
                "loss_ci_lower": ci_low_loss,
                "loss_ci_upper": ci_high_loss,
                "ar95_diff_mean": mean_d_ar,
                "ar95_p_value": p_ar,
                "ar95_ci_lower": ci_low_ar,
                "ar95_ci_upper": ci_high_ar,
                "pr_auc_diff_mean": mean_d_pr,
                "pr_auc_p_value": p_pr,
                "recall_ar3_diff_mean": mean_d_rec3,
                "recall_ar3_p_value": p_rec3,
            }
        )

    # Apply Holm-Bonferroni correction across k for primary metric (loss)
    df_paired = pd.DataFrame(paired_rows)
    raw_p_loss = df_paired["loss_p_value"].values
    adj_p_loss = holm_bonferroni(raw_p_loss)
    df_paired["loss_p_holm"] = adj_p_loss
    df_paired.to_csv(TABLES_DIR / "e5b_compact_vs_full_paired.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5b_compact_vs_full_paired.csv'}")

    df_compact_summary = pd.DataFrame(compact_summary_rows)
    df_compact_summary.to_csv(TABLES_DIR / "e5b_compact_summary.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5b_compact_summary.csv'}")

    # Feature Set Jaccard Stability across folds
    jaccard_stats = []
    n_splits_actual = len(selected_splits)
    for k in [10, 20, 30]:
        sets = fold_selected_features[k]
        j_vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                s_a, s_b = sets[i], sets[j]
                j_sim = len(s_a.intersection(s_b)) / len(s_a.union(s_b))
                j_vals.append(j_sim)
        j_mean = float(np.mean(j_vals)) if j_vals else 1.0
        j_min = float(np.min(j_vals)) if j_vals else 1.0
        j_max = float(np.max(j_vals)) if j_vals else 1.0
        jaccard_stats.append({"k": k, "jaccard_mean": j_mean, "jaccard_min": j_min, "jaccard_max": j_max})

    logger.info(f"Jaccard stability across folds: {jaccard_stats}")

    # Feature frequency across all 15 outer folds
    freq_rows = []
    for feat, count in all_features_frequency.items():
        ratio = count / n_splits_actual
        freq_rows.append(
            {
                "feature": feat,
                "frequency_in_top20": count,
                "total_folds": n_splits_actual,
                "frequency_ratio": ratio,
                "financial_dimension": map_financial_dimension(feat),
                "directly_computable": is_directly_computable(feat),
            }
        )
    df_freq = pd.DataFrame(freq_rows).sort_values("frequency_ratio", ascending=False).reset_index(drop=True)
    df_freq.to_csv(TABLES_DIR / "e5b_feature_frequency.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5b_feature_frequency.csv'}")

    # Core Features Rule: >= 80% (>= 12/15 folds). If < 10 features, relax to >= 60% (>= 9/15).
    core_rule_80 = df_freq[df_freq["frequency_ratio"] >= 0.80]
    if len(core_rule_80) >= 10:
        rule_applied = "Threshold >= 80% (appearance in >= 12 of 15 outer folds)"
        core_df = core_rule_80.copy()
    else:
        rule_applied = f"Threshold relaxed to >= 60% (appearance in >= 9 of 15 folds; >= 80% yielded only {len(core_rule_80)} features)"
        core_df = df_freq[df_freq["frequency_ratio"] >= 0.60].copy()

    core_df["selection_rule"] = rule_applied
    core_df.to_csv(TABLES_DIR / "e5b_core_features.csv", index=False)
    logger.info(f"Saved {TABLES_DIR / 'e5b_core_features.csv'} ({len(core_df)} core features under rule: {rule_applied})")

    # Save outputs/models/final/core_features.txt (plain text, strictly one feature per line)
    core_txt_path = FINAL_DIR / "core_features.txt"
    with open(core_txt_path, "w", encoding="utf-8") as f_out:
        for feat in core_df["feature"]:
            f_out.write(f"{feat}\n")
    logger.info(f"Saved {core_txt_path}")

    # ==============================================================================
    # Figures Generation (English titles and labels)
    # ==============================================================================
    logger.info("=== Generating E5 Figures ===")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: e5_permutation_top20.png
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)
    for idx, m in enumerate(MODELS):
        sub_top = df_perm_summary[df_perm_summary["model"] == m].head(20).sort_values("pr_auc_drop_mean", ascending=True)
        ax = axes[idx]
        y_pos = np.arange(len(sub_top))
        ax.barh(y_pos, sub_top["pr_auc_drop_mean"], xerr=sub_top["pr_auc_drop_std"], color="#2b5c8f", alpha=0.85, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_top["feature"], fontsize=9)
        ax.set_title(f"Model {m}: Top 20 Permutation Importance", fontsize=12, pad=10)
        ax.set_xlabel("PR-AUC Drop (5 Repeats, Outer Val Folds)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "e5_permutation_top20.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig1_path}")

    # Figure 2: e5_shap_summary_<model>.png
    for m in MODELS:
        if len(shap_plot_cache[m]["shap_vals"]) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_plot_cache[m]["shap_vals"],
                features=shap_plot_cache[m]["X_eval"],
                feature_names=shap_plot_cache[m]["feat_names"],
                max_display=20,
                show=False,
            )
            plt.title(f"SHAP Summary (TreeExplainer) - Model {m}", fontsize=14, pad=12)
            plt.tight_layout()
            fig2_path = FIGURES_DIR / f"e5_shap_summary_{m}.png"
            plt.savefig(fig2_path, dpi=300)
            plt.close()
            logger.info(f"Saved {fig2_path}")

    # Figure 3: e5_importance_stability.png
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"M3": "#2b5c8f", "M4": "#d95f02", "M5": "#7570b3"}
    for m in MODELS:
        sub_m = df_stability[df_stability["model"] == m].head(25).copy()
        sub_m["rank"] = np.arange(1, len(sub_m) + 1)
        ax.scatter(sub_m["rank"], sub_m["in_top20_ratio"], label=m, color=colors[m], s=50, alpha=0.8)
    ax.set_title("Feature Importance Stability Across 15 Outer Folds", fontsize=14, pad=12)
    ax.set_xlabel("Overall Permutation Rank (1 = Most Important)", fontsize=11)
    ax.set_ylabel("Selection Frequency in Top 20 (Folds / 15)", fontsize=11)
    ax.axhline(0.80, color="red", linestyle="--", alpha=0.6, label="80% Stability Threshold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "e5_importance_stability.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig3_path}")

    # Figure 4: e5_shap_dependence_top10.png
    # Find Top 10 features for E_A (based on average SHAP rank across M3, M4, M5)
    piv_rank_shap = df_shap_summary.pivot(index="feature", columns="model", values="overall_rank")
    piv_rank_shap["ea_rank"] = piv_rank_shap.mean(axis=1)
    top10_ea_feats = piv_rank_shap.sort_values("ea_rank").head(10).index.tolist()

    # Use M4 or M5 SHAP cache for dependence plots
    ref_model = "M4"
    if len(shap_plot_cache[ref_model]["shap_vals"]) > 0:
        fig, axes = plt.subplots(5, 2, figsize=(14, 18))
        axes = axes.flatten()
        feat_list = shap_plot_cache[ref_model]["feat_names"]
        vals = shap_plot_cache[ref_model]["shap_vals"]
        X_mat = shap_plot_cache[ref_model]["X_eval"]

        for i, fname in enumerate(top10_ea_feats):
            ax = axes[i]
            if fname in feat_list:
                f_idx = feat_list.index(fname)
                x_feat = X_mat[:, f_idx]
                s_feat = vals[:, f_idx]
                ax.scatter(x_feat, s_feat, alpha=0.6, color="#d95f02", s=25)
                # Fit lowess / trend line
                sort_idx = np.argsort(x_feat)
                ax.plot(x_feat[sort_idx], np.convolve(s_feat[sort_idx], np.ones(15)/15, mode="same"), color="black", linewidth=1.5)
                ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
                ax.set_title(f"{i+1}. {fname}", fontsize=10)
                ax.set_xlabel("Normalized Feature Value", fontsize=9)
                ax.set_ylabel("SHAP Value (Log-Odds Impact)", fontsize=9)
            else:
                ax.text(0.5, 0.5, f"{fname}\n(Not in {ref_model})", ha="center", va="center")
        plt.suptitle("Top 10 Features SHAP Dependence Plots (Risk Direction Analysis)", fontsize=14, y=0.995)
        plt.tight_layout()
        fig4_path = FIGURES_DIR / "e5_shap_dependence_top10.png"
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        logger.info(f"Saved {fig4_path}")

    # Figure 5: e5_compact_vs_full.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # AR@0.95 plot
    all_k_labels = [f"k={k}" for k in k_list] + ["Full"]
    ar95_data = [df_compact_fold[df_compact_fold["k"] == k]["ar95"].values for k in k_list]
    ar95_data.append(full_EA_folds["ar95_internal"].values[:len(ar95_data[0])])

    pr_data = [df_compact_fold[df_compact_fold["k"] == k]["pr_auc"].values for k in k_list]
    pr_data.append(full_EA_folds["pr_auc_internal"].values[:len(pr_data[0])])

    # Plot AR@0.95
    for idx_k, (lbl, vals) in enumerate(zip(all_k_labels, ar95_data)):
        x_jitter = np.random.RandomState(42).normal(idx_k + 1, 0.05, size=len(vals))
        ax1.scatter(x_jitter, vals, alpha=0.5, color="#2b5c8f", s=35)
        ax1.plot([idx_k + 0.8, idx_k + 1.2], [np.mean(vals), np.mean(vals)], color="red", linewidth=2.5)

    ax1.set_xticks(range(1, len(all_k_labels) + 1))
    ax1.set_xticklabels(all_k_labels, fontsize=11)
    ax1.set_title("AR@0.95 across Folds (Red Line = Mean)", fontsize=13)
    ax1.set_ylabel("Alert Ratio at 95% Recall (Lower is Better)", fontsize=11)

    # Plot PR-AUC
    for idx_k, (lbl, vals) in enumerate(zip(all_k_labels, pr_data)):
        x_jitter = np.random.RandomState(42).normal(idx_k + 1, 0.05, size=len(vals))
        ax2.scatter(x_jitter, vals, alpha=0.5, color="#2b5c8f", s=35)
        ax2.plot([idx_k + 0.8, idx_k + 1.2], [np.mean(vals), np.mean(vals)], color="red", linewidth=2.5)

    ax2.set_xticks(range(1, len(all_k_labels) + 1))
    ax2.set_xticklabels(all_k_labels, fontsize=11)
    ax2.set_title("PR-AUC across Folds (Red Line = Mean)", fontsize=13)
    ax2.set_ylabel("PR-AUC (Higher is Better)", fontsize=11)

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "e5_compact_vs_full.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig5_path}")

    logger.info("=== E5 Execution Completed Successfully ===")
    return {
        "compact_summary": df_compact_summary,
        "paired_tests": df_paired,
        "core_features_count": len(core_df),
        "rule_applied": rule_applied,
        "jaccard_stability": jaccard_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E5 Feature Importance & Compact Sets")
    parser.add_argument("--quick", action="store_true", help="Quick mode on repeat 0 only")
    parser.add_argument("--force", action="store_true", help="Force recalculation ignoring cache")
    args = parser.parse_args()

    run_e5(quick=args.quick, force=args.force)
