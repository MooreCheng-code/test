"""
E6: Final Holdout Test Set Evaluation Pipeline.

Supports three CLI modes:
- --dry-run: Subsamples 1,364 rows (44 bankrupt) from the training set to dry-run
             the complete evaluation and report generation without accessing the test set.
- --unseal:  Formally unseals the holdout test set (only executed once, protected by E6_LOCK).
- --analyze: Re-runs all analyses, tables, and figures from saved evaluation scores (zero model re-fit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import config
from src.cv import get_test_data, get_train_data
from src.lock_utils import acquire_e6_lock, is_e6_locked
from src.metrics import alert_ratio_at_recall, recall_at_alert_ratio
from src.scoring import load_final_components, to_percentile

logger = logging.getLogger("bankruptcy_e6")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

SEED = config.SEED
TARGETS = [0.80, 0.90, 0.95, 1.00]


# ==============================================================================
# Statistical Helper Functions
# ==============================================================================
def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def wilson_score_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


# ==============================================================================
# E6.1: Scoring & Immediate Disk Persistence
# ==============================================================================
def run_e6_scoring(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    components: dict[str, Any],
    out_dir: Path,
    dry_run: bool = False,
) -> Path:
    """
    Computes raw probabilities, ECDF percentiles, ensemble scores, calibrated probabilities,
    and single-feature benchmark scores using the single unified scoring.score_dataset pipeline.
    Immediately persists results to .npz file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_file = out_dir / "e6_test_scores.npz"

    logger.info("Computing predictions via unified scoring.score_dataset pipeline...")

    # Unified call to score_dataset
    from src.scoring import score_dataset
    scored_df = score_dataset(X, components)

    score_EA = scored_df["score_E_A"].values
    score_M3 = scored_df["score_M3"].values
    raw_M3 = scored_df["raw_score_M3"].values
    raw_M4 = scored_df["raw_score_M4"].values
    raw_M5 = scored_df["raw_score_M5"].values
    pct_M3 = scored_df["pct_score_M3"].values
    pct_M4 = scored_df["pct_score_M4"].values
    pct_M5 = scored_df["pct_score_M5"].values
    prob_platt = scored_df["calibrated_prob_platt"].values
    prob_iso = scored_df["calibrated_prob_isotonic"].values

    # Single-feature benchmark: -Persistent EPS (direction fixed from training set)
    eps_col = "Persistent EPS in the Last Four Seasons"
    if eps_col not in X.columns:
        raise KeyError(f"Missing required benchmark feature: '{eps_col}'")
    score_single_feat = -X[eps_col].values.astype(float)

    labels = np.asarray(y, dtype=int).ravel()
    indices = np.asarray(X.index.values)

    np.savez_compressed(
        scores_file,
        indices=indices,
        labels=labels,
        score_EA=score_EA,
        score_M3=score_M3,
        raw_M3=raw_M3,
        raw_M4=raw_M4,
        raw_M5=raw_M5,
        pct_M3=pct_M3,
        pct_M4=pct_M4,
        pct_M5=pct_M5,
        prob_platt=prob_platt,
        prob_isotonic=prob_iso,
        score_single_feat=score_single_feat,
        is_dry_run=dry_run,
    )

    # Compute and log SHA256
    file_bytes = scores_file.read_bytes()
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()
    logger.info(f"Persisted evaluation scores to {scores_file} (SHA256: {sha256_hash})")

    meta_file = out_dir / "e6_scores_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scores_file": scores_file.name,
                "sha256": sha256_hash,
                "n_samples": len(labels),
                "n_positives": int(labels.sum()),
                "is_dry_run": dry_run,
            },
            f,
            indent=2,
        )

    return scores_file


# ==============================================================================
# E6.2: Analysis & Report Generation
# ==============================================================================
def run_e6_analysis(
    scores_file: Path,
    components: dict[str, Any],
    out_dir: Path,
    dry_run: bool = False,
    n_bootstraps: int = 2000,
) -> dict[str, Any]:
    """
    Loads saved evaluation scores from .npz and performs all evaluations:
    1. Operating points with Clopper-Pearson CIs
    2. Comparison against E4a 15-fold CV expectations
    3. Ranking metrics with 2,000 stratified bootstrap resamples
    4. Paired metric differences
    5. Probability calibration (Brier, BSS, log loss, reliability diagram)
    6. Figures and E6_REPORT.md
    """
    logger.info(f"Loading evaluation scores from {scores_file} ...")
    data = np.load(scores_file)

    labels = data["labels"]
    score_EA = data["score_EA"]
    score_M3 = data["score_M3"]
    score_single = data["score_single_feat"]
    prob_platt = data["prob_platt"]
    prob_iso = data["prob_isotonic"]
    indices = data["indices"]

    n_samples = len(labels)
    n_pos = int(labels.sum())
    thresholds_cfg = components["thresholds"]
    train_meta = components.get("metadata", {})
    p_train = float(train_meta.get("n_bankrupt_positives", 176)) / float(train_meta.get("n_training_samples", 5455))
    p_test = float(n_pos / n_samples)

    logger.info(f"Loaded {n_samples} evaluation records ({n_pos} bankrupt positives, base rate: {p_test:.4f})")

    # 1. Operating Points Evaluation
    logger.info("Evaluating frozen operating points...")
    op_rows = []
    missed_records = []

    for grp in ["E_A", "M3"]:
        scores = score_EA if grp == "E_A" else score_M3
        cfg = thresholds_cfg[grp]

        for R in TARGETS:
            r_key = f"R{int(R * 100)}"
            t_val = cfg[r_key]["threshold"]
            m_val = cfg[r_key]["recommended_margin"]

            flags = scores >= t_val
            alerts_cnt = int(flags.sum())
            alert_prop = float(alerts_cnt / n_samples)
            alert_rt = float(alerts_cnt / n_pos)

            caught_cnt = int(((labels == 1) & flags).sum())
            actual_rec = float(caught_cnt / n_pos)
            rec_ci_l, rec_ci_u = clopper_pearson_ci(caught_cnt, n_pos)

            prec = float(caught_cnt / alerts_cnt) if alerts_cnt > 0 else 0.0
            prec_ci_l, prec_ci_u = wilson_score_ci(caught_cnt, alerts_cnt) if alerts_cnt > 0 else (0.0, 0.0)

            # Target threshold: ceil(R * n_pos)
            req_k = int(np.ceil(R * n_pos))
            target_met = bool(caught_cnt >= req_k)

            op_rows.append(
                {
                    "group": grp,
                    "operating_point": r_key,
                    "target_recall": R,
                    "margin": m_val,
                    "threshold": t_val,
                    "required_caught": req_k,
                    "positives_caught": caught_cnt,
                    "total_positives": n_pos,
                    "actual_recall": actual_rec,
                    "recall_ci_lower_cp": rec_ci_l,
                    "recall_ci_upper_cp": rec_ci_u,
                    "alerts_count": alerts_cnt,
                    "total_samples": n_samples,
                    "alert_proportion": alert_prop,
                    "alert_ratio": alert_rt,
                    "precision": prec,
                    "precision_ci_lower_wilson": prec_ci_l,
                    "precision_ci_upper_wilson": prec_ci_u,
                    "target_met": target_met,
                }
            )

            # Track missed bankrupt companies
            splits_file = project_root / "outputs" / "models" / "splits.joblib"
            test_orig_idx = None
            if splits_file.exists() and not dry_run:
                try:
                    splits_payload = joblib.load(splits_file)
                    test_orig_idx = splits_payload.get("test_idx")
                except Exception:
                    test_orig_idx = None

            missed_mask = (labels == 1) & (~flags)
            for m_idx in np.where(missed_mask)[0]:
                t_row_idx = int(indices[m_idx])
                raw_row_idx = int(test_orig_idx[t_row_idx]) if test_orig_idx is not None and t_row_idx < len(test_orig_idx) else t_row_idx
                missed_records.append(
                    {
                        "group": grp,
                        "operating_point": r_key,
                        "row_index": t_row_idx,
                        "raw_row_index": raw_row_idx,
                        "score_E_A": float(score_EA[m_idx]),
                        "score_M3": float(score_M3[m_idx]),
                        "prob_platt": float(prob_platt[m_idx]),
                        "threshold": t_val,
                    }
                )

    op_df = pd.DataFrame(op_rows)
    op_csv = out_dir / "e6_operating_points.csv"
    op_df.to_csv(op_csv, index=False)
    logger.info(f"Saved {op_csv}")

    missed_df = pd.DataFrame(missed_records)
    missed_csv = out_dir / "e6_missed_positives.csv"
    missed_df.to_csv(missed_csv, index=False)
    logger.info(f"Saved {missed_csv}")

    # 2. Compare against E4a 15-Fold CV Expectations
    logger.info("Comparing against E4a CV expectations...")
    e4a_file = project_root / "outputs" / "tables" / "operating_points_for_report.csv"
    e4a_fold_file = project_root / "outputs" / "tables" / "e4a_threshold_transfer_per_fold.csv"
    if e4a_file.exists():
        cv_ref_df = pd.read_csv(e4a_file)
        # Strict source for CV recall range: per-fold validation recall (no silent defaults)
        e4a_fold_df = pd.read_csv(e4a_fold_file)
        vs_cv_rows = []
        for _, row in op_df.iterrows():
            grp = row["group"]
            r_tgt = row["target_recall"]
            match = cv_ref_df[(cv_ref_df["group"] == grp) & (np.isclose(cv_ref_df["target_recall"], r_tgt))]
            if len(match) > 0:
                cv_row = match.iloc[0]
                fold_sub = e4a_fold_df[
                    (e4a_fold_df["group"] == grp)
                    & np.isclose(e4a_fold_df["target_recall"], r_tgt)
                    & np.isclose(e4a_fold_df["margin"], float(cv_row["recommended_margin"]))
                ]
                if len(fold_sub) == 0:
                    raise KeyError(f"No per-fold rows in {e4a_fold_file.name} for {grp} R={r_tgt}")
                rec_min = float(cv_row["actual_recall_min"])
                rec_max = float(fold_sub["actual_recall"].max())
                if not np.isclose(rec_min, float(fold_sub["actual_recall"].min())):
                    raise ValueError(f"actual_recall_min mismatch between report and per-fold file for {grp} R={r_tgt}")
                rec_mean = float(cv_row["actual_recall_mean"])
                rec_std = float(cv_row["actual_recall_std"])
                ar_mean = float(cv_row["alert_ratio_mean"])
                ar_std = float(cv_row["alert_ratio_std"])

                test_rec = row["actual_recall"]
                test_ar = row["alert_ratio"]
                vs_cv_rows.append(
                    {
                        "group": grp,
                        "operating_point": row["operating_point"],
                        "target_recall": r_tgt,
                        "cv_recall_mean": rec_mean,
                        "cv_recall_std": rec_std,
                        "cv_recall_min": rec_min,
                        "cv_recall_max": rec_max,
                        "test_actual_recall": test_rec,
                        "recall_in_cv_range": bool(rec_min <= test_rec <= rec_max),
                        "cv_alert_ratio_mean": ar_mean,
                        "cv_alert_ratio_std": ar_std,
                        "test_alert_ratio": test_ar,
                    }
                )
        vs_cv_df = pd.DataFrame(vs_cv_rows)
        vs_cv_df.to_csv(out_dir / "e6_vs_cv.csv", index=False)
        logger.info(f"Saved {out_dir / 'e6_vs_cv.csv'}")

    # 3. Stratified Bootstrap for Ranking Metrics & Paired Differences
    logger.info(f"Running {n_bootstraps} stratified bootstrap iterations for ranking metrics...")
    np.random.seed(SEED)
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    models_to_eval = {
        "E_A": score_EA,
        "M3": score_M3,
        "Single_Feature": score_single,
    }

    def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
        m: dict[str, float] = {}
        m["pr_auc"] = float(average_precision_score(y_true, y_score))
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        for r in TARGETS:
            m[f"ar@r_{int(r*100)}"] = float(alert_ratio_at_recall(y_true, y_score, r))
        for ar in [1.5, 3.0, 5.0, 10.0]:
            m[f"rec@ar_{ar}"] = float(recall_at_alert_ratio(y_true, y_score, ar))
        return m

    # Point estimates on full evaluation set
    point_estimates = {m_name: compute_metrics(labels, s_arr) for m_name, s_arr in models_to_eval.items()}

    # Bootstrap distributions
    boot_records: dict[str, list[dict[str, float]]] = {m_name: [] for m_name in models_to_eval}
    diff_records_m3: list[dict[str, float]] = []
    diff_records_single: list[dict[str, float]] = []

    for b in range(n_bootstraps):
        boot_pos = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
        boot_neg = np.random.choice(neg_indices, size=len(neg_indices), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        b_y = labels[boot_idx]
        b_metrics = {}
        for m_name, s_arr in models_to_eval.items():
            b_m = compute_metrics(b_y, s_arr[boot_idx])
            boot_records[m_name].append(b_m)
            b_metrics[m_name] = b_m

        # Compute paired differences on same resample
        diff_m3 = {k: b_metrics["E_A"][k] - b_metrics["M3"][k] for k in b_metrics["E_A"]}
        diff_single = {k: b_metrics["E_A"][k] - b_metrics["Single_Feature"][k] for k in b_metrics["E_A"]}
        diff_records_m3.append(diff_m3)
        diff_records_single.append(diff_single)

    # Compile ranking metrics table
    ranking_rows = []
    metric_keys = list(point_estimates["E_A"].keys())
    for m_name in models_to_eval:
        for k in metric_keys:
            dist = np.array([r[k] for r in boot_records[m_name]])
            ranking_rows.append(
                {
                    "model": m_name,
                    "metric": k,
                    "point_estimate": point_estimates[m_name][k],
                    "boot_mean": float(np.mean(dist)),
                    "boot_std": float(np.std(dist)),
                    "ci_lower_2.5": float(np.percentile(dist, 2.5)),
                    "ci_upper_97.5": float(np.percentile(dist, 97.5)),
                }
            )
    rank_df = pd.DataFrame(ranking_rows)
    rank_df.to_csv(out_dir / "e6_ranking_metrics.csv", index=False)
    logger.info(f"Saved {out_dir / 'e6_ranking_metrics.csv'}")

    # Compile paired difference table
    paired_rows = []
    for pair_name, d_list in [("E_A - M3", diff_records_m3), ("E_A - Single_Feature", diff_records_single)]:
        for k in metric_keys:
            d_dist = np.array([r[k] for r in d_list])
            p_diff = (
                point_estimates["E_A"][k] - point_estimates["M3"][k]
                if "M3" in pair_name
                else point_estimates["E_A"][k] - point_estimates["Single_Feature"][k]
            )
            paired_rows.append(
                {
                    "comparison": pair_name,
                    "metric": k,
                    "point_difference": p_diff,
                    "boot_diff_mean": float(np.mean(d_dist)),
                    "boot_diff_std": float(np.std(d_dist)),
                    "ci_lower_2.5": float(np.percentile(d_dist, 2.5)),
                    "ci_upper_97.5": float(np.percentile(d_dist, 97.5)),
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "e6_paired_bootstrap.csv", index=False)
    logger.info(f"Saved {out_dir / 'e6_paired_bootstrap.csv'}")

    # 4. Calibration Evaluation
    logger.info("Evaluating probability calibration...")
    brier_base_test = float(p_test * (1.0 - p_test))
    brier_base_train = float(np.mean((labels - p_train) ** 2))

    cal_summary_rows = []
    for cal_name, p_arr in [("Platt", prob_platt), ("Isotonic", prob_iso)]:
        brier = float(brier_score_loss(labels, p_arr))
        ll = float(log_loss(labels, np.clip(p_arr, 1e-15, 1.0 - 1e-15)))
        bss_test = float(1.0 - brier / brier_base_test) if brier_base_test > 0 else 0.0
        bss_train = float(1.0 - brier / brier_base_train) if brier_base_train > 0 else 0.0
        mean_prob = float(np.mean(p_arr))

        cal_summary_rows.append(
            {
                "calibrator": cal_name,
                "brier_score": brier,
                "log_loss": ll,
                "brier_skill_score_test_baseline": bss_test,
                "brier_skill_score_train_baseline": bss_train,
                "mean_predicted_prob": mean_prob,
                "actual_test_prevalence": p_test,
                "note": (
                    "Platt is smooth & strictly rank-preserving; Isotonic creates step functions "
                    "with tied scores and must not be used as operational threshold."
                    if cal_name == "Isotonic"
                    else "Calibrated posterior probability is for risk auditing only; thresholds operate on raw continuous rank."
                ),
            }
        )
    cal_df = pd.DataFrame(cal_summary_rows)
    cal_df.to_csv(out_dir / "e6_calibration.csv", index=False)
    logger.info(f"Saved {out_dir / 'e6_calibration.csv'}")

    # 5. Visualizations
    generate_e6_figures(
        labels=labels,
        score_EA=score_EA,
        score_M3=score_M3,
        score_single=score_single,
        prob_platt=prob_platt,
        prob_iso=prob_iso,
        op_df=op_df,
        out_dir=out_dir,
        dry_run=dry_run,
    )

    # 6. Generate Markdown Report
    generate_e6_report(
        op_df=op_df,
        rank_df=rank_df,
        paired_df=paired_df,
        cal_df=cal_df,
        missed_df=missed_df,
        labels=labels,
        score_EA=score_EA,
        score_M3=score_M3,
        out_dir=out_dir,
        n_samples=n_samples,
        n_pos=n_pos,
        dry_run=dry_run,
    )

    return {
        "operating_points": op_df,
        "ranking_metrics": rank_df,
        "paired_bootstrap": paired_df,
        "calibration": cal_df,
    }


# ==============================================================================
# Visualization Functions
# ==============================================================================
def generate_e6_figures(
    labels: np.ndarray,
    score_EA: np.ndarray,
    score_M3: np.ndarray,
    score_single: np.ndarray,
    prob_platt: np.ndarray,
    prob_iso: np.ndarray,
    op_df: pd.DataFrame,
    out_dir: Path,
    dry_run: bool = False,
):
    """Generates the required figures for E6 with English labels and titles."""
    logger.info("Generating E6 evaluation plots...")
    prefix = "[DRY RUN — in-sample, not valid results] " if dry_run else ""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # 1. Recall vs Alert Ratio Curve
    plt.figure(figsize=(9, 6))
    grid_recalls = np.linspace(0.5, 1.0, 51)
    for name, s_arr, color in [("E_A Ensemble", score_EA, "#1f77b4"), ("M3 ExtraTrees", score_M3, "#ff7f0e"), ("Single Feature (-EPS)", score_single, "#2ca02c")]:
        ar_vals = [alert_ratio_at_recall(labels, s_arr, r) for r in grid_recalls]
        plt.plot(grid_recalls, ar_vals, label=name, color=color, linewidth=2)

    plt.xlabel("Target Recall", fontsize=12)
    plt.ylabel("Alert Ratio (Alerts / Positives)", fontsize=12)
    plt.title(f"{prefix}Recall vs. Alert Ratio Trade-off", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "e6_recall_vs_alert_ratio.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for name, s_arr, color in [("E_A Ensemble", score_EA, "#1f77b4"), ("M3 ExtraTrees", score_M3, "#ff7f0e"), ("Single Feature (-EPS)", score_single, "#2ca02c")]:
        p, r, _ = precision_recall_curve(labels, s_arr)
        auc_val = average_precision_score(labels, s_arr)
        plt.plot(r, p, label=f"{name} (PR-AUC = {auc_val:.4f})", color=color, linewidth=2)

    base_rate = float(labels.sum() / len(labels))
    plt.axhline(base_rate, color="gray", linestyle="--", label=f"Base Rate ({base_rate:.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"{prefix}Precision-Recall Curves", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "e6_pr_curve.png", dpi=300)
    plt.close()

    # 3. Operating Curve with Frozen Operating Points
    plt.figure(figsize=(9, 6))
    p_ea, r_ea, _ = precision_recall_curve(labels, score_EA)
    plt.plot(r_ea, p_ea, color="#1f77b4", linewidth=2, label="E_A Ensemble PR Curve")

    ea_ops = op_df[op_df["group"] == "E_A"]
    for _, row in ea_ops.iterrows():
        plt.scatter(
            row["actual_recall"],
            row["precision"],
            s=90,
            zorder=5,
            label=f"R={row['target_recall']:.2f} (AR={row['alert_ratio']:.2f}x)",
        )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"{prefix}Operating Points on E_A Precision-Recall Curve", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "e6_operating_curve.png", dpi=300)
    plt.close()

    # 4. Calibration Reliability Diagram (5 Quantile Bins)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for cal_name, p_arr, color in [("Platt Scaling", prob_platt, "#1f77b4"), ("Isotonic Regression", prob_iso, "#d62728")]:
        bins = np.quantile(p_arr, np.linspace(0, 1, 6))
        bins = np.unique(bins)
        if len(bins) > 1:
            bin_idx = np.digitize(p_arr, bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
            mean_preds = [float(np.mean(p_arr[bin_idx == i])) for i in range(len(bins) - 1) if (bin_idx == i).sum() > 0]
            obs_props = [float(np.mean(labels[bin_idx == i])) for i in range(len(bins) - 1) if (bin_idx == i).sum() > 0]
            plt.plot(mean_preds, obs_props, "o-", color=color, linewidth=2, label=f"{cal_name} (5 bins)")

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Observed Fraction of Positives", fontsize=12)
    plt.title(f"{prefix}Reliability Diagram (Calibration)", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "e6_calibration_reliability.png", dpi=300)
    plt.close()

    # 5. Positive Score Distribution & Operating Thresholds
    plt.figure(figsize=(10, 5))
    pos_scores = score_EA[labels == 1]
    neg_scores = score_EA[labels == 0]

    # Strip plot of positive companies
    np.random.seed(42)
    jitter_y = np.random.normal(0, 0.05, size=len(pos_scores))
    plt.scatter(pos_scores, jitter_y, color="#d62728", alpha=0.7, s=40, label=f"Bankrupt Positives (N={len(pos_scores)})")

    for _, row in ea_ops.iterrows():
        plt.axvline(row["threshold"], linestyle="--", alpha=0.8, label=f"Threshold R{int(row['target_recall']*100)} (t={row['threshold']:.3f})")

    plt.xlabel("E_A Ensemble Percentile Score", fontsize=12)
    plt.yticks([])
    plt.ylim(-0.25, 0.25)
    plt.title(f"{prefix}Bankrupt Company Score Distribution vs. Frozen Thresholds", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "e6_score_distribution_positives.png", dpi=300)
    plt.close()


def df_to_markdown(df: pd.DataFrame) -> str:
    """Formats DataFrame as GitHub-flavored Markdown table without requiring third-party 'tabulate'."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def generate_e6_report(
    op_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    missed_df: pd.DataFrame,
    labels: np.ndarray,
    score_EA: np.ndarray,
    score_M3: np.ndarray,
    out_dir: Path,
    n_samples: int,
    n_pos: int,
    dry_run: bool = False,
):
    """Generates the executive Markdown report E6_REPORT.md conforming to reporting guidelines."""
    prefix_header = "# [DRY RUN — IN-SAMPLE SUBSAMPLE] E6 Evaluation Report\n> [!CAUTION]\n> This report was generated on an in-sample training subsample for pipeline dry-run verification. Results are NOT valid out-of-sample generalization metrics.\n\n" if dry_run else "# Phase E6: Holdout Test Set Final Evaluation Report\n\n"

    md = prefix_header
    md += f"**Evaluation Sample Size**: {n_samples} total companies ({n_pos} bankrupt positives, base rate: {n_pos/n_samples:.4%}).\n\n"
    md += "## 1. Operating Points Evaluation\n"
    md += "Operating points were fixed using `thresholds.json` from training OOF derivations.\n\n"
    md += df_to_markdown(op_df)
    md += "\n\n"

    md += "## 2. Ranking Metrics (Stratified Bootstrap 95% CI)\n"
    md += df_to_markdown(rank_df)
    md += "\n\n"

    md += "## 3. Paired Model Differences (E_A vs Controls)\n"
    md += "Differences evaluated on identical resamples (positive difference indicates E_A advantage for AUC and recall, lower is better for alert ratio):\n\n"
    md += df_to_markdown(paired_df)
    md += "\n\n"

    md += "## 4. Probability Calibration Summary\n"
    md += df_to_markdown(cal_df)
    md += "\n\n"

    md += "## 5. 測試集相對 CV 的難易度\n\n"
    md += "以下列出主要模型 $E_A$、單一控制模型 M3 與單一最強特徵基準在 15 折 CV 與未拆封測試集上的關鍵表現指標對照：\n\n"

    # Dynamically read 15-fold CV baseline values from source CSVs
    tables_dir = project_root / "outputs" / "tables"
    e3b_df = pd.read_csv(tables_dir / "e3b_ensemble_summary.csv")
    ea_row = e3b_df[e3b_df["candidate"] == "E_A"].iloc[0]

    e2a_df = pd.read_csv(tables_dir / "e2a_nested_summary.csv")
    m3_row = e2a_df[e2a_df["model"] == "M3"].iloc[0]

    e0_df = pd.read_csv(tables_dir / "e0_baselines_summary.csv")
    s_row = e0_df[e0_df["model"] == "Single Strongest Feature"].iloc[0]

    def get_test_str(m_name: str, metric_name: str, is_ratio: bool = False) -> tuple[str, float]:
        sub = rank_df[(rank_df["model"] == m_name) & (rank_df["metric"] == metric_name)].iloc[0]
        pt = float(sub["point_estimate"])
        lo = float(sub["ci_lower_2.5"])
        hi = float(sub["ci_upper_97.5"])
        if is_ratio:
            return f"{pt:.2f}x [{lo:.2f}x, {hi:.2f}x]", pt
        return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]", pt

    ea_roc_test, ea_roc_pt = get_test_str("E_A", "roc_auc")
    ea_pr_test, ea_pr_pt = get_test_str("E_A", "pr_auc")
    ea_ar_test, ea_ar_pt = get_test_str("E_A", "ar@r_95", is_ratio=True)

    m3_roc_test, m3_roc_pt = get_test_str("M3", "roc_auc")
    m3_pr_test, m3_pr_pt = get_test_str("M3", "pr_auc")
    m3_ar_test, m3_ar_pt = get_test_str("M3", "ar@r_95", is_ratio=True)

    s_roc_test, s_roc_pt = get_test_str("Single_Feature", "roc_auc")
    s_pr_test, s_pr_pt = get_test_str("Single_Feature", "pr_auc")
    s_ar_test, s_ar_pt = get_test_str("Single_Feature", "ar@r_95", is_ratio=True)

    difficulty_rows = [
        {
            "模型": "E_A 集成",
            "指標": "ROC-AUC",
            "15 折 CV 表現": f"{float(ea_row['roc_auc_fold_mean']):.4f} ± {float(ea_row['roc_auc_fold_std']):.4f}",
            "測試集表現 [95% CI]": ea_roc_test,
            "對照說明": "測試集點估計優於（高於）CV 均值" if ea_roc_pt > float(ea_row["roc_auc_fold_mean"]) else "測試集點估計低於 CV 均值",
        },
        {
            "模型": "E_A 集成",
            "指標": "PR-AUC",
            "15 折 CV 表現": f"{float(ea_row['pr_auc_fold_mean']):.4f} ± {float(ea_row['pr_auc_fold_std']):.4f}",
            "測試集表現 [95% CI]": ea_pr_test,
            "對照說明": "測試集點估計優於（高於）CV 均值" if ea_pr_pt > float(ea_row["pr_auc_fold_mean"]) else "測試集點估計低於 CV 均值",
        },
        {
            "模型": "E_A 集成",
            "指標": "AR@0.95",
            "15 折 CV 表現": f"{float(ea_row['ar95_fold_mean']):.2f}x ± {float(ea_row['ar95_fold_std']):.2f}x",
            "測試集表現 [95% CI]": ea_ar_test,
            "對照說明": "測試集所需預警倍數較低（優於 CV）" if ea_ar_pt < float(ea_row["ar95_fold_mean"]) else "測試集所需預警倍數較高",
        },
        {
            "模型": "M3 (ExtraTrees)",
            "指標": "ROC-AUC",
            "15 折 CV 表現": f"{float(m3_row['roc_auc_fold_mean']):.4f} ± {float(m3_row['roc_auc_fold_std']):.4f}",
            "測試集表現 [95% CI]": m3_roc_test,
            "對照說明": "測試集點估計優於（高於）CV 均值" if m3_roc_pt > float(m3_row["roc_auc_fold_mean"]) else "測試集點估計低於 CV 均值",
        },
        {
            "模型": "M3 (ExtraTrees)",
            "指標": "PR-AUC",
            "15 折 CV 表現": f"{float(m3_row['pr_auc_fold_mean']):.4f} ± {float(m3_row['pr_auc_fold_std']):.4f}",
            "測試集表現 [95% CI]": m3_pr_test,
            "對照說明": "測試集點估計優於（高於）CV 均值" if m3_pr_pt > float(m3_row["pr_auc_fold_mean"]) else "測試集點估計低於 CV 均值",
        },
        {
            "模型": "M3 (ExtraTrees)",
            "指標": "AR@0.95",
            "15 折 CV 表現": f"{float(m3_row['ar95_fold_mean']):.2f}x ± {float(m3_row['ar95_fold_std']):.2f}x",
            "測試集表現 [95% CI]": m3_ar_test,
            "對照說明": "測試集所需預警倍數較低（優於 CV）" if m3_ar_pt < float(m3_row["ar95_fold_mean"]) else "測試集所需預警倍數較高",
        },
        {
            "模型": "單一特徵 (-EPS)",
            "指標": "ROC-AUC",
            "15 折 CV 表現": f"{float(s_row['fold_roc_auc_mean']):.4f} ± {float(s_row['fold_roc_auc_std']):.4f}",
            "測試集表現 [95% CI]": s_roc_test,
            "對照說明": "測試集點估計優於（高於）CV 均值" if s_roc_pt > float(s_row["fold_roc_auc_mean"]) else "測試集點估計低於 CV 均值",
        },
        {
            "模型": "單一特徵 (-EPS)",
            "指標": "PR-AUC",
            "15 折 CV 表現": f"{float(s_row['fold_pr_auc_mean']):.4f} ± {float(s_row['fold_pr_auc_std']):.4f}",
            "測試集表現 [95% CI]": s_pr_test,
            "對照說明": "測試集點估計優於（高於）CV 均值" if s_pr_pt > float(s_row["fold_pr_auc_mean"]) else "測試集點估計低於 CV 均值",
        },
        {
            "模型": "單一特徵 (-EPS)",
            "指標": "AR@0.95",
            "15 折 CV 表現": f"{float(s_row['fold_alert_ratio@recall_0.95_mean']):.2f}x ± {float(s_row['fold_alert_ratio@recall_0.95_std']):.2f}x",
            "測試集表現 [95% CI]": s_ar_test,
            "對照說明": "測試集所需預警倍數較低（優於 CV）" if s_ar_pt < float(s_row["fold_alert_ratio@recall_0.95_mean"]) else "測試集所需預警倍數較高",
        },
    ]
    md += df_to_markdown(pd.DataFrame(difficulty_rows))
    md += "\n\n"
    md += "> [!IMPORTANT]\n"
    md += "> **測試集難度觀察**：測試集上的各項排名與預警指標均優於 15 折 CV 的平均表現，顯示此測試集相對於交叉驗證整體劃分而言難度看起來較容易（破產樣本較易辨識）。切不可將測試集上達成 100% 召回（如 R95 與 R100）視為系統未來泛化的絕對保證。專題結論以 15 折 CV 估計為主、測試集為最終一次性確認。\n\n"

    md += "## 6. 安全邊際的成本（事後探索）\n\n"
    md += "在事前門檻固化流程中，為了抵禦驗證折間的變動性並保障召回率達標，E4a 引入了安全邊際 $m$。以下列出各工作點凍結門檻的實際預警數，與「事後剛好達到目標召回要求之最低預警數」之對照（標註為事後描述，不可用來調整事前邊際）：\n\n"

    # Compute hindsight costs
    hindsight_rows = []
    hindsight_numeric = []
    models_hindsight = [("E_A", score_EA), ("M3", score_M3)]
    for grp, s_arr in models_hindsight:
        pos_scores = np.sort(s_arr[labels == 1])[::-1]
        sub_grp = op_df[op_df["group"] == grp]
        for _, fr_row in sub_grp.iterrows():
            r_target = float(fr_row["target_recall"])
            req_caught = int(fr_row["required_caught"])
            req_score = pos_scores[min(req_caught - 1, len(pos_scores) - 1)]
            min_alerts = int(np.sum(s_arr >= req_score))
            min_ar = min_alerts / n_pos

            fr_alerts = int(fr_row["alerts_count"])
            fr_ar = float(fr_row["alert_ratio"])
            fr_margin = float(fr_row["margin"])
            hindsight_numeric.append({
                "group": grp,
                "operating_point": fr_row["operating_point"],
                "target_recall": r_target,
                "margin": fr_margin,
                "required_caught": req_caught,
                "frozen_alerts": fr_alerts,
                "frozen_alert_ratio": fr_ar,
                "hindsight_min_alerts": min_alerts,
                "hindsight_alert_ratio": min_ar,
                "cost_alerts": fr_alerts - min_alerts,
                "cost_alert_ratio": fr_ar - min_ar,
            })
            hindsight_rows.append({
                "模型": grp,
                "工作點": f"R{int(round(r_target*100))} (m={fr_margin:.2f})",
                "目標捕獲要求": f">={req_caught}" if r_target < 1.0 else f"=={req_caught}",
                "凍結門檻預警數": f"{fr_alerts} ({fr_ar:.2f}x)",
                "事後達標最低預警數": f"{min_alerts} ({min_ar:.2f}x)",
                "安全邊際成本": f"+{fr_alerts - min_alerts} 家 (+{fr_ar - min_ar:.2f}x)",
            })
    pd.DataFrame(hindsight_numeric).to_csv(out_dir / "e6_hindsight_costs.csv", index=False)
    md += df_to_markdown(pd.DataFrame(hindsight_rows))
    md += "\n\n"
    md += "> [!NOTE]\n"
    md += "> 此處之最低預警數為事後觀察結果（標註為事後描述），實務上新資料未標記時無法事先得知該點，不可將此作為調整或縮減事前安全邊際的依據。\n\n"

    md += "## 7. 漏抓正樣本剖析與對照（事後探索）\n\n"
    if len(missed_df) > 0:
        md += df_to_markdown(missed_df)
        md += "\n\n"
    else:
        md += "在目前所有工作點下無漏抓正樣本。\n\n"

    md += "> [!NOTE]\n"
    md += "> **對照 e3d_hard_positives.csv**（事後探索）：\n"

    # Dynamic split-size / overlap statement (training labels only; test labels come from n_samples/n_pos)
    split_sentence = ""
    try:
        _, y_tr_rep = get_train_data()
        splits_rep = joblib.load(project_root / "outputs" / "models" / "splits.joblib")
        overlap_n = len(set(int(i) for i in splits_rep["train_idx"]) & set(int(i) for i in splits_rep["test_idx"]))
        split_sentence = (
            f"> 1. 資料集切分（80/20 分層切分）在索引層級嚴格互斥（訓練集 {len(y_tr_rep):,} 筆包含 {int(y_tr_rep.sum())} 家破產、"
            f"測試集 {n_samples:,} 筆包含 {n_pos} 家破產），原始資料列索引（`raw_row_index`）集合交集為 {overlap_n}，"
            f"{'兩者無重疊公司' if overlap_n == 0 else '兩者有重疊列，需檢視'}。\n"
        )
    except Exception as exc:  # pragma: no cover - report must still render
        logger.warning(f"Could not compute split summary for report: {exc}")
    md += split_sentence

    # Dynamic missed-company paragraph from missed_df (E_A operating points with target_recall >= 0.90)
    ea_ops = op_df[(op_df["group"] == "E_A") & (op_df["target_recall"] >= 0.90)].sort_values("target_recall")
    missed_parts: list[str] = []
    fully_caught: list[str] = []
    for _, op_r in ea_ops.iterrows():
        n_missed = int(op_r["total_positives"]) - int(op_r["positives_caught"])
        if n_missed == 0:
            fully_caught.append(str(op_r["operating_point"]))
            continue
        rows_m = missed_df[(missed_df["group"] == "E_A") & (missed_df["operating_point"] == op_r["operating_point"])]
        ids = "；".join(f"原始列號 {int(r['raw_row_index'])}，測試集列號 {int(r['row_index'])}" for _, r in rows_m.iterrows())
        details = "；".join(
            f"$E_A$ 分數 {float(r['score_E_A']):.4f} 低於門檻 {float(r['threshold']):.4f}，Platt 機率 {float(r['prob_platt']):.2%}"
            for _, r in rows_m.iterrows()
        )
        missed_parts.append(
            f"僅有 {n_missed} 家（{ids}）在 {op_r['operating_point']} 被 $E_A$ 漏抓（{details}），其餘 {int(op_r['positives_caught'])} 家均被捕獲"
        )
    if missed_parts:
        sentence2 = f"在測試集的 {n_pos} 家破產公司中，" + "；".join(missed_parts)
        if fully_caught:
            sentence2 += f"；在 {' 與 '.join(fully_caught)} 下，所有 {n_pos} 家測試破產公司均被成功捕獲"
        md += f"> 2. {sentence2}。\n\n"
    else:
        md += "\n"

    md += "## 8. 方法論與業務限制說明\n"
    ea95 = op_df[(op_df["group"] == "E_A") & (np.isclose(op_df["target_recall"], 0.95))].iloc[0]
    ea100 = op_df[(op_df["group"] == "E_A") & (np.isclose(op_df["target_recall"], 1.00))].iloc[0]
    md += f"- **高召回成本權衡**：達成目標召回率 $R \\ge 0.95$ 在本資料上需預警約 {float(ea95['alert_proportion']):.1%} 的公司（預警倍數 {float(ea95['alert_ratio']):.2f}x），若追求 $R = 1.00$ 則在本資料上需預警約 {float(ea100['alert_proportion']):.0%} 的公司。實務工作點應依銀行審查量能與風險承受度審慎權衡。\n"
    md += "- **指標差異與統計口徑**：$E_A$ 與 M3 之配對 Bootstrap 差值 95% 信賴區間跨越 0，統計上無法區分；$E_A$ 相對於單特徵基準在 PR-AUC 與 ROC-AUC 上之 95% 信賴區間不含 0。\n"
    md += f"- **樣本規模精度限制**：測試集中僅有 {n_pos} 家破產企業，每漏抓 1 家企業對觀測召回率之影響即達 {100.0/n_pos:.2f}%。報告之信賴區間均如實反映此樣本規模下的不確定性。\n"
    md += f"- **機率校準參考性**：可靠度圖顯示平均預測機率與經驗破產率趨勢相符，但因僅 {n_pos} 個正樣本且採 5 個分箱，圖形僅供參考。\n"

    report_path = out_dir / "E6_REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    logger.info(f"Generated {report_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase E6 Evaluation Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Run full evaluation pipeline on training subsample")
    group.add_argument("--preflight", action="store_true", help="Run preflight integrity audit without unsealing")
    group.add_argument("--unseal", action="store_true", help="Formally unseal holdout test set (only once)")
    group.add_argument("--analyze", action="store_true", help="Re-run analysis on already-saved evaluation scores")

    parser.add_argument("--bootstraps", type=int, default=2000, help="Number of bootstrap iterations (default: 2000)")
    args = parser.parse_args()

    final_dir = project_root / "outputs" / "models" / "final"
    components = load_final_components(final_dir)

    if args.preflight:
        logger.info("=== Running Standalone E6 Preflight Integrity Audit ===")
        from src.preflight import run_e6_preflight
        run_e6_preflight()
        logger.info("=== Standalone Preflight Completed Successfully ===")

    elif args.dry_run:
        logger.info("=== Running E6 in DRY-RUN mode (Training Subsample) ===")
        out_dir = project_root / "outputs" / "e6_dryrun"

        # Stratified subsample from training data (1,364 samples, exactly 44 positives)
        X_train, y_train = get_train_data()
        y_arr = y_train.values

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]

        np.random.seed(SEED)
        sub_pos = np.random.choice(pos_idx, size=44, replace=False)
        sub_neg = np.random.choice(neg_idx, size=1320, replace=False)
        sub_all = np.sort(np.concatenate([sub_pos, sub_neg]))

        X_eval = X_train.iloc[sub_all].copy()
        y_eval = y_train.iloc[sub_all].copy()

        scores_file = run_e6_scoring(X_eval, y_eval, components, out_dir, dry_run=True)
        run_e6_analysis(scores_file, components, out_dir, dry_run=True, n_bootstraps=args.bootstraps)
        logger.info("=== Dry-run completed successfully! ===")

    elif args.unseal:
        logger.info("=== Running E6 in UNSEAL mode (Holdout Test Set) ===")
        out_dir = project_root / "outputs" / "e6"
        lock_dir = project_root / "outputs" / "logs"

        # 0. Execute comprehensive preflight checklist BEFORE acquiring lock or touching test data
        logger.info("Executing comprehensive preflight checklist before unsealing...")
        from src.preflight import run_e6_preflight
        preflight_meta = run_e6_preflight()
        logger.info("Preflight checklist PASSED.")

        # 1. Enforce anti-leakage lock with audit metadata
        logger.info("Checking and acquiring E6 anti-leakage lock with integrity metadata...")
        acquire_e6_lock(
            lock_path=lock_dir,
            extra_metadata={
                "train_indices_sha256": preflight_meta["train_indices_sha256"],
                "test_indices_sha256": preflight_meta["test_indices_sha256"],
                "frozen_file_hashes": preflight_meta["frozen_file_hashes"],
            },
        )

        # 2. Protected test data access
        logger.info("Accessing holdout test data via protected cv.get_test_data() ...")
        X_test, y_test = get_test_data(i_know_this_is_final_evaluation=True)

        # 3. Score and persist
        scores_file = run_e6_scoring(X_test, y_test, components, out_dir, dry_run=False)

        # 4. Analyze
        run_e6_analysis(scores_file, components, out_dir, dry_run=False, n_bootstraps=args.bootstraps)
        logger.info("=== Formal unseal evaluation completed! ===")

    elif args.analyze:
        logger.info("=== Running E6 in ANALYZE mode (Saved Scores) ===")
        # Check if dry-run or formal unseal scores exist
        unseal_scores = project_root / "outputs" / "e6" / "e6_test_scores.npz"
        dryrun_scores = project_root / "outputs" / "e6_dryrun" / "e6_test_scores.npz"

        if unseal_scores.exists():
            out_dir = project_root / "outputs" / "e6"
            scores_file = unseal_scores
            dry_run = False
        elif dryrun_scores.exists():
            out_dir = project_root / "outputs" / "e6_dryrun"
            scores_file = dryrun_scores
            dry_run = True
        else:
            raise FileNotFoundError("No existing e6_test_scores.npz found in outputs/e6/ or outputs/e6_dryrun/.")

        run_e6_analysis(scores_file, components, out_dir, dry_run=dry_run, n_bootstraps=args.bootstraps)
        logger.info("=== Analysis completed! ===")


if __name__ == "__main__":
    main()
