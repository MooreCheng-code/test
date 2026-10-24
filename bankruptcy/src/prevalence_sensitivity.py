"""
E7a: Prevalence Sensitivity Analysis for Bankruptcy Prediction.

Evaluates how operating points translate into real-world operational performance
across varying market bankruptcy prevalence rates: pi in {0.5%, 1%, 2%, 3.2%, 5%}.

Analytical formulas:
- alert_proportion = pi * TPR + (1 - pi) * FPR
- precision = (pi * TPR) / alert_proportion
- inspection_effort_per_positive = 1.0 / precision
- expected_alerts_per_10k = 10000 * alert_proportion
- expected_caught_per_10k = 10000 * pi * TPR

Assumes invariant TPR and FPR conditional distributions P(x|y=1) and P(x|y=0).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
E6_DIR = OUTPUTS_DIR / "e6"
MODELS_FINAL_DIR = OUTPUTS_DIR / "models" / "final"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PREVALENCES = [0.005, 0.010, 0.020, 0.032, 0.050]


def load_configs() -> list[tuple[str, float, float, str]]:
    """Reads (group, target_recall, recommended_margin, key) from operating_points_for_report.csv."""
    op_file = TABLES_DIR / "operating_points_for_report.csv"
    if not op_file.exists():
        raise FileNotFoundError(f"Missing {op_file}")
    op_df = pd.read_csv(op_file)
    configs = []
    for grp in ["E_A", "M3"]:
        sub = op_df[op_df["group"] == grp].sort_values("target_recall")
        for _, row in sub.iterrows():
            r_tgt = float(row["target_recall"])
            configs.append((grp, r_tgt, float(row["recommended_margin"]), f"R{int(round(r_tgt * 100))}"))
    return configs


CONFIGS = load_configs()


def run_prevalence_sensitivity() -> pd.DataFrame:
    # 1. Load E4a per-fold results
    e4a_file = TABLES_DIR / "e4a_threshold_transfer_per_fold.csv"
    if not e4a_file.exists():
        raise FileNotFoundError(f"Missing {e4a_file}")
    e4a_df = pd.read_csv(e4a_file)

    # 2. Load E6 test scores (strictly without reading raw test data)
    e6_scores_file = E6_DIR / "e6_test_scores.npz"
    if not e6_scores_file.exists():
        raise FileNotFoundError(f"Missing {e6_scores_file}")
    test_data = np.load(e6_scores_file)
    test_labels = test_data["labels"]
    n_pos_test = int(test_labels.sum())
    n_neg_test = len(test_labels) - n_pos_test

    # 3. Load frozen thresholds
    thresh_file = MODELS_FINAL_DIR / "thresholds.json"
    with open(thresh_file, "r", encoding="utf-8") as f:
        thresh_cfg = json.load(f)

    records = []

    for grp, r_tgt, m_rec, r_key in CONFIGS:
        # A. 15-fold CV Statistics
        sub = e4a_df[
            (e4a_df["group"] == grp)
            & (np.isclose(e4a_df["target_recall"], r_tgt))
            & (np.isclose(e4a_df["margin"], m_rec))
        ]
        tot_samples = np.round(sub["alert_count"] / sub["alert_proportion"]).astype(int)
        pos_samples = np.round(sub["val_pos_caught"] / sub["actual_recall"]).astype(int)
        neg_samples = tot_samples - pos_samples

        cv_tpr_folds = sub["actual_recall"].values
        cv_fpr_folds = ((sub["alert_count"] - sub["val_pos_caught"]) / neg_samples).values

        cv_tpr_mean = float(np.mean(cv_tpr_folds))
        cv_tpr_std = float(np.std(cv_tpr_folds))
        cv_fpr_mean = float(np.mean(cv_fpr_folds))
        cv_fpr_std = float(np.std(cv_fpr_folds))

        # B. Holdout Test Statistics (from saved evaluation scores)
        score_arr = test_data["score_EA"] if grp == "E_A" else test_data["score_M3"]
        threshold_val = float(thresh_cfg[grp][r_key]["threshold"])
        test_alerts = (score_arr >= threshold_val)
        tp_test = int(np.sum(test_alerts & (test_labels == 1)))
        fp_test = int(np.sum(test_alerts & (test_labels == 0)))
        test_tpr = float(tp_test / n_pos_test)
        test_fpr = float(fp_test / n_neg_test)

        # C. Evaluate across prevalence rates
        for pi in PREVALENCES:
            # 1. CV-based analytical projection
            cv_alert_prop = pi * cv_tpr_mean + (1.0 - pi) * cv_fpr_mean
            cv_precision = (pi * cv_tpr_mean) / cv_alert_prop
            cv_effort = 1.0 / cv_precision
            cv_alerts_10k = 10000.0 * cv_alert_prop
            cv_caught_10k = 10000.0 * pi * cv_tpr_mean

            # Per-fold variation in precision & effort
            fold_precisions = (pi * cv_tpr_folds) / (pi * cv_tpr_folds + (1.0 - pi) * cv_fpr_folds)
            cv_prec_std = float(np.std(fold_precisions))

            # 2. Test-based analytical projection
            test_alert_prop = pi * test_tpr + (1.0 - pi) * test_fpr
            test_precision = (pi * test_tpr) / test_alert_prop
            test_effort = 1.0 / test_precision
            test_alerts_10k = 10000.0 * test_alert_prop
            test_caught_10k = 10000.0 * pi * test_tpr

            records.append(
                {
                    "group": grp,
                    "operating_point": r_key,
                    "target_recall": r_tgt,
                    "margin": m_rec,
                    "threshold": threshold_val,
                    "prevalence": pi,
                    "prevalence_pct": f"{pi*100:.1f}%",
                    # CV Metrics
                    "cv_tpr_mean": cv_tpr_mean,
                    "cv_tpr_std": cv_tpr_std,
                    "cv_fpr_mean": cv_fpr_mean,
                    "cv_fpr_std": cv_fpr_std,
                    "cv_alert_proportion": cv_alert_prop,
                    "cv_precision_mean": cv_precision,
                    "cv_precision_std": cv_prec_std,
                    "cv_effort_per_positive": cv_effort,
                    "cv_expected_alerts_per_10k": cv_alerts_10k,
                    "cv_expected_caught_per_10k": cv_caught_10k,
                    # Test Metrics
                    "test_tpr": test_tpr,
                    "test_fpr": test_fpr,
                    "test_alert_proportion": test_alert_prop,
                    "test_precision": test_precision,
                    "test_effort_per_positive": test_effort,
                    "test_expected_alerts_per_10k": test_alerts_10k,
                    "test_expected_caught_per_10k": test_caught_10k,
                    "note": "Assumes invariant class-conditional feature distributions P(x|y). Test metrics based on N_pos=44.",
                }
            )

    df_results = pd.DataFrame(records)
    out_csv = TABLES_DIR / "e7a_prevalence_sensitivity.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Plot figure: Precision and Audit Effort vs Prevalence for E_A
    generate_prevalence_figure(df_results)
    return df_results


def generate_prevalence_figure(df: pd.DataFrame) -> None:
    ea_df = df[df["group"] == "E_A"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {
        "R80": "#1f77b4",
        "R90": "#2ca02c",
        "R95": "#ff7f0e",
        "R100": "#d62728",
    }

    # Plot 1: Precision vs Prevalence
    ax1 = axes[0]
    for r_key in ["R80", "R90", "R95", "R100"]:
        sub = ea_df[ea_df["operating_point"] == r_key].sort_values("prevalence")
        x_pct = sub["prevalence"] * 100
        ax1.plot(
            x_pct,
            sub["cv_precision_mean"] * 100,
            marker="o",
            color=colors[r_key],
            linewidth=2,
            label=f"CV 15-Fold: {r_key}",
        )
        ax1.plot(
            x_pct,
            sub["test_precision"] * 100,
            marker="s",
            linestyle="--",
            alpha=0.6,
            color=colors[r_key],
            label=f"Test (N=44): {r_key}",
        )

    ax1.set_xlabel("Bankruptcy Prevalence (%)", fontsize=11)
    ax1.set_ylabel("Expected Precision (%)", fontsize=11)
    ax1.set_title("Expected Precision vs. Market Prevalence (E_A)", fontsize=12, pad=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    # Plot 2: Inspection Effort (Firms Audited per Caught Positive) vs Prevalence
    ax2 = axes[1]
    for r_key in ["R80", "R90", "R95", "R100"]:
        sub = ea_df[ea_df["operating_point"] == r_key].sort_values("prevalence")
        x_pct = sub["prevalence"] * 100
        ax2.plot(
            x_pct,
            sub["cv_effort_per_positive"],
            marker="o",
            color=colors[r_key],
            linewidth=2,
            label=f"CV 15-Fold: {r_key}",
        )
        ax2.plot(
            x_pct,
            sub["test_effort_per_positive"],
            marker="s",
            linestyle="--",
            alpha=0.6,
            color=colors[r_key],
            label=f"Test (N=44): {r_key}",
        )

    ax2.set_xlabel("Bankruptcy Prevalence (%)", fontsize=11)
    ax2.set_ylabel("Audit Effort (Firms Examined per Caught Positive)", fontsize=11)
    ax2.set_title("Audit Inspection Burden vs. Market Prevalence (E_A)", fontsize=12, pad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_fig = FIGURES_DIR / "e7_prevalence_precision.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    run_prevalence_sensitivity()
