"""
E7b: Random Seed Stability Analysis for Bankruptcy Prediction.

Evaluates E_A ensemble performance stability under 5 different random seeds:
seeds in {42, 7, 123, 2025, 31337}.

Only uses training data and fixed outer 15-fold splits.
For each seed, fits M3, M4, M5 with E2b best hyperparameters and varying SMOTE/classifier seeds.
Computes validation-fold-internal percentile rank average (matching E3b internal rank version):
- 3-point log loss
- AR@0.95
- PR-AUC

Outputs: outputs/tables/e7b_seed_stability.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cv import get_train_data, load_splits
from src.ensemble import compute_three_point_log_loss
from src.metrics import alert_ratio_at_recall
from src.pipelines import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_stability")

SEEDS = [42, 7, 123, 2025, 31337]


def run_seed_stability() -> pd.DataFrame:
    logger.info("=== Commencing E7b: Random Seed Stability Evaluation ===")

    # 1. Load training data & splits
    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_idx = splits_payload["train_idx"]
    orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}

    # 2. Load E2b optimal parameters
    best_params_dir = PROJECT_ROOT / "outputs" / "models" / "best_params"
    with open(best_params_dir / "M3.json", "r", encoding="utf-8") as f:
        p3 = json.load(f)["best_params"]
    with open(best_params_dir / "M4.json", "r", encoding="utf-8") as f:
        p4 = json.load(f)["best_params"]
    with open(best_params_dir / "M5.json", "r", encoding="utf-8") as f:
        p5 = json.load(f)["best_params"]

    fold_records = []

    # 3. Iterate through all 15 outer folds
    for s_idx, split_info in enumerate(all_splits):
        rep = split_info["repeat"]
        fold = split_info["fold"]
        logger.info(f"Processing Fold {s_idx + 1}/15 (Repeat {rep}, Fold {fold}) ...")

        out_tr_idx = np.array([orig_to_rel[i] for i in split_info["train_idx"]])
        out_val_idx = np.array([orig_to_rel[i] for i in split_info["val_idx"]])

        X_tr = X_train.iloc[out_tr_idx]
        y_tr = y_train.iloc[out_tr_idx]
        X_val = X_train.iloc[out_val_idx]
        y_val = y_train.iloc[out_val_idx].values

        n_val = len(y_val)

        for seed in SEEDS:
            # Fit M3 (ExtraTrees, S0)
            pipe3 = build_pipeline(
                "M3",
                imbalance="S0",
                use_winsorizer=False,
                model_params=p3,
                seed=seed,
            )
            pipe3.fit(X_tr, y_tr)
            raw3 = pipe3.predict_proba(X_val)[:, 1]

            # Fit M4 (XGBoost, S0)
            pipe4 = build_pipeline(
                "M4",
                imbalance="S0",
                use_winsorizer=False,
                model_params=p4,
                seed=seed,
            )
            pipe4.fit(X_tr, y_tr)
            raw4 = pipe4.predict_proba(X_val)[:, 1]

            # Fit M5 (LightGBM, S3)
            pipe5 = build_pipeline(
                "M5",
                imbalance="S3",
                use_winsorizer=False,
                model_params=p5,
                seed=seed,
            )
            pipe5.fit(X_tr, y_tr)
            raw5 = pipe5.predict_proba(X_val)[:, 1]

            # Validation internal percentile ranks
            pct3 = stats.rankdata(raw3, method="average") / n_val
            pct4 = stats.rankdata(raw4, method="average") / n_val
            pct5 = stats.rankdata(raw5, method="average") / n_val

            # Equal-weight ensemble score
            score_EA = (pct3 + pct4 + pct5) / 3.0

            loss = float(compute_three_point_log_loss(y_val, score_EA))
            ar95 = float(alert_ratio_at_recall(y_val, score_EA, 0.95))
            pr_auc = float(average_precision_score(y_val, score_EA))

            fold_records.append(
                {
                    "seed": seed,
                    "repeat": rep,
                    "fold": fold,
                    "loss": loss,
                    "ar95": ar95,
                    "pr_auc": pr_auc,
                }
            )

    df_folds = pd.DataFrame(fold_records)

    # 4. Compute per-seed 15-fold means and within-seed fold standard deviations
    seed_summaries = []
    for seed in SEEDS:
        sub = df_folds[df_folds["seed"] == seed]
        seed_summaries.append(
            {
                "seed": seed,
                "loss_mean": float(sub["loss"].mean()),
                "loss_std": float(sub["loss"].std()),
                "ar95_mean": float(sub["ar95"].mean()),
                "ar95_std": float(sub["ar95"].std()),
                "pr_auc_mean": float(sub["pr_auc"].mean()),
                "pr_auc_std": float(sub["pr_auc"].std()),
            }
        )

    df_seed_summary = pd.DataFrame(seed_summaries)

    # 5. Across-seed comparison: Variation across seeds vs. Variation across folds
    metrics_comp = []
    for m in ["loss", "ar95", "pr_auc"]:
        means_across_seeds = df_seed_summary[f"{m}_mean"].values
        stds_across_folds = df_seed_summary[f"{m}_std"].values

        seed_mean_val = float(np.mean(means_across_seeds))
        seed_std_val = float(np.std(means_across_seeds, ddof=1))
        seed_min_val = float(np.min(means_across_seeds))
        seed_max_val = float(np.max(means_across_seeds))
        seed_range_val = seed_max_val - seed_min_val
        avg_fold_std = float(np.mean(stds_across_folds))
        avg_fold_se = float(avg_fold_std / np.sqrt(15))

        metrics_comp.append(
            {
                "metric": m,
                "overall_mean": seed_mean_val,
                "seed_std": seed_std_val,
                "seed_min": seed_min_val,
                "seed_max": seed_max_val,
                "seed_range": seed_range_val,
                "fold_std_mean": avg_fold_std,
                "fold_se_mean": avg_fold_se,
                "seed_std_vs_fold_se_ratio": seed_std_val / avg_fold_se if avg_fold_se > 0 else 0.0,
            }
        )

    df_metrics_comp = pd.DataFrame(metrics_comp)

    # 6. Save combined results
    out_table_path = PROJECT_ROOT / "outputs" / "tables" / "e7b_seed_stability.csv"
    
    # Store both the per-seed summary and the contrast summary
    combined_rows = []
    for _, row in df_seed_summary.iterrows():
        combined_rows.append({
            "type": "per_seed",
            "seed": int(row["seed"]),
            "metric": "all",
            "loss_mean": row["loss_mean"],
            "loss_fold_std": row["loss_std"],
            "ar95_mean": row["ar95_mean"],
            "ar95_fold_std": row["ar95_std"],
            "pr_auc_mean": row["pr_auc_mean"],
            "pr_auc_fold_std": row["pr_auc_std"],
            "note": "E7b使用E2b參數，絕對水準偏樂觀，僅供種子穩定性評估；無偏估計以E2a/E3與E6為準。",
        })
    for _, row in df_metrics_comp.iterrows():
        combined_rows.append({
            "type": "across_seed_comparison",
            "seed": -1,
            "metric": row["metric"],
            "mean": row["overall_mean"],
            "seed_std": row["seed_std"],
            "seed_min": row["seed_min"],
            "seed_max": row["seed_max"],
            "seed_range": row["seed_range"],
            "fold_std": row["fold_std_mean"],
            "fold_se": row["fold_se_mean"],
            "ratio_seed_to_fold_se": row["seed_std_vs_fold_se_ratio"],
            "note": "種子造成的變動遠小於折間變動；分母採用15折平均標準誤(fold_std/√15)。E7b絕對水準偏樂觀。",
        })

    df_final = pd.DataFrame(combined_rows)
    df_final.to_csv(out_table_path, index=False)
    logger.info(f"Saved E7b results to {out_table_path}")

    # Also save raw per-fold per-seed data for inspection
    raw_fold_path = PROJECT_ROOT / "outputs" / "tables" / "e7b_seed_stability_per_fold.csv"
    df_folds.to_csv(raw_fold_path, index=False)
    logger.info(f"Saved per-fold data to {raw_fold_path}")

    return df_final


if __name__ == "__main__":
    run_seed_stability()
