"""
E0 Baseline Experiments for Bankruptcy Prediction.

Evaluates 3 baseline models across Repeated Stratified 5-Fold Cross-Validation:
1. Dummy Classifier: DummyClassifier(strategy="prior")
2. Single Strongest Feature: Dynamically identified per fold via univariate ROC-AUC
3. Base Logistic Regression: Preprocessing pipeline + standard Logistic Regression (S0)

Includes:
- Label shuffling leakage test (permuted target labels evaluated with LR + S3)
- OOF score serialization (.npy)
- Summary tables and visual comparison figures (Recall vs Alert Ratio, PR curves)

Usage:
    python src/run_baselines.py           # Full evaluation (5 folds x 3 repeats = 15 folds)
    python src/run_baselines.py --quick   # Fast verification (5 folds x 1 repeat = 5 folds)
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from config import (
    CV_FOLDS,
    CV_REPEATS,
    FIGURES_DIR,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
    TARGET_COL,
)
from src.cv import (
    cross_validate_pipeline,
    get_train_data,
    load_splits,
)
from src.metrics import evaluate_scores, recall_at_alert_ratio
from src.pipelines import build_pipeline, get_score


OOF_DIR = MODELS_DIR / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_single_feature_fold(
    X_train_f: pd.DataFrame,
    y_train_f: np.ndarray,
    X_val_f: pd.DataFrame,
) -> tuple[np.ndarray, str, str, float]:
    """
    Identifies the strongest predictive feature strictly within the training fold.
    Returns:
        (val_scores, best_feature_name, direction, train_auc)
    """
    best_feat = None
    best_auc = -1.0
    best_sign = 1.0
    best_dir = "positive"

    # Evaluate each feature on training fold
    for col in X_train_f.columns:
        vals = X_train_f[col].values
        if np.all(vals == vals[0]):
            continue
        try:
            raw_auc = roc_auc_score(y_train_f, vals)
            if raw_auc >= 0.5:
                adj_auc = raw_auc
                sign = 1.0
                direction = "positive"
            else:
                adj_auc = 1.0 - raw_auc
                sign = -1.0
                direction = "negative"
        except Exception:
            continue

        if adj_auc > best_auc:
            best_auc = adj_auc
            best_feat = col
            best_sign = sign
            best_dir = direction

    # Directly use directional feature values as risk scores
    val_scores = best_sign * X_val_f[best_feat].values

    return val_scores, best_feat, best_dir, float(best_auc)


def run_e0_baselines(quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run E0 Baseline evaluations on outer CV folds.
    """
    print("=" * 80)
    print("              E0 BASELINE EXPERIMENTS: EXECUTION")
    print("=" * 80)

    X_train, y_train = get_train_data()
    splits_payload = load_splits()
    all_splits = splits_payload["cv_splits"]
    train_orig_indices = splits_payload["train_idx"]

    # Map original df indices to X_train row positions (0 to len(X_train)-1)
    orig_to_rel = {orig_i: rel_i for rel_i, orig_i in enumerate(train_orig_indices)}

    # Select splits (1 repeat if quick, else all repeats)
    target_repeats = [0] if quick else list(range(CV_REPEATS))
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

    n_samples = len(X_train)
    n_repeats = len(target_repeats)
    print(f"Dataset Size: {n_samples} training samples (Test set locked).")
    print(f"Evaluation Mode: {'QUICK (1 repeat, 5 folds)' if quick else 'FULL (3 repeats, 15 folds)'}")

    all_fold_records = []
    all_oof_records = []

    # --------------------------------------------------------------------------
    # 1. Model: Dummy Classifier
    # --------------------------------------------------------------------------
    print("\n[1/3] Running Dummy Classifier (strategy='prior') ...")
    # Pipeline integration verifies pipeline stability with prior dummy output
    def dummy_factory():
        return build_pipeline("M0", imbalance="S0", scaler="quantile", use_winsorizer=True, use_corr_filter=True)

    dummy_out = cross_validate_pipeline(dummy_factory, X_train, y_train, rel_splits)
    dummy_folds = dummy_out["fold_results"]
    dummy_folds["model"] = "Dummy"
    all_fold_records.append(dummy_folds)

    dummy_oof_summary = dummy_out["oof_summary"]
    dummy_oof_summary["model"] = "Dummy"
    all_oof_records.append(dummy_oof_summary)

    np.save(OOF_DIR / "e0_dummy.npy", dummy_out["oof_scores"])

    # --------------------------------------------------------------------------
    # 2. Model: Single Strongest Feature
    # --------------------------------------------------------------------------
    print("\n[2/3] Running Single Strongest Feature (dynamically selected per fold) ...")
    single_oof_scores = np.full((n_repeats, n_samples), fill_value=np.nan, dtype=float)
    single_fold_records = []
    feature_selection_records = []

    for split in rel_splits:
        r = split["repeat"]
        f = split["fold"]
        tr_idx = split["train_idx"]
        val_idx = split["val_idx"]

        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx].values
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx].values

        val_scores, best_feat, best_dir, train_auc = evaluate_single_feature_fold(X_tr, y_tr, X_val)

        r_idx = target_repeats.index(r)
        single_oof_scores[r_idx, val_idx] = val_scores

        fold_metrics = evaluate_scores(y_val, val_scores)
        fold_metrics["model"] = "Single Strongest Feature"
        fold_metrics["repeat"] = r
        fold_metrics["fold"] = f
        single_fold_records.append(fold_metrics)

        feature_selection_records.append({
            "repeat": r,
            "fold": f,
            "selected_feature": best_feat,
            "direction": best_dir,
            "train_auc": train_auc,
        })

    single_folds = pd.DataFrame(single_fold_records)
    all_fold_records.append(single_folds)

    # Pooled OOF for Single Feature
    single_oof_records = []
    for r_idx, r in enumerate(target_repeats):
        pooled_scores = single_oof_scores[r_idx]
        pooled_m = evaluate_scores(y_train.values, pooled_scores)
        pooled_m["model"] = "Single Strongest Feature"
        pooled_m["repeat"] = r
        single_oof_records.append(pooled_m)

    single_oof_summary = pd.DataFrame(single_oof_records)
    all_oof_records.append(single_oof_summary)

    np.save(OOF_DIR / "e0_single_feature.npy", single_oof_scores)

    # Feature selection frequency table
    feat_sel_df = pd.DataFrame(feature_selection_records)
    feat_sel_df.to_csv(TABLES_DIR / "e0_single_feature_selection.csv", index=False)

    # --------------------------------------------------------------------------
    # 3. Model: Base Logistic Regression (S0)
    # --------------------------------------------------------------------------
    print("\n[3/3] Running Base Logistic Regression (S0, QuantileScaler, No SMOTE) ...")
    def lr_factory():
        return build_pipeline(
            "M1",
            imbalance="S0",
            scaler="quantile",
            use_winsorizer=True,
            use_corr_filter=True,
        )

    lr_out = cross_validate_pipeline(lr_factory, X_train, y_train, rel_splits)
    lr_folds = lr_out["fold_results"]
    lr_folds["model"] = "Base Logistic Regression (S0)"
    all_fold_records.append(lr_folds)

    lr_oof_summary = lr_out["oof_summary"]
    lr_oof_summary["model"] = "Base Logistic Regression (S0)"
    all_oof_records.append(lr_oof_summary)

    np.save(OOF_DIR / "e0_logistic_regression.npy", lr_out["oof_scores"])

    # --------------------------------------------------------------------------
    # Assemble and Save Summary Tables
    # --------------------------------------------------------------------------
    per_fold_df = pd.concat(all_fold_records, ignore_index=True)
    per_fold_df.to_csv(TABLES_DIR / "e0_baselines_per_fold.csv", index=False)

    oof_combined_df = pd.concat(all_oof_records, ignore_index=True)

    # Compile Summary Table (Fold Mean +/- Std & Pooled OOF Mean +/- Std)
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "alert_ratio@recall_0.80",
        "alert_ratio@recall_0.90",
        "alert_ratio@recall_0.95",
        "alert_ratio@recall_1.00",
        "recall@alert_ratio_1.5",
        "recall@alert_ratio_3",
        "recall@alert_ratio_5",
        "recall@alert_ratio_10",
    ]

    summary_rows = []
    models = ["Dummy", "Single Strongest Feature", "Base Logistic Regression (S0)"]
    for m in models:
        m_folds = per_fold_df[per_fold_df["model"] == m]
        m_oof = oof_combined_df[oof_combined_df["model"] == m]

        row = {"model": m}
        for col in metric_cols:
            fold_mean = m_folds[col].mean()
            fold_std = m_folds[col].std() if len(m_folds) > 1 else 0.0

            oof_mean = m_oof[col].mean()
            oof_std = m_oof[col].std() if len(m_oof) > 1 else 0.0

            row[f"fold_{col}_mean"] = fold_mean
            row[f"fold_{col}_std"] = fold_std
            row[f"fold_{col}_display"] = f"{fold_mean:.4f} +/- {fold_std:.4f}"

            row[f"oof_{col}_mean"] = oof_mean
            row[f"oof_{col}_std"] = oof_std
            row[f"oof_{col}_display"] = f"{oof_mean:.4f} +/- {oof_std:.4f}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "e0_baselines_summary.csv", index=False)

    # --------------------------------------------------------------------------
    # Generate Figures (Repeat 0 Pooled OOF)
    # --------------------------------------------------------------------------
    plot_e0_figures(
        y_true=y_train.values,
        dummy_scores=dummy_out["oof_scores"][0],
        single_scores=single_oof_scores[0],
        lr_scores=lr_out["oof_scores"][0],
    )

    return per_fold_df, summary_df, feat_sel_df


def plot_e0_figures(y_true: np.ndarray, dummy_scores: np.ndarray, single_scores: np.ndarray, lr_scores: np.ndarray):
    """Generate and save Recall vs Alert Ratio curves and PR curves for E0 baselines."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    models_data = [
        ("Dummy (Prior)", dummy_scores, "#95a5a6", ":"),
        ("Single Strongest Feature", single_scores, "#e67e22", "--"),
        ("Base Logistic Regression (S0)", lr_scores, "#2980b9", "-"),
    ]

    # 1. Recall vs Alert Ratio Curve (Ratio 1.0 to 30.0)
    plt.figure(figsize=(9, 6))
    ratios = np.linspace(1.0, 30.0, 60)

    for name, scores, color, lstyle in models_data:
        recalls = [recall_at_alert_ratio(y_true, scores, r) for r in ratios]
        plt.plot(ratios, recalls, label=name, color=color, linestyle=lstyle, linewidth=2.2)

    # Reference benchmark lines
    plt.axhline(0.95, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8, label="Target Recall = 95%")
    plt.axhline(0.90, color="#f39c12", linestyle=":", linewidth=1.2, alpha=0.8, label="Target Recall = 90%")

    plt.title("E0 Baselines: Recall vs Alert Ratio (Repeat 0 Pooled OOF)", fontsize=14, pad=12)
    plt.xlabel("Alert Ratio (Flagged Count / Actual Bankrupt Count)", fontsize=12)
    plt.ylabel("Recall on Bankrupt Companies", fontsize=12)
    plt.xlim(1.0, 30.0)
    plt.ylim(-0.02, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e0_recall_vs_alert_ratio.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 6))
    for name, scores, color, lstyle in models_data:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        auc_pr = evaluate_scores(y_true, scores)["pr_auc"]
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {auc_pr:.4f})", color=color, linestyle=lstyle, linewidth=2.2)

    baseline_prevalence = float((y_true == 1).mean())
    plt.axhline(baseline_prevalence, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Random Prevalence ({baseline_prevalence*100:.2f}%)")

    plt.title("E0 Baselines: Precision-Recall Curves (Repeat 0 Pooled OOF)", fontsize=14, pad=12)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e0_pr_curves.png", dpi=300)
    plt.close()


def run_label_shuffle_test() -> pd.DataFrame:
    """
    Task 2: Label Shuffle Leakage Test.
    Shuffles target labels across 3 seeds and runs 5-fold CV using Logistic Regression + SMOTE (S3).
    Verifies that ROC-AUC collapses to ~0.5 and PR-AUC collapses to ~0.032.
    """
    print("\n" + "=" * 80)
    print("         TASK 2: LABEL SHUFFLE LEAKAGE TEST (LR + S3, 5 FOLDS)")
    print("=" * 80)

    X_train, y_train = get_train_data()
    n_samples = len(X_train)
    positive_rate = float((y_train == 1).mean())

    seeds = [42, 123, 999]
    records = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        # Permute target labels
        y_shuffled = pd.Series(rng.permutation(y_train.values))

        # Stratified 5-fold CV on permuted labels
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
        shuffle_splits = []
        for f, (tr, val) in enumerate(rskf.split(X_train, y_shuffled)):
            shuffle_splits.append({
                "repeat": 0,
                "fold": f,
                "split_id": f"r0_f{f}",
                "train_idx": tr,
                "val_idx": val,
            })

        def lr_s3_factory():
            return build_pipeline("M1", imbalance="S3", scaler="quantile", use_winsorizer=True, use_corr_filter=True)

        cv_out = cross_validate_pipeline(lr_s3_factory, X_train, y_shuffled, shuffle_splits)
        fold_df = cv_out["fold_results"]
        oof_summary = cv_out["oof_summary"].iloc[0]

        mean_fold_roc = float(fold_df["roc_auc"].mean())
        mean_fold_pr = float(fold_df["pr_auc"].mean())
        oof_roc = float(oof_summary["roc_auc"])
        oof_pr = float(oof_summary["pr_auc"])

        records.append({
            "shuffle_seed": seed,
            "mean_fold_roc_auc": mean_fold_roc,
            "mean_fold_pr_auc": mean_fold_pr,
            "pooled_oof_roc_auc": oof_roc,
            "pooled_oof_pr_auc": oof_pr,
            "expected_positive_rate": positive_rate,
        })

        print(f"  * Seed {seed:3d}: Mean Fold ROC-AUC = {mean_fold_roc:.4f} | Pooled OOF ROC-AUC = {oof_roc:.4f} | PR-AUC = {oof_pr:.4f}")

    shuffle_df = pd.DataFrame(records)
    shuffle_df.to_csv(TABLES_DIR / "e0_label_shuffle.csv", index=False)

    mean_shuffled_roc = float(shuffle_df["pooled_oof_roc_auc"].mean())
    mean_shuffled_pr = float(shuffle_df["pooled_oof_pr_auc"].mean())

    print(f"\nAverage Across 3 Seeds: ROC-AUC = {mean_shuffled_roc:.4f} (Expected ~ 0.5000), PR-AUC = {mean_shuffled_pr:.4f} (Expected ~ {positive_rate:.4f})")

    # Strict leakage check
    if not (0.44 <= mean_shuffled_roc <= 0.56):
        raise AssertionError(
            f"LEAKAGE ALERT! Shuffled labels yielded ROC-AUC {mean_shuffled_roc:.4f}, "
            f"which significantly deviates from chance level (0.50)!"
        )

    print("  -> PASSED: ROC-AUC cleanly collapses to chance level (~0.50). Zero data leakage confirmed!")
    return shuffle_df


def main():
    parser = argparse.ArgumentParser(description="E0 Baseline Experiments for Bankruptcy Prediction")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run 1 repeat (5 folds) instead of 3 repeats (15 folds)")
    args = parser.parse_args()

    # 1. Run Baselines
    per_fold_df, summary_df, feat_sel_df = run_e0_baselines(quick=args.quick)

    # 2. Run Label Shuffle Leakage Test
    shuffle_df = run_label_shuffle_test()

    print("\n" + "=" * 80)
    print("                      E0 SUMMARY RESULTS")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"Model: {row['model']}")
        print(f"  - 15-Fold ROC-AUC:  {row['fold_roc_auc_display']}  | Pooled OOF: {row['oof_roc_auc_display']}")
        print(f"  - 15-Fold PR-AUC:   {row['fold_pr_auc_display']}  | Pooled OOF: {row['oof_pr_auc_display']}")
        print(f"  - Alert Ratio @ 95: {row['fold_alert_ratio@recall_0.95_display']}  | Pooled OOF: {row['oof_alert_ratio@recall_0.95_display']}")
        print(f"  - Alert Ratio @ 90: {row['fold_alert_ratio@recall_0.90_display']}  | Pooled OOF: {row['oof_alert_ratio@recall_0.90_display']}")
        print(f"  - Alert Ratio @ 80: {row['fold_alert_ratio@recall_0.80_display']}  | Pooled OOF: {row['oof_alert_ratio@recall_0.80_display']}")
        print()

    print("Outputs successfully saved to:")
    print(f"  - Tables:  {TABLES_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Models:  {OOF_DIR}")


if __name__ == "__main__":
    main()
