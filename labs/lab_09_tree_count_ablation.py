#!/usr/bin/env python3
"""
Lab 09: ExtraTrees (M3) 樹數消融實驗 (Tree Count Ablation)

探討極端隨機樹在極度不平衡資料（3.23%）下，樹數 n_estimators ∈ {80, 100, 300, 600}
對預測分數空隙（間距）、同分門檻判定與 AR@0.95 退化機制的實測影響。

注意：每棵樹的輸出為葉節點類別比例的平均，樹數越少，兩個相鄰分數之間的空隙越大，
高召回工作點門檻切在稀疏區時更容易涵蓋多個同分樣本。本實驗以唯一分數個數與逐折門檻佐證。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bankruptcy.src.pipelines import build_pipeline, get_score
from labs._common import (
    REPO_ROOT,
    evaluate_fold_predictions,
    get_cv_splitter,
    get_pipeline_flags,
    load_best_params,
    load_train_data,
    save_manifest,
)

PARAMS = {
    "tree_counts_quick": [80, 600],
    "tree_counts_standard": [80, 100, 300, 600],
    "tree_counts_full": [80, 100, 200, 300, 450, 600],
    "cv_folds": 5,
    "seed": 42,
}


def run_lab(mode: str = "standard", pipeline: str = "project") -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    start_time = time.time()
    pipe_flags = get_pipeline_flags(pipeline)
    out_dir = REPO_ROOT / "labs" / "outputs" / "lab_09"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_train_data()
    seed = PARAMS["seed"]
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=1, seed=seed)

    base_json = load_best_params("M3")
    base_params = base_json["best_params"].copy()
    imb = base_json["imbalance"]

    if mode == "quick":
        tree_counts = PARAMS["tree_counts_quick"]
    elif mode == "full":
        tree_counts = PARAMS["tree_counts_full"]
    else:
        tree_counts = PARAMS["tree_counts_standard"]

    print(f"Executing Lab 09 (ExtraTrees Tree Count Ablation) [{mode} mode, pipeline={pipeline}]...")
    print(f"Testing n_estimators: {tree_counts}")

    fold_rows = []
    summary_rows = []

    for n_trees in tree_counts:
        t_start = time.time()
        p = base_params.copy()
        p["n_estimators"] = n_trees

        ar95_list, pr_list, roc_list = [], [], []
        unique_counts, zero_pos_list = [], []

        for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
            t0 = time.time()
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe = build_pipeline(
                model_name="M3",
                imbalance=imb,
                model_params=p,
                seed=seed + fold_idx,
                **pipe_flags,
            )
            pipe.fit(X_tr, y_tr)
            score_val = get_score(pipe, X_val)
            m = evaluate_fold_predictions(y_val, score_val)

            # Detailed metrics on 95% target recall
            P = int(np.sum(y_val == 1))
            k_target = int(np.ceil(0.95 * P))
            pos_scores = np.sort(score_val[y_val == 1])
            thr = float(pos_scores[P - k_target])
            alerts = int(np.sum(score_val >= thr))
            ar95 = float(alerts / P) if P > 0 else 0.0
            zero_pos = int(np.sum(pos_scores == 0.0))
            n_unique = int(len(np.unique(score_val)))
            f_elapsed = time.time() - t0

            fold_rows.append({
                "n_estimators": n_trees,
                "fold": fold_idx,
                "P": P,
                "k_target": k_target,
                "pos_score_at_ceil_95": thr,
                "threshold": thr,
                "alerts_count": alerts,
                "ar95": ar95,
                "pr_auc": float(m["pr_auc"]),
                "roc_auc": float(m["roc_auc"]),
                "zero_score_positives": zero_pos,
                "unique_scores_count": n_unique,
                "elapsed_seconds": f_elapsed,
            })

            ar95_list.append(ar95)
            pr_list.append(m["pr_auc"])
            roc_list.append(m["roc_auc"])
            unique_counts.append(n_unique)
            zero_pos_list.append(zero_pos)

        total_trees_time = time.time() - t_start
        summary_rows.append({
            "n_estimators": n_trees,
            "ar95_mean": float(np.mean(ar95_list)),
            "ar95_std": float(np.std(ar95_list, ddof=1)),
            "ar95_min": float(np.min(ar95_list)),
            "ar95_max": float(np.max(ar95_list)),
            "pr_auc_mean": float(np.mean(pr_list)),
            "pr_auc_std": float(np.std(pr_list, ddof=1)),
            "roc_auc_mean": float(np.mean(roc_list)),
            "roc_auc_std": float(np.std(roc_list, ddof=1)),
            "unique_scores_mean": float(np.mean(unique_counts)),
            "zero_score_pos_mean": float(np.mean(zero_pos_list)),
            "elapsed_seconds": total_trees_time,
        })
        print(f"  n_trees={n_trees:3d} | AR@0.95: {np.mean(ar95_list):.2f}x (std={np.std(ar95_list, ddof=1):.2f}) | PR-AUC: {np.mean(pr_list):.4f} | Uniq: {np.mean(unique_counts):.0f} | Time: {total_trees_time:.1f}s")

    df_folds = pd.DataFrame(fold_rows)
    df_summary = pd.DataFrame(summary_rows)

    folds_csv = out_dir / "lab_09_tree_count_folds.csv"
    summary_csv = out_dir / "lab_09_tree_count_summary.csv"
    snap_csv = REPO_ROOT / "docs" / "code_learning" / "lab_snapshots" / "lab_09_tree_count_ablation_snapshot.csv"
    snap_folds_csv = REPO_ROOT / "docs" / "code_learning" / "lab_snapshots" / "lab_09_tree_count_folds_snapshot.csv"
    snap_csv.parent.mkdir(parents=True, exist_ok=True)

    df_folds.to_csv(folds_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    df_summary.to_csv(snap_csv, index=False)
    df_folds.to_csv(snap_folds_csv, index=False)

    # Plot figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. AR@0.95 vs Trees
    axes[0].errorbar(
        df_summary["n_estimators"], df_summary["ar95_mean"],
        yerr=df_summary["ar95_std"], fmt="o-", color="#e76f51", capsize=4, label="AR@0.95"
    )
    axes[0].set_title("樹數 vs AR@0.95 均值與標準差", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("樹數 (n_estimators)")
    axes[0].set_ylabel("預警倍數 (AR@0.95)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. PR-AUC vs Trees
    axes[1].plot(df_summary["n_estimators"], df_summary["pr_auc_mean"], "s-", color="#2b5c8f", label="PR-AUC")
    axes[1].set_title("樹數 vs PR-AUC", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("樹數 (n_estimators)")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Unique Scores vs Trees
    axes[2].bar(
        [str(x) for x in df_summary["n_estimators"]],
        df_summary["unique_scores_mean"], color="#457b9d", alpha=0.85
    )
    axes[2].set_title("樹數 vs 驗證折唯一預測分數個數", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("樹數 (n_estimators)")
    axes[2].set_ylabel("唯一分數個數 (Unique Scores)")
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig_path = out_dir / "lab_09_tree_count_ablation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    save_manifest(
        output_dir=out_dir,
        lab_id="lab_09",
        params=PARAMS,
        elapsed_seconds=elapsed,
        seed=seed,
        extra={"mode": mode, "pipeline": pipeline},
    )

    print("\n" + "=" * 70)
    print(f"LAB_09 執行完畢，總耗時: {elapsed:.2f} 秒")
    print(f"逐折 CSV : {folds_csv.as_posix()}")
    print(f"彙總 CSV : {summary_csv.as_posix()}")
    print(f"快照 CSV : {snap_csv.as_posix()}")
    print(f"圖表 PNG : {fig_path.as_posix()}")
    print("=" * 70)

    return df_folds, df_summary, summary_csv, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 09: ExtraTrees 樹數消融實驗")
    parser.add_argument("--full", action="store_true", help="完整模式 (6 種樹數)")
    parser.add_argument("--standard", action="store_true", default=True, help="標準模式 (4 種樹數 80, 100, 300, 600, 預設)")
    parser.add_argument("--quick", action="store_true", help="快速模式 (2 種樹數 80, 600)")
    parser.add_argument("--pipeline", choices=["project", "teaching"], default="project", help="前處理旗標 (project 關閉截尾, teaching 開啟截尾, 預設 project)")
    args = parser.parse_args()

    if args.full:
        exec_mode = "full"
    elif args.quick:
        exec_mode = "quick"
    else:
        exec_mode = "standard"

    run_lab(mode=exec_mode, pipeline=args.pipeline)
