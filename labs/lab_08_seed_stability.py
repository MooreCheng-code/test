"""
lab_08_seed_stability.py - 種子穩定性與隨機性分解實驗

【教學核心目標】
1. 隨機性來源正交分解 (Variance Decomposition)：
   - 條件 1【演算法隨機性 (Algorithm Randomness)】：
     固定資料切分種子 (split_seed=42)，僅改變模型種子與重抽樣種子 (seeds: 42, 7, 123, 2025, 31337)。
   - 條件 2【資料抽樣變異 (Data Sampling Variation)】：
     固定模型種子 (model_seed=42)，僅改變資料切分種子 (seeds: 42, 7, 123, 2025, 31337)。
   - 條件 3【複合變異 (Combined Variation)】：
     同時改變資料切分種子與模型種子。
2. 遵守 E7b 種子協定事實：
   - 專案正式實驗 E7b (seed_stability.py) 明確定義：固定外層 15 折切分，僅改變模型種子與 SMOTE 種子
     (42, 7, 123, 2025, 31337)，評估的是「演算法隨機性」而非資料切分變異。
   - 並排印出 bankruptcy/outputs/tables/e7b_seed_stability.csv 對照值。
3. 統計洞察：
   - 資料抽樣變異 (折間與切分變異) 顯著大於演算法隨機性 (種子間變異)。
   - 在 E7b 中，PR-AUC 的 seed_std 僅 0.0017，而 fold_std 高達 0.0791 (比率約 2.1%)；
     AR@0.95 的 seed_std 僅 0.148x，而 fold_std 高達 2.13x (比率約 6.9%)。

【防護守則】
- 僅呼叫 load_train_data()，絕不讀取測試集或 E6_LOCK。
- 產物輸出至 labs/outputs/lab_08/ 與 docs/code_learning/lab_snapshots/。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from labs._common import (
    SNAPSHOT_DIR,
    build_pipeline,
    evaluate_fold_predictions,
    get_score,
    load_best_params,
    load_train_data,
    save_manifest,
)

LAB_ID = "lab_08"
OUTPUT_DIR = os.path.join("labs", "outputs", LAB_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

SEED_LIST = [42, 7, 123, 2025, 31337]


def run_single_condition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    condition_name: str,
    split_seeds: list[int],
    model_seeds: list[int],
    model_code: str = "M3",
    n_splits: int = 5,
) -> pd.DataFrame:
    """執行特定條件下 5 次實驗並收集 CV 指標平均。"""
    cfg = load_best_params(model_code)
    m_params = cfg.get("best_params", {})
    imb = cfg.get("imbalance", "S0")

    runs_records = []
    for run_idx, (s_seed, m_seed) in enumerate(zip(split_seeds, model_seeds)):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s_seed)
        fold_praucs, fold_ar95s = [], []

        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe = build_pipeline(model_code, imbalance=imb, model_params=m_params, seed=m_seed)
            pipe.fit(X_tr, y_tr)
            score_val = get_score(pipe, X_val)
            m = evaluate_fold_predictions(y_val, score_val)

            fold_praucs.append(m["pr_auc"])
            fold_ar95s.append(m["ar_95"])

        runs_records.append({
            "condition": condition_name,
            "run_index": run_idx,
            "split_seed": s_seed,
            "model_seed": m_seed,
            "pr_auc_mean": float(np.mean(fold_praucs)),
            "pr_auc_fold_std": float(np.std(fold_praucs)),
            "ar95_mean": float(np.mean(fold_ar95s)),
            "ar95_fold_std": float(np.std(fold_ar95s)),
        })

    return pd.DataFrame(runs_records)


def run_variance_decomposition(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_code: str = "M3",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """執行三種正交條件之變異分解。"""
    fixed_seed = 42

    print(f"\n[條件 1] 演算法隨機性 (固定切分種子={fixed_seed}，改變模型種子)...")
    df_c1 = run_single_condition(
        X_train, y_train,
        condition_name="1_algorithm_randomness_only",
        split_seeds=[fixed_seed] * len(SEED_LIST),
        model_seeds=SEED_LIST,
        model_code=model_code,
        n_splits=n_splits,
    )

    print(f"\n[條件 2] 資料抽樣變異 (固定模型種子={fixed_seed}，改變切分種子)...")
    df_c2 = run_single_condition(
        X_train, y_train,
        condition_name="2_data_split_variation_only",
        split_seeds=SEED_LIST,
        model_seeds=[fixed_seed] * len(SEED_LIST),
        model_code=model_code,
        n_splits=n_splits,
    )

    print(f"\n[條件 3] 複合變異 (同時改變切分種子與模型種子)...")
    df_c3 = run_single_condition(
        X_train, y_train,
        condition_name="3_combined_variation",
        split_seeds=SEED_LIST,
        model_seeds=SEED_LIST,
        model_code=model_code,
        n_splits=n_splits,
    )

    df_all_runs = pd.concat([df_c1, df_c2, df_c3], ignore_index=True)

    # 匯總各條件的變異指標
    summary_rows = []
    for cond, grp in df_all_runs.groupby("condition"):
        summary_rows.append({
            "condition": cond,
            "n_runs": len(grp),
            "pr_auc_overall_mean": float(grp["pr_auc_mean"].mean()),
            "pr_auc_across_runs_std": float(grp["pr_auc_mean"].std()),
            "pr_auc_within_fold_std_mean": float(grp["pr_auc_fold_std"].mean()),
            "ar95_overall_mean": float(grp["ar95_mean"].mean()),
            "ar95_across_runs_std": float(grp["ar95_mean"].std()),
            "ar95_within_fold_std_mean": float(grp["ar95_fold_std"].mean()),
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_all_runs, df_summary


def print_formal_e7b_comparison():
    """印出正式實驗 E7b 種子穩定性數值。"""
    e7b_path = os.path.join("bankruptcy", "outputs", "tables", "e7b_seed_stability.csv")
    print("\n" + "=" * 80)
    print("【正式實驗 E7b 種子穩定性對照】")
    if os.path.exists(e7b_path):
        df_e7b = pd.read_csv(e7b_path)
        across = df_e7b[df_e7b["type"] == "across_seed_comparison"]
        print(">> E7b 跨種子 vs 折間標準差對照:")
        print(across[["metric", "mean", "seed_std", "fold_std", "ratio_seed_to_fold_se"]].to_string(index=False))
    print("=" * 80)


def plot_seed_variance_decomposition(df_summary: pd.DataFrame, out_path: str):
    """繪製三種條件下變異數分解長條圖。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    conditions = df_summary["condition"].tolist()
    labels = ["1. Algorithm Only\n(Seed Var)", "2. Data Split Only\n(Split Var)", "3. Combined\n(Both Var)"]
    x = np.arange(len(conditions))

    # 子圖 1: PR-AUC 跨輪標準差
    axes[0].bar(x, df_summary["pr_auc_across_runs_std"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title("PR-AUC Standard Deviation Across 5 Runs", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Std Dev of Run Means", fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    for i, v in enumerate(df_summary["pr_auc_across_runs_std"]):
        axes[0].text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    # 子圖 2: AR@0.95 跨輪標準差
    axes[1].bar(x, df_summary["ar95_across_runs_std"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title("AR@0.95 Standard Deviation Across 5 Runs", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Std Dev of Run Means (x-factor)", fontsize=10)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    for i, v in enumerate(df_summary["ar95_across_runs_std"]):
        axes[1].text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Lab 08: Variance Decomposition (Algorithm Seed vs Data Split Variation)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [圖檔已儲存] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Lab 08: Seed Stability and Variance Decomposition")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print(f"Lab 08: Seed Stability & Variance Decomposition (模式: {args.mode})")
    print("=" * 80)

    X_train, y_train = load_train_data()

    n_splits = 3 if args.mode == "quick" else 5
    df_runs, df_summary = run_variance_decomposition(
        X_train, y_train, model_code="M3", n_splits=n_splits
    )

    out_csv = os.path.join(OUTPUT_DIR, "seed_stability.csv")
    df_runs.to_csv(out_csv, index=False)
    df_runs.to_csv(os.path.join(SNAPSHOT_DIR, "lab_08_seed_stability_snapshot.csv"), index=False)

    summary_csv = os.path.join(OUTPUT_DIR, "seed_variance_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_summary.to_csv(os.path.join(SNAPSHOT_DIR, "lab_08_seed_variance_summary_snapshot.csv"), index=False)

    png_path = os.path.join(OUTPUT_DIR, "seed_variance_decomposition.png")
    plot_seed_variance_decomposition(df_summary, png_path)

    print("\n>> Lab 08 隨機性分解匯總表:")
    print(df_summary[["condition", "pr_auc_across_runs_std", "pr_auc_within_fold_std_mean", "ar95_across_runs_std", "ar95_within_fold_std_mean"]].to_string(index=False))

    print_formal_e7b_comparison()

    total_time = time.time() - start_time
    save_manifest(
        output_dir=OUTPUT_DIR,
        lab_id=LAB_ID,
        params={"mode": args.mode, "n_splits": n_splits, "seed_list": SEED_LIST},
        elapsed_seconds=total_time,
        seed=42,
        extra={
            "execution_seconds": total_time,
            "alg_prauc_std": float(df_summary[df_summary["condition"] == "1_algorithm_randomness_only"]["pr_auc_across_runs_std"].values[0]),
            "split_prauc_std": float(df_summary[df_summary["condition"] == "2_data_split_variation_only"]["pr_auc_across_runs_std"].values[0]),
        },
    )
    print(f"\n[完成] Lab 08 執行總耗時: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()
