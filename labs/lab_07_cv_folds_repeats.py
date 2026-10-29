"""
lab_07_cv_folds_repeats.py - 交叉驗證折數 (K) 與重複次數 (R) 效應與樣本稀少階梯化實驗

【教學核心目標】
1. 探索 K in {3, 5, 10} x R in {1, 3} 的折內正樣本數與評估變異：
   - 記錄每折驗證集正樣本數 (min, max, mean)。
   - 評估 PR-AUC 與 AR@0.95 的折間標準差 (Std Dev) 與重複平均後穩定度。
   - 記錄各設定之整體運算耗時 (線性隨 K * R 成長)。
2. 正樣本數對階梯化與評估不確定性的影響：
   - 嚴格僅在訓練集內部，透過分層子抽樣保留比例 {0.1, 0.2, 0.3} 進行 50 次隨機分層切分。
   - 當驗證集正樣本數極端稀少時 (例如 10% 比例下僅 ~3.5 家破產)，單一破產樣本的去留
     會使 Recall 產生 1/P 的劇烈離散跳躍，致使 Clopper-Pearson 95% 信心區間寬度顯著放大。
3. 驗證分層切分 (Stratified Split) 的正樣本分派：
   - 實作單元測試，驗證各折破產企業數之最大與最小差額 <= 1。

【防護守則】
- 僅呼叫 load_train_data()，絕不讀取測試集或 E6_LOCK。
- 產物輸出至 labs/outputs/lab_07/ 與 docs/code_learning/lab_snapshots/。
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
from scipy.stats import beta
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit

from labs._common import (
    SNAPSHOT_DIR,
    build_pipeline,
    evaluate_fold_predictions,
    get_score,
    load_best_params,
    load_train_data,
    save_manifest,
)

LAB_ID = "lab_07"
OUTPUT_DIR = os.path.join("labs", "outputs", LAB_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def clopper_pearson_width(k: int, n: int, alpha: float = 0.05) -> float:
    """計算二項分佈 Clopper-Pearson 95% 信心區間寬度。"""
    if n == 0:
        return 0.0
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return upper - lower


def verify_stratification_balance(y: pd.Series, n_splits: int, random_state: int = 42) -> bool:
    """單元測試：驗證分層抽樣在各折中的正樣本數最大差額 <= 1。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pos_counts = []
    for _, val_idx in skf.split(np.zeros(len(y)), y):
        pos_counts.append(int(y.iloc[val_idx].sum()))
    diff = max(pos_counts) - min(pos_counts)
    return diff <= 1


def run_cv_matrix_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_code: str = "M1",
    k_list: list[int] = [3, 5, 10],
    r_list: list[int] = [1, 3],
    random_state: int = 42,
) -> pd.DataFrame:
    """執行 K x R 交叉驗證矩陣實驗。"""
    cfg = load_best_params(model_code)
    m_params = cfg.get("best_params", {})
    imb = cfg.get("imbalance", "S0")

    records = []
    for K in k_list:
        for R in r_list:
            t0 = time.time()
            rskf = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=random_state)

            praucs, ar95s, pos_counts = [], [], []
            for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                pipe = build_pipeline(model_code, imbalance=imb, model_params=m_params, seed=random_state)
                pipe.fit(X_tr, y_tr)
                scores = get_score(pipe, X_val)
                m = evaluate_fold_predictions(y_val, scores)

                praucs.append(m["pr_auc"])
                ar95s.append(m["ar_95"])
                pos_counts.append(int(y_val.sum()))

            elapsed = time.time() - t0
            records.append({
                "model_code": model_code,
                "n_splits_K": K,
                "n_repeats_R": R,
                "total_folds": K * R,
                "val_positives_min": int(np.min(pos_counts)),
                "val_positives_max": int(np.max(pos_counts)),
                "val_positives_mean": float(np.mean(pos_counts)),
                "pr_auc_mean": float(np.mean(praucs)),
                "pr_auc_std": float(np.std(praucs)),
                "ar95_mean": float(np.mean(ar95s)),
                "ar95_std": float(np.std(ar95s)),
                "total_seconds": float(elapsed),
                "seconds_per_fold": float(elapsed / (K * R)),
            })
            print(
                f"  K={K:2d}, R={R} | 折內正樣本: {np.mean(pos_counts):.1f} (min={np.min(pos_counts)}, max={np.max(pos_counts)}) | "
                f"PR-AUC: {np.mean(praucs):.4f} (+/- {np.std(praucs):.4f}) | "
                f"AR@0.95: {np.mean(ar95s):.2f}x (+/- {np.std(ar95s):.2f}) | 耗時: {elapsed:.2f}s"
            )

    return pd.DataFrame(records)


def run_sample_size_ladder_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_code: str = "M1",
    subsample_ratios: list[float] = [0.1, 0.2, 0.3],
    n_iterations: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    在訓練集內部以不同子抽樣比例評估正樣本稀少帶來的評估變異放大。
    嚴格防護：僅在訓練集內部切分，不碰任何測試資料。
    """
    cfg = load_best_params(model_code)
    m_params = cfg.get("best_params", {})
    imb = cfg.get("imbalance", "S0")

    records = []
    print(f"\n[樣本階梯實驗] 評估保留比例 {subsample_ratios} 各進行 {n_iterations} 次隨機分層評估...")

    for ratio in subsample_ratios:
        sss = StratifiedShuffleSplit(n_splits=n_iterations, test_size=ratio, random_state=random_state)

        praucs, ar95s, cp_widths, pos_counts = [], [], [], []
        for seed_idx, (eval_tr_idx, eval_val_idx) in enumerate(sss.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[eval_tr_idx], y_train.iloc[eval_tr_idx]
            X_val, y_val = X_train.iloc[eval_val_idx], y_train.iloc[eval_val_idx]

            pipe = build_pipeline(model_code, imbalance=imb, model_params=m_params, seed=random_state + seed_idx)
            pipe.fit(X_tr, y_tr)
            scores = get_score(pipe, X_val)
            m = evaluate_fold_predictions(y_val, scores)

            p_val = int(y_val.sum())
            n_val = len(y_val)
            praucs.append(m["pr_auc"])
            ar95s.append(m["ar_95"])
            pos_counts.append(p_val)
            cp_widths.append(clopper_pearson_width(p_val, n_val))

        records.append({
            "subsample_ratio": ratio,
            "n_iterations": n_iterations,
            "val_samples_mean": float(len(X_train) * ratio),
            "val_positives_mean": float(np.mean(pos_counts)),
            "pr_auc_mean": float(np.mean(praucs)),
            "pr_auc_std": float(np.std(praucs)),
            "pr_auc_iqr": float(np.percentile(praucs, 75) - np.percentile(praucs, 25)),
            "ar95_mean": float(np.mean(ar95s)),
            "ar95_std": float(np.std(ar95s)),
            "ar95_iqr": float(np.percentile(ar95s, 75) - np.percentile(ar95s, 25)),
            "clopper_pearson_95ci_width_mean": float(np.mean(cp_widths)),
        })
        print(
            f"  保留比例: {ratio*100:4.1f}% | 驗證正樣本數: ~{np.mean(pos_counts):.1f} | "
            f"PR-AUC Std: {np.std(praucs):.4f} (IQR: {records[-1]['pr_auc_iqr']:.4f}) | "
            f"AR@0.95 Std: {np.std(ar95s):.2f}x | CP-95% CI 寬度: {np.mean(cp_widths):.4f}"
        )

    return pd.DataFrame(records)


def plot_cv_effects(df_matrix: pd.DataFrame, df_ladder: pd.DataFrame, out_path: str):
    """繪製 K x R 效應與樣本階梯化圖表。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子圖 1: K 與 R 對 PR-AUC 折間變異與耗時的影響
    ax1 = axes[0]
    for r in df_matrix["n_repeats_R"].unique():
        sub = df_matrix[df_matrix["n_repeats_R"] == r]
        ax1.plot(sub["n_splits_K"], sub["pr_auc_std"], marker="o", label=f"R={r} Repeats (PR-AUC Std)")
    ax1.set_title("Fold Variance (Std Dev) vs Splits K & Repeats R", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Splits (K)", fontsize=10)
    ax1.set_ylabel("PR-AUC Standard Deviation across Folds", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 子圖 2: 樣本稀少度對 Clopper-Pearson 寬度與 PR-AUC IQR 的影響
    ax2 = axes[1]
    ax2.plot(
        df_ladder["subsample_ratio"],
        df_ladder["pr_auc_std"],
        marker="s",
        color="#d62728",
        label="PR-AUC Std Dev (Empirical)",
    )
    ax2.plot(
        df_ladder["subsample_ratio"],
        df_ladder["clopper_pearson_95ci_width_mean"],
        marker="^",
        color="#2ca02c",
        label="Clopper-Pearson 95% CI Width",
    )
    ax2.set_title("Uncertainty Amplification by Sample Sparsity", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Subsample Ratio (Validation Set Size)", fontsize=10)
    ax2.set_ylabel("Uncertainty Measure (Std / CI Width)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [圖檔已儲存] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Lab 07: CV Folds and Repeats Effects")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print(f"Lab 07: CV Folds, Repeats, and Sample Sparsity Effects (模式: {args.mode})")
    print("=" * 80)

    X_train, y_train = load_train_data()

    # 1. 驗證分層切分正樣本平衡性 (差額 <= 1)
    for k in [3, 5, 10]:
        is_balanced = verify_stratification_balance(y_train, n_splits=k)
        print(f"  [分層單元測試] K={k:2d}: 正樣本分配極大極小差額 <= 1 驗證通過: {is_balanced}")
        assert is_balanced, f"Stratification failed for K={k}"

    # 2. K x R 矩陣實驗
    print("\n[1/2] 執行 K in {3, 5, 10} x R in {1, 3} 交叉驗證矩陣實驗...")
    k_list = [3, 5, 10]
    r_list = [1, 3] if args.mode != "quick" else [1, 2]
    df_matrix = run_cv_matrix_experiment(
        X_train, y_train, model_code="M1", k_list=k_list, r_list=r_list, random_state=42
    )
    matrix_csv = os.path.join(OUTPUT_DIR, "cv_matrix.csv")
    df_matrix.to_csv(matrix_csv, index=False)
    df_matrix.to_csv(os.path.join(SNAPSHOT_DIR, "lab_07_cv_matrix_snapshot.csv"), index=False)

    # 3. 樣本階梯實驗 (訓練集內部子抽樣)
    print("\n[2/2] 執行訓練集內部樣本階梯化實驗 (評估正樣本稀少之離散性)...")
    n_iters = 20 if args.mode == "quick" else 50
    df_ladder = run_sample_size_ladder_experiment(
        X_train, y_train, model_code="M1",
        subsample_ratios=[0.1, 0.2, 0.3],
        n_iterations=n_iters,
        random_state=42,
    )
    ladder_csv = os.path.join(OUTPUT_DIR, "sample_size_ladder.csv")
    df_ladder.to_csv(ladder_csv, index=False)
    df_ladder.to_csv(os.path.join(SNAPSHOT_DIR, "lab_07_sample_size_ladder_snapshot.csv"), index=False)

    # 4. 繪圖
    png_path = os.path.join(OUTPUT_DIR, "cv_effects.png")
    plot_cv_effects(df_matrix, df_ladder, png_path)

    total_time = time.time() - start_time
    save_manifest(
        output_dir=OUTPUT_DIR,
        lab_id=LAB_ID,
        params={"mode": args.mode, "k_list": k_list, "r_list": r_list, "ladder_iters": n_iters},
        elapsed_seconds=total_time,
        seed=42,
        extra={
            "execution_seconds": total_time,
            "k5_r1_prauc_mean": float(df_matrix[(df_matrix["n_splits_K"] == 5) & (df_matrix["n_repeats_R"] == 1)]["pr_auc_mean"].values[0]),
            "k5_r1_prauc_std": float(df_matrix[(df_matrix["n_splits_K"] == 5) & (df_matrix["n_repeats_R"] == 1)]["pr_auc_std"].values[0]),
            "k10_r1_val_pos_mean": float(df_matrix[(df_matrix["n_splits_K"] == 10) & (df_matrix["n_repeats_R"] == 1)]["val_positives_mean"].values[0]),
        },
    )
    print(f"\n[完成] Lab 07 執行總耗時: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()
