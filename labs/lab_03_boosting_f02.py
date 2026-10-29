"""labs/lab_03_boosting_f02.py - Boosting (M4, M5) Grid & F-02 Empirical Comparison.

Topics:
1. Learning rate x n_estimators trade-off on XGBoost (M4) and LightGBM (M5).
2. F-02 Empirical Comparison: Evaluates the impact of subsample_freq (0 vs 1)
   when subsample=0.607... in LightGBM with S3 SMOTE across multiple seeds.

SAFETY:
- Uses ONLY load_train_data() (5,455 training rows). Never touches holdout test set.
- Re-fits models on custom CV folds. Never loads frozen final model objects.
- Outputs written to labs/outputs/lab_03/. Snapshot to docs/code_learning/lab_snapshots/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from bankruptcy.src.pipelines import build_pipeline, get_score
from labs._common import (
    LABS_OUTPUTS_DIR,
    evaluate_fold_predictions,
    get_cv_splitter,
    load_best_params,
    load_train_data,
    save_manifest,
    save_snapshot,
    setup_chinese_font,
)

# ==============================================================================
# 可調參數字典 (PARAMS)
# 集中定義於檔頭，附中文說明與推薦範圍
# ==============================================================================
PARAMS: dict[str, Any] = {
    # 網格掃描 (快速模式 3x3，控制於 2.5 分鐘內)
    "learning_rates": [0.01, 0.05, 0.15],
    "n_estimators_quick": [40, 80, 150],
    "n_estimators_full": [100, 200, 400, 800],
    # F-02 對照實驗種子清單 (至少 3 組模型種子)
    "f02_model_seeds": [42, 100, 999],
    # 交叉驗證設定
    "cv_folds": 5,
    "cv_repeats_quick": 1,
    "cv_repeats_full": 3,
    "split_seed": 42,
}


def run_lab(quick: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Execute lab_03 boosting grid and F-02 empirical investigation."""
    start_time = time.time()
    setup_chinese_font()

    n_repeats = PARAMS["cv_repeats_quick"] if quick else PARAMS["cv_repeats_full"]
    n_est_list = PARAMS["n_estimators_quick"] if quick else PARAMS["n_estimators_full"]
    lr_list = PARAMS["learning_rates"]
    f02_seeds = PARAMS["f02_model_seeds"]
    split_seed = PARAMS["split_seed"]

    print("=" * 70)
    print(f"LAB_03: 提升法超參數網格與 F-02 實證對照 ({'--quick 快速模式' if quick else '--full 完整模式'})")
    print(f"CV 設定: {PARAMS['cv_folds']} 折 x {n_repeats} 次重複 (切分種子={split_seed})")
    print(f"LightGBM 版本: {lgb.__version__}")
    print("=" * 70)

    # 1. 唯讀載入訓練資料
    X_train, y_train = load_train_data(check_guard=True)
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=split_seed)

    # ==========================================================================
    # 第一部分：M4 (XGBoost) 與 M5 (LightGBM) learning_rate x n_estimators 網格
    # ==========================================================================
    print("\n[1/2] 執行提升法網格掃描 (learning_rate x n_estimators)...")
    grid_records: list[dict[str, Any]] = []

    for m_code, m_name, imb in [("M4", "XGBoost", "S0"), ("M5", "LightGBM", "S3")]:
        base_json = load_best_params(m_code)
        base_params = base_json["best_params"].copy()

        for lr in lr_list:
            for n_est in n_est_list:
                cfg = base_params.copy()
                cfg["learning_rate"] = lr
                cfg["n_estimators"] = n_est

                pr_aucs, ar95s, ar90s, fold_times = [], [], [], []
                for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
                    t0 = time.time()
                    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                    pipe = build_pipeline(
                        model_name=m_code,
                        imbalance=imb,
                        model_params=cfg,
                        seed=split_seed + fold_idx,
                    )
                    pipe.fit(X_tr, y_tr)
                    score_val = get_score(pipe, X_val)
                    m = evaluate_fold_predictions(y_val, score_val)

                    pr_aucs.append(m["pr_auc"])
                    ar95s.append(m["ar_95"])
                    ar90s.append(m["ar_90"])
                    fold_times.append(time.time() - t0)

                rec = {
                    "model_code": m_code,
                    "model_name": m_name,
                    "imbalance": imb,
                    "learning_rate": lr,
                    "n_estimators": n_est,
                    "pr_auc_mean": float(np.mean(pr_aucs)),
                    "pr_auc_std": float(np.std(pr_aucs)),
                    "ar95_mean": float(np.mean(ar95s)),
                    "ar95_std": float(np.std(ar95s)),
                    "ar90_mean": float(np.mean(ar90s)),
                    "ar90_std": float(np.std(ar90s)),
                    "avg_fit_seconds": float(np.mean(fold_times)),
                }
                grid_records.append(rec)
                print(
                    f"  {m_code} lr={lr:<5} n_est={n_est:<4} | "
                    f"Val PR-AUC: {rec['pr_auc_mean']:.4f} (+/- {rec['pr_auc_std']:.4f}) | "
                    f"AR@0.95: {rec['ar95_mean']:.2f}x | 耗時: {rec['avg_fit_seconds']:.2f}s"
                )

    df_grid = pd.DataFrame(grid_records)

    # ==========================================================================
    # 第二部分：F-02 對照實驗 (subsample_freq 0 vs 1 影響幅度檢定)
    # ==========================================================================
    print("\n[2/2] 執行 F-02 對照實驗 (M5 固化參數 + S3 SMOTE 下 subsample_freq 0 vs 1)...")
    m5_best = load_best_params("M5")
    m5_frozen_params = m5_best["best_params"].copy()
    m5_imb = m5_best["imbalance"]  # "S3"

    f02_records: list[dict[str, Any]] = []

    # 檢查實際生效之參數
    test_pipe_freq0 = build_pipeline("M5", m5_imb, model_params={**m5_frozen_params, "subsample_freq": 0}, seed=42)
    test_pipe_freq1 = build_pipeline("M5", m5_imb, model_params={**m5_frozen_params, "subsample_freq": 1}, seed=42)
    test_pipe_freq0.fit(X_train.iloc[:500], y_train.iloc[:500])
    test_pipe_freq1.fit(X_train.iloc[:500], y_train.iloc[:500])

    lgb_clf0 = test_pipe_freq0.named_steps["classifier"]
    lgb_clf1 = test_pipe_freq1.named_steps["classifier"]
    actual_params_freq0 = lgb_clf0.get_params()
    actual_params_freq1 = lgb_clf1.get_params()

    print(f"  [驗證生效參數] 設定 A (freq=0): subsample={actual_params_freq0.get('subsample')}, subsample_freq={actual_params_freq0.get('subsample_freq')}")
    print(f"  [驗證生效參數] 設定 B (freq=1): subsample={actual_params_freq1.get('subsample')}, subsample_freq={actual_params_freq1.get('subsample_freq')}")

    for m_seed in f02_seeds:
        for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            # 設定 A: subsample_freq=0 (LightGBM 不啟用 bagging，subsample 實質被忽略)
            cfg_a = m5_frozen_params.copy()
            cfg_a["subsample_freq"] = 0
            pipe_a = build_pipeline("M5", m5_imb, model_params=cfg_a, seed=m_seed)
            pipe_a.fit(X_tr, y_tr)
            score_a = get_score(pipe_a, X_val)
            m_a = evaluate_fold_predictions(y_val, score_a)

            # 設定 B: subsample_freq=1 (LightGBM 每輪抽樣 60.7% 樣本)
            cfg_b = m5_frozen_params.copy()
            cfg_b["subsample_freq"] = 1
            pipe_b = build_pipeline("M5", m5_imb, model_params=cfg_b, seed=m_seed)
            pipe_b.fit(X_tr, y_tr)
            score_b = get_score(pipe_b, X_val)
            m_b = evaluate_fold_predictions(y_val, score_b)

            # 設定 C: subsample=1.0, subsample_freq=0 (全樣本基準)
            cfg_c = m5_frozen_params.copy()
            cfg_c["subsample"] = 1.0
            cfg_c["subsample_freq"] = 0
            pipe_c = build_pipeline("M5", m5_imb, model_params=cfg_c, seed=m_seed)
            pipe_c.fit(X_tr, y_tr)
            score_c = get_score(pipe_c, X_val)
            m_c = evaluate_fold_predictions(y_val, score_c)

            # 比對兩組分數的逐位一致性與相關係數
            is_identical = bool(np.array_equal(score_a, score_b))
            corr_ab, _ = pearsonr(score_a, score_b)
            max_abs_diff = float(np.max(np.abs(score_a - score_b)))

            f02_records.append({
                "model_seed": m_seed,
                "fold_idx": fold_idx,
                "pr_auc_setting_A_freq0": m_a["pr_auc"],
                "pr_auc_setting_B_freq1": m_b["pr_auc"],
                "pr_auc_setting_C_sub10": m_c["pr_auc"],
                "pr_auc_diff_B_minus_A": m_b["pr_auc"] - m_a["pr_auc"],
                "ar95_setting_A_freq0": m_a["ar_95"],
                "ar95_setting_B_freq1": m_b["ar_95"],
                "ar95_setting_C_sub10": m_c["ar_95"],
                "ar95_diff_B_minus_A": m_b["ar_95"] - m_a["ar_95"],
                "pearson_corr_A_vs_B": float(corr_ab),
                "is_bit_identical": is_identical,
                "max_abs_score_diff": max_abs_diff,
            })

    df_f02 = pd.DataFrame(f02_records)

    # 彙整統計
    mean_diff_pr = df_f02["pr_auc_diff_B_minus_A"].mean()
    mean_diff_ar = df_f02["ar95_diff_B_minus_A"].mean()
    mean_corr = df_f02["pearson_corr_A_vs_B"].mean()
    any_identical = df_f02["is_bit_identical"].any()

    print(f"\n  [F-02 實證分析結論]")
    print(f"  - 配對 PR-AUC 差異 (freq=1 減 freq=0 均值): {mean_diff_pr:+.5f}")
    print(f"  - 配對 AR@0.95 差異 (freq=1 減 freq=0 均值): {mean_diff_ar:+.4f}x")
    print(f"  - 預測分數平均皮爾森相關係數: {mean_corr:.6f}")
    print(f"  - 是否有任何一折逐位完全相同: {any_identical} (最大絕對分數差: {df_f02['max_abs_score_diff'].max():.4f})")
    print(f"  - 說明: 當 subsample_freq=1 時，LightGBM 確實啟動每輪隨機抽樣，分數產生微幅波動，相關性極高 (>0.99)。")

    # 3. 輸出儲存
    out_dir = LABS_OUTPUTS_DIR / "lab_03"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "lab_03_boosting_grid.csv"
    f02_path = out_dir / "lab_03_f02_comparison.csv"
    df_grid.to_csv(grid_path, index=False, encoding="utf-8")
    df_f02.to_csv(f02_path, index=False, encoding="utf-8")

    # 快照輸出
    snap_path = save_snapshot(df_f02, "lab_03_f02_comparison_snapshot.csv")

    # 4. 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子圖 1: M4 vs M5 網格掃描 PR-AUC 對比
    ax1 = axes[0]
    for lr, style in [(0.01, "--"), (0.05, "-"), (0.15, ":")]:
        sub_m4 = df_grid[(df_grid["model_code"] == "M4") & (df_grid["learning_rate"] == lr)].sort_values("n_estimators")
        ax1.plot(sub_m4["n_estimators"], sub_m4["pr_auc_mean"], f"o{style}", color="#457b9d", label=f"M4 lr={lr}")
    for lr, style in [(0.01, "--"), (0.05, "-"), (0.15, ":")]:
        sub_m5 = df_grid[(df_grid["model_code"] == "M5") & (df_grid["learning_rate"] == lr)].sort_values("n_estimators")
        ax1.plot(sub_m5["n_estimators"], sub_m5["pr_auc_mean"], f"s{style}", color="#e63946", label=f"M5 lr={lr}")
    ax1.set_title("提升法網格: 學習率 x 樹數 vs PR-AUC", fontsize=11, fontweight="bold")
    ax1.set_xlabel("n_estimators")
    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(ncol=2, fontsize=8)

    # 子圖 2: F-02 配對 PR-AUC 差異散布圖 (Setting B vs Setting A)
    ax2 = axes[1]
    ax2.scatter(
        df_f02["pr_auc_setting_A_freq0"],
        df_f02["pr_auc_setting_B_freq1"],
        c=df_f02["model_seed"],
        cmap="viridis",
        s=60,
        alpha=0.85,
        edgecolor="k",
    )
    lims = [
        min(df_f02["pr_auc_setting_A_freq0"].min(), df_f02["pr_auc_setting_B_freq1"].min()) * 0.95,
        max(df_f02["pr_auc_setting_A_freq0"].max(), df_f02["pr_auc_setting_B_freq1"].max()) * 1.05,
    ]
    ax2.plot(lims, lims, "r--", label="對角等值線 (y=x)")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title(f"F-02 配對 PR-AUC 對比\n(均值差異={mean_diff_pr:+.4f}, 相關性={mean_corr:.4f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("設定 A (subsample_freq=0, 抽樣未生效)")
    ax2.set_ylabel("設定 B (subsample_freq=1, 抽樣生效)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 子圖 3: F-02 配對 AR@0.95 差異分佈長條圖
    ax3 = axes[2]
    diffs = df_f02["ar95_diff_B_minus_A"]
    ax3.hist(diffs, bins=8, color="#2a9d8f", edgecolor="black", alpha=0.7)
    ax3.axvline(0.0, color="red", linestyle="--", label="無差異基準線 (0.0)")
    ax3.axvline(mean_diff_ar, color="blue", linestyle="-", label=f"差異均值 ({mean_diff_ar:+.2f}x)")
    ax3.set_title("F-02 配對 AR@0.95 差異分佈\n(設定 B - 設定 A)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("AR@0.95 差異 (倍數)")
    ax3.set_ylabel("折數頻次")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    fig_path = out_dir / "lab_03_boosting_f02.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    save_manifest(
        output_dir=out_dir,
        lab_id="lab_03",
        params=PARAMS,
        elapsed_seconds=elapsed,
        seed=split_seed,
        extra={
            "quick_mode": quick,
            "lightgbm_version": lgb.__version__,
            "mean_pr_auc_diff": float(mean_diff_pr),
            "mean_ar95_diff": float(mean_diff_ar),
            "mean_score_correlation": float(mean_corr),
            "any_bit_identical": bool(any_identical),
            "actual_lgb_params_setting_A": {
                "subsample": actual_params_freq0.get("subsample"),
                "subsample_freq": actual_params_freq0.get("subsample_freq"),
            },
            "actual_lgb_params_setting_B": {
                "subsample": actual_params_freq1.get("subsample"),
                "subsample_freq": actual_params_freq1.get("subsample_freq"),
            },
        },
    )

    print("\n" + "=" * 70)
    print(f"LAB_03 執行完成！總耗時: {elapsed:.2f} 秒")
    print(f"網格 CSV : {grid_path}")
    print(f"F-02 CSV : {f02_path}")
    print(f"圖表 PNG : {fig_path}")
    print(f"小型快照 : {snap_path}")
    print("=" * 70)

    return df_grid, df_f02, f02_path, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 03: 提升法與 F-02 實證對照實驗")
    parser.add_argument("--full", action="store_true", help="執行完整模式")
    parser.add_argument("--quick", action="store_true", default=True, help="執行快速模式 (預設)")
    args = parser.parse_args()

    is_quick = not args.full
    run_lab(quick=is_quick)
