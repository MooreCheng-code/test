"""labs/lab_03_boosting_f02.py - Boosting (M4, M5) Grid & F-02 Empirical Comparison.

Topics:
1. Learning rate x n_estimators trade-off on XGBoost (M4) and LightGBM (M5).
2. F-02 Empirical Comparison: Evaluates the impact of subsample_freq (0 vs 1)
   when subsample=0.607... in LightGBM with S3 SMOTE.
   Supports --quick (fast), --standard, and --f02-extended (45 pairs with noise reference & CI).

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
from scipy.stats import pearsonr, spearmanr, t

from bankruptcy.src.pipelines import build_pipeline, get_score
from labs._common import (
    get_pipeline_flags,
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
    # 網格掃描學習率與樹數
    "learning_rates_quick": [0.02, 0.08],
    "learning_rates_standard": [0.01, 0.05, 0.15],
    "n_estimators_quick": [40, 100],
    "n_estimators_standard": [40, 80, 150],
    "n_estimators_full": [100, 200, 400, 800],
    # F-02 對照實驗種子清單
    "f02_seeds_quick": [42, 100],
    "f02_seeds_standard": [42, 100, 999],
    "f02_seeds_extended": [42, 100, 999],
    # 交叉驗證設定
    "cv_folds": 5,
    "cv_repeats_quick": 1,
    "cv_repeats_standard": 1,
    "cv_repeats_extended": 3,
    "split_seed": 42,
}


def compute_confidence_intervals(diffs: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Compute Student's t-interval and Bootstrap percentile interval for paired differences."""
    n = len(diffs)
    mean_val = float(np.mean(diffs))
    std_val = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_val = std_val / np.sqrt(n) if n > 0 else 0.0

    # Student's t-interval
    t_crit = float(t.ppf(1.0 - alpha / 2.0, df=n - 1)) if n > 1 else 1.96
    t_ci_lower = mean_val - t_crit * se_val
    t_ci_upper = mean_val + t_crit * se_val

    # Bootstrap percentile interval (2000 resamples)
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diffs, size=n, replace=True)) for _ in range(2000)]
    boot_ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    boot_ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": mean_val,
        "std": std_val,
        "se": se_val,
        "t_ci_95": [float(t_ci_lower), float(t_ci_upper)],
        "boot_ci_95": [boot_ci_lower, boot_ci_upper],
    }


def run_lab(mode: str = "quick", pipeline: str = "project") -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    pipe_flags = get_pipeline_flags(pipeline)
    """Execute lab_03 boosting grid and F-02 empirical investigation."""
    start_time = time.time()
    setup_chinese_font()

    split_seed = PARAMS["split_seed"]

    if mode == "f02_extended":
        n_repeats = PARAMS["cv_repeats_extended"]  # 3 repeats -> 15 folds
        f02_seeds = PARAMS["f02_seeds_extended"]   # 3 seeds -> 45 pairs
        run_grid = False  # Extended mode focuses exclusively on F-02 deep dive
        lr_list = []
        n_est_list = []
    elif mode == "standard":
        n_repeats = PARAMS["cv_repeats_standard"]  # 1 repeat -> 5 folds
        f02_seeds = PARAMS["f02_seeds_standard"]   # 3 seeds -> 15 pairs
        run_grid = True
        lr_list = PARAMS["learning_rates_standard"]
        n_est_list = PARAMS["n_estimators_standard"]
    elif mode == "full":
        n_repeats = 3
        f02_seeds = PARAMS["f02_seeds_standard"]
        run_grid = True
        lr_list = PARAMS["learning_rates_standard"]
        n_est_list = PARAMS["n_estimators_full"]
    else:  # quick (default)
        n_repeats = PARAMS["cv_repeats_quick"]     # 1 repeat -> 5 folds
        f02_seeds = PARAMS["f02_seeds_quick"]      # 2 seeds -> 10 pairs
        run_grid = True
        lr_list = PARAMS["learning_rates_quick"]
        n_est_list = PARAMS["n_estimators_quick"]

    print("=" * 70)
    print(f"LAB_03: 提升法超參數網格與 F-02 實證對照 ({mode} 模式)")
    print(f"CV 設定: {PARAMS['cv_folds']} 折 x {n_repeats} 次重複 (切分種子={split_seed})")
    print(f"LightGBM 執行版本: {lgb.__version__}")
    print("=" * 70)

    # 1. 唯讀載入訓練資料
    X_train, y_train = load_train_data(check_guard=True)
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=split_seed)

    # ==========================================================================
    # 第一部分：M4 (XGBoost) 與 M5 (LightGBM) learning_rate x n_estimators 網格
    # ==========================================================================
    grid_records: list[dict[str, Any]] = []
    if run_grid:
        print("\n[1/2] 執行提升法網格掃描 (learning_rate x n_estimators)...")
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
                            **pipe_flags,
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
                        f"AR@0.95: {rec['ar95_mean']:.2f}x | 耗時: {rec['avg_fit_seconds']:.2f}s/fold"
                    )
    else:
        print("\n[1/2] --f02-extended 模式跳過網格掃描，專注執行 F-02 深入對照...")

    df_grid = pd.DataFrame(grid_records)

    # ==========================================================================
    # 第二部分：F-02 對照實驗 (subsample_freq 0 vs 1 影響幅度檢定)
    # ==========================================================================
    print(f"\n[2/2] 執行 F-02 對照實驗 (M5 固化參數 + S3 SMOTE 下 subsample_freq 0 vs 1)...")
    m5_best = load_best_params("M5")
    m5_frozen_params = m5_best["best_params"].copy()
    m5_imb = m5_best["imbalance"]  # "S3"

    f02_records: list[dict[str, Any]] = []

    # 驗證實際生效之參數
    test_pipe_freq0 = build_pipeline("M5", m5_imb, model_params={**m5_frozen_params, "subsample_freq": 0}, seed=42, **pipe_flags)
    test_pipe_freq1 = build_pipeline("M5", m5_imb, model_params={**m5_frozen_params, "subsample_freq": 1}, seed=42, **pipe_flags)
    test_pipe_freq0.fit(X_train.iloc[:200], y_train.iloc[:200])
    test_pipe_freq1.fit(X_train.iloc[:200], y_train.iloc[:200])

    actual_params_freq0 = test_pipe_freq0.named_steps["classifier"].get_params()
    actual_params_freq1 = test_pipe_freq1.named_steps["classifier"].get_params()

    print(f"  [驗證生效參數] 設定 A (freq=0): subsample={actual_params_freq0.get('subsample')}, subsample_freq={actual_params_freq0.get('subsample_freq')}")
    print(f"  [驗證生效參數] 設定 B (freq=1): subsample={actual_params_freq1.get('subsample')}, subsample_freq={actual_params_freq1.get('subsample_freq')}")

    # 用於儲存各模型種子在 Setting B 下的分數與指標，以計算「種子雜訊參照」
    seed_b_scores: dict[int, dict[int, np.ndarray]] = {}  # {seed: {fold: scores}}
    seed_b_prauc: dict[int, list[float]] = {}
    seed_b_ar95: dict[int, list[float]] = {}

    for m_seed in f02_seeds:
        seed_b_scores[m_seed] = {}
        seed_b_prauc[m_seed] = []
        seed_b_ar95[m_seed] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            # 設定 A: subsample_freq=0 (LightGBM 官方規格：不啟用 bagging，subsample 實質被忽略)
            cfg_a = m5_frozen_params.copy()
            cfg_a["subsample_freq"] = 0
            pipe_a = build_pipeline("M5", m5_imb, model_params=cfg_a, seed=m_seed, **pipe_flags)
            pipe_a.fit(X_tr, y_tr)
            score_a = get_score(pipe_a, X_val)
            m_a = evaluate_fold_predictions(y_val, score_a)

            # 設定 B: subsample_freq=1 (LightGBM 每輪隨機抽樣 60.7% 樣本)
            cfg_b = m5_frozen_params.copy()
            cfg_b["subsample_freq"] = 1
            pipe_b = build_pipeline("M5", m5_imb, model_params=cfg_b, seed=m_seed, **pipe_flags)
            pipe_b.fit(X_tr, y_tr)
            score_b = get_score(pipe_b, X_val)
            m_b = evaluate_fold_predictions(y_val, score_b)

            # 記錄 Setting B 數據用於雜訊參照
            seed_b_scores[m_seed][fold_idx] = score_b
            seed_b_prauc[m_seed].append(m_b["pr_auc"])
            seed_b_ar95[m_seed].append(m_b["ar_95"])

            # 設定 C: subsample=1.0, subsample_freq=0 (全樣本基準)
            cfg_c = m5_frozen_params.copy()
            cfg_c["subsample"] = 1.0
            cfg_c["subsample_freq"] = 0
            pipe_c = build_pipeline("M5", m5_imb, model_params=cfg_c, seed=m_seed, **pipe_flags)
            pipe_c.fit(X_tr, y_tr)
            score_c = get_score(pipe_c, X_val)
            m_c = evaluate_fold_predictions(y_val, score_c)

            # 相關係數與逐位比對
            is_identical = bool(np.array_equal(score_a, score_b))
            corr_pearson, _ = pearsonr(score_a, score_b)
            corr_spearman, _ = spearmanr(score_a, score_b)
            max_abs_diff = float(np.max(np.abs(score_a - score_b)))

            f02_records.append({
                "model_seed": m_seed,
                "fold_idx": fold_idx,
                "pr_auc_setting_A_freq0": m_a["pr_auc"],
                "pr_auc_setting_B_freq1": m_b["pr_auc"],
                "pr_auc_setting_C_sub10": m_c["pr_auc"],
                "pr_auc_diff_B_minus_A": float(m_b["pr_auc"] - m_a["pr_auc"]),
                "ar95_setting_A_freq0": m_a["ar_95"],
                "ar95_setting_B_freq1": m_b["ar_95"],
                "ar95_setting_C_sub10": m_c["ar_95"],
                "ar95_diff_B_minus_A": float(m_b["ar_95"] - m_a["ar_95"]),
                "ar90_setting_A_freq0": m_a["ar_90"],
                "ar90_setting_B_freq1": m_b["ar_90"],
                "ar90_diff_B_minus_A": float(m_b["ar_90"] - m_a["ar_90"]),
                "brier_setting_A_freq0": m_a["brier"],
                "brier_setting_B_freq1": m_b["brier"],
                "brier_diff_B_minus_A": float(m_b["brier"] - m_a["brier"]),
                "pearson_corr_A_vs_B": float(corr_pearson),
                "spearman_corr_A_vs_B": float(corr_spearman),
                "is_bit_identical": is_identical,
                "max_abs_score_diff": max_abs_diff,
            })

    df_f02 = pd.DataFrame(f02_records)

    # 4. 統計推論與信賴區間分析
    pr_diffs = df_f02["pr_auc_diff_B_minus_A"].to_numpy()
    ar95_diffs = df_f02["ar95_diff_B_minus_A"].to_numpy()
    ar90_diffs = df_f02["ar90_diff_B_minus_A"].to_numpy()
    brier_diffs = df_f02["brier_diff_B_minus_A"].to_numpy()

    ci_pr = compute_confidence_intervals(pr_diffs)
    ci_ar95 = compute_confidence_intervals(ar95_diffs)
    ci_ar90 = compute_confidence_intervals(ar90_diffs)
    ci_brier = compute_confidence_intervals(brier_diffs)

    # 符號一致性比例
    sign_pos_pr = float(np.mean(pr_diffs > 0))
    sign_pos_ar95 = float(np.mean(ar95_diffs < 0))  # AR 越低越好

    # 5. 計算「模型種子雜訊參照」
    # 在同一切分下，改變 Setting B 的模型種子所帶來的指標差異量級
    seed_noise_pr_list = []
    seed_noise_ar95_list = []
    if len(f02_seeds) >= 2:
        for s1_idx in range(len(f02_seeds)):
            for s2_idx in range(s1_idx + 1, len(f02_seeds)):
                s1 = f02_seeds[s1_idx]
                s2 = f02_seeds[s2_idx]
                p1 = np.array(seed_b_prauc[s1])
                p2 = np.array(seed_b_prauc[s2])
                a1 = np.array(seed_b_ar95[s1])
                a2 = np.array(seed_b_ar95[s2])
                seed_noise_pr_list.extend(np.abs(p1 - p2).tolist())
                seed_noise_ar95_list.extend(np.abs(a1 - a2).tolist())

    mean_seed_noise_pr = float(np.mean(seed_noise_pr_list)) if seed_noise_pr_list else 0.0
    mean_seed_noise_ar95 = float(np.mean(seed_noise_ar95_list)) if seed_noise_ar95_list else 0.0

    # 判定結論措辭
    pr_ci_contains_zero = ci_pr["t_ci_95"][0] <= 0.0 <= ci_pr["t_ci_95"][1]
    diff_below_seed_noise = abs(ci_pr["mean"]) < mean_seed_noise_pr

    if pr_ci_contains_zero and diff_below_seed_noise:
        conclusion_text = (
            "在本 lab 設定下沒有偵測到顯著差異（配對差 95% 信賴區間涵蓋 0，"
            f"且差異均值 {ci_pr['mean']:+.5f} 小於同設定換模型種子之雜訊水準 {mean_seed_noise_pr:.5f}；"
            "樣本量與檢定力有限）。"
        )
    else:
        conclusion_text = (
            f"在本 lab 設定下觀察到差異：PR-AUC 配對差均值為 {ci_pr['mean']:+.5f} "
            f"(95% CI: [{ci_pr['t_ci_95'][0]:.5f}, {ci_pr['t_ci_95'][1]:.5f}])，"
            "此結果僅代表本訓練集 CV 分割下之觀察，不外推到部署版 M5 在未見 E6 盲測集上的表現。"
        )

    print("\n" + "=" * 70)
    print(f"F-02 實證分析綜合報告 (配對總數: {len(df_f02)} 組)")
    print("=" * 70)
    print(f"1. 預測一致性檢驗:")
    print(f"   - 逐位完全相同比例: {float(np.mean(df_f02['is_bit_identical'])) * 100:.1f}%")
    print(f"   - 最大絕對分數差: {df_f02['max_abs_score_diff'].max():.4f}")
    print(f"   - 皮爾森相關係數均值: {df_f02['pearson_corr_A_vs_B'].mean():.6f}")
    print(f"   - 斯皮爾曼等級相關均值: {df_f02['spearman_corr_A_vs_B'].mean():.6f}")
    print(f"\n2. 配對指標差異 (設定 B - 設定 A):")
    print(f"   - PR-AUC  差均值: {ci_pr['mean']:+.5f} (Std: {ci_pr['std']:.5f}) | t-CI 95%: [{ci_pr['t_ci_95'][0]:+.5f}, {ci_pr['t_ci_95'][1]:+.5f}] | 正向折佔比: {sign_pos_pr*100:.1f}%")
    print(f"   - AR@0.95 差均值: {ci_ar95['mean']:+.4f}x (Std: {ci_ar95['std']:.4f}) | t-CI 95%: [{ci_ar95['t_ci_95'][0]:+.4f}, {ci_ar95['t_ci_95'][1]:+.4f}] | 改善折佔比: {sign_pos_ar95*100:.1f}%")
    print(f"   - AR@0.90 差均值: {ci_ar90['mean']:+.4f}x (Std: {ci_ar90['std']:.4f}) | t-CI 95%: [{ci_ar90['t_ci_95'][0]:+.4f}, {ci_ar90['t_ci_95'][1]:+.4f}]")
    print(f"   - Brier   差均值: {ci_brier['mean']:+.6f} | t-CI 95%: [{ci_brier['t_ci_95'][0]:+.6f}, {ci_brier['t_ci_95'][1]:+.6f}]")
    print(f"\n3. 模型種子雜訊參照 (設定 B 下跨模型種子之平均絕對差):")
    print(f"   - PR-AUC  種子雜訊均值: {mean_seed_noise_pr:.5f}")
    print(f"   - AR@0.95 種子雜訊均值: {mean_seed_noise_ar95:.4f}x")
    print(f"\n4. 依規則判定之結論:")
    print(f"   {conclusion_text}")
    print("   (注意：CV 折間相依性會使 t 區間與 bootstrap 區間略為偏窄；此為【一般經驗，未經本專案驗證】之統計特性)")
    print("=" * 70)

    # 6. 輸出儲存
    out_dir = LABS_OUTPUTS_DIR / "lab_03"
    out_dir.mkdir(parents=True, exist_ok=True)
    f02_filename = "lab_03_f02_extended.csv" if mode == "f02_extended" else "lab_03_f02_comparison.csv"
    f02_snapname = "lab_03_f02_extended_snapshot.csv" if mode == "f02_extended" else "lab_03_f02_comparison_snapshot.csv"

    grid_path = out_dir / "lab_03_boosting_grid.csv"
    f02_path = out_dir / f02_filename

    if run_grid:
        df_grid.to_csv(grid_path, index=False, encoding="utf-8")
    df_f02.to_csv(f02_path, index=False, encoding="utf-8")

    snap_path = save_snapshot(df_f02, f02_snapname)

    # 7. 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子圖 1: 網格或延伸分佈
    ax1 = axes[0]
    if run_grid and not df_grid.empty:
        for lr, style in [(0.02, "-"), (0.08, "--")]:
            sub_m4 = df_grid[(df_grid["model_code"] == "M4") & (df_grid["learning_rate"] == lr)].sort_values("n_estimators")
            if not sub_m4.empty:
                ax1.plot(sub_m4["n_estimators"], sub_m4["pr_auc_mean"], f"o{style}", color="#457b9d", label=f"M4 lr={lr}")
            sub_m5 = df_grid[(df_grid["model_code"] == "M5") & (df_grid["learning_rate"] == lr)].sort_values("n_estimators")
            if not sub_m5.empty:
                ax1.plot(sub_m5["n_estimators"], sub_m5["pr_auc_mean"], f"s{style}", color="#e63946", label=f"M5 lr={lr}")
        ax1.set_title("提升法網格: 學習率 x 樹數 vs PR-AUC", fontsize=11, fontweight="bold")
        ax1.set_xlabel("n_estimators")
        ax1.set_ylabel("PR-AUC")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()
    else:
        # 在延伸模式下繪製 PR-AUC 配對差分佈
        ax1.hist(pr_diffs, bins=12, color="#457b9d", edgecolor="black", alpha=0.7)
        ax1.axvline(0.0, color="red", linestyle="--", label="0 基準線")
        ax1.axvline(ci_pr["mean"], color="blue", label=f"均值={ci_pr['mean']:+.4f}")
        ax1.set_title(f"F-02 PR-AUC 配對差分佈\n(95% CI: [{ci_pr['t_ci_95'][0]:+.4f}, {ci_pr['t_ci_95'][1]:+.4f}])", fontsize=11, fontweight="bold")
        ax1.set_xlabel("PR-AUC 差異 (設定 B - 設定 A)")
        ax1.set_ylabel("折數頻次")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

    # 子圖 2: 配對散布圖
    ax2 = axes[1]
    ax2.scatter(
        df_f02["pr_auc_setting_A_freq0"],
        df_f02["pr_auc_setting_B_freq1"],
        c=df_f02["model_seed"],
        cmap="viridis",
        s=50,
        alpha=0.85,
        edgecolor="k",
    )
    lims = [
        min(df_f02["pr_auc_setting_A_freq0"].min(), df_f02["pr_auc_setting_B_freq1"].min()) * 0.95,
        max(df_f02["pr_auc_setting_A_freq0"].max(), df_f02["pr_auc_setting_B_freq1"].max()) * 1.05,
    ]
    ax2.plot(lims, lims, "r--", label="等值線 (y=x)")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title(f"F-02 配對 PR-AUC 對比\n(Spearman r={df_f02['spearman_corr_A_vs_B'].mean():.4f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("設定 A (subsample_freq=0)")
    ax2.set_ylabel("設定 B (subsample_freq=1)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 子圖 3: AR@0.95 差異與種子雜訊參照
    ax3 = axes[2]
    ax3.hist(ar95_diffs, bins=10, color="#2a9d8f", edgecolor="black", alpha=0.7)
    ax3.axvline(0.0, color="red", linestyle="--", label="0 基準線")
    ax3.axvline(ci_ar95["mean"], color="blue", label=f"差異均值 ({ci_ar95['mean']:+.2f}x)")
    if mean_seed_noise_ar95 > 0:
        ax3.axvline(mean_seed_noise_ar95, color="orange", linestyle=":", label=f"種子雜訊參照 (+{mean_seed_noise_ar95:.2f}x)")
        ax3.axvline(-mean_seed_noise_ar95, color="orange", linestyle=":")
    ax3.set_title("F-02 AR@0.95 配對差分佈與種子雜訊", fontsize=11, fontweight="bold")
    ax3.set_xlabel("AR@0.95 差異 (倍數)")
    ax3.set_ylabel("頻次")
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
            "mode": mode,
            "n_pairs": len(df_f02),
            "lightgbm_version": lgb.__version__,
            "pr_auc_mean_diff": ci_pr["mean"],
            "pr_auc_t_ci_95": ci_pr["t_ci_95"],
            "ar95_mean_diff": ci_ar95["mean"],
            "ar95_t_ci_95": ci_ar95["t_ci_95"],
            "mean_seed_noise_pr": mean_seed_noise_pr,
            "mean_seed_noise_ar95": mean_seed_noise_ar95,
            "spearman_rank_correlation_mean": float(df_f02["spearman_corr_A_vs_B"].mean()),
            "conclusion": conclusion_text,
        },
    )

    print("\n" + "=" * 70)
    print(f"LAB_03 執行完成！總耗時: {elapsed:.2f} 秒")
    print(f"結果 CSV : {f02_path}")
    print(f"圖表 PNG : {fig_path}")
    print(f"小型快照 : {snap_path}")
    print("=" * 70)

    return df_grid, df_f02, f02_path, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 03: 提升法與 F-02 實證對照實驗")
    parser.add_argument("--f02-extended", action="store_true", help="執行 F-02 擴大對照模式 (45組配對, 5-10分鐘)")
    parser.add_argument("--standard", action="store_true", help="執行標準模式 (3x3網格, 3種子)")
    parser.add_argument("--full", action="store_true", help="執行完整模式")
    parser.add_argument("--quick", action="store_true", default=False, help="執行快速模式 (2x2網格, 2種子, <=1.5分鐘, 預設)")
    parser.add_argument("--pipeline", choices=["project", "teaching"], default="project", help="前處理旗標 (project 關閉截尾, teaching 開啟截尾, 預設 project)")
    args = parser.parse_args()

    if args.f02_extended:
        exec_mode = "f02_extended"
    elif args.full:
        exec_mode = "full"
    elif args.standard:
        exec_mode = "standard"
    else:
        exec_mode = "quick"

    run_lab(mode=exec_mode, pipeline=args.pipeline)
