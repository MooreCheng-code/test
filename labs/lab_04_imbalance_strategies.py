"""labs/lab_04_imbalance_strategies.py - Imbalance Handling Strategies (S0 to S6) & Probability Shift.

Topics:
1. Compares S0 (baseline), S1 (class_weight / adaptive scale_pos_weight),
   S2 (SMOTE 0.1), S3 (SMOTE 0.3), S4 (SMOTE 0.5), S5 (BorderlineSMOTE 0.3),
   and S6 (SMOTETomek 0.3) on M1 (Logistic Regression) and M3 (Extra Trees).
2. Investigates why resampling distorts probability calibration (mean predicted probability
   vs true prevalence ~0.0323 and Brier score) while ranking metrics (PR-AUC, AR) shift differently.
3. Compares results side-by-side with formal project experiment E1a (e1a_imbalance_comparison.csv).

SAFETY:
- Uses ONLY load_train_data() (5,455 training rows). Never touches holdout test set.
- Re-fits models on custom CV folds. Resampling occurs strictly inside training folds.
- Outputs written to labs/outputs/lab_04/. Snapshot to docs/code_learning/lab_snapshots/.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bankruptcy.src.pipelines import build_pipeline, get_score
from labs._common import (
    get_pipeline_flags,
    LABS_OUTPUTS_DIR,
    TABLES_DIR,
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
    # 測試的不平衡處理策略清單 (S0 到 S6)
    "strategies": ["S0", "S1", "S2", "S3", "S4", "S5", "S6"],
    # 比較的基準模型
    "models": ["M1", "M3"],
    # 交叉驗證設定
    "cv_folds": 5,
    "cv_repeats_quick": 1,
    "cv_repeats_full": 3,
    "seed": 42,
}


def run_lab(quick: bool = True, pipeline: str = "project") -> tuple[pd.DataFrame, Path, Path]:
    pipe_flags = get_pipeline_flags(pipeline)
    """Execute lab_04 imbalance strategies scan and calibration shift study."""
    start_time = time.time()
    setup_chinese_font()

    n_repeats = PARAMS["cv_repeats_quick"] if quick else PARAMS["cv_repeats_full"]
    strategies = PARAMS["strategies"]
    model_list = PARAMS["models"]
    seed = PARAMS["seed"]

    print("=" * 70)
    print(f"LAB_04: 不平衡策略與機率偏移實驗 ({'--quick 快速模式' if quick else '--full 完整模式'})")
    print(f"CV 設定: {PARAMS['cv_folds']} 折 x {n_repeats} 次重複 (種子={seed})")
    print(f"比較策略: {', '.join(strategies)}")
    print("=" * 70)

    # 1. 唯讀載入訓練資料
    X_train, y_train = load_train_data(check_guard=True)
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=seed)
    overall_prevalence = float(y_train.mean())

    records: list[dict[str, Any]] = []

    # 2. 執行評估
    for m_code in model_list:
        base_json = load_best_params(m_code)
        base_params = base_json["best_params"].copy()
        m_name = "LogisticRegression" if m_code == "M1" else "ExtraTrees"

        # 快速模式下適度縮減樹數加速
        if quick and m_code == "M3":
            base_params["n_estimators"] = min(base_params.get("n_estimators", 300), 80)

        print(f"\n--- 評估模型: {m_code} ({m_name}) ---")

        for strat in strategies:
            pr_aucs, roc_aucs, ar95s, ar90s, ar80s = [], [], [], [], []
            mean_probas, brier_scores = [], []
            fold_times = []

            for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
                t0 = time.time()
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                pipe = build_pipeline(
                    model_name=m_code,
                    imbalance=strat,
                    model_params=base_params,
                    seed=seed + fold_idx,
                )
                pipe.fit(X_tr, y_tr)
                score_val = get_score(pipe, X_val)
                val_metrics = evaluate_fold_predictions(y_val, score_val)

                pr_aucs.append(val_metrics["pr_auc"])
                roc_aucs.append(val_metrics["roc_auc"])
                ar95s.append(val_metrics["ar_95"])
                ar90s.append(val_metrics["ar_90"])
                ar80s.append(val_metrics["ar_80"])
                mean_probas.append(val_metrics["mean_proba"])
                brier_scores.append(val_metrics["brier"])
                fold_times.append(time.time() - t0)

            rec = {
                "model_code": m_code,
                "model_name": m_name,
                "strategy": strat,
                "pr_auc_mean": float(np.mean(pr_aucs)),
                "pr_auc_std": float(np.std(pr_aucs)),
                "ar95_mean": float(np.mean(ar95s)),
                "ar95_std": float(np.std(ar95s)),
                "ar90_mean": float(np.mean(ar90s)),
                "ar90_std": float(np.std(ar90s)),
                "ar80_mean": float(np.mean(ar80s)),
                "ar80_std": float(np.std(ar80s)),
                "roc_auc_mean": float(np.mean(roc_aucs)),
                "mean_proba_mean": float(np.mean(mean_probas)),
                "true_prevalence": overall_prevalence,
                "proba_shift_ratio": float(np.mean(mean_probas) / overall_prevalence),
                "brier_mean": float(np.mean(brier_scores)),
                "avg_fit_seconds": float(np.mean(fold_times)),
            }
            records.append(rec)
            print(
                f"  [{m_code}] 策略 {strat:<2} | "
                f"PR-AUC: {rec['pr_auc_mean']:.4f} | "
                f"AR@0.95: {rec['ar95_mean']:.2f}x | "
                f"預測機率均值: {rec['mean_proba_mean']:.4f} (盛行率: {overall_prevalence:.4f}, 偏移: {rec['proba_shift_ratio']:.1f}x) | "
                f"Brier: {rec['brier_mean']:.4f}"
            )

    df_results = pd.DataFrame(records)

    # 3. 讀取專案正式 E1a 結果進行對照
    e1a_file = TABLES_DIR / "e1a_imbalance_comparison.csv"
    if e1a_file.is_file():
        df_e1a = pd.read_csv(e1a_file)
        # 篩選出 M1 與 M3 的 quantile scaler
        e1a_sub = df_e1a[
            (df_e1a["model"].isin(["M1", "M3"])) & (df_e1a["scaler"] == "quantile")
        ][["model", "strategy", "pr_auc_mean", "alert_ratio@recall_0.95_mean"]].rename(
            columns={
                "pr_auc_mean": "e1a_formal_pr_auc",
                "alert_ratio@recall_0.95_mean": "e1a_formal_ar95",
            }
        )
        df_results = df_results.merge(
            e1a_sub,
            left_on=["model_code", "strategy"],
            right_on=["model", "strategy"],
            how="left",
        ).drop(columns=["model"], errors="ignore")

        # 計算策略在 Lab 與正式 E1a 中的排名 (PR-AUC 越高越優、AR@0.95 越低越優)
        df_results["lab_pr_rank"] = df_results.groupby("model_code")["pr_auc_mean"].rank(ascending=False).astype(int)
        df_results["lab_ar95_rank"] = df_results.groupby("model_code")["ar95_mean"].rank(ascending=True).astype(int)
        df_results["e1a_pr_rank"] = df_results.groupby("model_code")["e1a_formal_pr_auc"].rank(ascending=False).astype(int)
        df_results["e1a_ar95_rank"] = df_results.groupby("model_code")["e1a_formal_ar95"].rank(ascending=True).astype(int)

        print("\n" + "=" * 70)
        print("Lab 與正式 E1a 策略排名對照 (口徑差異說明：Lab 為 5折x1次重複，E1a 為 5折x3次重複):")
        print("=" * 70)
        for m_code in model_list:
            sub = df_results[df_results["model_code"] == m_code]
            print(f"\n[{m_code}] 策略對照表:")
            print(f"{'策略':<4} | {'Lab PR':<8} {'E1a PR':<8} {'PR秩差':<6} | {'Lab AR95':<10} {'E1a AR95':<10} {'AR秩差':<6} | {'機率偏移':<8}")
            print("-" * 70)
            for _, r in sub.iterrows():
                pr_rank_diff = r["lab_pr_rank"] - r["e1a_pr_rank"]
                ar_rank_diff = r["lab_ar95_rank"] - r["e1a_ar95_rank"]
                print(
                    f"{r['strategy']:<4} | {r['pr_auc_mean']:<8.4f} {r['e1a_formal_pr_auc']:<8.4f} {pr_rank_diff:+2d}     | "
                    f"{r['ar95_mean']:<10.2f} {r['e1a_formal_ar95']:<10.2f} {ar_rank_diff:+2d}     | {r['proba_shift_ratio']:<8.1f}x"
                )

    # 4. 輸出儲存
    out_dir = LABS_OUTPUTS_DIR / "lab_04"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lab_04_imbalance_strategies.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8")

    snap_path = save_snapshot(df_results, "lab_04_imbalance_strategies_snapshot.csv")

    # 5. 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子圖 1: 策略 vs PR-AUC
    ax1 = axes[0]
    x_indices = np.arange(len(strategies))
    width = 0.35
    for i, m_code in enumerate(model_list):
        sub = df_results[df_results["model_code"] == m_code].set_index("strategy")
        y_vals = [sub.loc[s, "pr_auc_mean"] for s in strategies]
        color = "#2b5c8f" if m_code == "M1" else "#e76f51"
        ax1.bar(x_indices + (i - 0.5) * width, y_vals, width, label=m_code, color=color, alpha=0.85)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(strategies)
    ax1.set_title("不平衡策略 (S0～S6) vs 驗證折 PR-AUC", fontsize=11, fontweight="bold")
    ax1.set_xlabel("策略代碼")
    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend()

    # 子圖 2: 策略 vs AR@0.95 (預警倍數)
    ax2 = axes[1]
    for m_code, color in [("M1", "#2b5c8f"), ("M3", "#e76f51")]:
        sub = df_results[df_results["model_code"] == m_code].set_index("strategy")
        y_vals = [sub.loc[s, "ar95_mean"] for s in strategies]
        ax2.plot(strategies, y_vals, "o-", color=color, label=f"{m_code} AR@0.95 (越低越好)")
    ax2.set_title("不平衡策略 vs AR@0.95 預警倍數", fontsize=11, fontweight="bold")
    ax2.set_xlabel("策略代碼")
    ax2.set_ylabel("AR@0.95 (預警倍數)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 子圖 3: 策略 vs 平均預測機率 (機率扭曲效應)
    ax3 = axes[2]
    for m_code, color in [("M1", "#2b5c8f"), ("M3", "#e76f51")]:
        sub = df_results[df_results["model_code"] == m_code].set_index("strategy")
        y_vals = [sub.loc[s, "mean_proba_mean"] for s in strategies]
        ax3.plot(strategies, y_vals, "s-", color=color, label=f"{m_code} 平均預測機率")
    ax3.axhline(overall_prevalence, color="green", linestyle="--", linewidth=1.5, label=f"真實盛行率 ({overall_prevalence:.3f})")
    ax3.set_title("不平衡策略 vs 平均預測機率 (校準偏移現象)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("策略代碼")
    ax3.set_ylabel("驗證折平均預測機率")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    fig_path = out_dir / "lab_04_imbalance_strategies.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    save_manifest(
        output_dir=out_dir,
        lab_id="lab_04",
        params=PARAMS,
        elapsed_seconds=elapsed,
        seed=seed,
        extra={
            "quick_mode": quick,
            "overall_prevalence": overall_prevalence,
        },
    )

    print("\n" + "=" * 70)
    print(f"LAB_04 執行完成！總耗時: {elapsed:.2f} 秒")
    print(f"結果 CSV : {csv_path}")
    print(f"圖表 PNG : {fig_path}")
    print(f"小型快照 : {snap_path}")
    print("=" * 70)

    return df_results, csv_path, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 04: 不平衡處理策略實驗")
    parser.add_argument("--full", action="store_true", help="執行完整模式")
    parser.add_argument("--quick", action="store_true", default=True, help="執行快速模式 (預設)")
    parser.add_argument("--pipeline", choices=["project", "teaching"], default="project", help="前處理旗標 (project 關閉截尾, teaching 開啟截尾, 預設 project)")
    args = parser.parse_args()

    is_quick = not args.full
    run_lab(quick=is_quick, pipeline=args.pipeline)
