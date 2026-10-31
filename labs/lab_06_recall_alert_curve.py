"""
lab_06_recall_alert_curve.py - 召回率與預警率 (Recall-Alert) 權衡曲線與門檻轉移實驗

【教學核心目標】
1. 觀察 Recall 在 [0.80, 0.99] (步長 0.01) 與 1.0 的非線性膨脹：
   在高召回率極端區間 (Recall > 0.90)，為了多捕獲最後幾家破產企業，預警率 (Alert Ratio)
   呈指數級急速攀升。
2. 隨機猜測基準線 (Random Guess Baseline)：
   隨機預警 k% 樣本，期望捕獲 k% 破產企業，故 AR = (k% * N) / (k% * P) = N / P = 31.00x (平坦線)。
3. 門檻轉移達標率 (Threshold Transfer Attainment)：
   模擬 E4a 協定：在訓練折內部自建 3 折 OOF，呼叫純計算函式 derive_operating_threshold 導出门檻，
   套用至外層驗證折，評估 Recall 是否達標 (Recall >= target_recall) 與 Margin (m) 的安全緩衝效應。
4. 20 筆玩具示範：
   說明當 P 很小時，目標 Recall 由 0.90 升至 0.95 時，所需正樣本數 ceil(R * P) 之離散階梯跳躍。

【防護守則】
- 僅呼叫 load_train_data()，絕不讀取測試集或 E6_LOCK。
- 產物輸出至 labs/outputs/lab_06/ 與 docs/code_learning/lab_snapshots/。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from labs._common import (
    get_pipeline_flags,
    SNAPSHOT_DIR,
    build_pipeline,
    evaluate_fold_predictions,
    get_score,
    load_best_params,
    load_train_data,
    save_manifest,
)
from bankruptcy.src.thresholding import derive_operating_threshold

LAB_ID = "lab_06"
OUTPUT_DIR = os.path.join("labs", "outputs", LAB_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def run_toy_recall_demonstration() -> pd.DataFrame:
    """示範 20 筆資料下，不同正樣本數 P 與目標 Recall 如何造成 ceil(R * P) 階梯跳躍。"""
    recalls = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 1.00]
    p_cases = [1, 2, 5, 10, 35]  # 35 為正式驗證折破產數約值

    rows = []
    for P in p_cases:
        for R in recalls:
            needed_pos = int(np.ceil(R * P))
            actual_achievable_recall = needed_pos / P
            rows.append({
                "total_positives_P": P,
                "target_recall": R,
                "needed_positives_ceil": needed_pos,
                "actual_min_recall": actual_achievable_recall,
                "step_jump_note": (
                    f"P={P}: 目標 {R:.2f} 需捕獲 {needed_pos}/{P} 正樣本 (實質達標率 {actual_achievable_recall:.4f})"
                ),
            })
    df_toy = pd.DataFrame(rows)
    return df_toy


def run_recall_alert_tradeoff_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_code: str = "M3",
    n_splits: int = 5,
    random_state: int = 42,
    pipeline: str = "project",
) -> tuple[pd.DataFrame, float]:
    """計算 Recall [0.80, 1.00] 與 AR 的實測權衡曲線。"""
    pipe_flags = get_pipeline_flags(pipeline)
    cfg = load_best_params(model_code)
    m_params = cfg.get("best_params", {})
    imb = cfg.get("imbalance", "S0")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target_recalls = np.round(np.append(np.arange(0.80, 1.00, 0.01), [1.00]), 2)

    # 收集全部折預測以建立全折與折平均 AR
    curve_records: list[dict[str, Any]] = []

    # 預先擬合各折取得驗證預測分數
    fold_scores = []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        pipe = build_pipeline(model_code, imbalance=imb, model_params=m_params, seed=random_state, **pipe_flags)
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        val_s = get_score(pipe, X_train.iloc[val_idx])
        fold_scores.append((y_train.iloc[val_idx].to_numpy(), val_s))

    total_N = len(y_train)
    total_P = int(y_train.sum())
    random_guess_ar = total_N / total_P  # 31.00x

    for R in target_recalls:
        fold_ars = []
        fold_alerts = []
        fold_precs = []
        for y_true, s_val in fold_scores:
            m_res = evaluate_fold_predictions(pd.Series(y_true), s_val)
            # 依門檻計算該 R 下的 AR
            P_f = np.sum(y_true == 1)
            N_f = len(y_true)
            needed_P = int(np.ceil(R * P_f))
            # 悲觀排序
            pos_scores = s_val[y_true == 1]
            pos_scores_sorted = np.sort(pos_scores)[::-1]
            thr = pos_scores_sorted[needed_P - 1] if needed_P <= len(pos_scores_sorted) else 0.0
            n_alerts = int(np.sum(s_val >= thr))
            ar = n_alerts / P_f
            prec = np.sum((s_val >= thr) & (y_true == 1)) / n_alerts if n_alerts > 0 else 0.0
            fold_ars.append(ar)
            fold_alerts.append(n_alerts)
            fold_precs.append(prec)

        curve_records.append({
            "target_recall": float(R),
            "alert_ratio_mean": float(np.mean(fold_ars)),
            "alert_ratio_std": float(np.std(fold_ars)),
            "alert_count_mean": float(np.mean(fold_alerts)),
            "precision_mean": float(np.mean(fold_precs)),
            "random_guess_ar": float(random_guess_ar),
            "ratio_to_random": float(np.mean(fold_ars) / random_guess_ar),
        })

    df_curve = pd.DataFrame(curve_records)
    return df_curve, random_guess_ar


def run_threshold_transfer_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_code: str = "M3",
    outer_splits: int = 5,
    inner_splits: int = 3,
    margins: list[float] = [0.0, 0.02, 0.04, 0.06],
    target_recalls: list[float] = [0.80, 0.90, 0.95],
    random_state: int = 42,
    pipeline: str = "project",
) -> pd.DataFrame:
    """模擬 E4a 門檻轉移協定：訓練折內層 3 折 OOF 導出门檻，外層驗證折評估達標率。"""
    pipe_flags = get_pipeline_flags(pipeline)
    cfg = load_best_params(model_code)
    m_params = cfg.get("best_params", {})
    imb = cfg.get("imbalance", "S0")

    outer_skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    transfer_records = []

    print(f"\n[門檻轉移實驗] 模型: {model_code}, 外層: {outer_splits} 折, 內層: {inner_splits} 折...")

    for fold_idx, (tr_idx, val_idx) in enumerate(outer_skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        y_val_arr = y_val.to_numpy()

        # 1. 內層 OOF 交叉驗證取得訓練折內預測分數
        inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold_idx)
        inner_oof_scores = np.zeros(len(X_tr), dtype=float)

        for in_tr_idx, in_val_idx in inner_skf.split(X_tr, y_tr):
            pipe_inner = build_pipeline(model_code, imbalance=imb, model_params=m_params, seed=random_state, **pipe_flags)
            pipe_inner.fit(X_tr.iloc[in_tr_idx], y_tr.iloc[in_tr_idx])
            inner_oof_scores[in_val_idx] = get_score(pipe_inner, X_tr.iloc[in_val_idx])

        # 2. 全外層訓練折擬合以取得外層驗證折分數
        pipe_outer = build_pipeline(model_code, imbalance=imb, model_params=m_params, seed=random_state, **pipe_flags)
        pipe_outer.fit(X_tr, y_tr)
        val_scores = get_score(pipe_outer, X_val)

        # 3. 測試各 (target_recall, margin) 組合
        for R in target_recalls:
            for m in margins:
                thr_info = derive_operating_threshold(
                    scores_arr=inner_oof_scores,
                    y_arr=y_tr.to_numpy(),
                    target_recall=R,
                    margin=m,
                )
                thr = thr_info["threshold"]

                # 套用至外層驗證折
                val_preds = (val_scores >= thr)
                val_pos = np.sum(y_val_arr == 1)
                val_caught = np.sum(val_preds & (y_val_arr == 1))
                val_actual_recall = val_caught / val_pos if val_pos > 0 else 0.0
                target_met = int(val_actual_recall >= R)
                val_alerts = np.sum(val_preds)
                val_ar = val_alerts / val_pos if val_pos > 0 else 0.0

                transfer_records.append({
                    "fold": fold_idx,
                    "model_code": model_code,
                    "target_recall": R,
                    "margin": m,
                    "derived_threshold": thr,
                    "val_actual_recall": val_actual_recall,
                    "target_met": target_met,
                    "val_alert_ratio": val_ar,
                    "val_alert_count": int(val_alerts),
                })

    df_raw = pd.DataFrame(transfer_records)

    # 匯總平均與達標率 (attainment rate)
    summary_rows = []
    for (R, m), grp in df_raw.groupby(["target_recall", "margin"]):
        summary_rows.append({
            "model_code": model_code,
            "target_recall": R,
            "margin": m,
            "n_folds": len(grp),
            "attainment_rate": float(grp["target_met"].mean()),
            "actual_recall_mean": float(grp["val_actual_recall"].mean()),
            "actual_recall_std": float(grp["val_actual_recall"].std()),
            "alert_ratio_mean": float(grp["val_alert_ratio"].mean()),
            "alert_ratio_std": float(grp["val_alert_ratio"].std()),
            "threshold_mean": float(grp["derived_threshold"].mean()),
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary


def plot_recall_alert_curve(df_curve: pd.DataFrame, random_guess_ar: float, out_path: str):
    """繪製 Recall-AR 權衡曲線與隨機基準線。"""
    plt.figure(figsize=(10, 6))

    plt.plot(
        df_curve["target_recall"],
        df_curve["alert_ratio_mean"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="ExtraTrees (M3) AR@Recall",
    )
    plt.fill_between(
        df_curve["target_recall"],
        df_curve["alert_ratio_mean"] - df_curve["alert_ratio_std"],
        df_curve["alert_ratio_mean"] + df_curve["alert_ratio_std"],
        alpha=0.2,
        color="#1f77b4",
        label="Fold Std Dev (+/- 1 SD)",
    )

    # 隨機基準線
    plt.axhline(
        y=random_guess_ar,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Random Guess Baseline (AR = N/P = {random_guess_ar:.2f}x)",
    )

    # 關鍵召回率標記
    for target_r in [0.90, 0.95]:
        row = df_curve[df_curve["target_recall"] == target_r]
        if not row.empty:
            ar_val = row["alert_ratio_mean"].values[0]
            plt.scatter([target_r], [ar_val], color="red", s=80, zorder=5)
            plt.annotate(
                f"R={target_r:.2f}: AR={ar_val:.2f}x\n({ar_val/random_guess_ar*100:.1f}% of random)",
                xy=(target_r, ar_val),
                xytext=(target_r - 0.04, ar_val + 3.0),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9,
                fontweight="bold",
            )

    plt.title("Lab 06: Recall vs Alert Ratio Trade-off Curve & Random Baseline", fontsize=12, fontweight="bold")
    plt.xlabel("Target Recall (破產企業捕獲率)", fontsize=11)
    plt.ylabel("Alert Ratio (倍數相對於正樣本數 N/P)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [圖檔已儲存] {out_path}")


def print_formal_e4a_comparison():
    """讀取並並排印出正式實驗 e4a 表格對照值。"""
    summary_path = os.path.join("bankruptcy", "outputs", "tables", "e4a_threshold_transfer_summary.csv")
    margin_path = os.path.join("bankruptcy", "outputs", "tables", "e4a_margin_recommendation.csv")

    print("\n" + "=" * 80)
    print("【正式實驗 E4a 對照參考】")
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        m3_sum = df_sum[df_sum["group"] == "M3"][["group", "target_recall", "margin", "attainment_rate", "actual_recall_mean", "alert_ratio_mean"]]
        print(">> 正式 E4a 門檻轉移匯總 (M3 部分):")
        print(m3_sum.to_string(index=False))
    if os.path.exists(margin_path):
        df_m = pd.read_csv(margin_path)
        print("\n>> 正式 E4a Margin 推薦表:")
        print(df_m[["group", "target_recall", "recommended_margin", "rule_status"]].to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Lab 06: Recall-Alert Curve & Threshold Transfer")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="quick")
    parser.add_argument("--quick", action="store_true", help="執行快速模式")
    parser.add_argument("--pipeline", choices=["project", "teaching"], default="project", help="前處理旗標 (project 關閉截尾, teaching 開啟截尾, 預設 project)")
    args = parser.parse_args()
    if getattr(args, "quick", False):
        args.mode = "quick"

    start_time = time.time()
    print("=" * 80)
    print(f"Lab 06: Recall-Alert Curve & Threshold Transfer (模式: {args.mode})")
    print("=" * 80)

    X_train, y_train = load_train_data()

    # 1. 20 筆資料玩具示範
    print("\n[1/3] 執行 20 筆資料階梯跳躍示範 (P 數值 vs ceil(R * P))...")
    df_toy = run_toy_recall_demonstration()
    toy_csv = os.path.join(OUTPUT_DIR, "toy_recall_steps.csv")
    df_toy.to_csv(toy_csv, index=False)
    df_toy.to_csv(os.path.join(SNAPSHOT_DIR, "lab_06_toy_recall_steps_snapshot.csv"), index=False)
    print(df_toy.head(10).to_string(index=False))

    # 2. Recall-AR 曲線
    print("\n[2/3] 計算 Recall [0.80, 1.00] 權衡曲線...")
    n_splits = 3 if args.mode == "quick" else 5
    df_curve, random_guess_ar = run_recall_alert_tradeoff_curve(
        X_train, y_train, model_code="M3", n_splits=n_splits, random_state=42, pipeline=args.pipeline
    )
    curve_csv = os.path.join(OUTPUT_DIR, "recall_alert_curve.csv")
    df_curve.to_csv(curve_csv, index=False)
    df_curve.to_csv(os.path.join(SNAPSHOT_DIR, "lab_06_recall_alert_curve_snapshot.csv"), index=False)

    png_path = os.path.join(OUTPUT_DIR, "recall_alert_curve.png")
    plot_recall_alert_curve(df_curve, random_guess_ar, png_path)

    # 3. 門檻轉移實驗
    print("\n[3/3] 執行門檻轉移達標率實驗...")
    outer_splits = 3 if args.mode == "quick" else 5
    inner_splits = 2 if args.mode == "quick" else 3
    df_transfer = run_threshold_transfer_experiment(
        X_train, y_train, model_code="M3",
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        margins=[0.0, 0.02, 0.04, 0.06],
        target_recalls=[0.80, 0.90, 0.95],
        random_state=42,
        pipeline=args.pipeline,
    )
    transfer_csv = os.path.join(OUTPUT_DIR, "threshold_transfer.csv")
    df_transfer.to_csv(transfer_csv, index=False)
    df_transfer.to_csv(os.path.join(SNAPSHOT_DIR, "lab_06_threshold_transfer_snapshot.csv"), index=False)

    print("\n>> Lab 06 門檻轉移達標率結果 (M3):")
    print(df_transfer[["target_recall", "margin", "attainment_rate", "actual_recall_mean", "alert_ratio_mean"]].to_string(index=False))

    # 4. 印出正式 E4a 對照
    print_formal_e4a_comparison()

    total_time = time.time() - start_time
    save_manifest(
        output_dir=OUTPUT_DIR,
        lab_id=LAB_ID,
        params={"mode": args.mode, "outer_splits": outer_splits, "inner_splits": inner_splits},
        elapsed_seconds=total_time,
        seed=42,
        extra={
            "execution_seconds": total_time,
            "random_guess_ar": random_guess_ar,
            "ar90_mean": float(df_curve[df_curve["target_recall"] == 0.90]["alert_ratio_mean"].values[0]),
            "ar95_mean": float(df_curve[df_curve["target_recall"] == 0.95]["alert_ratio_mean"].values[0]),
        },
    )
    print(f"\n[完成] Lab 06 執行總耗時: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()
