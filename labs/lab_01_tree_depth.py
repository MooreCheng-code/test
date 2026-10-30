"""labs/lab_01_tree_depth.py - Decision Tree & Forest Depth Scan.

Topic: Tree depth (max_depth) and overfitting vs underfitting trade-offs.
Scans single DecisionTreeClassifier depth across [1, 2, 3, 4, 6, 8, 12, None]
and compares with tree ensembles M2 (RandomForest), M3 (ExtraTrees), M4 (XGBoost).

SAFETY:
- Uses ONLY load_train_data() (5,455 training rows). Never touches holdout test set.
- Re-fits models on custom CV folds. Never loads frozen final models.
- Outputs written to labs/outputs/lab_01/. Snapshot to docs/code_learning/lab_snapshots/.
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
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from bankruptcy.src.pipelines import build_pipeline, get_score
from bankruptcy.src.preprocessing import (
    CorrelationFilter,
    DropConstantAndDuplicate,
    QuantileScaler,
    Winsorizer,
)
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
    # 單棵決策樹掃描深度清單 (快速模式與標準模式)
    "dt_depths_quick": [1, 2, 4, 8, None],
    "dt_depths_standard": [1, 2, 3, 4, 6, 8, 12, None],
    # 集成模型比較清單與各自深度
    "m2_depths_quick": [4, 8, 12, None],
    "m2_depths_standard": [4, 6, 8, 12, None],
    "m3_depths_quick": [4, 8, 12, None],
    "m3_depths_standard": [4, 6, 8, 12, None],
    "m4_depths_quick": [2, 4, 6],
    "m4_depths_standard": [2, 3, 4, 6],
    # 樹木數量 (快速模式 40 棵以保證在 1 分鐘內跑完，標準模式 80 棵，完整模式 300 棵)
    "n_estimators_quick": 40,
    "n_estimators_standard": 80,
    "n_estimators_full": 300,
    # 交叉驗證設定
    "cv_folds": 5,
    "cv_repeats_quick": 1,
    "cv_repeats_full": 3,
    "seed": 42,
}


def build_decision_tree_pipeline(max_depth: int | None, seed: int = 42, pipeline: str = "project") -> Pipeline:
    flags = get_pipeline_flags(pipeline)
    """Build preprocessing + single DecisionTreeClassifier pipeline."""
    return Pipeline([
        ("drop_const_dup", DropConstantAndDuplicate()),
        ("winsorizer", Winsorizer(enabled=flags["use_winsorizer"])),
        ("scaler", QuantileScaler(kind=flags["scaler"])),
        ("corr_filter", CorrelationFilter(enabled=flags["use_corr_filter"])),
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=seed,
            class_weight=None,
        )),
    ])


def run_lab(mode: str = "quick", pipeline: str = "project") -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    pipe_flags = get_pipeline_flags(pipeline)
    """Execute lab_01 tree depth scan."""
    start_time = time.time()
    setup_chinese_font()

    if mode == "quick":
        n_repeats = PARAMS["cv_repeats_quick"]
        n_estimators = PARAMS["n_estimators_quick"]
        dt_depths = PARAMS["dt_depths_quick"]
        m2_depths = PARAMS["m2_depths_quick"]
        m3_depths = PARAMS["m3_depths_quick"]
        m4_depths = PARAMS["m4_depths_quick"]
    elif mode == "standard":
        n_repeats = PARAMS["cv_repeats_quick"]
        n_estimators = PARAMS["n_estimators_standard"]
        dt_depths = PARAMS["dt_depths_standard"]
        m2_depths = PARAMS["m2_depths_standard"]
        m3_depths = PARAMS["m3_depths_standard"]
        m4_depths = PARAMS["m4_depths_standard"]
    else:  # full
        n_repeats = PARAMS["cv_repeats_full"]
        n_estimators = PARAMS["n_estimators_full"]
        dt_depths = PARAMS["dt_depths_standard"]
        m2_depths = PARAMS["m2_depths_standard"]
        m3_depths = PARAMS["m3_depths_standard"]
        m4_depths = PARAMS["m4_depths_standard"]

    seed = PARAMS["seed"]

    print("=" * 70)
    print(f"LAB_01: 樹深度掃描實驗 ({mode} 模式)")
    print(f"CV 設定: {PARAMS['cv_folds']} 折 x {n_repeats} 次重複 (種子={seed})")
    print("=" * 70)

    # 1. 唯讀安全載入訓練資料
    X_train, y_train = load_train_data(check_guard=True)
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=seed)

    # 2. 單棵決策樹掃描
    print("\n[1/2] 執行單棵 DecisionTreeClassifier 深度掃描...")
    records: list[dict[str, Any]] = []
    
    for depth in dt_depths:
        depth_label = "None (無限制)" if depth is None else str(depth)
        train_pr_aucs, val_pr_aucs = [], []
        train_ar95s, val_ar95s = [], []
        train_ar90s, val_ar90s = [], []
        val_n_uniques = []
        fold_times = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
            t0 = time.time()
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe = build_decision_tree_pipeline(max_depth=depth, seed=seed + fold_idx, pipeline=pipeline)
            pipe.fit(X_tr, y_tr)

            # 訓練折評估
            score_tr = pipe.predict_proba(X_tr)[:, 1]
            tr_metrics = evaluate_fold_predictions(y_tr, score_tr)

            # 驗證折評估
            score_val = pipe.predict_proba(X_val)[:, 1]
            val_metrics = evaluate_fold_predictions(y_val, score_val)

            train_pr_aucs.append(tr_metrics["pr_auc"])
            val_pr_aucs.append(val_metrics["pr_auc"])
            train_ar95s.append(tr_metrics["ar_95"])
            val_ar95s.append(val_metrics["ar_95"])
            train_ar90s.append(tr_metrics["ar_90"])
            val_ar90s.append(val_metrics["ar_90"])
            val_n_uniques.append(val_metrics["n_unique_scores"])
            fold_times.append(time.time() - t0)

        records.append({
            "model_type": "Single_DecisionTree",
            "model_name": "DT",
            "max_depth": depth_label,
            "depth_numeric": 99 if depth is None else depth,
            "train_pr_auc_mean": np.mean(train_pr_aucs),
            "train_pr_auc_std": np.std(train_pr_aucs),
            "val_pr_auc_mean": np.mean(val_pr_aucs),
            "val_pr_auc_std": np.std(val_pr_aucs),
            "pr_auc_gap": np.mean(train_pr_aucs) - np.mean(val_pr_aucs),
            "train_ar95_mean": np.mean(train_ar95s),
            "val_ar95_mean": np.mean(val_ar95s),
            "val_ar95_std": np.std(val_ar95s),
            "ar95_gap": np.mean(train_ar95s) - np.mean(val_ar95s),
            "val_ar90_mean": np.mean(val_ar90s),
            "n_unique_scores": float(np.mean(val_n_uniques)),
            "avg_fit_seconds": np.mean(fold_times),
        })
        print(
            f"  DT depth={depth_label:<14} | Val PR-AUC: {np.mean(val_pr_aucs):.4f} "
            f"(Train: {np.mean(train_pr_aucs):.4f}, Gap: {np.mean(train_pr_aucs)-np.mean(val_pr_aucs):+.4f}) | "
            f"Val AR@0.95: {np.mean(val_ar95s):.2f}x | 唯一分數數: {np.mean(val_n_uniques):.1f}"
        )

    # 3. 集成模型深度對照 (M2 RandomForest, M3 ExtraTrees, M4 XGBoost)
    print("\n[2/2] 執行樹集成模型 (M2, M3, M4) 深度掃描對照...")
    ensemble_configs = [
        ("M2", "RandomForest", m2_depths, "S3"),
        ("M3", "ExtraTrees", m3_depths, "S0"),
        ("M4", "XGBoost", m4_depths, "S0"),
    ]

    for m_code, m_name, depths, imb in ensemble_configs:
        for depth in depths:
            depth_label = "None (無限制)" if depth is None else str(depth)
            val_pr_aucs, val_ar95s = [], []
            val_n_uniques = []
            fold_times = []

            for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
                t0 = time.time()
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                model_params = {
                    "max_depth": depth,
                    "n_estimators": n_estimators,
                }
                pipe = build_pipeline(
                    model_name=m_code,
                    imbalance=imb,
                    model_params=model_params,
                    seed=seed + fold_idx,
                    **pipe_flags,
                )
                pipe.fit(X_tr, y_tr)
                score_val = get_score(pipe, X_val)
                val_metrics = evaluate_fold_predictions(y_val, score_val)

                val_pr_aucs.append(val_metrics["pr_auc"])
                val_ar95s.append(val_metrics["ar_95"])
                val_n_uniques.append(val_metrics["n_unique_scores"])
                fold_times.append(time.time() - t0)

            records.append({
                "model_type": "Ensemble",
                "model_name": m_code,
                "max_depth": depth_label,
                "depth_numeric": 99 if depth is None else depth,
                "train_pr_auc_mean": np.nan,
                "train_pr_auc_std": np.nan,
                "val_pr_auc_mean": np.mean(val_pr_aucs),
                "val_pr_auc_std": np.std(val_pr_aucs),
                "pr_auc_gap": np.nan,
                "train_ar95_mean": np.nan,
                "val_ar95_mean": np.mean(val_ar95s),
                "val_ar95_std": np.std(val_ar95s),
                "ar95_gap": np.nan,
                "val_ar90_mean": np.nan,
                "n_unique_scores": float(np.mean(val_n_uniques)),
                "avg_fit_seconds": np.mean(fold_times),
            })
            print(
                f"  {m_code} ({m_name:<12}) depth={depth_label:<14} | "
                f"Val PR-AUC: {np.mean(val_pr_aucs):.4f} (+/- {np.std(val_pr_aucs):.4f}) | "
                f"Val AR@0.95: {np.mean(val_ar95s):.2f}x | 唯一分數數: {np.mean(val_n_uniques):.1f}"
            )

    df_results = pd.DataFrame(records)

    # 4. 讀取專案最佳參數標註固化深度
    m2_best = load_best_params("M2")["best_params"].get("max_depth")
    m3_best = load_best_params("M3")["best_params"].get("max_depth")
    m4_best = load_best_params("M4")["best_params"].get("max_depth")

    # 5. 輸出目錄與繪圖
    out_dir = LABS_OUTPUTS_DIR / "lab_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lab_01_tree_depth.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8")

    # 快照輸出至 docs/code_learning/lab_snapshots/
    snap_path = save_snapshot(df_results, "lab_01_tree_depth_snapshot.csv")

    # 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子圖 1: 單棵決策樹 Train vs Val PR-AUC (過擬合軌跡)
    dt_df = df_results[df_results["model_type"] == "Single_DecisionTree"].copy()
    x_labels = dt_df["max_depth"].tolist()
    x_pos = np.arange(len(x_labels))

    ax1 = axes[0]
    ax1.plot(x_pos, dt_df["train_pr_auc_mean"], "o-", color="#2b5c8f", label="訓練折 PR-AUC (Train)")
    ax1.plot(x_pos, dt_df["val_pr_auc_mean"], "s-", color="#e76f51", label="驗證折 PR-AUC (Val)")
    ax1.fill_between(
        x_pos,
        dt_df["val_pr_auc_mean"],
        dt_df["train_pr_auc_mean"],
        color="#e76f51",
        alpha=0.15,
        label="過擬合差距 (Gap)",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=30)
    ax1.set_title("單棵決策樹 (DT): 深度 vs PR-AUC (過擬合軌跡)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # 子圖 2: 單棵決策樹 AR@0.95 (預警倍數)
    ax2 = axes[1]
    ax2.plot(x_pos, dt_df["val_ar95_mean"], "D-", color="#2a9d8f", label="驗證折 AR@0.95 (越低越好)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=30)
    ax2.set_title("單棵決策樹 (DT): 深度 vs 驗證折 AR@0.95\n(粗階梯同分導致失真至 N/P≈31.0x)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("AR@0.95 (預警倍數)")
    ax2.text(0.5, 0.40, "注意：單樹葉節點機率為粗階梯\n悲觀同分迫使門檻納入全折同分樣本\n導致 AR 退化至全樣本倍數 N/P≈31x",
             transform=ax2.transAxes, fontsize=8, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#ffeeba", alpha=0.9))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 子圖 3: 樹集成模型 (M2, M3, M4) 驗證 PR-AUC 對比
    ax3 = axes[2]
    colors = {"M2": "#457b9d", "M3": "#1d3557", "M4": "#e63946"}
    for m_code in ["M2", "M3", "M4"]:
        sub = df_results[df_results["model_name"] == m_code].copy()
        x_pts = sub["depth_numeric"].tolist()
        y_pts = sub["val_pr_auc_mean"].tolist()
        # Sort by depth numeric
        xy = sorted(zip(x_pts, y_pts), key=lambda t: t[0])
        x_sorted = [t[0] for t in xy]
        y_sorted = [t[1] for t in xy]
        labels = ["None" if x == 99 else str(x) for x in x_sorted]
        ax3.plot(range(len(labels)), y_sorted, "o-", color=colors[m_code], label=f"{m_code} (Val PR-AUC)")

    # 標註固化值
    ax3.set_title(
        f"樹集成驗證 PR-AUC 對照\n(固化深度: M2={m2_best}, M3={m3_best}, M4={m4_best})",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("深度層級")
    ax3.set_ylabel("驗證折 PR-AUC")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    fig_path = out_dir / "lab_01_tree_depth.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    save_manifest(
        output_dir=out_dir,
        lab_id="lab_01",
        params=PARAMS,
        elapsed_seconds=elapsed,
        seed=seed,
        extra={
            "mode": mode,
            "m2_frozen_depth": str(m2_best),
            "m3_frozen_depth": str(m3_best),
            "m4_frozen_depth": str(m4_best),
        },
    )

    print("\n" + "=" * 70)
    print(f"LAB_01 執行完成！總耗時: {elapsed:.2f} 秒")
    print(f"結果 CSV : {csv_path}")
    print(f"圖表 PNG : {fig_path}")
    print(f"小型快照 : {snap_path}")
    print("=" * 70)

    return df_results, dt_df, csv_path, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 01: 樹深度掃描實驗")
    parser.add_argument("--full", action="store_true", help="執行完整模式 (5折x3重複, 300棵樹)")
    parser.add_argument("--standard", action="store_true", help="執行標準模式 (5折x1重複, 80棵樹, 8個深度)")
    parser.add_argument("--quick", action="store_true", default=False, help="執行快速模式 (5折x1重複, 40棵樹, 5個深度, 預設)")
    parser.add_argument("--pipeline", choices=["project", "teaching"], default="project", help="前處理旗標 (project 關閉截尾, teaching 開啟截尾, 預設 project)")
    args = parser.parse_args()

    if args.full:
        exec_mode = "full"
    elif args.standard:
        exec_mode = "standard"
    else:
        exec_mode = "quick"

    run_lab(mode=exec_mode, pipeline=args.pipeline)
