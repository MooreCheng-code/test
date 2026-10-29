"""labs/lab_02_forest_hyperparams.py - Random Forest & Extra Trees Hyperparameters.

Topic: Forest trees count (n_estimators), feature subsampling (max_features),
and leaf constraint (min_samples_leaf) on M2 (RandomForest) and M3 (ExtraTrees).

SAFETY:
- Uses ONLY load_train_data() (5,455 training rows). Never touches holdout test set.
- Re-fits models on custom CV folds. Never loads frozen final model objects.
- Outputs written to labs/outputs/lab_02/. Snapshot to docs/code_learning/lab_snapshots/.
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
    # 掃描參數設定 (快速模式縮減以保證於 2 分鐘內完成)
    "n_estimators_quick": [100, 200, 400],
    "n_estimators_full": [100, 300, 500, 800],
    "max_features_options": ["sqrt", "log2", 0.3],
    "min_samples_leaf_options": [1, 2, 5, 10],
    # 快速模式下進行單變量掃描時的基底樹數
    "base_n_est_quick": 150,
    # 交叉驗證設定
    "cv_folds": 5,
    "cv_repeats_quick": 1,
    "cv_repeats_full": 3,
    "seed": 42,
}


def run_lab(quick: bool = True) -> tuple[pd.DataFrame, Path, Path]:
    """Execute lab_02 forest hyperparameters scan."""
    start_time = time.time()
    setup_chinese_font()

    n_repeats = PARAMS["cv_repeats_quick"] if quick else PARAMS["cv_repeats_full"]
    n_est_list = PARAMS["n_estimators_quick"] if quick else PARAMS["n_estimators_full"]
    max_features_list = PARAMS["max_features_options"]
    min_leaf_list = PARAMS["min_samples_leaf_options"]
    seed = PARAMS["seed"]

    print("=" * 70)
    print(f"LAB_02: 森林超參數掃描實驗 ({'--quick 快速模式' if quick else '--full 完整模式'})")
    print(f"CV 設定: {PARAMS['cv_folds']} 折 x {n_repeats} 次重複 (種子={seed})")
    print("=" * 70)

    # 1. 唯讀載入訓練資料與最佳基準參數
    X_train, y_train = load_train_data(check_guard=True)
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=seed)

    m2_base_json = load_best_params("M2")
    m3_base_json = load_best_params("M3")

    m2_base = m2_base_json["best_params"].copy()
    m2_imb = m2_base_json["imbalance"]

    m3_base = m3_base_json["best_params"].copy()
    m3_imb = m3_base_json["imbalance"]

    models_info = [
        ("M2", "RandomForest", m2_base, m2_imb),
        ("M3", "ExtraTrees", m3_base, m3_imb),
    ]

    records: list[dict[str, Any]] = []

    # 輔助評估函式
    def eval_config(
        m_code: str,
        m_name: str,
        param_group: str,
        param_name: str,
        param_val: Any,
        cfg_params: dict[str, Any],
        imbalance: str,
    ) -> None:
        pr_aucs, roc_aucs, ar95s, ar90s, ar80s = [], [], [], [], []
        fold_times = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
            t0 = time.time()
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe = build_pipeline(
                model_name=m_code,
                imbalance=imbalance,
                model_params=cfg_params,
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
            fold_times.append(time.time() - t0)

        record = {
            "model_code": m_code,
            "model_name": m_name,
            "param_group": param_group,
            "param_name": param_name,
            "param_value": str(param_val),
            "n_estimators": cfg_params.get("n_estimators"),
            "max_features": str(cfg_params.get("max_features")),
            "min_samples_leaf": cfg_params.get("min_samples_leaf"),
            "max_depth": str(cfg_params.get("max_depth")),
            "imbalance": imbalance,
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
            "ar95_mean": float(np.mean(ar95s)),
            "ar95_std": float(np.std(ar95s)),
            "ar90_mean": float(np.mean(ar90s)),
            "ar90_std": float(np.std(ar90s)),
            "ar80_mean": float(np.mean(ar80s)),
            "ar80_std": float(np.std(ar80s)),
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "roc_auc_std": float(np.std(roc_aucs)),
            "avg_fit_seconds": float(np.mean(fold_times)),
        }
        records.append(record)
        print(
            f"  [{m_code}] {param_name}={str(param_val):<8} | "
            f"Val PR-AUC: {record['pr_auc_mean']:.4f} (+/- {record['pr_auc_std']:.4f}) | "
            f"AR@0.95: {record['ar95_mean']:.2f}x (+/- {record['ar95_std']:.2f}) | "
            f"耗時: {record['avg_fit_seconds']:.2f}s/fold"
        )

    # 2. 依序單變量掃描
    for m_code, m_name, base_params, imb in models_info:
        print(f"\n--- 掃描模型: {m_code} ({m_name}) [基準不平衡策略: {imb}] ---")

        # 2.1 樹數 (n_estimators)
        print(f"\n[A] 掃描樹數量 n_estimators...")
        for n_est in n_est_list:
            cfg = base_params.copy()
            cfg["n_estimators"] = n_est
            eval_config(m_code, m_name, "n_estimators", "n_estimators", n_est, cfg, imb)

        # 快速模式下為保持在 2 分鐘內，單變量掃描時適度縮小樹數至 150
        base_scan_cfg = base_params.copy()
        if quick:
            base_scan_cfg["n_estimators"] = min(base_scan_cfg.get("n_estimators", 300), PARAMS["base_n_est_quick"])

        # 2.2 特徵抽樣 (max_features)
        print(f"\n[B] 掃描特徵抽樣 max_features...")
        for mf in max_features_list:
            cfg = base_scan_cfg.copy()
            cfg["max_features"] = mf
            eval_config(m_code, m_name, "max_features", "max_features", mf, cfg, imb)

        # 2.3 葉節點最小樣本 (min_samples_leaf)
        print(f"\n[C] 掃描葉節點最小樣本 min_samples_leaf...")
        for ml in min_leaf_list:
            cfg = base_scan_cfg.copy()
            cfg["min_samples_leaf"] = ml
            eval_config(m_code, m_name, "min_samples_leaf", "min_samples_leaf", ml, cfg, imb)

    df_results = pd.DataFrame(records)

    # 3. 輸出儲存
    out_dir = LABS_OUTPUTS_DIR / "lab_02"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lab_02_forest_hyperparams.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8")

    snap_path = save_snapshot(df_results, "lab_02_forest_hyperparams_snapshot.csv")

    # 4. 繪製圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 子圖 1: n_estimators vs PR-AUC (含標準差誤差帶)
    ax1 = axes[0, 0]
    nest_df = df_results[df_results["param_group"] == "n_estimators"].copy()
    for m_code, color in [("M2", "#457b9d"), ("M3", "#e63946")]:
        sub = nest_df[nest_df["model_code"] == m_code].sort_values("n_estimators")
        x = sub["n_estimators"].values
        y = sub["pr_auc_mean"].values
        y_std = sub["pr_auc_std"].values
        ax1.plot(x, y, "o-", color=color, label=f"{m_code} PR-AUC 均值", linewidth=2)
        ax1.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.15)
    ax1.set_title("樹數量 (n_estimators) vs 驗證折 PR-AUC (含折間標準差)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("n_estimators (樹數量)")
    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 子圖 2: n_estimators vs 擬合時間
    ax2 = axes[0, 1]
    for m_code, color in [("M2", "#457b9d"), ("M3", "#e63946")]:
        sub = nest_df[nest_df["model_code"] == m_code].sort_values("n_estimators")
        ax2.plot(sub["n_estimators"], sub["avg_fit_seconds"], "s-", color=color, label=f"{m_code} 每折擬合秒數")
    ax2.set_title("樹數量 (n_estimators) vs 擬合時間 (線性增長)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("n_estimators")
    ax2.set_ylabel("擬合時間 (秒/折)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 子圖 3: max_features vs PR-AUC
    ax3 = axes[1, 0]
    mf_df = df_results[df_results["param_group"] == "max_features"].copy()
    x_indices = np.arange(len(max_features_list))
    width = 0.35
    for i, (m_code, color) in enumerate([("M2", "#457b9d"), ("M3", "#e63946")]):
        sub = mf_df[mf_df["model_code"] == m_code].set_index("param_value")
        y_vals = [sub.loc[str(mf), "pr_auc_mean"] for mf in max_features_list]
        ax3.bar(x_indices + (i - 0.5) * width, y_vals, width, label=m_code, color=color, alpha=0.85)
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([str(mf) for mf in max_features_list])
    ax3.set_title("特徵抽樣 (max_features) vs 驗證折 PR-AUC", fontsize=11, fontweight="bold")
    ax3.set_xlabel("max_features")
    ax3.set_ylabel("PR-AUC")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend()

    # 子圖 4: min_samples_leaf vs AR@0.95
    ax4 = axes[1, 1]
    ml_df = df_results[df_results["param_group"] == "min_samples_leaf"].copy()
    for m_code, color in [("M2", "#457b9d"), ("M3", "#e63946")]:
        sub = ml_df[ml_df["model_code"] == m_code].sort_values("min_samples_leaf")
        ax4.plot(sub["min_samples_leaf"], sub["ar95_mean"], "D-", color=color, label=f"{m_code} AR@0.95 (越低越好)")
    ax4.set_title("葉節點約束 (min_samples_leaf) vs AR@0.95 (預警倍數)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("min_samples_leaf (最小葉樣本數)")
    ax4.set_ylabel("AR@0.95 (預警倍數)")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    plt.tight_layout()
    fig_path = out_dir / "lab_02_forest_hyperparams.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    save_manifest(
        output_dir=out_dir,
        lab_id="lab_02",
        params=PARAMS,
        elapsed_seconds=elapsed,
        seed=seed,
        extra={
            "quick_mode": quick,
            "m2_frozen_base": m2_base,
            "m3_frozen_base": m3_base,
        },
    )

    print("\n" + "=" * 70)
    print(f"LAB_02 執行完成！總耗時: {elapsed:.2f} 秒")
    print(f"結果 CSV : {csv_path}")
    print(f"圖表 PNG : {fig_path}")
    print(f"小型快照 : {snap_path}")
    print("=" * 70)

    return df_results, csv_path, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 02: 森林超參數掃描實驗")
    parser.add_argument("--full", action="store_true", help="執行完整模式")
    parser.add_argument("--quick", action="store_true", default=True, help="執行快速模式 (預設)")
    args = parser.parse_args()

    is_quick = not args.full
    run_lab(quick=is_quick)
