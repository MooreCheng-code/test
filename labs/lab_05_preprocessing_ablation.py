"""labs/lab_05_preprocessing_ablation.py - Preprocessing Ablation Study.

Topics:
1. Collinearity filtering thresholds: [None (Off), 0.99, 0.95, 0.90, 0.80]
2. Outlier winsorization: Winsorizer On vs Off
3. Feature scaling: QuantileScaler vs StandardScaler
Evaluated on linear model M1 (LogisticRegression) and tree ensemble M3 (ExtraTrees).

SAFETY:
- Uses ONLY load_train_data() (5,455 training rows). Never touches holdout test set.
- Re-fits models on custom CV folds. Never loads frozen final model objects.
- Outputs written to labs/outputs/lab_05/. Snapshot to docs/code_learning/lab_snapshots/.
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
    # 相關性過濾門檻 (None 表示關閉過濾)
    "corr_thresholds_quick": [None, 0.95, 0.80],
    "corr_thresholds_standard": [None, 0.99, 0.95, 0.90, 0.80],
    # 截尾開關
    "winsorizer_options": [True, False],
    # 縮放器選項
    "scaler_options": ["quantile", "standard"],
    # 測試模型
    "models": ["M1", "M3"],
    # 交叉驗證設定
    "cv_folds": 5,
    "cv_repeats_quick": 1,
    "cv_repeats_standard": 1,
    "cv_repeats_full": 3,
    "seed": 42,
    "noise_ref_seed": 100,  # 雜訊參照種子
}


def run_lab(mode: str = "quick", pipeline: str = "project") -> tuple[pd.DataFrame, Path, Path]:
    base_flags = get_pipeline_flags(pipeline)
    base_win = base_flags["use_winsorizer"]
    base_scaler = base_flags["scaler"]
    """Execute lab_05 preprocessing ablation study."""
    start_time = time.time()
    setup_chinese_font()

    if mode == "full":
        n_repeats = PARAMS["cv_repeats_full"]
        corr_threshs = PARAMS["corr_thresholds_standard"]
    elif mode == "standard":
        n_repeats = PARAMS["cv_repeats_standard"]
        corr_threshs = PARAMS["corr_thresholds_standard"]
    else:  # quick (default)
        n_repeats = PARAMS["cv_repeats_quick"]
        corr_threshs = PARAMS["corr_thresholds_quick"]

    winsor_opts = PARAMS["winsorizer_options"]
    scaler_opts = PARAMS["scaler_options"]
    models = PARAMS["models"]
    seed = PARAMS["seed"]
    noise_seed = PARAMS["noise_ref_seed"]

    print("=" * 70)
    print(f"LAB_05: 前處理消融實驗 ({mode} 模式)")
    print(f"CV 設定: {PARAMS['cv_folds']} 折 x {n_repeats} 次重複 (基準種子={seed}, 雜訊種子={noise_seed})")
    print("=" * 70)

    # 1. 唯讀載入訓練資料
    X_train, y_train = load_train_data(check_guard=True)
    splitter = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=seed)
    splitter_noise = get_cv_splitter(n_splits=PARAMS["cv_folds"], n_repeats=n_repeats, seed=noise_seed)

    records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    # 輔助評估函式
    def eval_pipeline_setting(
        m_code: str,
        ablation_group: str,
        setting_name: str,
        use_corr: bool,
        corr_th: float | None,
        use_win: bool,
        s_kind: str,
        sp_splitter,
        tag: str = "standard",
    ) -> dict[str, Any]:
        base_json = load_best_params(m_code)
        base_params = base_json["best_params"].copy()
        imb = base_json["imbalance"]

        # 快速模式下 ExtraTrees 樹數微調
        if mode == "quick" and m_code == "M3":
            base_params["n_estimators"] = min(base_params.get("n_estimators", 300), 80)

        pr_aucs, roc_aucs, ar95s, ar90s = [], [], [], []
        retained_feats = []
        fold_times = []

        for fold_idx, (tr_idx, val_idx) in enumerate(sp_splitter.split(X_train, y_train)):
            t0 = time.time()
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe = build_pipeline(
                model_name=m_code,
                imbalance=imb,
                scaler=s_kind,
                use_winsorizer=use_win,
                use_corr_filter=use_corr,
                model_params=base_params,
                seed=seed + fold_idx,
            )

            if use_corr and corr_th is not None:
                pipe.named_steps["corr_filter"].threshold = float(corr_th)

            pipe.fit(X_tr, y_tr)
            score_val = get_score(pipe, X_val)
            m = evaluate_fold_predictions(y_val, score_val)

            # 統計保留特徵數
            if use_corr:
                n_feats = len(pipe.named_steps["corr_filter"].selected_features_)
            else:
                n_feats = len(pipe.named_steps["corr_filter"].feature_names_in_)

            pr_aucs.append(m["pr_auc"])
            roc_aucs.append(m["roc_auc"])
            ar95s.append(m["ar_95"])
            ar90s.append(m["ar_90"])
            retained_feats.append(n_feats)
            fold_times.append(time.time() - t0)

            fold_records.append({
                "model_code": m_code,
                "ablation_group": ablation_group,
                "setting_name": setting_name,
                "fold": fold_idx,
                "ar95": float(m["ar_95"]),
                "pr_auc": float(m["pr_auc"]),
                "retained_features": int(n_feats),
                "n_pos": int(np.sum(y_val == 1)),
            })

        rec = {
            "model_code": m_code,
            "ablation_group": ablation_group,
            "setting_name": setting_name,
            "use_corr_filter": use_corr,
            "corr_threshold": "None" if corr_th is None else str(corr_th),
            "use_winsorizer": use_win,
            "scaler": s_kind,
            "eval_tag": tag,
            "retained_features_mean": float(np.mean(retained_feats)),
            "retained_features_min": int(np.min(retained_feats)),
            "retained_features_max": int(np.max(retained_feats)),
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
            "ar95_mean": float(np.mean(ar95s)),
            "ar95_std": float(np.std(ar95s)),
            "ar90_mean": float(np.mean(ar90s)),
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "avg_fit_seconds": float(np.mean(fold_times)),
        }
        records.append(rec)
        print(
            f"  [{m_code}] {ablation_group:<12} | {setting_name:<16} | "
            f"保留特徵: {rec['retained_features_mean']:<4.1f} | "
            f"Val PR-AUC: {rec['pr_auc_mean']:.4f} | "
            f"AR@0.95: {rec['ar95_mean']:.2f}x | 耗時: {rec['avg_fit_seconds']:.2f}s/fold"
        )
        return rec

    # 2. 執行消融掃描
    for m_code in models:
        print(f"\n--- 執行消融: {m_code} ---")

        # 2.1 相關性過濾門檻掃描 (基準: Winsorizer=True, Scaler='quantile')
        print("  [A] 相關性過濾門檻掃描...")
        for th in corr_threshs:
            use_c = th is not None
            s_name = "關閉過濾" if th is None else f"門檻={th}"
            eval_pipeline_setting(m_code, "corr_filter", s_name, use_c, th, base_win, base_scaler, splitter)

        # 2.2 截尾開關掃描 (基準: Corr=0.95, Scaler='quantile')
        print("  [B] 截尾開關 (Winsorizer) 掃描...")
        for w in winsor_opts:
            s_name = "截尾開啟" if w else "截尾關閉"
            eval_pipeline_setting(m_code, "winsorizer", s_name, True, 0.95, w, base_scaler, splitter)

        # 2.3 縮放器比較 (基準: Corr=0.95, Winsorizer=True)
        print("  [C] 縮放器 (Scaler) 掃描...")
        for sc in scaler_opts:
            s_name = f"Scaler={sc}"
            eval_pipeline_setting(m_code, "scaler", s_name, True, 0.95, base_win, sc, splitter)

        # 2.4 雜訊參照：以基準設定換切分種子 (seed=100)
        print("  [D] 切分種子雜訊參照評估...")
        rec_noise = eval_pipeline_setting(
            m_code, "noise_ref", "基準設定_種子100", True, 0.95, base_win, base_scaler, splitter_noise, tag="noise_ref"
        )

    df_results = pd.DataFrame(records)

    # 3. 計算基準與雜訊量級
    print("\n" + "=" * 70)
    print("消融實驗與切分雜訊參照對照表:")
    print("=" * 70)
    for m_code in models:
        sub = df_results[df_results["model_code"] == m_code]
        base_rec = sub[(sub["ablation_group"] == "corr_filter") & (sub["corr_threshold"] == "0.95")].iloc[0]
        noise_rec = sub[sub["ablation_group"] == "noise_ref"].iloc[0]

        noise_diff_pr = abs(base_rec["pr_auc_mean"] - noise_rec["pr_auc_mean"])
        noise_diff_ar = abs(base_rec["ar95_mean"] - noise_rec["ar95_mean"])

        print(f"\n[{m_code}] 基準 vs 雜訊水準 (換切分種子造成的變異):")
        print(f"  - 基準 PR-AUC (種子42): {base_rec['pr_auc_mean']:.4f} vs (種子100): {noise_rec['pr_auc_mean']:.4f} | 雜訊差: {noise_diff_pr:.4f}")
        print(f"  - 基準 AR@0.95 (種子42): {base_rec['ar95_mean']:.2f}x vs (種子100): {noise_rec['ar95_mean']:.2f}x | 雜訊差: {noise_diff_ar:.2f}x")

    # 4. 輸出儲存
    out_dir = LABS_OUTPUTS_DIR / "lab_05"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lab_05_preprocessing_ablation.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8")

    df_folds = pd.DataFrame(fold_records)
    folds_csv_path = out_dir / "lab_05_project_folds.csv"
    df_folds.to_csv(folds_csv_path, index=False, encoding="utf-8")

    snap_path = save_snapshot(df_results, "lab_05_preprocessing_ablation_snapshot.csv")
    save_snapshot(df_folds, "lab_05_project_folds_snapshot.csv")

    # 5. 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子圖 1: 相關性門檻 vs 保留特徵數與 PR-AUC
    ax1 = axes[0]
    corr_df = df_results[df_results["ablation_group"] == "corr_filter"].copy()
    for m_code, color in [("M1", "#2b5c8f"), ("M3", "#e76f51")]:
        sub = corr_df[corr_df["model_code"] == m_code]
        ax1.plot(sub["corr_threshold"], sub["pr_auc_mean"], "o-", color=color, label=f"{m_code} PR-AUC")
    ax1.set_title("相關性過濾門檻 vs PR-AUC", fontsize=11, fontweight="bold")
    ax1.set_xlabel("過濾門檻 (None=關閉)")
    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 子圖 2: 相關性門檻 vs 保留特徵數
    ax2 = axes[1]
    m1_corr = corr_df[corr_df["model_code"] == "M1"]
    ax2.bar(m1_corr["corr_threshold"], m1_corr["retained_features_mean"], color="#457b9d", alpha=0.8, width=0.4)
    ax2.set_title("相關性過濾門檻 vs 折內平均保留特徵數", fontsize=11, fontweight="bold")
    ax2.set_xlabel("過濾門檻")
    ax2.set_ylabel("保留特徵數")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 子圖 3: 截尾與縮放器比較長條圖
    ax3 = axes[2]
    other_df = df_results[df_results["ablation_group"].isin(["winsorizer", "scaler"])].copy()
    x_labels = other_df[other_df["model_code"] == "M1"]["setting_name"].tolist()
    x_indices = np.arange(len(x_labels))
    width = 0.35
    for i, m_code in enumerate(models):
        sub = other_df[other_df["model_code"] == m_code]
        y_vals = sub["pr_auc_mean"].tolist()
        color = "#2b5c8f" if m_code == "M1" else "#e76f51"
        ax3.bar(x_indices + (i - 0.5) * width, y_vals, width, label=m_code, color=color, alpha=0.85)
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(x_labels, rotation=20)
    ax3.set_title("截尾與縮放器消融 vs PR-AUC", fontsize=11, fontweight="bold")
    ax3.set_xlabel("前處理設定")
    ax3.set_ylabel("PR-AUC")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend()

    plt.tight_layout()
    fig_path = out_dir / "lab_05_preprocessing_ablation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    save_manifest(
        output_dir=out_dir,
        lab_id="lab_05",
        params=PARAMS,
        elapsed_seconds=elapsed,
        seed=seed,
        extra={
            "mode": mode,
        },
    )

    print("\n" + "=" * 70)
    print(f"LAB_05 執行完成！總耗時: {elapsed:.2f} 秒")
    print(f"結果 CSV : {csv_path}")
    print(f"圖表 PNG : {fig_path}")
    print(f"小型快照 : {snap_path}")
    print("=" * 70)

    return df_results, csv_path, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 05: 前處理消融實驗")
    parser.add_argument("--full", action="store_true", help="執行完整模式 (15折)")
    parser.add_argument("--standard", action="store_true", help="執行標準模式 (5折, 完整門檻)")
    parser.add_argument("--quick", action="store_true", default=True, help="執行快速模式 (5折, 3門檻, <=1.5分鐘, 預設)")
    parser.add_argument("--pipeline", choices=["project", "teaching"], default="project", help="前處理旗標 (project 關閉截尾, teaching 開啟截尾, 預設 project)")
    args = parser.parse_args()

    if args.full:
        exec_mode = "full"
    elif args.standard:
        exec_mode = "standard"
    else:
        exec_mode = "quick"

    run_lab(mode=exec_mode, pipeline=args.pipeline)
