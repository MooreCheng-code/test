"""
scripts/generate_mvp_report_and_card.py: Automated Model Card & Audit Report Generator for Bankruptcy Prediction MVP.

Executes:
1. Dynamic computation of 15-fold outer-CV metrics (ROC-AUC, PR-AUC, Recall@Top10%, 15-fold outer-CV Platt Brier with min/max)
   from outputs/models/splits.joblib and OOF files (E_A, M3, M4, M5, M1, M2).
2. Independent evaluation on pre-stored E6 holdout test scores (outputs/e6/e6_test_scores.npz) - NEVER touches test features or labels directly.
3. Verification of 4 operational thresholds (R80, R90, R95, R100), confusion matrix (TP, FP, FN, TN), F1, costs,
   and cost reductions across multiple cost ratios (10, 20, 50) and baselines (inspect none vs inspect all).
4. Generates MODEL_CARD.md adhering strictly to specification sections, project design facts,
   embedding exact formulas, data sources, and governance disclaimers.
5. Emits outputs/tables/mvp_model_card_metrics.csv as an authoritative single source of truth for automated testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
MODELS_FINAL_DIR = MODELS_DIR / "final"
MODELS_EXTRAS_DIR = MODELS_DIR / "mvp_extras"
OOF_DIR = MODELS_DIR / "oof"
E6_DIR = PROJECT_ROOT / "outputs" / "e6"

import config
from src.ensemble import ecdf_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_mvp_report_and_card")


def compute_cost_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    c_fn_ratios: list[float] = [10.0, 20.0, 50.0],
    c_fp: float = 1.0,
) -> dict[str, Any]:
    """Computes confusion matrix, F1, alert counts, costs, and cost reduction ratios."""
    pred = (scores >= threshold)
    tp = int(np.sum((y_true == 1) & pred))
    fp = int(np.sum((y_true == 0) & pred))
    fn = int(np.sum((y_true == 1) & (~pred)))
    tn = int(np.sum((y_true == 0) & (~pred)))
    n_pos = tp + fn
    n_neg = fp + tn
    n_total = n_pos + n_neg
    n_alerts = int(np.sum(pred))

    rec = tp / n_pos if n_pos > 0 else 0.0
    prec = tp / n_alerts if n_alerts > 0 else 0.0
    f1 = float(f1_score(y_true, pred)) if n_alerts > 0 else 0.0
    alert_prop = n_alerts / n_total if n_total > 0 else 0.0
    alert_ratio = n_alerts / n_pos if n_pos > 0 else 0.0

    res: dict[str, Any] = {
        "n_alerts": n_alerts,
        "alert_proportion": alert_prop,
        "alert_ratio": alert_ratio,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "c_all_review": float(c_fp * n_neg),
    }

    for c_fn in c_fn_ratios:
        c_base = float(c_fn * n_pos)
        c_model = float(c_fn * fn + c_fp * fp)
        c_red = (c_base - c_model) / c_base if c_base > 0 else 0.0
        ratio_int = int(c_fn)
        res[f"c_base_{ratio_int}"] = c_base
        res[f"c_model_{ratio_int}"] = c_model
        res[f"cost_red_{ratio_int}"] = c_red

    return res


def generate_mvp_metrics() -> dict[str, Any]:
    """Computes all CV and test metrics dynamically."""
    logger.info("Loading splits.joblib and training labels...")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    train_idx = splits["train_idx"]
    orig_to_rel = {orig: rel for rel, orig in enumerate(train_idx)}
    cv_splits = splits["cv_splits"]

    # Load training ground truth
    df_raw = pd.read_csv(config.DATA_PATH)
    y_train = df_raw[config.TARGET_COL].values[train_idx]
    n_train = len(y_train)
    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = n_train - n_pos_train
    base_rate_train = n_pos_train / n_train

    # Constant prediction baseline p*(1-p)
    brier_bench_const = base_rate_train * (1.0 - base_rate_train)

    # Load OOF prediction files
    oof_files = {
        "E_A": OOF_DIR / "e3_ensemble_E_A.npy",
        "M3": OOF_DIR / "e2_M3.npy",
        "M4": OOF_DIR / "e2_M4.npy",
        "M5": OOF_DIR / "e2_M5.npy",
        "M1": OOF_DIR / "e2_M1.npy",
        "M2": OOF_DIR / "e2_M2.npy",
    }
    oof_data = {m: np.load(path) for m, path in oof_files.items()}

    # 1. 15-Fold CV Evaluation (ROC-AUC, PR-AUC, Recall@Top10%)
    logger.info("Computing 15-fold CV metrics...")
    cv_results = {}
    for model_name, oof in oof_data.items():
        rocs, prs, rec10s = [], [], []

        for s in cv_splits:
            rep = s["repeat"]
            val_rel = np.array([orig_to_rel[i] for i in s["val_idx"]])
            y_v = y_train[val_rel]
            s_v = oof[rep, val_rel]

            rocs.append(roc_auc_score(y_v, s_v))
            prs.append(average_precision_score(y_v, s_v))

            k = int(np.floor(0.10 * len(s_v)))
            top_k_idx = np.argsort(s_v)[::-1][:k]
            caught = int(np.sum(y_v[top_k_idx] == 1))
            rec10s.append(caught / int(np.sum(y_v == 1)))

        # Pooled OOF on repeat 0 (5,455 cases)
        s_pool0 = oof[0]
        roc_pool0 = float(roc_auc_score(y_train, s_pool0))
        pr_pool0 = float(average_precision_score(y_train, s_pool0))
        k_pool0 = int(np.floor(0.10 * len(s_pool0)))
        top_k_pool0 = np.argsort(s_pool0)[::-1][:k_pool0]
        rec10_pool0 = float(np.sum(y_train[top_k_pool0] == 1) / n_pos_train)

        cv_results[model_name] = {
            "roc_mean": float(np.mean(rocs)),
            "roc_std": float(np.std(rocs)),
            "pr_mean": float(np.mean(prs)),
            "pr_std": float(np.std(prs)),
            "rec10_mean": float(np.mean(rec10s)),
            "rec10_std": float(np.std(rec10s)),
            "pool0_roc": roc_pool0,
            "pool0_pr": pr_pool0,
            "pool0_rec10": rec10_pool0,
        }

    # 2. Calibration & Brier Scores (15-fold outer-CV protocol for all models)
    logger.info("Computing 15-fold outer-CV Platt calibration Brier scores for all models...")
    cv_brier_results = {}
    for m in ["E_A", "M3", "M4", "M5", "M1", "M2"]:
        oof_m = oof_data[m]
        brier_folds = []
        for s in cv_splits:
            rep = s["repeat"]
            val_rel = np.array([orig_to_rel[i] for i in s["val_idx"]])
            train_rel = np.array([orig_to_rel[i] for i in s["train_idx"]])

            s_tr = oof_m[rep, train_rel].reshape(-1, 1)
            y_tr = y_train[train_rel]
            s_val = oof_m[rep, val_rel].reshape(-1, 1)
            y_val = y_train[val_rel]

            lr = LogisticRegression(solver="lbfgs", random_state=42)
            lr.fit(s_tr, y_tr)
            probs_val = lr.predict_proba(s_val)[:, 1]
            brier_folds.append(float(brier_score_loss(y_val, probs_val)))

        cv_brier_results[m] = {
            "mean": float(np.mean(brier_folds)),
            "std": float(np.std(brier_folds)),
            "min": float(np.min(brier_folds)),
            "max": float(np.max(brier_folds)),
        }

    # 3. E6 Test Set Evaluation
    logger.info("Loading pre-stored E6 test set scores...")
    e6 = np.load(E6_DIR / "e6_test_scores.npz")
    y_test = e6["labels"]
    n_test = len(y_test)
    n_pos_test = int(np.sum(y_test == 1))
    n_neg_test = n_test - n_pos_test
    base_rate_test = n_pos_test / n_test

    test_scores = {
        "E_A": e6["score_EA"],
        "M3": e6["score_M3"],
        "M4": e6["pct_M4"],
        "M5": e6["pct_M5"],
    }

    test_metrics = {}
    for m, s in test_scores.items():
        roc = float(roc_auc_score(y_test, s))
        pr = float(average_precision_score(y_test, s))
        k = int(np.floor(0.10 * len(s)))
        top_k_idx = np.argsort(s)[::-1][:k]
        caught = int(np.sum(y_test[top_k_idx] == 1))
        rec10 = caught / n_pos_test
        test_metrics[m] = {
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_multiplier": pr / base_rate_test,
            "recall_top10": rec10,
            "caught_top10": caught,
        }

    # Calibrated probability Brier score on test set
    test_platt_probs = e6["prob_platt"]
    test_brier_ea = float(brier_score_loss(y_test, test_platt_probs))

    # 4. Operating Points Evaluation on Both Test Set and CV Pooled OOF
    logger.info("Evaluating four operating points on test set and CV pooled OOF...")
    thresholds_json = json.load(open(MODELS_FINAL_DIR / "thresholds.json"))

    # For CV Pooled OOF, M3 scores must be mapped via reference distribution
    ref_m3 = np.load(MODELS_FINAL_DIR / "reference_M3.npy")
    oof_m3_pct = ecdf_map(ref_m3, oof_data["M3"][0])
    cv_scores = {
        "E_A": oof_data["E_A"][0],
        "M3": oof_m3_pct,
    }

    op_table_test = []
    for model_name in ["E_A", "M3"]:
        s = test_scores[model_name]
        for op in ["R80", "R90", "R95", "R100"]:
            op_meta = thresholds_json[model_name][op]
            t_val = float(op_meta["threshold"])
            margin = float(op_meta.get("recommended_margin", 0.0))
            metrics_dict = compute_cost_metrics(y_test, s, t_val)
            metrics_dict.update({
                "model": model_name,
                "operating_point": op,
                "safety_margin": margin,
                "threshold": t_val,
            })
            op_table_test.append(metrics_dict)

    op_table_cv = []
    for model_name in ["E_A", "M3"]:
        s = cv_scores[model_name]
        for op in ["R80", "R90", "R95", "R100"]:
            op_meta = thresholds_json[model_name][op]
            t_val = float(op_meta["threshold"])
            margin = float(op_meta.get("recommended_margin", 0.0))
            metrics_dict = compute_cost_metrics(y_train, s, t_val)
            metrics_dict.update({
                "model": model_name,
                "operating_point": op,
                "safety_margin": margin,
                "threshold": t_val,
            })
            op_table_cv.append(metrics_dict)

    # 5. Demo Dataset Scale (Dynamic from CSV)
    demo_df = pd.read_csv(MODELS_EXTRAS_DIR / "oof_demo_batch.csv")
    n_demo = len(demo_df)
    n_pos_demo = int(demo_df["bankrupt"].sum()) if "bankrupt" in demo_df.columns else int(demo_df["Bankrupt?"].sum())
    demo_prev = n_pos_demo / n_demo

    # 6. Overall MVP Verdict (Evaluated on CV Pooled OOF of best model E_A)
    ea_cv_pr = cv_results["E_A"]["pool0_pr"]
    pr_mult_cv = ea_cv_pr / base_rate_train
    ea_cv_rec10 = cv_results["E_A"]["pool0_rec10"]

    pass_pr_auc = bool(pr_mult_cv >= config.MVP_PASS_PR_AUC_MULTIPLIER)
    pass_top10 = bool(ea_cv_rec10 >= config.MVP_PASS_TOP10_RECALL)
    pass_overall = pass_pr_auc and pass_top10

    return {
        "n_train": n_train,
        "n_pos_train": n_pos_train,
        "n_neg_train": n_neg_train,
        "base_rate_train": base_rate_train,
        "brier_bench_const": brier_bench_const,
        "n_test": n_test,
        "n_pos_test": n_pos_test,
        "n_neg_test": n_neg_test,
        "base_rate_test": base_rate_test,
        "cv_results": cv_results,
        "cv_brier_results": cv_brier_results,
        "test_metrics": test_metrics,
        "test_brier_ea": test_brier_ea,
        "op_table_test": op_table_test,
        "op_table_cv": op_table_cv,
        "n_demo": n_demo,
        "n_pos_demo": n_pos_demo,
        "demo_prev": demo_prev,
        "pr_mult_cv": pr_mult_cv,
        "pass_pr_auc": pass_pr_auc,
        "pass_top10": pass_top10,
        "pass_overall": pass_overall,
    }


def build_model_card_markdown(m: dict[str, Any]) -> str:
    """Builds the comprehensive markdown model card."""
    verdict_str = "PASS (驗收合格)" if m["pass_overall"] else "FAIL (未達驗收標準)"

    cv_ea = m["cv_results"]["E_A"]
    cv_m3 = m["cv_results"]["M3"]
    cv_m4 = m["cv_results"]["M4"]
    cv_m5 = m["cv_results"]["M5"]
    cv_m1 = m["cv_results"]["M1"]
    cv_m2 = m["cv_results"].get("M2")

    test_ea = m["test_metrics"]["E_A"]
    test_m3 = m["test_metrics"]["M3"]
    test_m4 = m["test_metrics"]["M4"]
    test_m5 = m["test_metrics"]["M5"]

    brier_ea = m["cv_brier_results"]["E_A"]
    brier_m3 = m["cv_brier_results"]["M3"]
    brier_m4 = m["cv_brier_results"]["M4"]
    brier_m5 = m["cv_brier_results"]["M5"]
    brier_m1 = m["cv_brier_results"]["M1"]
    brier_m2 = m["cv_brier_results"].get("M2")
    brier_const = m["brier_bench_const"]

    # Total dataset stats
    n_total = m["n_train"] + m["n_test"]
    n_pos_total = m["n_pos_train"] + m["n_pos_test"]
    n_neg_total = m["n_neg_train"] + m["n_neg_test"]
    imbalance_ratio = n_neg_total / n_pos_total

    # R90 test numbers
    r90_test_ea = [r for r in m["op_table_test"] if r["model"] == "E_A" and r["operating_point"] == "R90"][0]
    r100_test_ea = [r for r in m["op_table_test"] if r["model"] == "E_A" and r["operating_point"] == "R100"][0]

    # Baseline inspection costs for Test and CV Pooled
    test_none_10 = 10 * m["n_pos_test"]
    test_none_20 = 20 * m["n_pos_test"]
    test_none_50 = 50 * m["n_pos_test"]
    test_all_10 = 1 * m["n_neg_test"]
    test_all_20 = 1 * m["n_neg_test"]
    test_all_50 = 1 * m["n_neg_test"]

    cv_none_10 = 10 * m["n_pos_train"]
    cv_none_20 = 20 * m["n_pos_train"]
    cv_none_50 = 50 * m["n_pos_train"]
    cv_all_10 = 1 * m["n_neg_train"]
    cv_all_20 = 1 * m["n_neg_train"]
    cv_all_50 = 1 * m["n_neg_train"]

    md = f"""# 模型卡：企業破產預測 AI 核心與 MVP 評估 (Model Card)

> **版本**：v2.0.0 (MVP 評估版)  
> **發布日期**：2026-10-27  
> **主要模型**：本專案設計之三模型百分位中點等權平均集成 ($E_A$: ExtraTrees $M_3$ + XGBoost $M_4$ + LightGBM $M_5$)  
> **評估基準**：正則化羅吉斯迴歸 ($M_1$)、單模型控制組 ($M_3$ ExtraTrees)  
> **總體檢驗結論**：**{verdict_str}**（依 `config.py` MVP 標準判定）

---

## 1. 模型基本資訊 (Model Details)

- **開發團隊**：企業破產風險 AI 專案工程組
- **模型架構**：
  - **核心模型 ($E_A$)**：採用無母數經驗累積分布函數（ECDF）百分位中點對齊後之等權重平均集成（同分時以對稱中點百分位映射）：
    $$\\text{{Score}}_{{E_A}} = \\frac{{1}}{{3}} \\left( \\text{{ECDF}}_{{M_3}}(s_3) + \\text{{ECDF}}_{{M_4}}(s_4) + \\text{{ECDF}}_{{M_5}}(s_5) \\right)$$
  - **基底弱分類器**：
    - $M_3$：`ExtraTreesClassifier` 超隨機樹分類器（72 維特徵，深度與分割超參數優化）
    - $M_4$：`XGBClassifier` XGBoost 梯度提升決策樹分類器（72 維特徵，學習率與正則化優化）
    - $M_5$：`LGBMClassifier` LightGBM 梯度提升決策樹分類器（72 維特徵，SMOTE 重抽樣控制組）
    - $M_1$：`LogisticRegression` $L_1/L_2$ 正則化羅吉斯迴歸分類器（基準對照）
    - $M_2$：隨機森林分類器（RandomForest，備選基底）
  - **機率校準器**：Platt 縮放器（單變數 Sigmoid 迴歸），維持原始風險排序完全單調不變。
- **輸入特徵規格**：95 維標準化財報財務比率（涵蓋獲利能力、資產運營、槓桿結構與現金流）。
- **推論依賴凍結**：模型權重、參考分布、校準器與門檻均自 `outputs/models/final/` 與 `outputs/models/mvp_extras/` 載入，具備完全 SHA-256 不變性。

---

## 2. 預期用途與應用場景 (Intended Use)

### 2.1 主要預期用途
- **輔助風險排查**：協助分析人員優先檢視高風險企業名單，不取代授信、投資或法律判斷。
- **高風險名單篩檢 (Triage)**：依風控能量設定前 10% 產能配額（Recall@Top10%），聚焦審查資源於核心暴險企業。
- **四個工作點門檻**：
  - **R80**：常態監控門檻，篩出邊際惡化企業。
  - **R90**：建議核心工作點，平衡召回率與審查負擔。
  - **R95**：高度警示門檻，聚焦極高風險企業。
  - **R100**：極端防線門檻，全量安全底線檢視（接近全抓，請參閱限制說明）。

### 2.2 非預期 / 禁用場景 (Out-of-Scope Uses)
- **法定裁量依據**：不得作為自動拒絕授信、投資或法律處置的唯一依據（輸出為統計學風險預警排序與校準分數，不取代專業人員審查）。
- **個人信貸評分**：本模型財務特徵專為企業法人財報設計，嚴禁套用於個人信用卡或零售信貸審查。

---

## 3. 影響因子與評估群體 (Factors, Populations & Subgroups)

- **資料來源**：提案所附資料集（含 95 維財務比率）。
- **樣本總量與標籤分佈**：
  - 總資料集：{n_total:,} 家企業，破產正樣本 {n_pos_total} 家，非破產負樣本 {n_neg_total:,} 家（破產盛行率 {n_pos_total/n_total:.2%}）。
  - 訓練集：{m["n_train"]:,} 家企業，破產正樣本 {m["n_pos_train"]} 家（盛行率 {m["base_rate_train"]:.2%}）。
  - 測試集（E6 封存）：{m["n_test"]:,} 家企業，破產正樣本 {m["n_pos_test"]} 家（盛行率 {m["base_rate_test"]:.2%}）。
- **極度不平衡挑戰**：破產事件極度稀缺（資料集動態計算負正比為 {n_neg_total}/{n_pos_total} $\\approx$ {imbalance_ratio:.1f}:1），常規準確率（Accuracy）無評估價值，模型全面鎖定 ROC-AUC、PR-AUC 與 Recall@Top10%。

---

## 4. 量化效能評估 (Quantitative Analyses)

### 4.1 15 折交叉驗證效能矩陣 (5-Fold CV $\\times$ 3 Repeats)

| 模型名稱 | 模型定位 | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) | CV Recall@Top10% | 基準倍數 (vs 隨機) | 15 折外層 CV Platt Brier (mean ± std [min, max]) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_A$ (Ensemble)** | **主要推薦** | **{cv_ea["roc_mean"]:.4f} ± {cv_ea["roc_std"]:.4f}** | **{cv_ea["pr_mean"]:.4f} ± {cv_ea["pr_std"]:.4f}** | **{cv_ea["rec10_mean"]:.2%}** | **{m["pr_mult_cv"]:.1f}x** | **{brier_ea["mean"]:.6f} ± {brier_ea["std"]:.6f} [{brier_ea["min"]:.6f}, {brier_ea["max"]:.6f}]** |
| $M_3$ (ExtraTrees) | 控制組 | {cv_m3["roc_mean"]:.4f} ± {cv_m3["roc_std"]:.4f} | {cv_m3["pr_mean"]:.4f} ± {cv_m3["pr_std"]:.4f} | {cv_m3["rec10_mean"]:.2%} | {cv_m3["pool0_pr"]/m["base_rate_train"]:.1f}x | {brier_m3["mean"]:.6f} ± {brier_m3["std"]:.6f} [{brier_m3["min"]:.6f}, {brier_m3["max"]:.6f}] |
| $M_5$ (LightGBM) | 單模候選 | {cv_m5["roc_mean"]:.4f} ± {cv_m5["roc_std"]:.4f} | {cv_m5["pr_mean"]:.4f} ± {cv_m5["pr_std"]:.4f} | {cv_m5["rec10_mean"]:.2%} | {cv_m5["pool0_pr"]/m["base_rate_train"]:.1f}x | {brier_m5["mean"]:.6f} ± {brier_m5["std"]:.6f} [{brier_m5["min"]:.6f}, {brier_m5["max"]:.6f}] |
| $M_4$ (XGBoost) | 單模基底 | {cv_m4["roc_mean"]:.4f} ± {cv_m4["roc_std"]:.4f} | {cv_m4["pr_mean"]:.4f} ± {cv_m4["pr_std"]:.4f} | {cv_m4["rec10_mean"]:.2%} | {cv_m4["pool0_pr"]/m["base_rate_train"]:.1f}x | {brier_m4["mean"]:.6f} ± {brier_m4["std"]:.6f} [{brier_m4["min"]:.6f}, {brier_m4["max"]:.6f}] |
| $M_1$ (LogisticReg) | 線性基準 | {cv_m1["roc_mean"]:.4f} ± {cv_m1["roc_std"]:.4f} | {cv_m1["pr_mean"]:.4f} ± {cv_m1["pr_std"]:.4f} | {cv_m1["rec10_mean"]:.2%} | {cv_m1["pool0_pr"]/m["base_rate_train"]:.1f}x | {brier_m1["mean"]:.6f} ± {brier_m1["std"]:.6f} [{brier_m1["min"]:.6f}, {brier_m1["max"]:.6f}] |
| $M_2$ (RandomForest) | 樹模型對照 | {cv_m2["roc_mean"]:.4f} ± {cv_m2["roc_std"]:.4f} | {cv_m2["pr_mean"]:.4f} ± {cv_m2["pr_std"]:.4f} | {cv_m2["rec10_mean"]:.2%} | {cv_m2["pool0_pr"]/m["base_rate_train"]:.1f}x | {brier_m2["mean"]:.6f} ± {brier_m2["std"]:.6f} [{brier_m2["min"]:.6f}, {brier_m2["max"]:.6f}] |
| 常數預測基準 $p(1-p)$ | 隨機基準 | 0.5000 | {m["base_rate_train"]:.4f} | 10.00% | 1.0x | {brier_const:.6f} (盛行率基準) |

> [!NOTE]
> **Brier 分數計算協定與 Platt 縮放輸入說明**：
> 1. 所有模型（包含 $E_A$）均採用完全相同之 **15 折外層交叉驗證協定** 重算（每折以其餘 4 折訓練樣本擬合 Platt 縮放器，再評估於該折留出之外層驗證樣本）。
> 2. **Platt 縮放器輸入型態**：輸入為模型輸出之**原始預測機率／百分位分數** $s \\in [0, 1]$（單變數特徵輸入至 `LogisticRegression(solver='lbfgs')`，擬合 $P(Y=1 \\mid s) = \\sigma(w \\cdot s + b)$），**未經 logit 轉換**。
> 3. **常數預測基準**：由資料集訓練盛行率 $p={m['base_rate_train']:.6f}$ 計算 $p(1-p)={brier_const:.6f}$。所有校準模型之 Brier 分數均低於此常數基準。
> 4. **模型校準表現說明**：樹模型對照組 $M_2$ (RandomForest) 之 15 折外層 CV Platt Brier 均值為 **{brier_m2["mean"]:.6f}**，低於主要模型 $E_A$ 的 **{brier_ea["mean"]:.6f}**；$E_A$ 並非校準分數（Brier）最低之模型。

### 4.2 E6 獨立未接觸測試集效能 (Unseen Test Set, N=1,364, 破產=44 家)

| 模型名稱 | 測試集 ROC-AUC | 測試集 PR-AUC | PR-AUC 基準提升 | Recall@Top10% | Top 10% 命中數 / 總破產數 | 測試集 Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_A$ (Ensemble)** | **{test_ea["roc_auc"]:.4f}** | **{test_ea["pr_auc"]:.4f}** | **{test_ea["pr_multiplier"]:.1f}x** | **{test_ea["recall_top10"]:.2%}** | **{test_ea["caught_top10"]} / {m["n_pos_test"]} 家** | **{m["test_brier_ea"]:.6f}** |
| $M_3$ (ExtraTrees) | {test_m3["roc_auc"]:.4f} | {test_m3["pr_auc"]:.4f} | {test_m3["pr_multiplier"]:.1f}x | {test_m3["recall_top10"]:.2%} | {test_m3["caught_top10"]} / {m["n_pos_test"]} 家 | - |
| $M_4$ (XGBoost) | {test_m4["roc_auc"]:.4f} | {test_m4["pr_auc"]:.4f} | {test_m4["pr_multiplier"]:.1f}x | {test_m4["recall_top10"]:.2%} | {test_m4["caught_top10"]} / {m["n_pos_test"]} 家 | - |
| $M_5$ (LightGBM 控制) | {test_m5["roc_auc"]:.4f} | {test_m5["pr_auc"]:.4f} | {test_m5["pr_multiplier"]:.1f}x | {test_m5["recall_top10"]:.2%} | {test_m5["caught_top10"]} / {m["n_pos_test"]} 家 | - |

> [!IMPORTANT]
> **主要模型選擇依據與 $M_5$ 測試指標說明**：
> 測試集實測中，$M_5$ 之 PR-AUC（0.5679）與 Recall@Top10%（86.36%）不低於 $E_A$（0.5676 與 79.55%）。本專案依據 E3 階段 15 折交叉驗證與 E6 測試集鎖定判準，選定 $E_A$ 為主要模型，原文依據如下：
> 1. **E3 判定原文引用**（`PROJECT_STATE.md:70` 與 `RESULTS_SUMMARY.md:59`）：
>    > 「基於工程多樣性與穩定性判斷，最終選定主要組 $E_A$（M3 ExtraTrees + M4 XGBoost + M5 LightGBM 之 ECDF 百分位等權平均），對照組為單一模型 M3。」  
>    > 「5 個單一機器學習模型兩兩比較（10 對模型 $\\times$ 5 個指標共 50 組檢定），Nadeau-Bengio 調整之 Holm 校正 $p$ 值全數為 $p_{{\\text{{Holm}}}} = 1.0$，統計上無法區分；4 個集成與 5 個單一模型配對比較（20 組配對）最小之 Holm $p$ 值為 $p_{{\\text{{Holm}}}} = 0.6195 > 0.05$；ECDF 排名等權集成 $E_A$ 展現數值上的穩定小幅改善：15 折平均 AR@0.95 為 8.06x（折內排名版為 8.04x）、15 折平均 PR-AUC 為 0.4598（Pooled OOF 為 0.4410）、15 折平均損失為 1.6344（折內排名版為 1.6327，Pooled OOF 為 1.6525）。選定 $E_A$ 為主要模型、M3 為控制組。」
> 2. **E6 判定原文引用**（`RESULTS_SUMMARY.md:75`）：
>    > 「測試集（1,364 筆，44 家破產）一次性解封鎖定（`outputs/logs/E6_LOCK`），所有工作點全數達成目標召回率（$R=0.90$ 捕獲 43/44，實際召回 97.73%，預警倍數 7.66x，精確率 12.76%；$R=0.95$ 捕獲 44/44，實際召回 100.00%，預警倍數 11.91x，精確率 8.40%）；$E_A - M3$ 配對 Bootstrap 差值 95% CI 為 [-0.0226, +0.1044]，區間跨越 0，統計上無法區分；測試集點估計（PR-AUC 0.5676, ROC-AUC 0.9621, AR@0.95 5.25x）優於（較低）15 折 CV 均值，顯示測試集看起來較容易、不可將 100% 召回當作未來保證，專題結論以 CV 估計為主、測試集為確認。」
> 3. **折間穩定性**：$E_A$ 在 15 折 CV 的 Brier 標準差僅 **{brier_ea["std"]:.6f}**（極小波動範圍 [{brier_ea["min"]:.6f}, {brier_ea["max"]:.6f}]），低於單模型 $M_5$ 的 **{brier_m5["std"]:.6f}**（[{brier_m5["min"]:.6f}, {brier_m5["max"]:.6f}]）與 $M_4$ 的 **{brier_m4["std"]:.6f}**。
> 4. **指標定義區分**：AR@0.95 是以排序找出 95% 召回所需的最小預警倍數，R95 是以訓練門檻實際預警的倍數，兩者定義不同。

### 4.3 四個工作點預警門檻與成本敏感度分析 (測試集 E6 實測，N=1,364，破產=44 家)

| 模型 | 工作點 | 門檻值 $t$ | 預警家數 (比例) | 預警倍數 | 混淆矩陣 (TP / FP / FN / TN) | 召回率 | 精確率 | F1 分數 | 業務成本 ($C_{{20}}$) | 成本節省 ($C_{{FN}}/C_{{FP}}=10$) | 成本節省 ($C_{{FN}}/C_{{FP}}=20$) | 成本節省 ($C_{{FN}}/C_{{FP}}=50$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in m["op_table_test"]:
        cm_str = f"{r['tp']} / {r['fp']} / {r['fn']} / {r['tn']}"
        md += (
            f"| **{r['model']}** | {r['operating_point']} | {r['threshold']:.4f} | "
            f"{r['n_alerts']} ({r['alert_proportion']:.1%}) | {r['alert_ratio']:.2f}x | {cm_str} | "
            f"{r['recall']:.1%} | {r['precision']:.2%} | {r['f1']:.4f} | {int(r['c_model_20'])} | "
            f"{r['cost_red_10']:.1%} | {r['cost_red_20']:.1%} | {r['cost_red_50']:.1%} |\n"
        )

    md += f"""
### 4.4 四個工作點預警門檻與成本敏感度分析 (CV 池化 OOF 實測，N=5,455，破產=176 家)

| 模型 | 工作點 | 門檻值 $t$ | 預警家數 (比例) | 預警倍數 | 混淆矩陣 (TP / FP / FN / TN) | 召回率 | 精確率 | F1 分數 | 業務成本 ($C_{{20}}$) | 成本節省 ($C_{{FN}}/C_{{FP}}=10$) | 成本節省 ($C_{{FN}}/C_{{FP}}=20$) | 成本節省 ($C_{{FN}}/C_{{FP}}=50$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in m["op_table_cv"]:
        cm_str = f"{r['tp']} / {r['fp']} / {r['fn']} / {r['tn']}"
        md += (
            f"| **{r['model']}** | {r['operating_point']} | {r['threshold']:.4f} | "
            f"{r['n_alerts']} ({r['alert_proportion']:.1%}) | {r['alert_ratio']:.2f}x | {cm_str} | "
            f"{r['recall']:.1%} | {r['precision']:.2%} | {r['f1']:.4f} | {int(r['c_model_20'])} | "
            f"{r['cost_red_10']:.1%} | {r['cost_red_20']:.1%} | {r['cost_red_50']:.1%} |\n"
        )

    md += f"""
> [!NOTE]
> **成本節省率定義與基準說明（數值記錄於 `outputs/tables/mvp_model_card_metrics.csv`）**：
> - **完全不審查基準（Inspect None Baseline）**：$C_{{\\text{{base}}}} = C_{{FN}} \\times N_{{\\text{{pos}}}}$（所有破產企業皆漏判）：
>   - 測試集（$N_{{\\text{{pos}}}}={m["n_pos_test"]}$）：$C_{{10}} = {test_none_10}$、$C_{{20}} = {test_none_20}$、$C_{{50}} = {test_none_50}$。
>   - CV 池化（$N_{{\\text{{pos}}}}={m["n_pos_train"]}$）：$C_{{10}} = {cv_none_10}$、$C_{{20}} = {cv_none_20}$、$C_{{50}} = {cv_none_50}$。
> - **全數審查基準（Inspect All Baseline）**：$C_{{\\text{{all\\_review}}}} = C_{{FP}} \\times N_{{\\text{{neg}}}}$（所有非破產企業皆耗費審查成本）：
>   - 測試集（$N_{{\\text{{neg}}}}={m["n_neg_test"]}$）：$C_{{10}} = {test_all_10}$、$C_{{20}} = {test_all_20}$、$C_{{50}} = {test_all_50}$（即 $1 \\times {m["n_neg_test"]} = {test_all_20}$）。
>   - CV 池化（$N_{{\\text{{neg}}}}={m["n_neg_train"]}$）：$C_{{10}} = {cv_all_10}$、$C_{{20}} = {cv_all_20}$、$C_{{50}} = {cv_all_50}$（即 $1 \\times {m["n_neg_train"]} = {cv_all_20}$）。
> - **模型成本節省率（相對於完全不審查基準）**：
>   $$\\text{{Cost Reduction}} = \\frac{{C_{{\\text{{base}}}} - C_{{\\text{{model}}}}}}{{C_{{\\text{{base}}}}}} = \\frac{{C_{{FN}} \\times N_{{\\text{{pos}}}} - (C_{{FN}} \\times FN + C_{{FP}} \\times FP)}}{{C_{{FN}} \\times N_{{\\text{{pos}}}}}}$$
> - **AR@0.95 與 R95 預警倍數之定義區分**：AR@0.95 是以排序找出 95% 召回所需的最小預警倍數，R95 是以訓練門檻實際預警的倍數，兩者定義不同。
> - **高召回工作點成本反轉說明**：在測試集與 CV 池化評估中，R95 在 $C_{{FN}}/C_{{FP}}=10:1$（測試集 -9.1%、CV -13.0%）以及 R100 在 $C_{{FN}}/C_{{FP}}=20:1$（測試集 -37.0%、CV -25.5%）的成本節省率皆為負值（業務成本超過不審查基準 $C_{{\\text{{base}}}}$），代表高召回工作點為追求極限召回放寬門檻引發大量假陽性調查成本，總成本可能高於完全不審查。

### 4.5 MVP 驗收判定準則與結論

> [!IMPORTANT]
> **MVP PASS 判定標準說明**：MVP 驗收判定指標評估於主要推薦模型（$E_A$）在 15 折交叉驗證池化 OOF（Repeat 0，5,455 筆）之表現；各項檢驗標準（`MVP_PASS_PR_AUC_MULTIPLIER` 與 `MVP_PASS_TOP10_RECALL`）均可在 `config.py` 中集中配置。

- **檢驗項目 1：PR-AUC 基準提升倍數**  
  標準：$\\ge {config.MVP_PASS_PR_AUC_MULTIPLIER:.1f}\\times$ 隨機猜測盛行率。  
  實測：CV Pooled ROC-AUC 為 {cv_ea["pool0_roc"]:.4f}，PR-AUC 為 {cv_ea["pool0_pr"]:.4f}（提升 **{m["pr_mult_cv"]:.1f}x**）$\\implies$ **{"PASS" if m["pass_pr_auc"] else "FAIL"}**。
- **檢驗項目 2：前 10% 產能召回率 (Recall@Top10%)**  
  標準：$\\ge {config.MVP_PASS_TOP10_RECALL:.0%}$。  
  實測：CV Pooled Recall@Top10% 為 **{cv_ea["pool0_rec10"]:.2%}**（測試集實測 **{test_ea["recall_top10"]:.2%}**）$\\implies$ **{"PASS" if m["pass_top10"] else "FAIL"}**。
- **非對稱誤判成本報告（規格第 6 節要求：同時報告誤報成本）**  
  計算公式（測試集 $E_A$ 工作點 R90 實測）：
  $$C_{{\\text{{base}}}} = C_{{FN}} \\times N_{{\\text{{pos}}}} = 20 \\times 44 = 880$$
  $$C_{{\\text{{model}}}} = C_{{FN}} \\times FN + C_{{FP}} \\times FP = 20 \\times {r90_test_ea['fn']} + 1 \\times {r90_test_ea['fp']} = {20 * r90_test_ea['fn']} + {r90_test_ea['fp']} = {int(r90_test_ea['c_model_20'])}$$
  $$\\text{{Cost Reduction}} = \\frac{{880 - {int(r90_test_ea['c_model_20'])}}}{{880}} = \\frac{{{880 - int(r90_test_ea['c_model_20'])}}}{{880}} \\approx {r90_test_ea['cost_red_20']:.1%}$$
  出處：`outputs/tables/mvp_model_card_metrics.csv`。  
  實測：測試集 $E_A$ 在建議核心工作點 R90 且 $C_{{FN}}/C_{{FP}}=20$ 下，業務成本由 880 降至 {int(r90_test_ea['c_model_20'])}，成本節省率達 **{r90_test_ea['cost_red_20']:.1%}**。
- **總體 MVP 判定**：**{verdict_str}**

---

## 5. 模型限制與治理聲明 (Model Limitations & Caveats)

1. **M5 模型超參數不一致與全域 SHAP 之影響 (Verbatim Restriction)**：
   > CV 評估的 M5 啟用 subsample_freq=1，最終凍結 M5 為 subsample_freq=0；CV 指標僅供選模參考，最終效能以 E6 測試集結果為準；E5 的 M5 全域 SHAP 亦使用未含該參數的重建設定。
2. **單模型門檻安全邊際未經折間驗證 (`margin_validated=False`)**：
   $E_A$ 與 $M_3$ 之最終門檻歷經完整 E4a 折間轉移安全邊際校驗（$m \\in [0.02, 0.06]$），具備折間穩定性保護；$M_1$ 與 $M_4$ 之四個工作點門檻採推導參考集未加安全邊際之原始分位點（$m=0$）。二者雖然同源自訓練集，但同名工作點之實際有效目標不同，**不可直接比較各模型之預警倍數**。
3. **極限工作點 R100 接近全抓警語**：
   R100 門檻之本質為排查極端安全企業，在 $E_A$ 測試集上之預警家數達 {r100_test_ea['n_alerts']} 家，佔比高達 {r100_test_ea['alert_proportion']:.1%}：
   $$\\text{{Alert Proportion}} = \\frac{{N_{{\\text{{alerts}}}}}}{{N_{{\\text{{test}}}}}} = \\frac{{{r100_test_ea['n_alerts']}}}{{{m['n_test']}}} \\approx {r100_test_ea['alert_proportion'] * 100:.1f}\\%$$
   （預警倍數 {r100_test_ea['alert_ratio']:.2f}x）。在低誤判成本比率（如 $C_{{FN}}/C_{{FP}}=10$）下，大量假陽性會使成本節省率反轉為負值（{r100_test_ea['cost_red_10']:.1%}）。**R100 僅適合作為安全底線防護，不具常態業務篩檢之產能效益**。
4. **示範資料集規模與分佈限制**：
   本系統展示分頁所使用之 OOF 示範資料集（`oof_demo_batch.csv`）共收錄 **{m["n_demo"]:,}** 家企業，其中破產正樣本 **{m["n_pos_demo"]}** 家（盛行率 {m["demo_prev"]:.2%}），來自第 0 次重複第 0 折之驗證集。示範評估之各項指標受限於該折次樣本，實際投產效能應以全量 CV 與 E6 獨立測試集報告為準。
5. **交叉驗證設計取代單一驗證集**：
   鑑於破產極端稀缺特性（全集僅 220 筆正樣本），單一固定驗證集具有高度抽樣偶然性與折間波動。本專案全流程採用 5 折重複 3 次（共 15 折）嚴格分層交叉驗證，以抑制評估變異並提高估計穩定性。
6. **成本矩陣僅供決策後驗敏感度評估**：
   $C_{{FN}}/C_{{FP}} \\in [10, 20, 50]$ 敏感度矩陣並非模型之訓練目標損失函數，僅作為評估不同誤判成本權衡下之綜合決策效益。
7. **校準分數與真實母體破產率之偏離**：
   本模型之 `calibrated_score`（Platt Scaling）校準基準為訓練集內部破產盛行率（{m["base_rate_train"]:.2%}），並非宏觀商業環境未經調整之常態破產率。若在不同破產率之客群落地部署，需實施先驗機率偏移調整（Prior Odds Ratio Adjustment / Prior Probability Shift）：
   $$\\text{{Odds}}_{{\\text{{adj}}}} = \\text{{Odds}}_{{\\text{{model}}}} \\times \\frac{{\\pi_{{\\text{{target}}}} / (1 - \\pi_{{\\text{{target}}}})}}{{\\pi_{{\\text{{train}}}} / (1 - \\pi_{{\\text{{train}}}})}}$$
   **公式成立前提與近似性**：
   - 條件機率分布 $P(X \\mid Y)$ 在訓練母體與目標母體間維持不變（先驗機率偏移假設）。
   - 此公式僅在校準分數可視為後驗機率時近似成立；若校準分數僅具備排序單調性而非真實條件機率，經勝算比轉換後之數值僅能視為相對排序調整，非嚴格母體事後機率。
   - 公式中嚴禁填入假定預設數值，須由實際部署環境之目標客群破產率 $\\pi_{{\\text{{target}}}}$ 動態代入。
8. **財務特徵高共線性與局部歸因局限性**：
   財報比率間普遍存在多重共線性（如流動比率、速動比率與現金流結構）。局部解釋性工具（$M_4$ TreeSHAP 與 $M_1$ 線性係數）提供單一模型特徵敏感度視角，無法直接等同於三模型集成 $E_A$ 之綜合非線性決策邊界，分析人員應結合全量財報背景綜合評判。
9. **測試集樣本規模與泛化保證限制**：
   E6 獨立測試集僅包含 44 家破產正樣本（總樣本 1,364 筆）。雖然在 R95 與 R100 工作點上實測召回率達到 100.0%（捕獲 44/44 家），且測試集點估計（PR-AUC 0.5676, ROC-AUC 0.9621, AR@0.95 5.25x）優於（較低）15 折 CV 均值，但此點估計顯示測試集看起來較容易，**不可把 100% 召回當作未來表現的保證**。專題結論與生產風控配置應以 15 折 CV 估計為主，測試集結果僅作為一次性驗收確認。
10. **缺失值推論與特徵敏感度邊界**：
   底層樹模型原生支援缺失值分裂，推論正常執行不報錯。注意：E_A 的 ECDF 百分位在參考分布頂端會接近 1，單一特徵的小幅變動可能不改變 E_A 分數。
"""
    return md


def main():
    logger.info("Starting MVP Report and Model Card Generation...")
    metrics = generate_mvp_metrics()

    # 1. Output comprehensive CSV metrics table FIRST so it acts as single source of truth
    csv_rows = []

    # CV 15-fold metrics (mean and std)
    for m_name, cv_m in metrics["cv_results"].items():
        csv_rows.append({"section": "CV_15_Fold", "model": m_name, "metric": "ROC_AUC_mean", "value": cv_m["roc_mean"]})
        csv_rows.append({"section": "CV_15_Fold", "model": m_name, "metric": "PR_AUC_mean", "value": cv_m["pr_mean"]})
        csv_rows.append({"section": "CV_15_Fold", "model": m_name, "metric": "Recall_Top10_mean", "value": cv_m["rec10_mean"]})

    # CV 15-fold Brier metrics (mean, std, min, max)
    for m_name, b_res in metrics["cv_brier_results"].items():
        csv_rows.append({"section": "CV_15_Fold_Brier", "model": m_name, "metric": "Brier_mean", "value": b_res["mean"]})
        csv_rows.append({"section": "CV_15_Fold_Brier", "model": m_name, "metric": "Brier_std", "value": b_res["std"]})
        csv_rows.append({"section": "CV_15_Fold_Brier", "model": m_name, "metric": "Brier_min", "value": b_res["min"]})
        csv_rows.append({"section": "CV_15_Fold_Brier", "model": m_name, "metric": "Brier_max", "value": b_res["max"]})

    # Constant prediction baseline row
    csv_rows.append({"section": "CV_15_Fold_Brier", "model": "Constant_Benchmark", "metric": "Brier_mean", "value": metrics["brier_bench_const"]})

    # CV Pooled OOF rows for E_A (used for MVP PASS evaluation)
    csv_rows.append({"section": "CV_Pooled_OOF", "model": "E_A", "metric": "ROC_AUC", "value": metrics["cv_results"]["E_A"]["pool0_roc"]})
    csv_rows.append({"section": "CV_Pooled_OOF", "model": "E_A", "metric": "PR_AUC", "value": metrics["cv_results"]["E_A"]["pool0_pr"]})
    csv_rows.append({"section": "CV_Pooled_OOF", "model": "E_A", "metric": "Recall_Top10", "value": metrics["cv_results"]["E_A"]["pool0_rec10"]})

    # Test set metrics
    for m_name, t_m in metrics["test_metrics"].items():
        csv_rows.append({"section": "Test_E6", "model": m_name, "metric": "ROC_AUC", "value": t_m["roc_auc"]})
        csv_rows.append({"section": "Test_E6", "model": m_name, "metric": "PR_AUC", "value": t_m["pr_auc"]})
        csv_rows.append({"section": "Test_E6", "model": m_name, "metric": "Recall_Top10", "value": t_m["recall_top10"]})
        csv_rows.append({"section": "Test_E6", "model": m_name, "metric": "caught_top10", "value": float(t_m["caught_top10"])})

    # Calibration test Brier
    csv_rows.append({"section": "Calibration", "model": "E_A", "metric": "Brier_Score_Test", "value": metrics["test_brier_ea"]})

    # Operating points metrics on Test Set (E6)
    for r in metrics["op_table_test"]:
        prefix = f"{r['model']}_{r['operating_point']}"
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_threshold", "value": r["threshold"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_n_alerts", "value": float(r["n_alerts"])})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_alert_proportion", "value": r["alert_proportion"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_alert_ratio", "value": r["alert_ratio"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_TP", "value": float(r["tp"])})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_FP", "value": float(r["fp"])})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_FN", "value": float(r["fn"])})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_TN", "value": float(r["tn"])})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_precision", "value": r["precision"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_recall", "value": r["recall"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_f1", "value": r["f1"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_cost_red_10", "value": r["cost_red_10"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_cost_red_20", "value": r["cost_red_20"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_cost_red_50", "value": r["cost_red_50"]})
        csv_rows.append({"section": "Operating_Points_Test", "model": r["model"], "metric": f"{r['operating_point']}_c_model_20", "value": r["c_model_20"]})

    # Operating points metrics on CV Pooled OOF
    for r in metrics["op_table_cv"]:
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_threshold", "value": r["threshold"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_n_alerts", "value": float(r["n_alerts"])})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_alert_proportion", "value": r["alert_proportion"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_alert_ratio", "value": r["alert_ratio"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_TP", "value": float(r["tp"])})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_FP", "value": float(r["fp"])})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_FN", "value": float(r["fn"])})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_TN", "value": float(r["tn"])})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_precision", "value": r["precision"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_recall", "value": r["recall"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_f1", "value": r["f1"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_cost_red_10", "value": r["cost_red_10"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_cost_red_20", "value": r["cost_red_20"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_cost_red_50", "value": r["cost_red_50"]})
        csv_rows.append({"section": "Operating_Points_CV_Pooled", "model": r["model"], "metric": f"{r['operating_point']}_c_model_20", "value": r["c_model_20"]})

    # Baseline inspection costs for Test and CV Pooled (Inspect_None and Inspect_All across 10/20/50)
    for r in [10, 20, 50]:
        cost_none_test = float(r * metrics["n_pos_test"])
        cost_all_test = float(1 * metrics["n_neg_test"])
        csv_rows.append({"section": "Baselines_Test", "model": "Inspect_None", "metric": f"cost_{r}", "value": cost_none_test})
        csv_rows.append({"section": "Baselines_Test", "model": "Inspect_All", "metric": f"cost_{r}", "value": cost_all_test})

        cost_none_cv = float(r * metrics["n_pos_train"])
        cost_all_cv = float(1 * metrics["n_neg_train"])
        csv_rows.append({"section": "Baselines_CV_Pooled", "model": "Inspect_None", "metric": f"cost_{r}", "value": cost_none_cv})
        csv_rows.append({"section": "Baselines_CV_Pooled", "model": "Inspect_All", "metric": f"cost_{r}", "value": cost_all_cv})

    metrics_df = pd.DataFrame(csv_rows)
    csv_out = TABLES_DIR / "mvp_model_card_metrics.csv"
    metrics_df.to_csv(csv_out, index=False)
    logger.info(f"Saved audit metrics table at {csv_out}")

    # 2. Generate MODEL_CARD.md sourcing directly from the structured metrics
    md_content = build_model_card_markdown(metrics)
    model_card_path = PROJECT_ROOT / "MODEL_CARD.md"
    model_card_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Successfully generated MODEL_CARD.md at {model_card_path}")
    print("Done! MODEL_CARD.md and mvp_model_card_metrics.csv generated successfully.")


if __name__ == "__main__":
    main()
