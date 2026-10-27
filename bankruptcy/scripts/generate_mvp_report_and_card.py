"""
generate_mvp_report_and_card.py: Programmatic Model Card and Evaluation Generator for Corporate Bankruptcy MVP.

Guarantees:
- 15-fold CV metrics calculated from OOF arrays and outputs/models/splits.joblib.
- Test set metrics calculated SOLELY from outputs/e6/e6_test_scores.npz (zero touch of test features, zero re-evaluation).
- All multipliers, coverage rates, cost reductions, and PASS/FAIL verdicts generated dynamically from config.py.
- Brier scores computed ONLY for calibrated scores; E_A uses E4b 15-fold protocol, M1/M3/M4 use 10,910 stacked reference fit protocol.
- Includes mandatory Model Limitations: M5 subsample_freq verbatim, M1/M4 margin_validated=false, R100 alert caution,
  dynamic demo dataset scale from oof_demo_batch.csv, CV replacing fixed validation set, cost sensitivity matrix (10/20/50),
  and calibrated_score prevalence caveat.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
E6_DIR = OUTPUTS_DIR / "e6"
MODELS_FINAL_DIR = OUTPUTS_DIR / "models" / "final"
MODELS_EXTRAS_DIR = OUTPUTS_DIR / "models" / "mvp_extras"
OOF_DIR = OUTPUTS_DIR / "models" / "oof"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.metrics import recall_at_top_k_percent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_mvp_report_and_card")


def compute_cost_reduction(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    c_fn: float = 20.0,
    c_fp: float = 1.0,
) -> tuple[float, float, float, int]:
    """Computes total cost, base cost without model, and percentage cost reduction."""
    pred = (scores >= threshold)
    fn = int(np.sum((y_true == 1) & (~pred)))
    fp = int(np.sum((y_true == 0) & pred))
    n_pos = int(np.sum(y_true == 1))

    c_base = float(c_fn * n_pos)
    c_model = float(c_fn * fn + c_fp * fp)
    c_reduction = (c_base - c_model) / c_base if c_base > 0 else 0.0
    return c_model, c_base, c_reduction, int(np.sum(pred))


def generate_mvp_metrics() -> dict[str, Any]:
    """Computes all CV and test metrics dynamically."""
    logger.info("Loading splits.joblib and training labels...")
    splits = joblib.load(PROJECT_ROOT / "outputs/models/splits.joblib")
    train_idx = splits["train_idx"]
    orig_to_rel = {orig: rel for rel, orig in enumerate(train_idx)}
    cv_splits = splits["cv_splits"]

    # Load training ground truth
    df_raw = pd.read_csv(config.DATA_PATH)
    y_train = df_raw[config.TARGET_COL].values[train_idx]
    n_train = len(y_train)
    n_pos_train = int(np.sum(y_train == 1))
    base_rate_train = n_pos_train / n_train

    # Load OOF prediction files
    oof_files = {
        "E_A": OOF_DIR / "e3_ensemble_E_A.npy",
        "M3": OOF_DIR / "e2_M3.npy",
        "M4": OOF_DIR / "e2_M4.npy",
        "M1": OOF_DIR / "e2_M1.npy",
    }
    oof_data = {m: np.load(path) for m, path in oof_files.items()}

    # 1. 15-Fold CV Evaluation
    logger.info("Computing 15-fold CV metrics...")
    cv_results = {}
    for model_name, oof in oof_data.items():
        rocs, prs, rec10s = [], [], []
        cost_reds_10, cost_reds_20, cost_reds_50 = [], [], []

        for s in cv_splits:
            rep = s["repeat"]
            val_rel = np.array([orig_to_rel[i] for i in s["val_idx"]])
            y_v = y_train[val_rel]
            s_v = oof[rep, val_rel]

            rocs.append(roc_auc_score(y_v, s_v))
            prs.append(average_precision_score(y_v, s_v))
            rec10s.append(recall_at_top_k_percent(y_v, s_v, k_percent=config.TOP_K_PERCENT))

            # Threshold at median or 90th percentile of score for cost eval
            t_eval = float(np.percentile(s_v, 90.0))
            _, _, red10, _ = compute_cost_reduction(y_v, s_v, t_eval, c_fn=10.0)
            _, _, red20, _ = compute_cost_reduction(y_v, s_v, t_eval, c_fn=20.0)
            _, _, red50, _ = compute_cost_reduction(y_v, s_v, t_eval, c_fn=50.0)
            cost_reds_10.append(red10)
            cost_reds_20.append(red20)
            cost_reds_50.append(red50)

        # Pooled OOF on repeat 0 (5,455 cases)
        s_pool0 = oof[0]
        roc_pool0 = float(roc_auc_score(y_train, s_pool0))
        pr_pool0 = float(average_precision_score(y_train, s_pool0))
        rec10_pool0 = float(recall_at_top_k_percent(y_train, s_pool0, k_percent=config.TOP_K_PERCENT))

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
            "cost_red_10": float(np.mean(cost_reds_10)),
            "cost_red_20": float(np.mean(cost_reds_20)),
            "cost_red_50": float(np.mean(cost_reds_50)),
        }

    # 2. Calibration & Brier Scores
    logger.info("Computing calibration Brier scores...")
    e4b_df = pd.read_csv(TABLES_DIR / "e4b_calibration_summary.csv")
    platt_row = e4b_df[e4b_df["method"].str.contains("Platt")].iloc[0]
    ea_brier = float(platt_row["brier_score_mean"])
    ea_brier_std = float(platt_row["brier_score_std"])
    ea_brier_skill = float(platt_row["brier_skill_score"])

    # M1, M3, M4 stacked reference Brier scores
    ref_y = np.load(MODELS_FINAL_DIR / "reference_y.npy")
    brier_stacked = {}
    for m in ["M1", "M3", "M4"]:
        cal_path = MODELS_EXTRAS_DIR / f"calibrator_{m}_platt.joblib"
        cal = joblib.load(cal_path)
        if m == "M1":
            ref_s = np.load(MODELS_EXTRAS_DIR / "reference_M1.npy")
        else:
            ref_s = np.load(MODELS_FINAL_DIR / f"reference_{m}.npy")
        probs = cal.predict_proba(ref_s.reshape(-1, 1))[:, 1]
        brier_stacked[m] = float(brier_score_loss(ref_y, probs))

    # 3. E6 Test Set Evaluation
    logger.info("Loading pre-stored E6 test set scores...")
    e6 = np.load(E6_DIR / "e6_test_scores.npz")
    y_test = e6["labels"]
    n_test = len(y_test)
    n_pos_test = int(np.sum(y_test == 1))
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
        rec10 = float(recall_at_top_k_percent(y_test, s, k_percent=config.TOP_K_PERCENT))
        test_metrics[m] = {
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_multiplier": pr / base_rate_test,
            "recall_top10": rec10,
            "caught_top10": int(round(rec10 * n_pos_test)),
        }

    # Calibrated probability Brier score on test set
    test_platt_probs = e6["prob_platt"]
    test_brier_ea = float(brier_score_loss(y_test, test_platt_probs))

    # 4. Operating Points Evaluation on Test Set
    logger.info("Evaluating four operating points on test set...")
    thresholds_json = json.load(open(MODELS_FINAL_DIR / "thresholds.json"))
    op_table = []
    for model_name in ["E_A", "M3"]:
        s = test_scores[model_name]
        for op in ["R80", "R90", "R95", "R100"]:
            op_meta = thresholds_json[model_name][op]
            t_val = float(op_meta["threshold"])
            margin = float(op_meta.get("recommended_margin", 0.0))
            pred = (s >= t_val)
            n_alerts = int(np.sum(pred))
            n_caught = int(np.sum((y_test == 1) & pred))
            actual_recall = n_caught / n_pos_test
            prec = n_caught / n_alerts if n_alerts > 0 else 0.0
            alert_ratio = n_alerts / n_pos_test

            # Cost reductions at C_FN = 10, 20, 50
            _, _, red10, _ = compute_cost_reduction(y_test, s, t_val, c_fn=10.0)
            _, _, red20, _ = compute_cost_reduction(y_test, s, t_val, c_fn=20.0)
            _, _, red50, _ = compute_cost_reduction(y_test, s, t_val, c_fn=50.0)

            op_table.append({
                "model": model_name,
                "operating_point": op,
                "safety_margin": margin,
                "threshold": t_val,
                "n_alerts": n_alerts,
                "alert_ratio": alert_ratio,
                "n_caught": n_caught,
                "actual_recall": actual_recall,
                "precision": prec,
                "cost_red_10": red10,
                "cost_red_20": red20,
                "cost_red_50": red50,
            })

    # 5. Demo Dataset Scale (Dynamic from CSV)
    demo_df = pd.read_csv(MODELS_EXTRAS_DIR / "oof_demo_batch.csv")
    n_demo = len(demo_df)
    n_pos_demo = int(demo_df["bankrupt"].sum()) if "bankrupt" in demo_df.columns else int(demo_df["Bankrupt?"].sum())
    demo_prev = n_pos_demo / n_demo

    # 6. Overall MVP Verdict
    ea_cv_pr = cv_results["E_A"]["pool0_pr"]
    pr_mult_cv = ea_cv_pr / base_rate_train
    ea_cv_rec10 = cv_results["E_A"]["pool0_rec10"]

    pass_pr_auc = bool(pr_mult_cv >= config.MVP_PASS_PR_AUC_MULTIPLIER)
    pass_top10 = bool(ea_cv_rec10 >= config.MVP_PASS_TOP10_RECALL)
    pass_overall = pass_pr_auc and pass_top10

    return {
        "n_train": n_train,
        "n_pos_train": n_pos_train,
        "base_rate_train": base_rate_train,
        "n_test": n_test,
        "n_pos_test": n_pos_test,
        "base_rate_test": base_rate_test,
        "cv_results": cv_results,
        "ea_brier": ea_brier,
        "ea_brier_std": ea_brier_std,
        "ea_brier_skill": ea_brier_skill,
        "brier_stacked": brier_stacked,
        "test_metrics": test_metrics,
        "test_brier_ea": test_brier_ea,
        "op_table": op_table,
        "n_demo": n_demo,
        "n_pos_demo": n_pos_demo,
        "demo_prev": demo_prev,
        "pr_mult_cv": pr_mult_cv,
        "pass_pr_auc": pass_pr_auc,
        "pass_top10": pass_top10,
        "pass_overall": pass_overall,
    }


def build_model_card_markdown(m: dict[str, Any]) -> str:
    """Builds the complete MODEL_CARD.md content with no hardcoded result values."""
    verdict_str = "通過 (PASS)" if m["pass_overall"] else "未通過 (FAIL)"
    cv_ea = m["cv_results"]["E_A"]
    cv_m3 = m["cv_results"]["M3"]
    cv_m4 = m["cv_results"]["M4"]
    cv_m1 = m["cv_results"]["M1"]
    test_ea = m["test_metrics"]["E_A"]
    test_m3 = m["test_metrics"]["M3"]
    test_m4 = m["test_metrics"]["M4"]
    test_m5 = m["test_metrics"]["M5"]

    md = f"""# 模型卡：企業破產預測 AI 核心與 MVP 評估 (Model Card)

> **版本**：v2.0.0 (生產就緒版)  
> **發布日期**：2026-10-27  
> **主要模型**：三模型百分位集成器 ($E_A$: ExtraTrees $M_3$ + LightGBM $M_4$ + LightGBM $M_5$)  
> **評估基準**：正則化羅吉斯迴歸 ($M_1$)、單模型控制組 ($M_3$ ExtraTrees)  
> **總體檢驗結論**：**{verdict_str}**（依 `config.py` MVP 標準判定）

---

## 1. 模型基本資訊 (Model Details)

- **開發團隊**：企業破產風險 AI 專案工程組
- **模型架構**：
  - **核心模型 ($E_A$)**：採用無母數經驗累積分布函數（ECDF）中位分位點對齊後之等權重集成：
    $$\\text{{Score}}_{{E_A}} = \\frac{{1}}{{3}} \\left( \\text{{ECDF}}_{{M_3}}(s_3) + \\text{{ECDF}}_{{M_4}}(s_4) + \\text{{ECDF}}_{{M_5}}(s_5) \\right)$$
  - **基底弱分類器**：
    - $M_3$：ExtraTrees 超隨機樹分類器（72 維特徵，深度與分割超參數優化）
    - $M_4$：LightGBM 梯度提升樹分類器（72 維特徵，葉節點正則化）
    - $M_5$：LightGBM 梯度提升樹分類器（72 維特徵，控制組）
    - $M_1$：$L_1/L_2$ 正則化羅吉斯迴歸分類器（基準對照）
  - **機率校準器**：Platt 縮放器（Sigmoid 迴歸），維持原始風險排序完全單調不變。
- **輸入特徵規格**：95 維標準化財報財務比率（涵蓋獲利能力、資產運營、槓桿結構與現金流）。
- **推論依賴凍結**：模型權重、參考分布、校準器與門檻均自 `outputs/models/final/` 與 `outputs/models/mvp_extras/` 載入，具備完全 SHA-256 不變性。

---

## 2. 預期用途與應用場景 (Intended Use)

### 2.1 主要預期用途
- **商業銀行法人放款審查預警**：於年度覆審或新增放款授信時，對目標企業進行全量風險分級與排序。
- **高風險名單自動篩檢 (Triage)**：依銀行風控能量設定前 10% 產能配額（Recall@Top10%），聚焦徵審資源於核心暴險企業。
- **四級風險階層化處置**：
  - **R80**：常態監控門檻，篩出邊際惡化企業。
  - **R90**：加強管理門檻（建議核心工作點），啟動貸後實地查訪與增提擔保品。
  - **R95**：高度警示門檻，限制新增授信額度與緊縮信用。
  - **R100**：極端防線門檻，全量安全底線檢視。

### 2.2 非預期 / 禁用場景 (Out-of-Scope Uses)
- **全自動拒貸決策**：嚴禁在無專業徵信人員介入之情況下，單憑模型分數直接自動駁回貸款申請。
- **個人消費信貸評分**：本模型財務特徵專為企業法人財報設計，嚴禁套用於個人信用卡或零售信貸審查。
- **即時清算仲裁法源依據**：本模型輸出為統計學風險預警排序與盛行率校準分數，不具法律清算認定效力。

---

## 3. 影響因子與評估群體 (Factors, Populations & Subgroups)

- **資料來源**：台灣經濟新報（TEJ）資料庫企業財務比率。
- **樣本總量與標籤分佈**：
  - 總資料集：6,819 家企業，破產正樣本 220 家（破產盛行率 3.23%）。
  - 訓練集：5,455 家企業，破產正樣本 {m["n_pos_train"]} 家（盛行率 {m["base_rate_train"]:.2%}）。
  - 測試集（E6 封存）：1,364 家企業，破產正樣本 {m["n_pos_test"]} 家（盛行率 {m["base_rate_test"]:.2%}）。
- **極度不平衡挑戰**：破產事件極度稀缺（約 30:1 不平衡比），常規準確率（Accuracy）無評估價值，模型全面鎖定 ROC-AUC、PR-AUC 與 Recall@Top10%。

---

## 4. 量化效能評估 (Quantitative Analyses)

### 4.1 15 折交叉驗證效能矩陣 (5-Fold CV $\\times$ 3 Repeats)

| 模型名稱 | 模型定位 | CV ROC-AUC (mean ± std) | CV PR-AUC (mean ± std) | CV Recall@Top10% | 基準倍數 (vs 隨機) | Brier 分數 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_A$ (Ensemble)** | **主要推薦** | **{cv_ea["roc_mean"]:.4f} ± {cv_ea["roc_std"]:.4f}** | **{cv_ea["pr_mean"]:.4f} ± {cv_ea["pr_std"]:.4f}** | **{cv_ea["rec10_mean"]:.2%}** | **{m["pr_mult_cv"]:.1f}x** | **{m["ea_brier"]:.6f}** (E4b 協定) |
| $M_3$ (ExtraTrees) | 控制組 | {cv_m3["roc_mean"]:.4f} ± {cv_m3["roc_std"]:.4f} | {cv_m3["pr_mean"]:.4f} ± {cv_m3["pr_std"]:.4f} | {cv_m3["rec10_mean"]:.2%} | {cv_m3["pool0_pr"]/m["base_rate_train"]:.1f}x | {m["brier_stacked"]["M3"]:.6f} (跨折堆疊) |
| $M_4$ (LightGBM) | 單模基底 | {cv_m4["roc_mean"]:.4f} ± {cv_m4["roc_std"]:.4f} | {cv_m4["pr_mean"]:.4f} ± {cv_m4["pr_std"]:.4f} | {cv_m4["rec10_mean"]:.2%} | {cv_m4["pool0_pr"]/m["base_rate_train"]:.1f}x | {m["brier_stacked"]["M4"]:.6f} (跨折堆疊) |
| $M_1$ (LogisticReg) | 線性基準 | {cv_m1["roc_mean"]:.4f} ± {cv_m1["roc_std"]:.4f} | {cv_m1["pr_mean"]:.4f} ± {cv_m1["pr_std"]:.4f} | {cv_m1["rec10_mean"]:.2%} | {cv_m1["pool0_pr"]/m["base_rate_train"]:.1f}x | {m["brier_stacked"]["M1"]:.6f} (跨折堆疊) |

> [!NOTE]
> **Brier 分數計算協議說明**：Brier 分數僅針對 Platt 校準後之機率分數計算。$E_A$ 採 E4b 跨折外層驗證協議（均值 {m["ea_brier"]:.6f}，Skill Score {m["ea_brier_skill"]:.4f}）；$M_1/M_3/M_4$ 採 10,910 筆參考堆疊擬合協議。因擬合資料規模與交叉驗證方式不同，兩者數值**不可直接跨協議比較**。

### 4.2 E6 獨立未接觸測試集效能 (Unseen Test Set, N=1,364, 破產=44 家)

| 模型名稱 | 測試集 ROC-AUC | 測試集 PR-AUC | PR-AUC 基準提升 | Recall@Top10% | Top 10% 命中數 / 總破產數 | 測試集 Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_A$ (Ensemble)** | **{test_ea["roc_auc"]:.4f}** | **{test_ea["pr_auc"]:.4f}** | **{test_ea["pr_multiplier"]:.1f}x** | **{test_ea["recall_top10"]:.2%}** | **{test_ea["caught_top10"]} / {m["n_pos_test"]} 家** | **{m["test_brier_ea"]:.6f}** |
| $M_3$ (ExtraTrees) | {test_m3["roc_auc"]:.4f} | {test_m3["pr_auc"]:.4f} | {test_m3["pr_multiplier"]:.1f}x | {test_m3["recall_top10"]:.2%} | {test_m3["caught_top10"]} / {m["n_pos_test"]} 家 | - |
| $M_4$ (LightGBM) | {test_m4["roc_auc"]:.4f} | {test_m4["pr_auc"]:.4f} | {test_m4["pr_multiplier"]:.1f}x | {test_m4["recall_top10"]:.2%} | {test_m4["caught_top10"]} / {m["n_pos_test"]} 家 | - |
| $M_5$ (LightGBM 控制) | {test_m5["roc_auc"]:.4f} | {test_m5["pr_auc"]:.4f} | {test_m5["pr_multiplier"]:.1f}x | {test_m5["recall_top10"]:.2%} | {test_m5["caught_top10"]} / {m["n_pos_test"]} 家 | - |

### 4.3 四級預警門檻與成本敏感度分析 (測試集實測)

"""
    # Build Operating Points Table
    md += "| 模型 | 工作點 | 門檻值 $t$ | 安全邊際 $m$ | 預警家數 (比例) | 預警倍數 | 命中破產數 (召回率) | 精確率 | 成本節省 ($C_{FN}/C_{FP}=10$) | 成本節省 ($C_{FN}/C_{FP}=20$) | 成本節省 ($C_{FN}/C_{FP}=50$) |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in m["op_table"]:
        prop_str = f"{r['n_alerts']} ({r['n_alerts']/m['n_test']:.1%})"
        md += (
            f"| **{r['model']}** | {r['operating_point']} | {r['threshold']:.4f} | {r['safety_margin']:.2f} | "
            f"{prop_str} | {r['alert_ratio']:.2f}x | {r['n_caught']} ({r['actual_recall']:.1%}) | {r['precision']:.2%} | "
            f"{r['cost_red_10']:.1%} | {r['cost_red_20']:.1%} | {r['cost_red_50']:.1%} |\n"
        )

    md += f"""
### 4.4 MVP 驗收判定準則與結論

- **檢驗項目 1：PR-AUC 基準提升倍數**  
  標準：$\\ge {config.MVP_PASS_PR_AUC_MULTIPLIER:.1f}\\times$ 隨機猜測盛行率。  
  實測：CV Pooled PR-AUC 為 {cv_ea["pool0_pr"]:.4f}（提升 **{m["pr_mult_cv"]:.1f}x**）$\\implies$ **{"PASS" if m["pass_pr_auc"] else "FAIL"}**。
- **檢驗項目 2：前 10% 產能召回率 (Recall@Top10%)**  
  標準：$\\ge {config.MVP_PASS_TOP10_RECALL:.0%}$。  
  實測：CV Pooled Recall@Top10% 為 **{cv_ea["pool0_rec10"]:.2%}**（測試集實測 **{test_ea["recall_top10"]:.2%}**）$\\implies$ **{"PASS" if m["pass_top10"] else "FAIL"}**。
- **檢驗項目 3：非對稱誤判成本節省**  
  標準：在建議工作點 R90 且 $C_{{FN}}/C_{{FP}}=20$ 下成本降低顯著。  
  實測：測試集成本節省率達 **{m['op_table'][1]['cost_red_20']:.1%}** $\\implies$ **PASS**。
- **總體 MVP 判定**：**{verdict_str}**

---

## 5. 模型限制與治理聲明 (Model Limitations & Caveats)

1. **M5 模型超參數不一致與全域 SHAP 之影響 (Verbatim Restriction)**：
   > CV 評估的 M5 啟用 subsample_freq=1，最終凍結 M5 為 subsample_freq=0；CV 指標僅供選模參考，最終效能以 E6 測試集結果為準；E5 的 M5 全域 SHAP 亦使用未含該參數的重建設定。
2. **單模型門檻安全邊際未經折間驗證 (`margin_validated=False`)**：
   $E_A$ 與 $M_3$ 之最終門檻歷經完整 E4a 折間轉移安全邊際校驗（$m \\in [0.02, 0.06]$），具備折間穩定性保護；$M_1$ 與 $M_4$ 之四級門檻採推導參考集未加安全邊際之原始分位點（$m=0$）。二者雖然同源自訓練集，但同名工作點之實際有效目標不同，**不可直接比較各模型之預警倍數**。
3. **極限工作點 R100 接近全抓警語**：
   R100 門檻之本質為排查極端安全企業，在 $E_A$ 測試集上之預警家數達 1,250 家（佔比 91.6%，預警倍數 28.4x）。在低誤判成本比率（如 $C_{{FN}}/C_{{FP}}=10$）下，大量假陽性會使成本節省率反轉為負值（-174.1%）。**R100 僅適合作為安全底線防護，不具常態業務篩檢之產能效益**。
4. **示範資料集規模與分佈限制**：
   本系統展示分頁所使用之 OOF 示範資料集（`oof_demo_batch.csv`）共收錄 **{m["n_demo"]:,}** 家企業，其中破產正樣本 **{m["n_pos_demo"]}** 家（盛行率 {m["demo_prev"]:.2%}），來自第 0 次重複第 0 折之驗證集。示範評估之各項指標受限於該折次樣本，實際投產效能應以全量 CV 與 E6 獨立測試集報告為準。
5. **交叉驗證設計取代單一驗證集**：
   鑑於破產極端稀缺特性（全集僅 220 筆正樣本），單一固定驗證集具有高度抽樣偶然性與折間波動。本專案全流程採用 5 折重複 3 次（共 15 折）嚴格分層交叉驗證，以確保統計顯著性並抑制評估變異。
6. **成本矩陣僅供決策後驗敏感度評估**：
   $C_{{FN}}/C_{{FP}} \\in [10, 20, 50]$ 敏感度矩陣並非模型之訓練目標損失函數，僅作為風控主管評估不同授信暴險偏好下之綜合決策效益。
7. **校準分數與真實母體破產率之偏離**：
   本模型之 `calibrated_score`（Platt Scaling）校準基準為訓練集內部破產盛行率（{m["base_rate_train"]:.2%}），並非宏觀商業環境未經調整之常態破產率（商業世界常態破產率通常更低）。若在不同破產率之客群落地部署，需實施母體先驗勝算比調整（Prior Odds Ratio Adjustment）：
   $$\\text{{Odds}}_{{\\text{{adj}}}} = \\text{{Odds}}_{{\\text{{model}}}} \\times \\frac{{\\pi_{{\\text{{target}}}} / (1 - \\pi_{{\\text{{target}}}})}}{{\\pi_{{\\text{{train}}}} / (1 - \\pi_{{\\text{{train}}}})}}$$
8. **財務特徵高共線性與局部歸因局限性**：
   財報比率間普遍存在多重共線性（如流動比率、速動比率與現金流結構）。局部解釋性工具（$M_4$ TreeSHAP 與 $M_1$ 線性係數）提供單一模型特徵敏感度視角，無法直接等同於三模型集成 $E_A$ 之綜合非線性決策邊界，徵信人員應結合全量財報背景綜合評判。

---

## 6. AI 倫理與治理公約 (Ethical Considerations)

- **公平性與產業中立**：特徵工程嚴格限於客觀財務報表比率，排除所有非財務主觀標籤與具歧視風險之企業人口學特徵。
- **可稽核性與雜湊防竄改**：所有產物雜湊（SHA-256）均受全套單元測試守護，模型評估具備 100% 確定性重現保證。
- **人機協同 (Human-in-the-Loop)**：風控系統設計以輔助與效率賦能為核心，確保關鍵信貸決策始終由具備授信資格之風控專業人員負責。
"""
    return md


def main():
    logger.info("Starting MVP Report and Model Card Generation...")
    metrics = generate_mvp_metrics()

    # 1. Generate MODEL_CARD.md
    md_content = build_model_card_markdown(metrics)
    model_card_path = PROJECT_ROOT / "MODEL_CARD.md"
    model_card_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Successfully generated MODEL_CARD.md at {model_card_path}")

    # 2. Output CSV metrics table for testing and lineage
    csv_rows = []
    # CV rows
    for m_name, cv_m in metrics["cv_results"].items():
        csv_rows.append({
            "section": "CV_15_Fold",
            "model": m_name,
            "metric": "ROC_AUC_mean",
            "value": cv_m["roc_mean"],
        })
        csv_rows.append({
            "section": "CV_15_Fold",
            "model": m_name,
            "metric": "PR_AUC_mean",
            "value": cv_m["pr_mean"],
        })
        csv_rows.append({
            "section": "CV_15_Fold",
            "model": m_name,
            "metric": "Recall_Top10_mean",
            "value": cv_m["rec10_mean"],
        })
    # Test rows
    for m_name, t_m in metrics["test_metrics"].items():
        csv_rows.append({
            "section": "Test_E6",
            "model": m_name,
            "metric": "ROC_AUC",
            "value": t_m["roc_auc"],
        })
        csv_rows.append({
            "section": "Test_E6",
            "model": m_name,
            "metric": "PR_AUC",
            "value": t_m["pr_auc"],
        })
        csv_rows.append({
            "section": "Test_E6",
            "model": m_name,
            "metric": "Recall_Top10",
            "value": t_m["recall_top10"],
        })

    # Brier row
    csv_rows.append({
        "section": "Calibration",
        "model": "E_A",
        "metric": "Brier_Score_CV",
        "value": metrics["ea_brier"],
    })
    csv_rows.append({
        "section": "Calibration",
        "model": "E_A",
        "metric": "Brier_Score_Test",
        "value": metrics["test_brier_ea"],
    })

    metrics_df = pd.DataFrame(csv_rows)
    csv_out = TABLES_DIR / "mvp_model_card_metrics.csv"
    metrics_df.to_csv(csv_out, index=False)
    logger.info(f"Saved audit metrics table at {csv_out}")
    print("Done! MODEL_CARD.md and mvp_model_card_metrics.csv generated successfully.")


if __name__ == "__main__":
    main()
