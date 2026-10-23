# 破產預測專案成果摘要 (RESULTS_SUMMARY.md)

本文件為專案最終技術報告之唯一數據入口與核心成果彙整。所有數據均直接引自 `outputs/` 目錄下之已固化與評估表格，並附上精確來源路徑。

---

## 目錄 (Table of Contents)
1. [資料與實驗設定摘要](#1-資料與實驗設定摘要)
2. [各階段一句話結論與關鍵數字 (E0～E7)](#2-各階段一句話結論與關鍵數字-e0e7)
3. [主要結果數據表](#3-主要結果數據表)
   - [3.1 工作點泛化評估 (CV vs. 測試集)](#31-工作點泛化評估-cv-vs-測試集)
   - [3.2 市場盛行率敏感度重點表](#32-市場盛行率敏感度重點表)
   - [3.3 隨機種子穩定性對照表](#33-隨機種子穩定性對照表)
4. [專案已知限制清單](#4-專案已知限制清單)
5. [圖表檔案清單與說明](#5-圖表檔案清單與說明)

---

## 1. 資料與實驗設定摘要

- **資料規模與標籤分布**：
  - 全體資料：6,819 筆企業財報樣本，95 個財務特徵，破產標籤（`Bankrupt?`）共 220 筆（盛行率 3.2258%）。
  - 資料切分：80/20 固定分層切分（Stratified Split，種子 42）：
    - **訓練集 (Training Set)**：5,455 筆，包含 176 家破產企業（盛行率 3.2264%）。
    - **最終測試集 (Holdout Test Set)**：1,364 筆，包含 44 家破產企業（盛行率 3.2258%），嚴格一次性解封鎖定。
  - 交叉驗證架構：外層 5 折 $\times$ 3 次重複（15 折重複分層交叉驗證，Repeated Stratified K-Fold）。切分索引固化於 `outputs/models/splits.joblib`。
- **前處理管線 (Preprocessing Pipeline)**：
  - 特徵選擇器 (`FeatureSelector`) $\to$ 常數與完全重複特徵剔除 (`DropConstantAndDuplicate`，剔除 1 常數 + 2 重複特徵) $\to$ Winsorizer 關閉 $\to$ LargeValueIndicator 關閉 $\to$ 分位數轉換 (`QuantileScaler`, output_distribution='uniform') $\to$ 相關性過濾 (`CorrelationFilter`, $|r| > 0.95$ 貪婪剔除，保留 72 個特徵)。
- **最終模型組成與位置**：
  - **主要模型 $E_A$ (Primary Ensemble)**：M3 (ExtraTrees) + M4 (XGBoost) + M5 (LightGBM) 的經驗累積分布函數 (ECDF) 百分位等權平均。
  - **控制組單一模型 M3 (Control Single Model)**：ExtraTreesClassifier。
  - **最佳超參數檔路徑**：
    - M3: `outputs/models/best_params/M3.json` (`n_estimators: 600, min_samples_leaf: 2, max_features: 0.3, S0`)
    - M4: `outputs/models/best_params/M4.json` (`n_estimators: 250, learning_rate: 0.0468, max_depth: 2, S0`)
    - M5: `outputs/models/best_params/M5.json` (`n_estimators: 300, learning_rate: 0.0137, num_leaves: 43, S3-SMOTE`)
  - **固化模型元件目錄**：`outputs/models/final/`。

---

## 2. 各階段一句話結論與關鍵數字 (E0～E7)

- **E0（基準線建立）**：
  隨機猜測 PR-AUC 僅 0.0323、AR@0.95 為 31.0x；單一最強財務特徵（`Persistent EPS in the Last Four Seasons`）AR@0.95 達到 16.1x（PR-AUC 0.2842）；基準邏輯斯迴歸（S0）AR@0.95 達到 10.3x（PR-AUC 0.3546）。證實財務指標具備強大非隨機預測力。
  *數據來源：`outputs/tables/e0_baselines_summary.csv`*

- **E1（前處理與不平衡策略消融）**：
  42 組前處理與重抽樣策略比較顯示，SMOTE 等重抽樣策略相對無重抽樣（S0）的 95% Nadeau-Bengio 配對信賴區間全數跨越 0（無檢出統計差異）；關閉 Winsorizer 略優並保留 72 個過濾特徵。確立前處理設定：Winsorizer 關、CorrFilter 開、QuantileScaler 開。
  *數據來源：`outputs/tables/e1a_paired_vs_s0.csv`、`outputs/tables/e1b_ablation.csv`*

- **E2（巢狀超參數調優）**：
  M1～M5 經 15 折巢狀調參後，相對預設參數均無檢出統計差異（Nadeau-Bengio 區間均跨越 0，$p_{\text{Holm}} = 1.0$），調參地景極度平坦；M3 (ExtraTrees) 折間標準差最小（1.35x）。確立最佳參數固化。
  *數據來源：`outputs/tables/e2a_nested_summary.csv`、`outputs/tables/e2b_final_params_summary.csv`*

- **E3（模型比較與集成）**：
  5 個單一機器學習模型在統計上無法區分（20 組配對經 Holm 校正後 $p_{\text{Holm}} = 0.6195 \gg 0.05$）；ECDF 排名等權集成 $E_A$ 相比單一最佳模型展現穩定性小幅改善（折均 AR@0.95 為 8.04x、PR-AUC 為 0.4439、3 點預警損失為 1.6366）。選定 $E_A$ 為主要模型、M3 為控制組。
  *數據來源：`outputs/tables/e3a_pairwise_models.csv`、`outputs/tables/e3b_ensemble_summary.csv`*

- **E4（工作點固化與機率校準）**：
  確定在未校準連續排名上固化門檻以避免 ties，引入安全邊際 $m$（$E_A$: R80 $m=0.02$, R90 $m=0.06$, R95 $m=0.04$, R100 $m=0.00$）；Platt Scaling 嚴格保序維持 AR@0.95（8.06x，BSS 0.1839），Isotonic 出現階梯 ties 破壞尾端排名（AR 惡化至 10.02x）。固化門檻與校準器存於 `outputs/models/final/`。
  *數據來源：`outputs/tables/operating_points_for_report.csv`、`outputs/tables/e4b_calibration_summary.csv`*

- **E5（特徵重要度、可解釋性與精簡特徵集）**：
  SHAP 分析確立獲利能力、流動性與負債比率為三大關鍵風險因子（Family 38 負債比率家族 100% 進入前 20 名）；精簡特徵巢狀消融顯示 $k=10$ 時損失顯著增加 $+0.1335$（$p_{\text{Holm}} = 0.0251$），而 $k=20$ 與 $k=30$ 與全特徵無檢出統計差異；產出 12 個核心特徵（`core_features.txt`）與 18 個擴大核心特徵（`core_features_extended.txt`）。
  *數據來源：`outputs/tables/e5b_compact_vs_full_paired.csv`、`outputs/tables/e5b_core_features.csv`*

- **E6（最終測試集正式評估，嚴格一次性解封）**：
  測試集（1,364 筆，44 家破產）一次性解封鎖定（`outputs/logs/E6_LOCK`），所有工作點全數達成目標召回率（$R=0.90$ 捕獲 43/44 家、$R=0.95$ 捕獲 44/44 家）；$E_A - M3$ 配對 Bootstrap 差值 95% CI 跨越 0，統計上無法區分；測試集點估計（PR-AUC 0.5676, AR@0.95 5.25x）高於 15 折 CV 均值，顯示測試集看起來較容易、不可將 100% 召回當作未來保證，專題結論以 CV 為主、測試集為確認。
  *數據來源：`outputs/e6/e6_operating_points.csv`、`outputs/e6/e6_vs_cv.csv`、`outputs/e6/e6_ranking_metrics.csv`、`outputs/e6/e6_paired_bootstrap.csv`*

- **E7（盛行率敏感度與隨機種子穩定性）**：
  E7a 解析推導顯示，在真實世界較低破產盛行率（例如 1.0%）下，$E_A$ $R=0.90$ 精確率為 3.61%（CV）/ 4.24%（測試集），每抓 1 家需審查 27.7 家（CV）/ 23.6 家（測試集）；E7b 跨 5 組隨機種子評估顯示，種子標準差為 15 折平均標準誤（$\sigma_{\text{fold}} / \sqrt{15}$）的 8.3%～26.9%，證實種子造成的變動遠小於折間變動（註明：E7b 使用 E2b 參數，絕對水準偏樂觀，僅供種子穩定性評估）。
  *數據來源：`outputs/tables/e7a_prevalence_sensitivity.csv`、`outputs/tables/e7b_seed_stability.csv`*

---

## 3. 主要結果數據表

### 3.1 工作點泛化評估 (CV vs. 測試集)
*數據來源：`outputs/tables/operating_points_for_report.csv` (15 折 CV) 與 `outputs/e6/e6_operating_points.csv` (測試集 N=1364, Pos=44)*

| 模型 | 工作點 | 固化門檻 $t$ | 目標召回要求 | CV 15 折實際召回 (均值±標準差) [min, max] | 測試集捕獲數 | 測試集實際召回率 (95% CP CI) | CV 15 折預警倍數 (均值±標準差) | 測試集預警倍數 (比例) | 測試集精確率 (95% Wilson CI) | 達標判定 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_A$** | R80 ($m=0.02$) | 0.8845 | $\ge 36$ | 82.54% ± 6.65% [68.57%, 100%] | **37 / 44** | **84.09%** [69.93%, 93.36%] | 3.70x ± 0.47x | **3.32x** (10.70%) | **25.34%** [18.98%, 32.97%] | **True** |
| **$E_A$** | R90 ($m=0.06$) | 0.7400 | $\ge 40$ | 95.45% ± 4.79% [82.86%, 100%] | **43 / 44** | **97.73%** [87.98%, 99.94%] | 8.68x ± 0.76x | **7.66x** (24.71%) | **12.76%** [9.61%, 16.75%] | **True** |
| **$E_A$** | R95 ($m=0.04$) | 0.5983 | $\ge 42$ | 98.68% ± 1.82% [94.29%, 100%] | **44 / 44** | **100.0%** [91.96%, 100%] | 13.34x ± 2.16x | **11.91x** (38.42%) | **8.40%** [6.31%, 11.08%] | **True** |
| **$E_A$** | R100 ($m=0.00$) | 0.1708 | $= 44$ | 99.43% ± 1.17% [97.14%, 100%] | **44 / 44** | **100.0%** [91.96%, 100%] | 23.48x ± 5.96x | **28.41x** (91.64%) | **3.52%** [2.63%, 4.69%] | **True** |
| **M3** | R80 ($m=0.04$) | 0.8743 | $\ge 36$ | 83.87% ± 7.07% [68.57%, 100%] | **36 / 44** | **81.82%** [67.29%, 91.81%] | 4.32x ± 0.49x | **3.73x** (12.02%) | **21.95%** [16.30%, 28.89%] | **True** |
| **M3** | R90 ($m=0.04$) | 0.7801 | $\ge 40$ | 94.50% ± 5.77% [77.14%, 100%] | **43 / 44** | **97.73%** [87.98%, 99.94%] | 7.59x ± 1.02x | **6.59x** (21.26%) | **14.83%** [11.20%, 19.38%] | **True** |
| **M3** | R95 ($m=0.04$) | 0.6449 | $\ge 42$ | 98.87% ± 1.44% [97.14%, 100%] | **44 / 44** | **100.0%** [91.96%, 100%] | 12.63x ± 1.34x | **10.73x** (34.60%) | **9.32%** [7.02%, 12.28%] | **True** |
| **M3** | R100 ($m=0.00$) | 0.2673 | $= 44$ | 99.43% ± 1.17% [97.14%, 100%] | **44 / 44** | **100.0%** [91.96%, 100%] | 23.74x ± 6.68x | **22.34x** (72.07%) | **4.48%** [3.35%, 5.96%] | **True** |

> **安全邊際成本（事後探索）**：
> - $E_A$ R90 ($m=0.06$)：凍結門檻預警 337 家（7.66x）vs 事後最低預警 205 家（4.66x），安全邊際成本為 +132 家（+3.00x AR）。
> - $E_A$ R95 ($m=0.04$)：凍結門檻預警 524 家（11.91x）vs 事後最低預警 231 家（5.25x），安全邊際成本為 +293 家（+6.66x AR）。
> - 註：事後最低預警數無法事先得知，不可用來調整或縮減事前安全邊際。

---

### 3.2 市場盛行率敏感度重點表
*數據來源：`outputs/tables/e7a_prevalence_sensitivity.csv` (假設條件特徵分布不變，測試集註記僅 44 個正樣本)*

| 模型與工作點 | 市場破產盛行率 $\pi$ | CV 預期精確率 (均值±標準差) | 測試集精確率 (N=44) | 每抓到 1 家需審查家數 (CV) | 每抓到 1 家需審查家數 (測試集) | 每萬家預期預警數 (CV) | 每萬家預期捕獲破產數 (CV) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_A$ R90** ($m=0.06$) | **0.5%** | 1.83% ± 0.22% | **2.16%** | **54.7 家** | **46.4 家** | 2,610 家 | 47.7 家 |
| | **1.0%** | 3.61% ± 0.42% | **4.24%** | **27.7 家** | **23.6 家** | 2,645 家 | 95.4 家 |
| | **2.0%** | 7.03% ± 0.77% | **8.22%** | **14.2 家** | **12.2 家** | 2,715 家 | 190.9 家 |
| | **3.2%** (資料集) | 10.91% ± 1.13% | **12.67%** | **9.2 家** | **7.9 家** | 2,799 家 | 305.4 家 |
| | **5.0%** | 16.32% ± 1.55% | **18.76%** | **6.1 家** | **5.3 家** | 2,924 家 | 477.2 家 |
| **$E_A$ R95** ($m=0.04$) | **0.5%** | 1.19% ± 0.20% | **1.36%** | **84.0 家** | **73.4 家** | 4,146 家 | 49.3 家 |
| | **1.0%** | 2.36% ± 0.38% | **2.70%** | **42.3 家** | **37.0 家** | 4,174 家 | 98.7 家 |
| | **2.0%** | 4.66% ± 0.71% | **5.31%** | **21.4 家** | **18.8 家** | 4,232 家 | 197.4 家 |
| | **3.2%** (資料集) | 7.34% ± 1.05% | **8.33%** | **13.6 家** | **12.0 家** | 4,301 家 | 315.8 家 |
| | **5.0%** | 11.20% ± 1.48% | **12.64%** | **8.9 家** | **7.9 家** | 4,404 家 | 493.4 家 |

---

### 3.3 隨機種子穩定性對照表
*數據來源：`outputs/tables/e7b_seed_stability.csv` (種子：42, 7, 123, 2025, 31337)*

| 評估指標 | 5 組種子 15 折均值 | 種子標準差 ($\sigma_{\text{seed}}$) | 15 折平均之標準誤 ($\text{SE}_{\text{fold}} = \sigma_{\text{fold}}/\sqrt{15}$) | 變動比值 ($\sigma_{\text{seed}} / \text{SE}_{\text{fold}}$) | 結論口徑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3 點 log 預警損失** | 1.6150 (範圍: [1.5940, 1.6303]) | **0.0147** | **0.0620** | **23.74%** | 種子變動遠小於折間變動 |
| **AR@0.95** | 7.9045x (範圍: [7.7266, 8.0489]) | **0.1479x** | **0.5496x** | **26.91%** | 種子變動遠小於折間變動 |
| **PR-AUC** | 0.4758 (範圍: [0.4743, 0.4785]) | **0.0017** | **0.0204** | **8.33%** | 種子變動遠小於折間變動 |

> **方法學註記**：E7b 使用 E2b 最終參數，這些參數是在完整訓練集上調出，因此 E7b 絕對水準偏樂觀，不可當作預期表現；無偏泛化估計以巢狀 CV（E2a/E3）與 E6 為準。E7b 僅用於評估隨機種子穩定性。

---

## 4. 專案已知限制清單

1. **缺乏企業識別碼與時間戳記**：資料集無公司 ID 與財報年份，無法排除同一公司跨年度被重複抽樣的可能性，亦無法執行真實世界的時間序列切分（Out-of-Time 驗證）。
2. **正樣本極度稀缺與高折間噪音**：外層每折僅約 35 個破產正樣本，95% 召回高度取決於邊界 1～2 家樣本，單折評估指標具有不可避免的長右尾變異。
3. **破產盛行率與母體偏離**：資料集破產率 3.23% 遠高於真實商業世界的母體破產率（通常 $< 0.5\% \sim 1\%$），絕對機率與預警倍數在實際落地時需做母體先驗勝算比調整。
4. **ECDF 參考分布在外部資料的代表性未驗證**：目前推論依賴訓練集 10 折 OOF 分布進行百分位轉換，若外部真實資料分布發生宏觀共變異數漂移，百分位映射可能失真。

---

## 5. 圖表檔案清單與說明

| 階段 | 圖表路徑 | 一句話說明 |
| :--- | :--- | :--- |
| **EDA** | `outputs/figures/target_distribution.png` | 顯示資料集破產（220 家）與非破產（6,599 家）極度不平衡比例（約 1:30）。 |
| **EDA** | `outputs/figures/univariate_auc_top20.png` | 展現單特徵 ROC-AUC 排名前 20 名指標（最高為 Persistent EPS，AUC=0.8878）。 |
| **EDA** | `outputs/figures/correlation_heatmap_top30.png` | 揭示財務指標群聚現象與高重疊特徵家族（如負債比與淨值比之 $|r|=1.0$）。 |
| **E0** | `outputs/figures/e0_pr_curves.png` | 基準模型 PR 曲線對照，證實邏輯斯迴歸與最強單特徵顯著超越隨機基準線。 |
| **E0** | `outputs/figures/e0_recall_vs_alert_ratio.png` | 基準模型的召回率與預警倍數權衡曲線，展示高召回下的審查代價。 |
| **E1** | `outputs/figures/e1_strategy_comparison_boxplots.png` | 42 組不平衡與前處理組合之箱形圖，顯示不同重抽樣策略間分佈高度重疊。 |
| **E1** | `outputs/figures/e1_paired_diff_forest.png` | 各不平衡策略與 S0 配對差異的森林圖，95% CI 均跨越 0，無檢出統計差異。 |
| **E1** | `outputs/figures/e1b_ablation_bar.png` | 前處理消融條形圖，支持關閉 Winsorizer 並維持相關性過濾。 |
| **E2** | `outputs/figures/e2_tuned_vs_default.png` | 巢狀調參對比預設參數散布圖，顯示多數模型點落在對角線附近（地景平坦）。 |
| **E2** | `outputs/figures/e2_recall_vs_alert_ratio_tuned.png` | 調參後 5 個模型的召回率 vs 預警倍數曲線，M3 具備最穩定的預警折間標準差。 |
| **E3** | `outputs/figures/e3_rank_correlation_heatmap.png` | 5 個模型驗證折預測排名的 Spearman 相關矩陣，顯示樹模型間高達 0.85+ 相關性。 |
| **E3** | `outputs/figures/e3_model_rank_heatmap.png` | 各外層折模型 Friedman 排名熱力圖，呈現各模型互有勝負無單一主導現象。 |
| **E3** | `outputs/figures/e3_ensemble_vs_single_ar95.png` | 各集成候選組對比單一模型 M3 之 AR@0.95 分布圖，確認 $E_A$ 具小幅改善。 |
| **E3** | `outputs/figures/e3_recall_vs_alert_ratio_ensembles.png` | 集成候選模型之召回率 vs 預警倍數曲線對照。 |
| **E3** | `outputs/figures/e3_hard_positives_percentile.png` | 訓練集 176 家破產企業在 15 次評估中平均排名分布，識別 2 家最難抓公司。 |
| **E4** | `outputs/figures/e4_threshold_transfer.png` | 內層門檻遷移至外層驗證折之召回率達成率與漂移度分析。 |
| **E4** | `outputs/figures/e4_operating_curve.png` | 候選安全邊際 $m$ 在 15 折驗證折上的預警倍數與精確率權衡曲線。 |
| **E4** | `outputs/figures/e4_calibration_reliability.png` | Platt 與 Isotonic 校準器在訓練 OOF 上的 5 分箱可靠度對齊曲線。 |
| **E5** | `outputs/figures/e5_shap_summary_M3.png` | M3 (ExtraTrees) 核心財務特徵 SHAP Beeswarm 摘要圖。 |
| **E5** | `outputs/figures/e5_shap_summary_M4.png` | M4 (XGBoost) 核心財務特徵 SHAP Beeswarm 摘要圖。 |
| **E5** | `outputs/figures/e5_shap_summary_M5.png` | M5 (LightGBM) 核心財務特徵 SHAP Beeswarm 摘要圖。 |
| **E5** | `outputs/figures/e5_shap_dependence_top10.png` | 前 10 大關鍵財務特徵之 SHAP 依賴圖，確認邊際風險非線性驅動方向。 |
| **E5** | `outputs/figures/e5_permutation_top20.png` | 置換重要性 (Permutation Importance) 前 20 名條形圖與折間誤差線。 |
| **E5** | `outputs/figures/e5_importance_stability.png` | 特徵重要性跨 15 折穩定度指標散布圖。 |
| **E5** | `outputs/figures/e5_compact_vs_full.png` | 精簡特徵集 ($k \in \{10, 20, 30\}$) 巢狀消融損失增加與 AR@0.95 變化曲線。 |
| **E6** | `outputs/e6/e6_pr_curve.png` | 最終測試集上 $E_A$、M3 與單特徵之 PR 曲線對照圖（$E_A$ PR-AUC=0.5676）。 |
| **E6** | `outputs/e6/e6_recall_vs_alert_ratio.png` | 最終測試集上 3 基準之召回率 vs 預警倍數曲線對照圖。 |
| **E6** | `outputs/e6/e6_operating_curve.png` | $E_A$ 固化工作點在測試集 PR 曲線上的落點分布圖。 |
| **E6** | `outputs/e6/e6_calibration_reliability.png` | 最終測試集上的 Platt 與 Isotonic 5 分箱可靠度圖（僅 44 個正樣本，供參考）。 |
| **E6** | `outputs/e6/e6_score_distribution_positives.png` | 44 家破產企業在 $E_A$ 連續風險分數上的分布與固化門檻對照圖。 |
| **E7** | `outputs/figures/e7_prevalence_precision.png` | $E_A$ 各工作點在不同破產盛行率（0.5%～5%）下的精確率與審查成本變化雙聯圖。 |
