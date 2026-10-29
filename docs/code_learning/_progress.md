# 《程式碼學習資料包》製作進度與接續紀錄 (`_progress.md`)

- **所依據的程式碼版本（Base Commit）**：`c35f833`（`c35f83371712aa59b7284c0e99944b47b2801a7a`，分支 `docs/code-learning`）
- **Python 與測試環境**：Python 3.14.4，`bankruptcy/tests/` 內已提交之 24 個測試檔共 190 項測試（若含工作區未追蹤之 `test_model_score_report.py` 則為 25 個檔案、193 項測試）。
- **最後更新階段**：階段 3a-fix（參數登錄表 v2 完成：134 筆參數、12 大類別、T1/T2/T3 分層；完整性檢查 0 未對應；抽取器單元測試通過；強化版檢查器完成）

---

## 1. 階段進度總覽

| 階段 | 名稱與產出檔案 | 狀態 | Commit 紀錄 |
| :--- | :--- | :--- | :--- |
| **階段 0** | 專案唯讀盤點、`_progress.md`、`_findings.md` | ✅ 已完成 | `9576888` (`docs(code-learning): stage 0 - inventory and initial findings`) |
| **階段 1** | `01_專案地圖與資料流.md`、`_facts_verified.md` | ✅ 已完成 | `ef8ac14` (`docs(code-learning): stage 1 - project map and data flow`) |
| **階段 2a** | `02_模組導讀/`：`config.md`, `data_audit.md`, `metrics.md`, `preprocessing.md`；`_stale_claims.csv` | ✅ 已完成 | `9978104` (`docs(code-learning): stage 2a - module guides part 1`) |
| **階段 2b** | `02_模組導讀/`：`cv_and_pipelines.md`, `tuning_E2.md`, `ensemble_E3.md`, `calibration_and_threshold.md` | ✅ 已完成 | `c291360` (Part 1) + `3ad26f3` (Part 2) |
| **階段 2c** | `02_模組導讀/`：`explain_E5.md`, `e6_lock_and_final_eval.md`, `e7_sensitivity.md`, `experiments_E0_E7.md`, `score_new_data.md`, `app_streamlit.md`；最小版檢查器 `tools/check_code_docs.py` | ✅ 已完成 | `55c4191` (Part 1) + `3ad26f3` (Part 2) |
| **階段 3a-fix** | `03a_param_registry.csv` v2（134 筆，T1 84 筆、T2 26 筆、T3 24 筆；12 大類別）、`param_notes.csv`、`param_exclusions.csv`（24 項排除）、`tools/build_param_registry.py`、`tools/check_code_docs.py --registry`（完整性檢查 0 未對應）、`tools/test_registry_extractor.py`（單元測試通過） | ✅ 已完成 | 本次提交 (`docs(code-learning): stage 3a-fix - registry v2, completeness check, independent verification`) |
| **階段 3c-fix** | F1～F9 全量事實修復、`_facts_verified.md` 機器驗證達成 0 錯誤、`03c_速查表.md` 完整重構、`check_code_docs.py` AST/證據覆蓋升級 | ✅ 已完成 | `506a535` (`docs(code-learning): stage 3c-fix - E6/Brier/margin corrections, evidence coverage, facts verification`) |
| **階段 4a** | `labs/_common.py` 與 `lab_01`～`lab_04`（安全守衛、自建 CV、唯讀原則、quick 模式實測全部通過）、`labs/README.md`、`labs/lab_map.csv`、`lab_snapshots/` | ✅ 已完成 | 本次提交 (`docs(code-learning): stage 4a - lab common module and lab_01 to lab_04`) |
| **階段 4b** | `lab_05`～`lab_08` 實作、卡片規劃中標記替換、速查表對照更新 | ⏸️ 待放行 | — |
| **階段 5** | `04_實驗操作指南.md` | ⏸️ 待執行 | — |
| **階段 6** | `05_動手改程式配方.md` | ⏸️ 待執行 | — |
| **階段 7** | `06_練習題與解答.md`、`labs/bugs/` | ⏸️ 待執行 | — |
| **階段 8** | `00_讀我_學習路線.md`、`07_常見錯誤與除錯.md`、`08_術語表.md`、`09_面試說明稿.md` | ⏸️ 待執行 | — |
| **階段 9** | `tools/check_code_docs.py` 與最終驗收 | ⏸️ 待執行 | — |

### 1.1 過時說法清單統計（`_stale_claims.csv` 統計報告）

全專案共掃描出 **288 筆**歷史過時說法，分佈如下：
- **被 Git 追蹤之檔案**：**117 筆**（包含 `bankruptcy/src/final_eval.py` 13 筆、`bankruptcy/MODEL_CARD.md` 6 筆、`PROJECT_STATE.md` 5 筆、`bankruptcy/src/ensemble.py` 2 筆、`bankruptcy/src/run_imbalance.py` 2 筆、`bankruptcy/src/seed_stability.py` 2 筆等）。
- **未被 Git 追蹤之工作區檔案**：**171 筆**（包含教學簡報資料腳本 `bankruptcy/scripts/student_slides_data_v2.py` 23 筆、`student_slides_data.py` 19 筆、`docs/model_score_learning_report.html` 13 筆、`docs/student_presentation_deck.md` 11 筆、`docs/student_learning_handbook.md` 10 筆等）。
- **處理原則**：僅於清單中精確記錄「檔案, 行號, 錯誤說法, 正確說法, 證據, 是否被追蹤」，**不更動任何原始檔案**，供後續專項任務依表修正。

---

## 2. 專案核心模組清單與 `02_模組導讀/` 對映表

本專案實際程式碼位於 `bankruptcy/` 目錄下，共包含 **22 個核心 `.py` 模組**（不含 `__init__.py` 與報告生成輔助腳本 `bankruptcy/scripts/`）及 **24 個已提交之 `tests/test_*.py` 測試檔**：

| 導讀檔案 (`docs/code_learning/02_模組導讀/`) | 對應實際程式碼路徑（由 repo 根目錄起算） | 程式碼行數 | 主要公開函式／類別與職責 |
| :--- | :--- | ---: | :--- |
| `config.md` | `bankruptcy/config.py` | 45 | 全域路徑常數、`SEED=42`、`TEST_SIZE=0.2`、`CV_FOLDS=5`、`CV_REPEATS=3`、`TARGET_RECALLS`、`ALERT_RATIOS`、MVP 營運與成本常數 |
| `data_audit.md` | `bankruptcy/src/data_audit.py` | 526 | `load_and_clean_data`, `audit_basic_info`, `audit_column_anomalies`, `audit_univariate_signals`, `audit_near_duplicates`, `generate_figures` |
| `metrics.md` | `bankruptcy/src/metrics.py`<br>`bankruptcy/src/stats_utils.py` | 607<br>128 | `alert_ratio_at_recall`, `recall_at_alert_ratio`, `evaluate_scores`, `evaluate_at_threshold`, `brier`, `compute_top_k_quota`, `recall_at_top_k_percent`, `misclassification_cost`, `cost_sensitivity_table`；`nadeau_bengio_correction_factor`, `nadeau_bengio_paired_test`, `wilcoxon_test`, `holm_bonferroni` |
| `preprocessing.md` | `bankruptcy/src/preprocessing.py` | 429 | `TrainingMedianImputer`, `FeatureSelector`, `DropConstantAndDuplicate`, `Winsorizer`, `LargeValueIndicator`, `QuantileScaler`, `CorrelationFilter` |
| `cv_and_pipelines.md` | `bankruptcy/src/cv.py`<br>`bankruptcy/src/pipelines.py` | 386<br>296 | `compute_y_hash`, `make_holdout_split`, `make_cv_splits`, `save_splits`, `load_splits`, `get_train_data`, `get_test_data`, `make_inner_splits`, `cross_validate_pipeline`；`AdaptiveScaleWeightClassifier`, `_get_resampler`, `_get_classifier`, `build_pipeline`, `get_score`, `get_feature_names_after_preprocessing` |
| `tuning_E2.md` | `bankruptcy/src/tune.py` | 1,131 | `compute_three_point_log_loss`, `sample_hyperparameters`, `create_inner_objective`, `run_single_outer_fold`, `get_untuned_baseline_results`, `run_e2_tuning` (E2a), `run_e2b_final_tuning` (E2b) |
| `ensemble_E3.md` | `bankruptcy/src/ensemble.py` | 1,669 | `ecdf_map`, `compute_ensemble_scores`, `get_or_generate_reference_distribution`, `run_e3a_single_models_comparison`, `run_e3b_and_e3c`, `run_e3b_paired_comparisons`, `run_e3d_diagnostics`, `apply_e3_selection_rule` |
| `calibration_and_threshold.md` | `bankruptcy/src/thresholding.py` | 1,103 | `wilson_score_interval`, `derive_operating_threshold`, `fit_single_model_platt`, `run_e4a_threshold_transfer`, `run_e4b_calibration`, `freeze_final_components` (E4c) |
| `explain_E5.md` | `bankruptcy/src/explain.py`<br>`bankruptcy/src/local_explain.py` | 1,253<br>293 | `map_financial_dimension`, `transform_through_preprocessing`, `compute_permutation_importance`, `compute_group_permutation_importance`, `compute_shap_importance`, `run_e5` (E5a/E5b)；`LocalExplainer`, `get_local_explainer`, `explain_case` |
| `e6_lock_and_final_eval.md` | `bankruptcy/src/preflight.py`<br>`bankruptcy/src/lock_utils.py`<br>`bankruptcy/src/final_eval.py` | 318<br>93<br>956 | `run_e6_preflight`；`compute_codebase_hash`, `acquire_e6_lock`, `is_e6_locked`；`clopper_pearson_ci`, `wilson_score_ci`, `run_e6_scoring`, `run_e6_analysis`（已鎖定，禁止重跑） |
| `e7_sensitivity.md` | `bankruptcy/src/prevalence_sensitivity.py`<br>`bankruptcy/src/seed_stability.py` | 242<br>237 | `run_prevalence_sensitivity` (E7a 母體盛行率解析推導)；`run_seed_stability` (E7b 5 組隨機種子穩定性) |
| `experiments_E0_E7.md` | `bankruptcy/src/run_baselines.py`<br>`bankruptcy/src/run_imbalance.py` | 441<br>697 | `evaluate_single_feature_fold`, `run_e0_baselines`, `run_label_shuffle_test` (E0)；`run_single_setting_cached`, `run_e1_experiments`, `compile_e1a_tables`, `compile_e1b_tables`, `determine_selection_recommendations` (E1)；以及 E0～E7 全實驗總覽索引 |
| `score_new_data.md` | `bankruptcy/src/scoring.py`<br>`bankruptcy/score_new_data.py` | 599<br>246 | `to_percentile`, `ensemble_score`, `flag`, `load_final_components`, `score_dataset`, `check_and_deduplicate_columns`, `load_mvp_components`, `score_mvp_batch`；CLI `get_required_feature_names`, `score_csv`, `main` |
| `app_streamlit.md` | `bankruptcy/app/app.py`<br>`bankruptcy/app/logic.py`<br>`bankruptcy/app/ui_text.py`<br>`bankruptcy/src/batch_ui_helper.py`<br>`bankruptcy/api/main.py` | 889<br>596<br>227<br>212<br>469 | Streamlit 5 大分頁與互動模擬邏輯、批次評估 UI 輔助函式、FastAPI 端點服務 |

---

## 3. 實驗 ID → 目的 → 產出檔案 → 主要結論對照表

| 實驗 ID | 對應模組 | 實驗目的 | 核心產出檔案（位於 `bankruptcy/outputs/`） | 主要結論（依實測結果檔） |
| :--- | :--- | :--- | :--- | :--- |
| **EDA** | `src/data_audit.py` | 資料完整性、欄位異常、單變數訊號與近重複樣本稽核 | `tables/audit_summary.csv`<br>`tables/constant_columns.csv`<br>`tables/duplicate_columns.csv`<br>`tables/large_value_columns.csv`<br>`tables/high_correlation_pairs.csv`<br>`tables/univariate_auc.csv`<br>`tables/near_duplicates.csv` | 總樣本 6,819 筆（破產 220 筆，3.23%），無缺失值與完全重複列；1 個常數欄（`Net Income Flag`）、2 對完全重複欄、24 個雙峰大值欄（最大值 $>1.0$）；最強單變數 `Persistent EPS in the Last Four Seasons` 調整後 AUC 達 0.8892。 |
| **E0** | `src/run_baselines.py` | 建立隨機猜測、單一最強特徵 LR、未調參多元 LR 基準線與標籤洗牌防洩漏檢驗 | `tables/e0_baselines_summary.csv`<br>`tables/e0_baselines_per_fold.csv`<br>`tables/e0_label_shuffle.csv`<br>`tables/e0_single_feature_selection.csv` | 隨機猜測 5 折 × 3 次重複 = 15 次外層評估平均 PR-AUC 為 0.0323、AR@0.95 為 31.01x；單一最強特徵 LR 平均 PR-AUC 為 0.3012、AR@0.95 為 16.08x；未調參多元 LR (S0) 平均 PR-AUC 為 0.3938、AR@0.95 為 10.31x；標籤洗牌 ROC-AUC 約 0.50，確認無資料洩漏。 |
| **E1a** | `src/run_imbalance.py` | 比較 7 種不平衡策略（S0～S6）× 2 種縮放（quantile, standard）× 3 個模型（M1, M2, M4）共 42 組設定 | `tables/e1a_imbalance_comparison.csv`<br>`tables/e1a_paired_vs_s0.csv`<br>`tables/e1_selection_recommendation.csv` | Quantile 縮放全面優於 Standard 縮放；各重抽樣策略（S1～S6）相對於不處理（S0）之 95% Nadeau-Bengio 配對信賴區間皆跨越 0（未檢出統計顯著差異），依簡約原則推薦（S0, quantile），並保留 S0 與 S3 進入 E2 作為調參類別選項。 |
| **E1b** | `src/run_imbalance.py` | 前處理消融實驗（Winsorizer 開關、CorrelationFilter 開關、LargeValueIndicator 開關） | `tables/e1b_ablation.csv`<br>`tables/e1b_paired_vs_default.csv` | 關閉 Winsorizer 與預設相比未檢出統計差異且在 M2 上點估計略佳（AR@0.95 8.25x vs 9.07x）；保留 `CorrelationFilter(threshold=0.95)` 可安全剔除高共線冗餘特徵。確立後續固定前處理：Winsorizer 關、CorrFilter 開、LargeValueIndicator 關、QuantileScaler 開。 |
| **E2a** | `src/tune.py` | 5 折 × 3 次重複 = 15 次外層評估搭配內層 3 折 Optuna TPE 巢狀超參數調優（M1～M5） | `tables/e2a_nested_summary.csv`<br>`tables/e2a_nested_per_fold.csv`<br>`tables/e2a_paired_vs_default.csv`<br>`tables/e2a_best_param_stability.csv`<br>`tables/e2_default_matched_per_fold.csv` | M1～M5 調參後與同條件預設參數相比，Nadeau-Bengio 95% CI 全數跨越 0（$p_{\text{Holm}}=1.0$），超參數地景平坦；單一模型平均 AR@0.95 介於 8.09x～9.07x，其中 M3 (ExtraTrees) 評估間標準差最小（1.35x）。 |
| **E2b** | `src/tune.py` | 在全量訓練集（5,455 筆）以 5 折 × 2 次重複（10 次內層評估）尋優並固化 M1～M5 最終超參數 | `models/best_params/M1.json` ～ `M5.json`<br>`tables/e2b_final_params_summary.csv` | 固化最終超參數：M1 (S0)、M2 (S3)、M3 (S0)、M4 (S0)、M5 (S3)；各模型前 5 名試驗之損失極差僅 0.0014～0.0162，再次印證超參數地景平坦。 |
| **E3a–d** | `src/ensemble.py` | 單一模型成對比較（E3a）、ECDF 百分位等權集成（E3b）、Stacking（E3c）與難抓破產樣本診斷（E3d） | `tables/e3a_pairwise_models.csv`<br>`tables/e3a_friedman_ranks.csv`<br>`tables/e3b_ensemble_summary.csv`<br>`tables/e3b_ensemble_vs_single_paired.csv`<br>`tables/e3c_stacking_summary.csv`<br>`tables/e3d_hard_positives.csv`<br>`tables/e3_selection.csv` | 5 個單一模型兩兩比較（$p_{\text{Holm}}=1.0$）與集成對單一模型比較（最小 $p_{\text{Holm}}=0.6195$）均無統計顯著差異；依預設決策規則（Rule 2），選出三點損失最低且 AR@0.95 變異較小之 $E_A$（M3+M4+M5 之 ECDF 百分位等權平均，平均損失 1.6344、AR@0.95 8.06x、PR-AUC 0.4598）為主要模型，M3 為對照組。E3d 發現僅 1 家破產公司（原始列 2246）在 15 次評估中全數漏抓。 |
| **E4a–c** | `src/thresholding.py` | 跨折門檻轉移與安全邊際 $m$ 評估（E4a）、Platt vs Isotonic 機率校準比較（E4b）、全訓練集最終元件固化（E4c） | `tables/e4a_threshold_transfer_summary.csv`<br>`tables/e4a_margin_recommendation.csv`<br>`tables/e4b_calibration_summary.csv`<br>`tables/operating_points_for_report.csv`<br>`models/final/*`（11 個固化檔） | 無安全邊際（$m=0$）跨折召回達標率不足 80%；加入安全邊際（$E_A$: R80 $m=0.02$, R90 $m=0.06$, R95 $m=0.04$, R100 $m=0.00$）後達標率 $\ge 80\%$。Platt 校準嚴格保序（AR@0.95 維持 8.06x，BSS 0.1839），而 Isotonic 因階梯同分使 AR@0.95 惡化至 10.02x。確立門檻切於未校準連續排名，Platt 僅供機率查詢。 |
| **E5a–b** | `src/explain.py` | 跨模型特徵重要性（內建、置換、家族置換、TreeSHAP）與精簡特徵集（$k \in \{10, 20, 30\}$）巢狀消融 | `tables/e5a_permutation_importance.csv`<br>`tables/e5a_group_importance.csv`<br>`tables/e5a_shap_importance.csv`<br>`tables/e5b_compact_summary.csv`<br>`tables/e5b_compact_vs_full_paired.csv`<br>`tables/e5b_core_features.csv` | M3、M4、M5 前 20 名 SHAP 特徵交集達 13 個，負債比率家族（Family 38）100% 進入前 20 名；精簡特徵 $k=10$ 顯著劣於全特徵（$p_{\text{Holm}}=0.0251$），$k=20$（$p_{\text{Holm}}=0.1423$）與 $k=30$（$p_{\text{Holm}}=0.4485$）未檢出統計顯著差異；產出 12 個核心特徵與 18 個擴大核心特徵。 |
| **E6** | `src/preflight.py`<br>`src/lock_utils.py`<br>`src/final_eval.py` | 通過 8 項 preflight 檢查後，對封存之 20% 測試集（1,364 筆，44 家破產）進行一次性解封評估並永久鎖定 | `e6/e6_operating_points.csv`<br>`e6/e6_ranking_metrics.csv`<br>`e6/e6_paired_bootstrap.csv`<br>`e6/e6_calibration.csv`<br>`e6/e6_vs_cv.csv`<br>`e6/e6_hindsight_costs.csv` | $E_A$ 與 M3 在全部四個工作點皆達成目標召回（$E_A$ R90 捕獲 43/44 = 97.73%，預警 24.71% = 7.66x；R95 捕獲 44/44 = 100.0%，預警 38.42% = 11.91x）；$E_A$ 測試集 PR-AUC 為 0.5676、ROC-AUC 為 0.9621、AR@0.95 為 5.25x；$E_A - M3$ 配對 Bootstrap 95% CI 為 $[-0.0226, +0.1044]$ 跨越 0。 |
| **E7a–b** | `src/prevalence_sensitivity.py`<br>`src/seed_stability.py` | 市場母體破產盛行率（0.5%～5.0%）敏感度解析推導（E7a）與 5 組隨機種子穩定性評估（E7b） | `tables/e7a_prevalence_sensitivity.csv`<br>`tables/e7b_seed_stability.csv`<br>`tables/e7b_seed_stability_per_fold.csv` | E7a 揭示當母體破產率降至 1.0% 時，$E_A$ R90 預期精確率降至 3.61%（CV 估計），每抓 1 家破產需審查 27.7 家；E7b 顯示 5 組種子間標準差僅為外層評估平均標準誤的 8.33%（PR-AUC）～26.91%（AR@0.95），證實隨機種子波動遠小於資料切分變動。 |

---

## 4. 已讀取檔案與關鍵發現摘要

- **已讀核心檔案**：`bankruptcy/config.py`, `bankruptcy/src/cv.py`, `bankruptcy/src/data_audit.py`, `bankruptcy/src/preprocessing.py`, `bankruptcy/src/pipelines.py`, `bankruptcy/src/run_imbalance.py`, `bankruptcy/src/tune.py`, `bankruptcy/src/ensemble.py`, `bankruptcy/src/thresholding.py`, `bankruptcy/src/explain.py`, `bankruptcy/src/lock_utils.py`, `bankruptcy/src/seed_stability.py`, `bankruptcy/tests/test_frozen_hashes.py`, `bankruptcy/tests/test_integrity_phase0.py`, `bankruptcy/tests/test_mvp_completeness.py`, `bankruptcy/tests/test_lock.py`，以及 `bankruptcy/outputs/tables/` 與 `bankruptcy/outputs/e6/` 內之 CSV/JSON 彙總檔。
- **關鍵發現**：詳見同目錄下之 `_findings.md`（包含既有文件與實際程式碼在內層 CV 折數、外層評估正樣本數、相關性門檻上的差異，以及 `thresholding.py:634` 與 `tune.py:577, 945` 的疑似程式問題）。
- **已驗證事實表**：詳見同目錄下之 `_facts_verified.md`（涵蓋資料切分、CV 各層正樣本數、特徵漏斗、M0～M5 與 $E_A$、SMOTE 使用情況、Optuna 設定、校準與門檻、190/193 項測試清單與 E6 結果）。
- **已完成項目**：
  - 階段 0：現況盤點與事實查證完成（`_facts_verified.md`、`_findings.md`）。
  - 階段 1：導讀目錄、檢查工具與測試案例就緒（`00_導讀目錄.md`、`check_code_docs.py`）。
  - 階段 2a：前處理、特徵、指標與 CV 導讀完成。
  - 階段 2b：基準模型與調參流程導讀完成。
  - 階段 2c：集成、門檻校準與評估驗證導讀完成。
  - 階段 3a & 3a-fix：參數登錄表升級 v2，165 筆全量收錄（115 項 T1、38 項 T2、12 項 T3），分級與三層校驗通過。
  - 階段 3b & 3b-fix：37 張參數卡片全部完成並通過全量審計，小表自動同步，結構化證據標籤全量驗證，8 大標準 lab 映射與路徑存在性校驗完成。
  - 階段 3c：目標導向調參速查表（6 大業務工程目標）與參數交互作用矩陣（10 組高風險交互作用）完成（`03c_速查表.md`）。
- **當前項目**：階段 3（參數手冊與速查表體系）全數完成，停下等待審閱放行階段 4。
- **下一步**：使用者審查通過後，進入階段 4（Lab 規劃與實作 lab_01～lab_08）。
