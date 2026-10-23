# 破產預測專案狀態檔 (PROJECT_STATE.md)
*更新時間：2026-10-22 | 本檔為新對話接續開發之核心交接文件*

## 1. 專案目標與資料
- **目標**：企業破產早期預警，業務場景為信貸審查與風險稽核。核心準則為「高召回（追求抓出 90%~95% 破產企業）、寧可錯殺不可放過」。
- **資料規模**：總樣本 6,819 筆，包含 95 個財務特徵欄位與 1 個目標欄位 (`Bankrupt?`)。
- **類別不平衡**：正常企業 6,599 家（96.77%），破產企業 220 家（3.23%），不平衡比例約 30.0 : 1。
  - 訓練集（80%）：5,455 筆（正常 5,279，破產 176）。
  - 測試集（20%）：1,364 筆（正常 1,320，破產 44），嚴格封存至 E6。
- **已知資料問題**（詳見 `outputs/tables/audit_summary.csv`）：
  - 常數欄位：1 個 (`Net Income Flag` 恆為 1)。
  - 完全重複欄位：2 對（`Current Liabilities/Liability` 與 `Current Liability to Liability`；`Current Liabilities/Equity` 與 `Current Liability to Equity`）。
  - 雙峰大值欄位：24 個特徵因科學記號/未標準化編碼，數值介於 0~1 與 10^8~10^10 雙峰分布。
  - 無缺失值，無完全重複列。

## 2. 環境
- **作業系統**：Windows 11 (Windows-11-10.0.26200-SP0)
- **Python 版本**：3.14.4 (64-bit)
- **核心套件**：
  - `scikit-learn`: 1.9.1 | `imbalanced-learn`: 0.14.2 | `xgboost`: 3.4.1 | `lightgbm`: 4.7.0
  - `optuna`: 5.0.0 | `scipy`: 1.18.1 | `numpy`: 2.4.6 | `pandas`: 3.0.6 | `pytest`: 9.1.1
- **測試執行方式**：在 `bankruptcy/` 目錄下執行 `pytest -q`（目前 42 項單元測試全數通過）。

## 3. 專案結構與檔案地圖
- **原始碼 (`bankruptcy/src/`)**：
  - `config.py`：全域固定種子 (`SEED = 42`)、路徑常數與共用參數。
  - `cv.py`：固化切分載入、外層 15 折/內層 5 折管理與測試集嚴格防洩漏隔離介面。
  - `data_audit.py`：資料完整性稽核、常數欄、重複欄與雙峰大值欄位偵測。
  - `metrics.py`：悲觀同分處理之 `alert_ratio@recall`、`recall@alert_ratio`、PR-AUC、ROC-AUC 計算。
  - `preprocessing.py`：自訂無洩漏轉換器（Winsorizer、CorrelationFilter、LargeValueIndicator、AdaptiveScaleWeightClassifier、QuantileScaler、FeatureSelector）。
  - `pipelines.py`：組裝 imbalanced-learn Pipeline（整合縮放、SMOTE 與分類器）。
  - `stats_utils.py`：Nadeau-Bengio 校正 paired t-test、Holm-Bonferroni 多重校正與 Wilcoxon 檢定。
  - `run_baselines.py`：E0 基準實驗執行腳本（隨機猜測、最常類別、標籤洗牌、單特徵 LR）。
  - `run_imbalance.py`：E1 不平衡策略比較（S0~S6 × 2 縮放 × 3 模型）與前處理消融實驗。
  - `tune.py`：E2 巢狀超參數調整（Optuna 內層最小化三點 log 預警倍數目標、外層無偏評估與 E2b 全量擬合）。
  - `ensemble.py`：E3 模型成對比較、ECDF 百分位等權集成、Stacking 與難抓樣本診斷。
  - `thresholding.py`：E4 門檻跨折轉移評估（E4a）、機率校準對照（E4b）與最終元件固化管線（E4c）。
  - `scoring.py`：即時與批次推論評分介面（基於固化參考分布之 ECDF 映射、門檻警示與 Platt 機率預測）。
- **關鍵資料與輸出檔案**：
  - 固化切分：`outputs/models/splits.joblib`（外層 15 折：5 折 × 3 重複；內層 5 折；E2b 10 折；測試集索引）。
  - 稽核表格：`outputs/tables/audit_summary.csv`, `constant_columns.csv`, `duplicate_columns.csv`, `large_value_columns.csv`。
  - 實驗報告表：`outputs/tables/`（E0 至 E4 各階段彙總與單折明細表）。
  - OOF 分數儲存庫：`outputs/models/oof/`（各階段各模型驗證折預測分數，格式 `.npy` / `.npz`）。
  - E2b 全訓練集最佳超參數：`outputs/models/best_params/` (`M1.json` ~ `M5.json`) 與 `outputs/tables/e2b_final_params_summary.csv`。
  - E4c 最終固化元件目錄：`outputs/models/final/`（含 `model_M3.joblib`, `model_M4.joblib`, `model_M5.joblib`, `reference_M3.npy`, `reference_M4.npy`, `reference_M5.npy`, `thresholds.json`, `thresholds.csv`, `calibrator_platt.joblib`, `calibrator_isotonic.joblib`, `version_metadata.json`）。
  - 圖表目錄：`outputs/figures/`。

## 4. 鐵則（必須逐字保留）
- 所有 fit（前處理、特徵過濾、SMOTE、門檻、校準）只能在訓練資料內進行。
- 測試集只能透過 cv.get_test_data(i_know_this_is_final_evaluation=True) 讀取，只在 E6 使用一次；在此之前任何階段不得讀取或評估。
- 每次只做一個步驟，完成後回報並停下來等我確認，不得提前進入下一步。
- 所有隨機性固定 config.SEED；長時間任務要有 --quick 模式、快取與續跑。
- 報告用語：不得在無校正檢定支持時宣稱「顯著」或「證明」；事後挑出的比較須標註為事後；折平均為主要指標，合併 OOF 只作輔助（巢狀 CV 下各折參數不同、分數尺度不一致）。

## 5. 指標與統計慣例
- **`alert_ratio@recall` 定義**：達到指定召回率 $R$ 時，發出警示的總樣本數相對於真實正樣本數（$P$）的倍數：$\text{AR@R} = \frac{N_{\text{alert}}}{P} = \frac{\text{alert\_proportion}}{\text{base\_rate}}$。
- **`recall@alert_ratio` 定義**：在預警總量限制為正樣本數的 $K$ 倍時，所能捕獲的正樣本比例：$\frac{\text{TP}}{P}$。
- **悲觀同分處理**：若門檻落在多個相同分數樣本上，所有相同分數樣本一律納入警示（$\text{score} \ge t$），完整計入誤警，絕不隨機或樂觀切分。
- **調參目標**：最小化內層 5 折合併 OOF 後的三點 log 預警倍數平均：
  $$\text{loss} = \frac{1}{3} \sum_{r \in \{0.80, 0.90, 0.95\}} \ln(\text{alert\_ratio}@r)$$
  *理由*：單點 recall 0.95 折間噪音過大（標準差約 4），三點 log 平均可平滑評估並加重懲罰長尾劣況；絕不納入 recall 1.0（避免受單一最難樣本牽制）。
- **Nadeau-Bengio 校正**：針對 5 折 × 3 重複（$k=15$ 折）之非獨立配對 t 檢定，校正因子 $c = \frac{1}{k} + \frac{n_{\text{val}}}{n_{\text{train}}} = \frac{1}{15} + \frac{1}{4} = 0.3167$，$SE_{\text{corr}} = \sqrt{c \cdot s^2}$，防止虛假顯著。
- **Holm-Bonferroni 多重校正**：對多組配對檢定排序後進行逐步 step-down 顯著性控制。
- **$\lceil R \cdot P \rceil$ 離散化**：驗證折正樣本數僅約 35 個，目標召回在單折層級為階梯跳躍。外層（$P \approx 35$）與內層（$P \approx 141$）正樣本量不同，離散化間距不同，指標僅能作為相對參考。

## 6. 已完成步驟與關鍵結論
- **E0（基準實驗）**：隨機猜測 PR-AUC 0.0323、AR@0.95 達 31.0x；單特徵羅吉斯回歸（`Persistent EPS in the Last Four Seasons`）AR@0.95 為 16.1x；未調參多元 LR AR@0.95 為 10.3x。證實多特徵模型顯著優於單一變數與隨機基準。詳見 `outputs/tables/e0_baselines_summary.csv`。
- **E1（不平衡策略與消融）**：42 組設定比較顯示，S0（不處理）與 S1（類別權重）在樹模型上表現最佳，SMOTE 變體無顯著優勢；Quantile 縮放全面優於 Standard。消融確認關閉 Winsorizer 對尾端預警無害，保留 CorrelationFilter 可安全去冗餘。**固化前處理：Winsorizer 關、CorrFilter 開、LargeValueIndicator 關、quantile 縮放、imbalance ∈ {S0, S3}**。詳見 `outputs/tables/e1a_imbalance_comparison.csv`。
- **E2（巢狀超參數調整）**：M1～M5 在 15 個外層折上調參後相對於同條件預設**無統計顯著差異**（Nadeau-Bengio CI 全數跨越 0，Holm 校正後 $p_{Holm} = 1.0$），超參數地景極度平坦。五個模型調參後 AR@0.95 介於 8.1x～9.1x，折間標準差（1.4～3.9）遠大於模型間差異；M3（ExtraTrees）折間標準差最小（1.35）。全訓練集參數固化於 `outputs/models/best_params/` 與 `outputs/tables/e2b_final_params_summary.csv`。
- **E3（模型集成與診斷）**：比較單一模型、4 種 ECDF 百分位平均集成與 Stacking。**所有集成相對於單一模型在統計上無顯著差異**（20 組配對 Holm 校正後 $p_{Holm} = 0.6195 \gg 0.05$）。基於工程多樣性與穩定性判斷，**最終選定主要組 $E_A$（M3 ExtraTrees + M4 XGBoost + M5 LightGBM 之 ECDF 百分位等權平均），對照組為單一模型 M3**。最難抓公司特徵診斷（口徑修正：每家公司只在 3 個外層折被驗證，定義為 5 個模型 × 3 個 repeat 共 15 次評估中漏抓超過 12 次）確認其核心財務指標介於正常與破產之間。詳見 `outputs/tables/e3b_ensemble_summary.csv` 與 `outputs/tables/e3d_hard_positives.csv`。
- **E4（門檻轉移、機率校準與元件固化）**：
  - 門檻轉移（E4a）：無安全邊界（$m=0$）時驗證折達成率僅 53%~73%；加入安全邊界可達 $\ge 80\%$。$E_A$ 建議邊界：$R=0.80 \to m=0.02$, $R=0.90 \to m=0.06$, $R=0.95 \to m=0.04$。$R=1.00$ 依規則固定 $m=0.0$（達成率 80%，需預警 75.7% 樣本，追求零漏抓在本資料上缺乏泛化支持）。
  - 機率校準（E4b）：Platt Scaling 嚴格保序且維持 AR@0.95（8.06x），Brier Skill Score 為 0.1839；Isotonic Regression 形成階梯常數段破壞尾端排序，使 AR@0.95 惡化至 10.02x（BSS 0.2650）。確立門檻設於未校準連續排名、Platt 校準僅供後驗機率查詢。訓練分布上平均預測機率對齊真實盛行率為擬合必然結果，以可靠度圖為準。
  - 元件固化（E4c）：全量擬合模型、10 折 OOF 參考分布與門檻表存於 `outputs/models/final/`。詳見 `outputs/tables/operating_points_for_report.csv`、`outputs/tables/e4a_margin_recommendation.csv` 與 `outputs/tables/e4b_calibration_summary.csv`。

- **E5（特徵重要度、可解釋性與精簡特徵集評估）[狀態：待驗證]**：
  - *稽核與核對備忘*：E5a 彙整分母錯誤已修正（未入選折補 0、以 15 折為全分母）；凍結門檻表待核對已完成（經 `tests/test_thresholds_reproduce.py` 與 `thresholds_recomputed.json` 驗證 100% 吻合）；過濾後特徵數待核對已完成（確認全訓練集與外層折皆為 92 -> 72 左右，與獨立貪婪參考實作 100% 一致）。
  - 特徵重要度（E5a）：15 個外層驗證折評估顯示，M3 (ExtraTrees)、M4 (XGBoost)、M5 (LightGBM) 在關鍵財務特徵上高度共識（三模型前 20 名 SHAP 交集達 13 個，M4 與 M5 之 SHAP 相關係數達 0.892）。SHAP 依賴圖確認風險驅動方向：獲利指標（`Persistent EPS in the Last Four Seasons`, `Net Income to Total Assets`）與流動性（`Quick Ratio`）越低風險顯著越高；槓桿指標（`Borrowing dependency`, `Debt ratio %`）越高風險顯著越高。
  - 精簡特徵巢狀評估（E5b）：每折於外層訓練集獨立選取 Top k 原始特徵重擬合，在外層驗證折以折內百分位等權集成：
    - $k=10$：3 點預警損失增加 $+0.1335$（95% CI $[0.0401, 0.2269]$，$p_{Holm} = 0.0251$），表現顯著衰退，AR@0.95 惡化至 9.35x。
    - $k=20$：損失差異 $+0.0841$（95% CI $[-0.0083, 0.1765]$，$p_{Holm} = 0.1423$），在本資料下沒有檢出統計差異（檢定力受每折約 35 個正樣本限制），AR@0.95 均值 8.62x。
    - $k=30$：損失差異 $+0.0354$（95% CI $[-0.0619, 0.1327]$，$p_{Holm} = 0.4485$），AR@0.95 均值 8.18x（全特徵 8.04x），極度接近全特徵集。
    - 核心特徵集：以 $\ge 80\%$ 出現比例篩選出 12 個核心特徵，固化於 `outputs/models/final/core_features.txt`。詳見 `outputs/tables/e5b_core_features.csv` 與 `outputs/tables/e5b_compact_vs_full_paired.csv`。

## 7. 已知限制
1. **缺乏企業識別碼與時間戳記**：資料集無公司 ID 與財報年份，無法排除同一公司跨年度被重複抽樣的可能性，亦無法執行真實世界的時間序列切分（Out-of-Time 驗證）。
2. **正樣本極度稀缺與高折間噪音**：外層每折僅約 35 個破產正樣本，95% 召回高度取決於邊界 1～2 家樣本，單折評估指標具有不可避免的長右尾變異。
3. **破產盛行率與母體偏離**：資料集破產率 3.23% 遠高於真實商業世界的母體破產率（通常 < 0.5%~1%），絕對機率與預警倍數在實際落地時需做母體先驗勝算比調整。
4. **ECDF 參考分布在外部資料的代表性未驗證**：目前推論依賴訓練集 10 折 OOF 分布進行百分位轉換，若外部真實資料分布發生宏觀共變異數漂移，百分位映射可能失真。

## 8. 待辦與順序
1. **E6（最終測試集正式評估，嚴格只做一次）**：解封 `cv.get_test_data(i_know_this_is_final_evaluation=True)`，以固化管線評估 $E_A$ 與 M3 的無偏泛化指標。
2. **E7（敏感度與壓力測試分析）**：盛行率調整測試、種子擾動與極端情境抗壓評估。
3. **`score_new_data.py`**：建立獨立命令列工具，支援新輸入 CSV 檔案自動輸出連續風險分數、Platt 破產機率與各召回門檻警示 Flag。
4. **最終專案技術報告**。
- **延後項目**：
  - M6（SVM）與 M7（MLP）深度神經網路調參評估。
  - 真實資料外部驗證與遷移學習（待流程全數完成後，使用 E5 產出之精簡特徵集於真實未標記資料重做）。

## 9. 下一步
- **下一個執行步驟：E6（最終測試集正式評估，嚴格只做一次）**
- **規格摘要**：
  - 解封測試集（1,364 筆，正常 1,320，破產 44）。
  - 使用 `outputs/models/final/` 已凍結之元件（$E_A$ 模型、OOF 參考分布、門檻表、Platt 校準器）進行一次性無偏推論評估。
  - 產出全集評估報告與 CI。
  - **重要提醒**：需等使用者確認所有前期規格、指標口徑與財務構面無誤後，始得執行 E6。不得自行提早解封。
