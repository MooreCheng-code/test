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
- **測試執行方式**：在 `bankruptcy/` 目錄下執行 `pytest -q`（目前 64 項單元與整合測試全數通過）。

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

- **E5（特徵重要度、可解釋性與精簡特徵集評估）**：
  - 特徵重要度（E5a）：15 個外層驗證折評估顯示，M3 (ExtraTrees)、M4 (XGBoost)、M5 (LightGBM) 在關鍵財務特徵上高度共識（三模型前 20 名 SHAP 交集達 13 個，M4 與 M5 之 SHAP 相關係數達 0.892）。特徵家族層級（|r| > 0.95）分析確認，負債比率家族（Family 38，包含 Total debt/Total net worth、Debt ratio %、Net worth/Assets 等 5 個特徵）在 E5b 中 100%（15/15 折）入選前 20 名，置換重要性穩定度達 80.0%。SHAP 依賴圖確認風險驅動方向：獲利與流動性越低風險越高，槓桿指標越高風險越高。
  - 精簡特徵巢狀評估（E5b）：每折於外層訓練集獨立選取 Top k 原始特徵重擬合，在外層驗證折以折內百分位等權集成：
    - $k=10$：3 點預警損失增加 $+0.1335$（95% CI $[0.0401, 0.2269]$，$p_{Holm} = 0.0251$），表現顯著衰退，AR@0.95 惡化至 9.35x。
    - $k=20$：損失差異 $+0.0841$（95% CI $[-0.0083, 0.1765]$，$p_{Holm} = 0.1423$），在本資料下沒有檢出統計差異（檢定力受每折約 35 個正樣本限制），AR@0.95 均值 8.62x。
    - $k=30$：損失差異 $+0.0354$（95% CI $[-0.0619, 0.1327]$，$p_{Holm} = 0.4485$），AR@0.95 均值 8.18x（全特徵 8.04x），極度接近全特徵集。
    - 核心特徵集：以 $\ge 80\%$ 篩選出 12 個核心特徵固化於 `core_features.txt`，並以 $\ge 60\%$ 產出 18 個特徵的擴大核心集於 `core_features_extended.txt`。

- **E6（最終測試集正式評估，嚴格一次性解封）[狀態：已鎖定完成]**：
  - *防護與執行*：經 preflight 8 大項驗證（含 61 項單元測試 100% 通過、前處理 bitwise 比對、訓練索引雜湊無相交），於 2026-10-23 正式解封，建立 `outputs/logs/E6_LOCK`。測試分數固化於 `outputs/e6/e6_test_scores.npz`（SHA256: `ca3b5205ac69d005c39ef6af0bca0521990a1a309932e04a6a401c7e2390f6f1`）。
  - *主要工作點評估*（1,364 筆測試樣本，44 個破產公司，基礎盛行率 3.23%）：
    - $R=0.80$（$t=0.8845$）：$E_A$ 捕獲 37/44 家（84.09%，95% Clopper-Pearson CI $[69.93\%, 93.36\%]$，達標 $\ge 36$），預警 146 家（比例 10.70%，預警倍數 3.32x），精確率 25.34%。
    - $R=0.90$（$t=0.7400$）：$E_A$ 捕獲 43/44 家（97.73%，95% CI $[87.98\%, 99.94\%]$，達標 $\ge 40$），預警 337 家（比例 24.71%，預警倍數 7.66x），精確率 12.76%。
    - $R=0.95$（$t=0.5983$）：$E_A$ 捕獲 44/44 家（100.0%，95% CI $[91.96\%, 100.0\%]$，達標 $\ge 42$），預警 524 家（比例 38.42%，預警倍數 11.91x），精確率 8.40%。
    - $R=1.00$（$t=0.1708$）：$E_A$ 捕獲 44/44 家（100.0%），預警 1,250 家（比例 91.64%，預警倍數 28.41x），精確率 3.52%。
    - 對照組 M3：各工作點捕獲數為 36、43、44、44 家，全數達成要求；全工作點實際召回率與預警倍數均落在 CV 15 折最小～最大預期區間內。
  - *排名指標與配對 Bootstrap*（2,000 次分層重抽樣 95% 百分位 CI）：
    - $E_A$：PR-AUC 0.5676 $[0.4441, 0.7158]$，ROC-AUC 0.9621 $[0.9436, 0.9777]$，AR@0.95 5.25x $[3.93x, 8.18x]$。
    - M3：PR-AUC 0.5275 $[0.4101, 0.6727]$，ROC-AUC 0.9606 $[0.9422, 0.9759]$，AR@0.95 5.16x $[3.89x, 8.20x]$。
    - 單特徵基準（-EPS）：PR-AUC 0.3593 $[0.2459, 0.5045]$，AR@0.95 10.48x $[8.39x, 26.48x]$。
    - 配對差異：$E_A - M3$ 之 PR-AUC 差異 95% CI 為 $[-0.0226, +0.1044]$，ROC-AUC 差異 CI 為 $[-0.0048, +0.0068]$，兩者區間跨越 0 且大幅重疊，統計上**無法區分**；$E_A$ 相對於單特徵基準之 PR-AUC 差值 95% CI 不含 0（$[+0.0743, +0.3427]$）。
  - *機率校準*：Platt Scaling 預測平均破產機率為 3.18%（測試集真實值 3.23%），Brier 0.0243，BSS 0.2207；可靠度圖顯示平均預測機率與經驗破產率趨勢相符（僅 44 個正樣本、5 個分箱，供參考）。詳見 `outputs/e6/`。

- **E7（敏感度與隨機種子穩定性分析）[狀態：完成]**：
  - 盛行率敏感度（E7a）：解析推導 $\pi \in \{0.5\%, 1\%, 2\%, 3.2\%, 5\%\}$ 下的營運指標。假設類別條件特徵分布不變，在母體盛行率為 1% 時：$E_A$ $R=0.90$（$m=0.06$）之精確率為 3.61%（15 折 CV）/ 4.24%（測試集），每捕獲 1 家破產需審查 27.7 家（CV）/ 23.6 家（測試集）；$R=0.95$（$m=0.04$）之精確率為 2.36%（CV）/ 2.70%（測試集），每捕獲 1 家需審查 42.3 家（CV）/ 37.0 家（測試集）。詳見 `outputs/tables/e7a_prevalence_sensitivity.csv` 與 `outputs/figures/e7_prevalence_precision.png`。
  - 隨機種子穩定性（E7b）：對 5 組不同隨機種子（42, 7, 123, 2025, 31337，涵蓋 SMOTE 與模型內部隨機性）重跑外層 15 折，以折內百分位等權集成：
    - 3 點預警損失均值 1.6150，跨種子標準差 0.0147（折均標準誤 $\sigma_{\text{fold}}/\sqrt{15} = 0.0620$，比值 23.7%）。
    - AR@0.95 均值 7.90x，跨種子標準差 0.148x（折均標準誤 0.5496x，比值 26.9%）。
    - PR-AUC 均值 0.4758，跨種子標準差 0.0017（折均標準誤 0.0204，比值 8.3%）。
    - 證實「種子造成的變動遠小於折間變動」。註明：E7b 使用 E2b 最終參數，這些參數是在完整訓練集上調出，因此 E7b 絕對水準偏樂觀，不可當作預期表現；無偏泛化估計以巢狀 CV（E2a/E3）與 E6 為準。E7b 僅用於評估隨機種子穩定性。詳見 `outputs/tables/e7b_seed_stability.csv`。

- **推論工具 `score_new_data.py` [狀態：完成]**：
  - 支援 `--input`、`--output`、`--operating-point`、`--model`。
  - 輸出 CSV 第一列嚴格為欄位名（無檔頭說明），方法學註記獨立存為同名 `*.README.txt`。
  - 自動清除欄名空格、缺欄位嚴格檢核報錯、多餘欄位保留為識別標籤，經 `tests/test_score_new_data.py` 驗證與 `scoring.py` 輸出完全一致（最大差異 0.00e+00）。

## 7. 已知限制
1. **缺乏企業識別碼與時間戳記**：資料集無公司 ID 與財報年份，無法排除同一公司跨年度被重複抽樣的可能性，亦無法執行真實世界的時間序列切分（Out-of-Time 驗證）。
2. **正樣本極度稀缺與高折間噪音**：外層每折僅約 35 個破產正樣本，95% 召回高度取決於邊界 1～2 家樣本，單折評估指標具有不可避免的長右尾變異。
3. **破產盛行率與母體偏離**：資料集破產率 3.23% 遠高於真實商業世界的母體破產率（商業世界常態破產率通常更低），絕對機率與預警倍數在實際落地時需做母體先驗勝算比調整。
4. **ECDF 參考分布在外部資料的代表性未驗證**：目前推論依賴訓練集 10 折 OOF 分布進行百分位轉換，若外部真實資料分布發生宏觀共變異數漂移，百分位映射可能失真。

## 8. 專案狀態：MVP 全量拓展完成 (v2.0.0 Production Ready)
- **測試驗證全數通過**：全套 189 項測試 100% 通過（含無快取乾淨虛擬環境驗證），包含單元測試、FastAPI 整合測試、Streamlit 冒煙測試、產物完整性與 SHA-256 防竄改測試。
- **凍結產物不變性**：`outputs/models/final/`（11 個凍結檔）持續維持零變更；`outputs/models/mvp_extras/` 經 SHA-256 清單驗證。
- **全模組架構落地**：
  1. `src/scoring.py` & `score_new_data.py`：四模型（E_A, M3, M4, M1）與四級門檻（R80, R90, R95, R100）批次推論、20% 缺失率保護與中位數補值。
  2. `src/local_explain.py`：單筆財務個案歸因（M4 TreeSHAP 水平展開與 M1 線性迴歸對照）。
  3. `api/main.py`：FastAPI 批次與解釋服務（10MB 區塊串流防護、1,000 筆上限、本機綁定與嚴格無 NaN JSON）。
  4. `app/app.py` & `src/batch_ui_helper.py`：Streamlit 第 5 分頁「批次評估與示範」（示範模式、CSV 上傳評估、Waterfall 水平貢獻展開、UTF-8-BOM 匯出）。
  5. `scripts/generate_mvp_report_and_card.py` & `MODEL_CARD.md`：程式化模型卡生成器，動態由 15 折 CV 與 E6 分數檔計算所有指標，含 8 大治理限制宣告。
  6. `tests/test_mvp_completeness.py`：模型卡、特徵中位數與產物雜湊完整性自動化測試。

## 9. 實驗階段歷史紀錄
- **E6 鎖定狀態**：
  - 鎖檔位置：`outputs/logs/E6_LOCK`
  - 測試分數固化檔案：`outputs/e6/e6_test_scores.npz`（SHA256: `ca3b5205ac69d005c39ef6af0bca0521990a1a309932e04a6a401c7e2390f6f1`）
  - Git Tags：`pre-e6-unseal`、`post-e6`、`final-experiments`
- **成果彙整入口**：`RESULTS_SUMMARY.md`（統整 E0～E7 所有數據、圖表路徑與限制；由 `scripts/make_results_summary.py` 全自動自 `outputs/` 讀取產生，完整溯源記錄於 `outputs/tables/results_summary_sources.csv`，並由 `tests/test_results_summary.py` 完成自動化驗證）
  - Git Tags：`pre-e6-unseal`、`post-e6`、`final-experiments`、`final-experiments-v2`、`final-experiments-v3`、`final-experiments-v4`
  - v4 重構：報告與摘要中的數字全由 outputs/ 檔案產生（`e6_hindsight_costs.csv` 為新增來源表）；`e6_vs_cv.csv` 的 CV 召回最大值改為嚴格讀取 `e4a_threshold_transfer_per_fold.csv`（R80 兩組為 0.9444，原為預設值 1.0）。

## 9. 真實資料重做時的注意事項（備忘，不於現有資料動手）
1. **資料欄位要求**：真實資料必須具備公司識別碼、財報年度、產業代碼（若有）與破產標籤，才能執行時間序列切分（Out-of-Time）與同公司分組，避免同一家公司跨折洩漏。
2. **特徵定義對齊**：特徵先以 `core_features_extended.txt` 或「特徵家族」層級進行對照，逐一確認每個特徵在資料來源中的定義與計算公式；單期資料無法計算的特徵（例如需要連續四季之 EPS）需另行處理。
3. **外部驗證優先**：先以目前凍結的模型管線直接對真實資料進行外部驗證（觀察特徵分布漂移程度與預測表現衰退幅度），評估後再決定是否在真實資料上重新訓練模型；目前的凍結門檻不可直接沿用於新母體。
4. **配置切換彈性**：重新執行時，`config` 需支援快速切換資料路徑、特徵清單以及切分方式（隨機分層／時間序列／同公司分組）。
5. **破產標籤口徑一致性**：破產定義（如聲請重整、跳票、清算）與預測時間窗（如預測 1 年內、2 年內破產）必須在重做前與真實資料主管方明確釐清並保持一致。

## 10. 下一步
- **下一個執行步驟：最終專案技術報告撰寫或使用者後續指令**
- **唯一數據入口**：`RESULTS_SUMMARY.md`。
