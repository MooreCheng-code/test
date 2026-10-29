# 模組導讀：門檻轉移、機率校準與最終元件固化（thresholding.py，實驗 E4）

## 一句話用途
評估跨折門檻轉移能力並引入安全邊際 $m$、比對 Platt 與 Isotonic 機率校準之保序性，並在全量訓練集上一次性擬合與固化 11 項可部署的正式模型元件。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「門檻調校、機率校準與生產固化層（E4 實驗）」。
- **上游**：接收 E3 產出的 $E_A$ 與 M3 的 15 次外層評估分數（5 折 × 3 次重複），以及 `outputs/models/best_params/*.json`。
- **下游**：產出 `outputs/models/final/` 下的 11 個固化檔案，直接提供正式推論（`src/scoring.py`、`score_new_data.py`）、Web 儀表板（`app/`）與最終解封測試（E6，`src/final_eval.py`）載入。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - 訓練集 OOF 預測矩陣與標籤 `y_train`（長度 5,455）。
  - 全量訓練集特徵 `X_train` `(5455, 95)`。
- **輸出**：
  - 門檻跨折分析表：`outputs/tables/e4a_threshold_transfer_summary.csv`、`outputs/tables/e4a_margin_recommendation.csv`。
  - 機率校準比對表：`outputs/tables/e4b_calibration_summary.csv`。
  - 最終生產固化目錄：`outputs/models/final/*`（包含 11 個核心元件檔案）。

---

## 核心演算法與設計

### 1. 門檻定義、安全邊際 $m$ 與跨折轉移（E4a，`bankruptcy/src/thresholding.py:120-250`）
- **做什麼**：在歷史資料（訓練折）求出的破產預警門檻，直接套用到未來的未知資料（驗證折）時，往往會因為資料波動而無法達成預期召回率（達標率低於 80%）。本模組透過跨折模擬尋找最佳的「安全邊際（Safety Margin, $m$）」，確保在未知資料上能以 $\ge 80\%$ 的機率穩健達標。
- **門檻與安全邊際公式**：
  若在訓練折達到目標召回率 $R$ 的原始門檻為 $\theta_{\text{raw}}$，則引入安全邊際後的實際操作門檻為：
  $$\theta_{\text{operating}} = \theta_{\text{raw}} - m$$
  門檻下調 $m$，意味著發出更多警報，寧可稍微增加調查成本，也要確保「召回率底線」不被突破。
- **專案實測安全邊際推薦（`outputs/tables/e4a_margin_recommendation.csv` 原文）**：
  針對主力模型 $E_A$：
  - **R80 工作點**：推薦安全邊際 $m = \mathbf{0.02}$，跨折召回達標率由無邊際的不足 75% 提升至 **80.0%**（平均實際召回率 82.5%）。
  - **R90 工作點**：推薦安全邊際 $m = \mathbf{0.06}$，跨折達標率提升至 **93.3%**（平均實際召回率 95.4%）。
  - **R95 工作點**：推薦安全邊際 $m = \mathbf{0.04}$，跨折達標率提升至 **93.3%**（平均實際召回率 98.7%）。
  - **R100 工作點**：因 $R=1.00$ 門檻完全受限於全市場最難抓的單一極端企業，固定設定 $m = \mathbf{0.00}$（達標率 80.0%）。

---

### 2. Platt Scaling vs. Isotonic Regression（E4b，`bankruptcy/src/thresholding.py:270-390`）

專案在 15 次外層評估中，嚴格比對兩種主流機率校準演算法對排序品質與極端尾部門檻的影響：

#### 實測比對表格（來自 `outputs/tables/e4b_calibration_summary.csv`）

| 校準方法 | Brier Skill Score (`BSS`) | AR@0.95 預警倍數 | 是否 100% 保序 | 輸出唯一值數量 (`unique_values_mean`) | 行為特性與影響 |
|---|:---:|:---:|:---:|:---:|---|
| **未校準（ECDF 百分位）** | 不適用 (N/A) | **8.06x** | 是 | 1,091.0 | 原始連續排序，無機率語意 |
| **Platt 縮放（Sigmoid）** | **0.1839** | **8.06x** | **✅ 100% 嚴格保序** | **1,091.0** | 嚴格單調遞增，完全保留原始排序，預警倍數完全不變 |
| **等保回歸（Isotonic）** | **0.2650** | **10.02x** | ❌ 產生階梯同分 | **23.9** | 階梯式常數區間，產生大量尾部同分，預警成本暴增 24% |

#### 為什麼等保回歸（Isotonic）會破壞高危險門檻？
- 等保回歸是非參數階梯函數，容易將多個不同分數的樣本合併為同一個區間機率值。在驗證折的 1,091 筆樣本中，Isotonic 將其壓縮至僅剩 **約 23.9 個唯一值**！
- 當門檻剛好落在一個同分區間時，依據專案的保守同分規則（`score >= thr`，[metrics.md](metrics.md)），系統必須將該區間內的所有樣本全數發出警報，直接導致 **AR@0.95 從 8.06x 暴增至 10.02x**。
- **最小數字例子（使用 scikit-learn 實際執行）**：
  ```python
  from sklearn.isotonic import IsotonicRegression
  import numpy as np

  y = np.array([0, 0, 0, 1, 0, 0, 1, 1])
  s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
  iso = IsotonicRegression(out_of_bounds="clip")
  iso_probs = iso.fit_transform(s, y)
  # 實測輸出: [0.0, 0.0, 0.0, 0.3333, 0.3333, 0.3333, 1.0, 1.0]
  # 原始 8 個連續相異值被壓縮至僅剩 3 個階梯值！
  ```
  原本分數為 0.4、0.5、0.6 的三間公司機率全部變成 0.3333。若門檻切在 0.3333，三人全部被視為同分，系統無法分辨誰更危險。

#### 架構決策結論
專案確立「**雙軌分離原則**」：
1. **決策與警報分流**：嚴格以未校準的連續 ECDF 分位數排名（或 Platt 單調映射）設定門檻，確保排序純度；
2. **商業報表呈現**：僅使用 Platt Scaling 輸出實質後驗破產機率供人類查閱，杜絕 Isotonic 的階梯扭曲。

---

### 3. E4c 全量生產元件固化清單（`outputs/models/final/`）
在 E4c 階段（`bankruptcy/src/thresholding.py:600-845`），程式在全部 5,455 筆訓練資料上重新訓練並固化可直接部署的 **11 個核心元件檔案**：

| 元件檔案名稱 | 格式 | 內容與角色 | 載入模組 |
|---|:---:|---|---|
| `model_M3.joblib` | Joblib | 全訓練集擬合之 M3 Pipeline（HistGradientBoosting） | `scoring.py` |
| `model_M4.joblib` | Joblib | 全訓練集擬合之 M4 Pipeline（XGBoost） | `scoring.py` |
| `model_M5.joblib` | Joblib | 全訓練集擬合之 M5 Pipeline（LightGBM） | `scoring.py` |
| `reference_M3.npy` | NumPy | 5 折 × 2 次重複全訓練集 OOF 預測分佈（長度 5,455），供 ECDF 映射 | `scoring.py` |
| `reference_M4.npy` | NumPy | M4 參考分佈（長度 5,455） | `scoring.py` |
| `reference_M5.npy` | NumPy | M5 參考分佈（長度 5,455） | `scoring.py` |
| `reference_y.npy` | NumPy | 全訓練集真實目標向量（長度 5,455） | `scoring.py` |
| `calibrator_platt.joblib` | Joblib | 主要機率校準器（LogisticRegression Sigmoid） | `scoring.py` |
| `calibrator_isotonic.joblib` | Joblib | 備用校準器（IsotonicRegression） | 探索備份 |
| `thresholds.json` / `.csv` | JSON/CSV | 4 個工作點（R80, R90, R95, R100）之固化操作門檻與邊際 | `scoring.py`、`app/` |
| `version_metadata.json` | JSON | 固化時間戳記、隨機種子、訓練筆數與無偏性法律聲明 | 稽核與版本控管 |

---

### 4. 已知邊界情況：`freeze_final_components` 快取命中分支之 `UnboundLocalError`（F-01）
- **位置**：`bankruptcy/src/thresholding.py:631-636` 與 `672`
- **嚴重度**：**低（Low）**
- **觸發條件與現象**：
  在 `freeze_final_components()` 函式中，第 631～636 行檢查磁碟上是否已有各模型的快取檔案（`reference_{m}.npy`、`_tmp_oof_r0_{m}.npy` 等）：
  ```python
  if ref_file.exists() and oof_r0_file.exists() and oof_r1_file.exists():
      oof_r0[m] = np.load(oof_r0_file)
      oof_r1[m] = np.load(oof_r1_file)
      final_references[m] = np.load(ref_file)  # 第 634 行
      continue
  ```
  然而區域變數 `final_references = {}` 延遲至迴圈外的**第 672 行**才初始化。如果磁碟上已經存在這些快取檔案且使用者「再次執行」此函式，程式會在第 634 行拋出 `UnboundLocalError`。
- **為什麼正常流程完全不會遇到？**
  1. `freeze_final_components()` 是離線一次性生成腳本，在初始空白目錄下一氣呵成執行完成。
  2. 線上推論與正式生產環境（`bankruptcy/src/scoring.py:86`、`bankruptcy/score_new_data.py:102`）**僅會呼叫 `load_final_components()`**，它只讀取成品，絕不會呼叫 `freeze_final_components()`。
- **守則宣告**：**不要在已固化的專案環境中重新執行 `freeze_final_components()`**！

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **生產門檻設定依據** | A. 直接採用訓練折門檻（$m=0$）<br>B. 引入安全邊際（$m>0$）下調門檻 | **方案 B（安全邊際 $m$）** | $m=0$ 時跨折泛化達標率不足 80%；引入安全邊際使 R90 與 R95 達標率達到 93.3%，大幅降低漏報倒閉企業的致命風險。 | `bankruptcy/src/thresholding.py:120-250`；`outputs/tables/e4a_margin_recommendation.csv` |
| **機率校準演算法** | A. Isotonic Regression<br>B. Platt Scaling (Sigmoid) | **方案 B（Platt Scaling）** | Isotonic 雖然 Brier 分數略優，但階梯同分使連續排序破裂，AR@0.95 惡化至 10.02x；Platt 嚴格保持連續排序（AR 維持 8.06x）。 | `outputs/tables/e4b_calibration_summary.csv` |

---

## 機器學習概念白話框

> **白話概念：機率校準（Probability Calibration）**
> - **什麼叫沒有校準？** 一個模型預測某公司破產風險分數為「0.85」，這代表它「有 85% 機率破產」嗎？不是！在樹模型中，0.85 可能只是某個葉節點的雜湊比率，該公司真實破產機率可能只有 4%。
> - **Platt 縮放做了什麼？** 它在模型的原始輸出分數與真實標籤之間，擬合一條 Sigmoid S 型平滑曲線（一維 Logistic Regression），將抽象的排序分數精準拉回符合真實市場破產率（3.23%）的後驗機率尺度。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「Platt 縮放」與「Brier 分數」。

---

## 本模組的單元測試
對應單元測試涵蓋：
1. `bankruptcy/tests/test_thresholds_reproduce.py`：驗證固化之 `thresholds.json` 能被 `scoring.py` 正確載入並產出一致的預警決策。
2. `bankruptcy/tests/test_results_summary.py`：驗證 E4 表格檔案之欄位完整性與命名規範。

---

## 讀碼練習
1. **練習 1**：在 `derive_operating_threshold` 中，為什麼安全邊際 $m$ 是以「減去 $m$（$\theta - m$）」的方式操作，而不是「加上 $m$」？
2. **練習 2**：在 `e4b_calibration_summary.csv` 中，為什麼未校準的 ECDF Percentile 其 Brier 分數欄位為 `NaN`？
3. **練習 3**：根據 F-01，如果在磁碟上已經存在 `reference_M3.npy` 的情況下再次執行 `freeze_final_components()`，Python 會在第幾行拋出什麼異常？如何從根本上修復它？（提示：變數初始化順序）

---

## 延伸：想改這裡時要注意什麼
1. **不得手動修改 `thresholds.json` 與 `thresholds.csv`**：任何門檻數值的變動會直接破壞前向相容性與後續 E6 測試集的驗證邏輯。
2. **生產推論依賴完整 11 個元件**：`outputs/models/final/` 內的任何檔案均不得個別單獨刪除或更名。
