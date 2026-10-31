# 模組導讀：資料前處理管線（preprocessing.py）

## 一句話用途
以嚴格相容 scikit-learn API 的類別架構，封裝特徵清洗、縮尾處理、分位數常態化與高相關過濾等 7 大轉換器（Transformers），徹底阻絕資料洩漏。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「資料前處理與特徵管線層」。
- **上游**：接收由 `bankruptcy/src/cv.py` 切分出的純訓練集折（$X_{\text{train}}$）或推論新資料。
- **下游**：將清洗並轉換後的特徵矩陣（形狀由 95 欄依序過濾為 70～73 欄）餵入分類器（M1～M5）或特徵重要性分析模組（E5）。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - `X`: `pd.DataFrame`，訓練時形狀為 `(4364, 95)`（外層折）或 `(5455, 95)`（全訓練集）。
  - `y`: `pd.Series` 或 `None`（特徵轉換器多數為非監督轉換）。
- **輸出**：
  - `transform(X)`: `pd.DataFrame`，全訓練集轉換後形狀固定為 `(5455, 72)`；外層折形狀為 `(4364, 70~73)`。

---

## 核心轉換器（Transformers）

### 1. `DropConstantAndDuplicate`（`bankruptcy/src/preprocessing.py:132-181`）
- **做什麼**：在 `fit` 階段找出訓練資料中的「0 變異常數欄位」與「數值 100% 完全重複的欄位」，並在 `transform` 階段將其剔除。
- **重要屬性**：
  - `self.constant_cols_`: 訓練集常數欄位清單（實測固定為 `['Net Income Flag']`，1 欄）。
  - `self.duplicate_cols_`: 訓練集重複欄位清單（實測固定為 `['Current Liability to Liability', 'Current Liability to Equity']`，2 欄）。
- **三個核心問題**：
  - *它解決什麼問題？* 徹底清除資訊量為 0 的常數與完全重複特徵，避免下游矩陣運算奇異或浪費計算資源。
  - *沒有它會出什麼錯？* 常數欄位在 Z-score 標準化時標準差為 0，會引發除以零錯誤產生 `NaN`；完全重複欄位會使未正則化的線性迴歸產生奇異共線性。
  - *有沒有更簡單的做法，為什麼沒選？* 可以用 `df.drop(columns=...)` 硬編碼欄位名稱，但這違反動態防禦原則；封裝成 Transformer 能在未來換用全新資料集時自動動態辨識無效欄位。
- **重點程式碼（`bankruptcy/src/preprocessing.py:145-167`）**：
  ```python
  def fit(self, X: pd.DataFrame, y=None):
      # 找出常數特徵
      self.constant_cols_ = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]

      # 找出完全重複特徵
      self.duplicate_cols_ = []
      seen = set()
      for i, c1 in enumerate(X.columns):
          if c1 in seen or c1 in self.constant_cols_: continue
          for c2 in X.columns[i + 1:]:
              if c2 in seen or c2 in self.constant_cols_: continue
              if X[c1].equals(X[c2]):
                  self.duplicate_cols_.append(c2)
                  seen.add(c2)

      self.drop_cols_ = self.constant_cols_ + self.duplicate_cols_
      self.feature_names_out_ = [c for c in X.columns if c not in self.drop_cols_]
      return self
  ```
- **逐段解說**：
  - `fit()` 嚴格只在訓練集上學習哪幾欄是常數與重複（第 147、157 行），將刪除名單鎖定在 `self.drop_cols_`。
  - `transform()`（第 174 行）僅呼叫 `X.drop(columns=self.drop_cols_, errors="ignore")`，確保測試集永遠套用訓練集的剔除決策，防範標籤或樣本洩漏。

---

### 2. `QuantileScaler`（`bankruptcy/src/preprocessing.py:287-336`）
- **做什麼**：使用 `QuantileTransformer` 將偏態嚴重的特徵投影至標準常態分佈，或使用 `StandardScaler` 進行標準化。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `kind` | `str` | `"quantile"` | 縮放模式，可選 `"quantile"` 或 `"standard"` |
- **三個核心問題**：
  - *它解決什麼問題？* 財務比率普遍具備極端厚尾特性，少數破產公司的極端財務比率會拉偏常規統計量，使梯度下降與距離運算退化。
  - *沒有它會出什麼錯？* 若使用一般的 Min-Max 或 Z-score，99% 的公司數值會被壓成一條直線，模型無法捕捉關鍵中位區間的細微變化。
  - *有沒有更簡單的做法，為什麼沒選？* 可以取對數（`log1p`），但財務比率大量包含負數（例如負營運資金、虧損淨利率），對數轉換必須處理負值平移；`QuantileTransformer` 依排序百分比轉換，天然不受負值與極端值干擾。
- **重點程式碼（`bankruptcy/src/preprocessing.py:303-324`）**：
  ```python
  def fit(self, X: pd.DataFrame, y=None):
      self.feature_names_in_ = list(X.columns)

      if self.kind == "quantile":
          n_quantiles = min(1000, len(X))
          self.scaler_ = QuantileTransformer(
              output_distribution="normal",
              n_quantiles=n_quantiles,
              random_state=SEED,
          )
      elif self.kind == "standard":
          self.scaler_ = StandardScaler()
      self.scaler_.fit(X)
      return self
  ```
- **重要發現與差異（F-03 深度分析）**：
  > [!NOTE]
  > **F-03：`n_quantiles` 的動態防護**
  > - 在 `bankruptcy/src/preprocessing.py:311` 中，程式寫為 `n_quantiles = min(1000, len(X))`。
  > - **為什麼要這樣寫？**
  >   當輸入的樣本數 $N < 1000$ 時（例如在微型單元測試或單一批次推論），若直接指定 `n_quantiles=1000`，scikit-learn 會強制降低分位數並拋出 `UserWarning: n_quantiles is greater than the total number of samples`。
  >   此處透過 `min(1000, len(X))`，當樣本足夠大時（全訓練集 $5,455$ 筆、外層訓練折 $4,364$ 筆）使用 1000 分位數；當樣本小於 1000 時自動降為樣本數，優雅消除警告。

---

### 3. `CorrelationFilter`（`bankruptcy/src/preprocessing.py:338-433`）
- **做什麼**：以貪婪演算法（Greedy Algorithm）檢查所有特徵配對的 Pearson 相關係數，當兩特徵絕對相關 $|r| > \text{threshold}$ 時，剔除後者。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `threshold` | `float` | `0.95` | 高共線性過濾門檻 |
- **三個核心問題**：
  - *它解決什麼問題？* 財務報表存在大量會計恆等式（例如「流動負債/總負債」與「流動負債/淨值」高度同動），高共線性會破壞線性模型的係數解釋力與數值穩定性。
  - *沒有它會出什麼錯？* Logistic Regression（M1）在特徵矩陣多重共線性下無法收斂，或係數符號產生違反經濟學直覺的荒謬反轉。
  - *有沒有更簡單的做法，為什麼沒選？* 可以用主成分分析（PCA），但 PCA 會將所有原始特徵混合成不具可解釋性的正交旋轉維度，面試官與金融風控團隊無法接受無法解釋的「PC1、PC2」作為核貸依據；貪婪過濾保留真實財務欄位名稱。
- **重點程式碼（`bankruptcy/src/preprocessing.py:365-385`）**：
  ```python
  # 計算相關係數矩陣
  corr_matrix = X.corr().abs()

  # 貪婪挑選剔除欄位
  drop_cols = set()
  for i in range(len(columns)):
      col_i = columns[i]
      if col_i in drop_cols: continue
      for j in range(i + 1, len(columns)):
          col_j = columns[j]
          if col_j in drop_cols: continue
          if corr_matrix.iloc[i, j] > self.threshold:
              drop_cols.add(col_j)  # 剔除後者

  self.drop_cols_ = sorted(list(drop_cols))
  self.feature_names_out_ = [c for c in columns if c not in self.drop_cols_]
  ```
- **逐段解說**：
  - 第 365 行在 `fit()` 中由訓練折特徵矩陣計算 Pearson 相關矩陣。
  - 第 370～380 行採用雙重迴圈由前向後比對，一旦發現 $|r| > 0.95$，固定將後者 `col_j` 加入剔除名單。
  - **本專案實測過濾數據**：
    - 在全量訓練集（5,455 筆）上，此步驟精確剔除 20 個高相關特徵，將欄位由 92 欄降至 **72 欄**。
    - 在 15 次外層交叉驗證折中（每折 4,364 筆），每折剔除 19～22 個特徵，保留 70～73 欄（平均 71.87 欄）。

---

### 4. 其他輔助轉換器
- **`TrainingMedianImputer`（`bankruptcy/src/preprocessing.py:34-87`）**：
  - 記錄訓練集各欄位中位數。雖然原始 `data.csv` 無遺失值，但生產環境 API（`app/`、`api/main.py`）可能收到缺漏欄位，此時以訓練集中位數安全填補。
- **`Winsorizer`（`bankruptcy/src/preprocessing.py:183-236`）**：
  - 在訓練集上計算 1% 與 99% 分位數閾值，將極端值夾在閾值之內（雙邊縮尾）。
- **`LargeValueIndicator`（`bankruptcy/src/preprocessing.py:238-285`）**：
  - 檢測數值超過指定臨界值（如 $10^6$）的欄位，派生出 `_is_large` 二元指示特徵。
- **`FeatureSelector`（`bankruptcy/src/preprocessing.py:89-130`）**：
  - 依據給定的特徵名稱清單選取子集，專供 E5 評估 Top 10、Top 20 精簡特徵集時使用。

---

## 特徵處理漏斗圖（Feature Funnel）

```
[原始資料 6,819 筆]
  │ (切分 80% 訓練 / 20% 測試)
  ▼
[訓練資料集 5,455 筆 × 96 欄]
  │ (移除目標欄位 'Bankrupt?')
  ▼
[初始特徵集 95 欄]
  │
  ├─ DropConstantAndDuplicate
  │    ├─ 剔除常數: Net Income Flag (1 欄)
  │    └─ 剔除完全重複: Current Liability to Liability, Current Liability to Equity (2 欄)
  ▼
[清洗後特徵集 92 欄]
  │
  ├─ QuantileScaler (分位數常態化，欄位數量不變)
  ▼
[分位數常態化特徵 92 欄]
  │
  ├─ CorrelationFilter (門檻 |r| > 0.95)
  │    └─ 全訓練集貪婪剔除: 20 欄
  │       (外層 15 次 CV 折每折剔除 19～22 欄)
  ▼
[最終建模特徵集 72 欄] (外層 CV 折為 70～73 欄，均值 71.87 欄)
```

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **特徵共線性處理** | A. PCA 主成分降維<br>B. CorrelationFilter (門檻 0.95 貪婪過濾)<br>C. 不做過濾交由正則化處理 | **方案 B（CorrelationFilter）** | PCA 產生的合成正交維度失去經濟學意義，無法在金融風控系統中向授信委員會解釋；相關過濾精確剔除 20 個冗餘維度，保留清晰的原生財務欄位。 | `bankruptcy/src/preprocessing.py:338-433`；`outputs/tables/data_audit_summary.json` |
| **資料縮放與常態化** | A. StandardScaler (Z-score)<br>B. MinMaxScaler<br>C. QuantileScaler (常態輸出) | **方案 C（QuantileScaler）** | 財務比率存在極端厚尾分佈（Kurtosis 極高），Z-score 與 Min-Max 會被極值壓縮至幾乎無區分度；QuantileScaler 投影為標準常態分佈，使神經網路與距離度量表現大幅提升。 | `bankruptcy/src/preprocessing.py:287-336`；E1 實驗對照 |

---

## 機器學習概念白話框

> **白話概念：前處理中的資料洩漏（Data Leakage）**
> - **什麼是資料洩漏？** 模型的任務是預測「未來的未知資料」。如果在處理資料時，不小心把「驗證集或測試集的統計資訊（如平均值、分位數、相關矩陣）」偷看進來，訓練出的模型就會出現「考試前偷看過答案」的虛假優秀現象，實際上線時必然慘敗。
> - **本專案如何做到 100% 阻絕洩漏？**
>   1. 永遠先切分出訓練折，所有 Transformer 的 `.fit()` 只能傳入訓練折（$X_{\text{train}}$）。
>   2. 驗證折（$X_{\text{val}}$）與測試折（$X_{\text{test}}$）只能調用 `.transform()`，套用訓練折算好的常數、中位數與剔除名單。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「資料洩漏」與「非監督前處理轉換」。

---

## 本模組的單元測試
對應單元測試涵蓋 `bankruptcy/tests/test_preprocessing.py` 內 12 項測試：
1. `test_drop_constant_and_duplicate_real_data`：驗證常數與重複特徵精確剔除。
2. `test_correlation_filter_logic`：驗證相關性高於門檻時嚴格剔除後者。
3. `test_quantile_scaler_fit_a_transform_b`：驗證在集合 A 上 fit 後轉換集合 B，不發生資料洩漏且輸出形狀一致。
4. `test_pipeline_assembly_and_clone`：驗證轉換器具備 scikit-learn 規範的 `clone()` 與 `Pipeline` 組裝能力。

---

## 讀碼練習
1. **練習 1**：在 `CorrelationFilter.fit()` 中，若欄位 X1 與 X2 的相關係數為 0.98，欄位 X2 與 X3 的相關係數為 0.96，程式碼會剔除哪些欄位？（提示：依序檢查，一旦欄位被加入 `drop_cols` 後是否會跳過？）
2. **練習 2**：在 `QuantileScaler` 中，為什麼需要設定 `random_state=SEED`？如果分位數轉換不固定隨機種子，會對模型預測帶來什麼後果？
3. **練習 3**：如果在全量資料（6,819 筆）上計算相關係數剔除欄位，然後才進行 5 折切分，這違反了機器學習的什麼鐵律？會對驗證分數產生什麼影響？

---

## 延伸：想改這裡時要注意什麼
1. **禁止在 `transform()` 中動態計算統計量**：所有的統計量（中位數、相關矩陣、最大最小值）必須且只能在 `fit()` 中記錄於 `self.*_` 屬性中。
2. **輸出格式必須維持 DataFrame**：下游管線與特徵名稱追蹤（`feature_names_out_`）高度依賴欄位名稱，自訂轉換器時回傳型別必須維持 `pd.DataFrame`。
