# 模組導讀：資料審計模組（data_audit.py）

## 一句話用途
在進入正式建模前對原始資料執行「全身健康檢查」，診斷遺失值、常數欄位、重複欄位、高相關特徵、單變量預測力與潛在樣本洩漏（近鄰重複樣本）。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「探索與審計層（E0 實驗）」。
- **上游**：讀取 `bankruptcy/data/data.csv` 原始資料（形狀 `(6819, 96)`）。
- **下游**：審計結果直接指引 `bankruptcy/src/preprocessing.py` 的特徵清洗規則（常數刪除、完全重複刪除、高度相關過濾）。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - `df`: `pd.DataFrame`，形狀 `(6819, 96)`，包含 95 個財務指標與 1 個目標欄位 `Bankrupt?`。
  - `target_col`: `str`，預設為 `"Bankrupt?"`。
- **輸出**：
  - 控制台格式化審計報告（由 `print_audit_report()` 輸出）。
  - 審計數值摘要：`outputs/tables/data_audit_summary.json`。
  - 欄位層級明細表：`outputs/tables/data_audit_column_details.csv`。
  - 審計視覺化圖表：`outputs/figures/data_audit_target_distribution.png`、`outputs/figures/data_audit_correlation_heatmap.png`、`outputs/figures/data_audit_top20_auc.png`。

---

## 核心函式與類別

### 1. `audit_column_anomalies()`（`bankruptcy/src/data_audit.py:99-177`）
- **做什麼**：全面清查特徵欄位中的「常數欄位（0 變異）」、「完全重複欄位」與「高相關性特徵配對（預設 $|r| > 0.95$）」。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `df` | `pd.DataFrame` | （必填） | 審計資料集 |
  | `features` | `list[str]` | （必填） | 欲檢查的特徵名稱清單 |
  | `threshold_corr` | `float` | `0.95` | 判定為強相關的 Pearson 相關係數絕對值門檻 |
- **回傳**：`dict`，包含 `constant_cols`、`duplicate_cols`、`high_corr_pairs` 與 `high_corr_cols` 等清單。
- **三個核心問題**：
  - *它解決什麼問題？* 找出毫無資訊量的常數特徵與提供重複資訊的冗餘特徵，預防共線性導致線性模型（M1）數值不穩定。
  - *沒有它會出什麼錯？* 常數欄位（如 `Net Income Flag` 全為 1）在部分模型標準化時會產生除以 0 的 `NaN` 錯誤；高度共線性則會膨脹係數變異數。
  - *有沒有更簡單的做法，為什麼沒選？* 可以直接仰賴樹模型自身的特徵篩選能力，但專案架構包含 Logistic Regression，且減少冗餘特徵能大幅加快超參數尋優速度。
- **重點程式碼（`bankruptcy/src/data_audit.py:108-132`）**：
  ```python
  # 1. 常數欄位（唯一值 <= 1）
  constant_cols = [c for c in features if df[c].nunique() <= 1]

  # 2. 完全重複欄位（內容 100% 相同）
  duplicate_cols = []
  seen = set()
  for i, c1 in enumerate(features):
      if c1 in seen:
          continue
      for c2 in features[i + 1:]:
          if c2 in seen:
              continue
          if df[c1].equals(df[c2]):
              duplicate_cols.append((c1, c2))
              seen.add(c2)

  # 3. 高相關性配對 (|r| > threshold_corr)
  corr_matrix = df[features].corr().abs()
  ```
- **逐段解說**：
  - 第 108 行：透過 `df[c].nunique() <= 1` 快速辨識常數欄位，本專案實測抓出 `Net Income Flag`（唯一值僅有 1）。
  - 第 111～122 行：使用雙重迴圈與 `equals()` 檢查兩欄數值是否完全一致，本專案實測抓出 2 對完全重複欄位（例如 `Current Liability to Liability` 與 `Current Liability to Equity`）。
  - 第 125～132 行：計算相關矩陣，篩選出 $|r| > 0.95$ 的高共線性特徵配對。
- **常見誤用或陷阱**：
  此處的相關係數是在「全資料集」上執行，僅供「探索性審計」！在正式建模的前處理管線中，**相關矩陣必須在訓練折內部計算**，嚴禁在全體資料集上計算特徵剔除清單，否則即構成資料洩漏（Data Leakage）。

---

### 2. `audit_near_duplicates()`（`bankruptcy/src/data_audit.py:223-334`）
- **做什麼**：使用常態分位數轉換（`QuantileTransformer`）搭配歐幾里得距離近鄰演算法（`NearestNeighbors`），檢測資料集中是否存在近乎相同的重複樣本，並評估標籤一致性。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `df` | `pd.DataFrame` | （必填） | 審計資料集 |
  | `features` | `list[str]` | （必填） | 參與計算的特徵名稱清單 |
  | `target_col` | `str` | （必填） | 目標標籤名稱 |
- **回傳**：`dict`，包含距離分位數（p1, p5, p25）、近鄰樣本標籤衝突比例與最相近前 10 筆樣本配對。
- **三個核心問題**：
  - *它解決什麼問題？* 檢驗資料集是否包含同間公司不同年份的近乎複製樣本，防止同一實體同時落入訓練集與驗證集造成虛假高分。
  - *沒有它會出什麼錯？* 若存在重複樣本且標籤相反（Label Noise），模型將無法收斂；若標籤相同且散落在 Train/Val，則 CV 分數會被嚴重高估。
  - *有沒有更簡單的做法，為什麼沒選？* 可以只比對 `df.duplicated()`，但真實世界資料常有微小的四捨五入誤差，必須透過高維空間的距離分佈才能抓出「近似重複（Near-duplicate）」。
- **重點程式碼（`bankruptcy/src/data_audit.py:230-244`）**：
  ```python
  y = df[target_col].values

  # Quantile normalization to normal distribution
  qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
  X_trans = qt.fit_transform(df[features])

  # 2 nearest neighbors (1st neighbor is self with dist 0, 2nd is true nearest neighbor)
  nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
  nn.fit(X_trans)
  distances, indices = nn.kneighbors(X_trans)

  nn_dist = distances[:, 1]
  nn_idx = indices[:, 1]
  ```
- **逐段解說**：
  - 第 231 行使用 `QuantileTransformer(output_distribution="normal")` 將偏態極大的財務比率映射至高斯分佈，避免少數離群值主導歐氏距離。
  - 第 235 行設定 `n_neighbors=2`，因為第 1 個近鄰永遠是樣本自身（距離為 0），第 2 個近鄰（索引 1）才是該樣本在特徵空間中的真實最近鄰居。
  - 第 241～242 行提取真實近鄰距離與樣本索引，並進一步比對兩者的 `y` 標籤是否一致。
- **重要發現與差異（F-03 深度分析）**：
  > [!NOTE]
  > **F-03：`data_audit.py` 與 `preprocessing.py` 中 `QuantileTransformer` 的參數差異**
  > - **程式碼差異**：
  >   - 在 `data_audit.py:231` 中，呼叫方式為 `QuantileTransformer(output_distribution="normal", random_state=SEED)`，未傳入 `n_quantiles` 參數，因此採用 scikit-learn 預設值 `n_quantiles=1000`。
  >   - 在 `preprocessing.py:311` 中，呼叫方式為 `n_quantiles = min(1000, len(X))`，動態依樣本數量進行上限截斷。
  > - **為什麼會這樣寫？在何種條件下會出現警告？**
  >   - scikit-learn 的規範要求 `n_quantiles <= n_samples`。當樣本總數小於 `n_quantiles` 時，scikit-learn 會強制降低分位數數量並噴出 `UserWarning: n_quantiles is greater than the total number of samples`。
  >   - 在 `data_audit.py` 中，處理的是全量原始資料（$N = 6,819$ 筆），$6,819 \ge 1000$，因此在全量資料審計時**未曾會觸發警告**。
  >   - 然而，若學習者在單元測試或小規模玩具樣本（例如 $N < 1000$）上直接調用未加保護的寫法，就會觸發 `UserWarning`。`preprocessing.py` 考量到未來可能接受微型批次推論或小樣本子集，故加入了 `min(1000, len(X))` 防禦性保護。

---

### 3. `audit_univariate_signals()`（`bankruptcy/src/data_audit.py:179-221`）
- **做什麼**：針對每一個特徵，計算單變量 ROC-AUC 與 Mann-Whitney U 檢定的 p 值，評估單一特徵對企業破產的獨立區別能力。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `df` | `pd.DataFrame` | （必填） | 審計資料集 |
  | `features` | `list[str]` | （必填） | 特徵清單 |
  | `target_col` | `str` | （必填） | 目標標籤名稱 |
- **回傳**：`pd.DataFrame`，按單變量 AUC 降冪排序，列出各特徵之 AUC、U 統計量與 p-value。
- **重點程式碼（`bankruptcy/src/data_audit.py:192-205`）**：
  ```python
  for f in features:
      vals = df[f].values
      # AUC 計算：若特徵與目標呈負相關（AUC < 0.5），取 max(auc, 1-auc)
      raw_auc = roc_auc_score(y, vals)
      directed_auc = max(raw_auc, 1.0 - raw_auc)
      stat, p_val = stats.mannwhitneyu(
          vals[y == 1], vals[y == 0], alternative="two-sided"
      )
  ```
- **逐段解說**：
  - 第 195～196 行：財務比率中，有些指標越高越安全（例如淨利率），有些越高越危險（例如負債比率）。`max(raw_auc, 1.0 - raw_auc)` 衡量的是指標的「純粹區分能力」，不論其物理方向為何。
  - 第 197～199 行：使用非參數的 Mann-Whitney U 檢定，不假設特徵服從常態分佈，檢定破產組與非破產組在中位數上是否存在統計顯著差異。

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **異常欄位檢驗時機** | A. 在前處理管線中默默刪除<br>B. 先於獨立審計腳本中完整驗證與出圖，再交由前處理執行 | **方案 B（獨立審計）** | 先驗證資料全貌能讓工程師掌握常數欄位（1 欄）與完全重複欄位（2 欄）的業務意義，並確保審計結論具備透明度與可重現性。 | `bankruptcy/src/data_audit.py:417-488`；`outputs/tables/data_audit_summary.json` |
| **近鄰樣本檢測空間** | A. 原始財務比率直接算歐氏距離<br>B. QuantileTransformer 常態化後計算 | **方案 B（QuantileTransformer 常態化）** | 財務比率存在極端厚尾與量綱懸殊（如資產金額 vs 比率），直接計算歐氏距離會被少數極端值壟斷；常態分位數轉換可將所有維度投影至相同刻度。 | `bankruptcy/src/data_audit.py:230-234`；`outputs/figures/data_audit_correlation_heatmap.png` |

---

## 機器學習概念白話框

> **白話概念：分位數常態化（QuantileTransformer）與厚尾分佈**
> - **什麼是厚尾分佈？** 財報數據往往極不均勻，多數公司的負債比在 0.3～0.6 之間，但少數瀕臨破產或特殊產業的公司負債比可能高達 50 甚至 500。如果使用一般標準化（Z-score），少數極大值會把平均值與標準差拉偏，導致 99% 的正常樣本被壓縮在 0 附近。
> - **QuantileTransformer 做了什麼？** 它不看數值的絕對差距，只看「排第幾名」（百分位數）。它把第 50% 名的樣本映射到標準常態分佈的 0，第 84% 名映射到 1，第 97.7% 名映射到 2。這讓極端離群值不再破壞高維空間的距離結構。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「分位數轉換」與「厚尾分佈」。

---

## 本模組的單元測試
對應單元測試主要位於：
1. `bankruptcy/tests/test_preprocessing.py:113-130`（`test_drop_constant_and_duplicate_real_data`）：使用真實資料驗證資料審計所發現的 1 個常數欄位與 2 個重複欄位能被前處理管線正確清除。
2. `bankruptcy/tests/test_integrity_phase0.py:16-50`：驗證資料欄位名稱是否包含非預期空白，且是否符合審計階段宣告的 schema 規格。

---

## 讀碼練習
1. **練習 1**：在 `audit_column_anomalies` 中，為什麼判定完全重複欄位時要維護一個 `seen` 集合？如果沒有 `seen` 集合，當欄位 A、B、C 彼此完全相等時，回傳的重複配對會長怎樣？
2. **練習 2**：在 `audit_near_duplicates` 中，為什麼 `NearestNeighbors` 的 `n_neighbors` 必須設為 2，而不能設為 1？若設為 1，`distances[:, 0]` 的數值必然是多少？
3. **練習 3**：根據 F-03，如果在只有 50 筆樣本的小資料集上呼叫 `QuantileTransformer(output_distribution="normal")` 且未指定 `n_quantiles`，scikit-learn 會發生什麼現象？在正式前處理管線中，程式碼是如何避免這個問題的？

---

## 延伸：想改這裡時要注意什麼
1. **不可在此模組中修改資料集**：`data_audit.py` 是唯讀的審計工具，只負責檢測、出圖與存檔報告，絕不直接修改原始 `data.csv`。
2. **審計輸出唯讀守則**：執行此腳本會更新 `outputs/tables/data_audit_summary.json` 等報告，不可在已凍結的生產環境下隨意重新執行。
