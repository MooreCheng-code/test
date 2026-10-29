# 模組導讀：交叉驗證與特徵管線（cv.py 與 pipelines.py）

## 一句話用途
建立完全隔離、防洩漏的重複分層交叉驗證（5 折 × 3 次重複）與權限鎖定機制，並封裝 scikit-learn／imblearn 的前處理與分類器端對端管線。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「資料切分與交叉驗證層」及「特徵管線組裝層」。
- **上游**：讀取 `bankruptcy/config.py` 常數與原始資料 `bankruptcy/data/data.csv`。
- **下游**：
  - `cv.py` 產出的 `train_idx` 與 15 次外層評估索引（5 折 × 3 次重複）驅動 E1～E5、E7 實驗的訓練與驗證分流；測試集 `test_idx` 受到嚴格權限鎖保護，僅限 E6 解封。
  - `pipelines.py` 構建的 `Pipeline` 物件直接傳入 `tune.py`（E2 調參）、`ensemble.py`（E3 融合）與正式推論模組。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - `df`: `pd.DataFrame`，全量原始資料 `(6819, 96)`。
  - `model_name`: `str`（`"M0"`～`"M5"`）。
  - `imbalance`: `str`（`"S0"`～`"S6"`）。
- **輸出**：
  - `splits.joblib`: 包含 metadata、`train_idx` (5,455 筆)、`test_idx` (1,364 筆) 與 15 個外層折字典的字典檔。
  - `get_train_data()`: `(X_train, y_train)`，形狀為 `(5455, 95)` 與 `(5455,)`。
  - `build_pipeline()`: `imblearn.pipeline.Pipeline` 實例，包含 8～9 個序列化轉換步驟。

---

## 核心設計與函式

### 1. `splits.joblib` 結構、SHA-256 驗證與 `get_test_data` 權限鎖（`bankruptcy/src/cv.py:119-252`）
- **做什麼**：一次性切分並持久化 80% 訓練集與 20% 測試集索引，在載入時自動校驗目標向量 SHA-256 雜湊值；對測試集建立嚴格的權限鎖，防止在 E6 之前的任何開發階段偷看測試集。
- **`splits.joblib` 字典結構**（`bankruptcy/src/cv.py:141-155`）：
  - `metadata`: 字典，內含 `seed=42`、`test_size=0.2`、`cv_folds=5`、`cv_repeats=3`、`n_total=6819`、`n_train=5455`、`n_test=1364` 與 `y_hash`。
  - `train_idx`: 長度 5,455 的整數陣列（破產 176 家，正常 5,279 家）。
  - `test_idx`: 長度 1,364 的整數陣列（破產 44 家，正常 1,320 家）。
  - `cv_splits`: 長度 15 的外層切分清單，每項包含 `repeat` (0~2)、`fold` (0~4)、`split_id` (如 `'r0_f1'`)、`train_idx` (4,364 筆) 與 `val_idx` (1,091 筆)。
- **SHA-256 雜湊完整性防護（`bankruptcy/src/cv.py:182-192`）**：
  ```python
  if len(df) != metadata["n_total"]:
      raise ValueError(f"Dataset row count mismatch! Expected {metadata['n_total']}, got {len(df)}.")

  current_y_hash = compute_y_hash(df[TARGET_COL].values)
  if current_y_hash != metadata["y_hash"]:
      raise ValueError("Target vector y SHA-256 hash mismatch! Data has changed or is corrupted.")
  ```
  程式在載入切分時，每次皆重算原始資料目標欄位的 SHA-256 雜湊。若底層 CSV 被任何外部程式修改、打散或竄改，程式立即拋出 `ValueError` 中斷執行。
- **`get_test_data` 權限鎖（`bankruptcy/src/cv.py:220-234`）**：
  ```python
  def get_test_data(
      i_know_this_is_final_evaluation: bool = False,
      data_path: Union[Path, str, None] = None,
  ) -> tuple[pd.DataFrame, pd.Series]:
      if not i_know_this_is_final_evaluation:
          raise PermissionError(
              "Access to the holdout test set is restricted until Phase E6 final evaluation! "
              "Call get_test_data(i_know_this_is_final_evaluation=True) only in final_eval.py."
          )
  ```
  非 E6 階段若誤調用 `get_test_data()`，預設參數為 `False`，會直接噴出 `PermissionError`，在語法與架構層面封殺「非故意偷看測試集」的可能性。

---

### 2. 種子衍生架構與「三種內層切分設定對照表」
- **外層切分種子（`bankruptcy/src/cv.py:97-101`）**：
  使用 `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`。scikit-learn 內部依序打散 3 次，產生 15 個外層折。
- **內層切分種子衍生（`bankruptcy/src/cv.py:273-277`）**：
  ```python
  # Deterministic seed unique to this outer fold
  id_hash = int(hashlib.md5(str(outer_id).encode("utf-8")).hexdigest()[:7], 16)
  inner_seed = (SEED + id_hash) % (2**31 - 1)
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=inner_seed)
  ```
  為每個外層折（如 `'r0_f1'`）計算獨立的 MD5 雜湊，加上 `SEED` 派生出獨一無二且 100% 確定性的 `inner_seed`。
- **為什麼內層切分不預存至磁碟？**
  1. 內層切分依據 `outer_id` + `SEED` 動態即時計算，運算耗時不到 1 毫秒，具備完全的可重現性。
  2. 避免在硬碟上儲存數十個甚至數百個零碎的索引檔案，降低檔案相依風險與硬碟 I/O 開銷。

#### 三種內層設定對照表

| 設定維度 | E2a 巢狀調參（內層） | E2b 全訓練集調參（內層） | E3 參考分布生成（內層） |
|---|---|---|---|
| **用途與目標** | 在各外層訓練折內尋優，產出無偏泛化估計 | 在全量訓練集上尋優，產出最終最佳超參數 | 產生各模型未見樣本的分數分佈，供 ECDF 融合 |
| **程式碼來源** | `bankruptcy/src/tune.py:206-214`；`cv.py:254` | `bankruptcy/src/tune.py:894-905` | `bankruptcy/src/ensemble.py:417-425` |
| **切分架構** | 單次 3 折（`StratifiedKFold(n_splits=3)`） | 5 折 × 2 次重複（10 次評估） | 單次 5 折（`StratifiedKFold(n_splits=5)`） |
| **每次評估之資料筆數** | 訓練折約 2,909 筆；驗證折約 1,455 筆 | 訓練折 4,364 筆；驗證折 1,091 筆 | 訓練折約 3,491 筆；驗證折約 873 筆 |
| **每次評估之正樣本數** | **訓練折 93～94 家；驗證折 46～47 家** | **訓練折 140～141 家；驗證折 35～36 家** | **訓練折約 112 家；驗證折約 28 家** |

---

### 3. `build_pipeline()` 步驟順序與防洩漏機制（`bankruptcy/src/pipelines.py:212-262`）
- **做什麼**：組裝符合 scikit-learn / imblearn 規範的序列化管線。
- **執行順序與各步驟行為**：
  1. `feature_selector` (`FeatureSelector`)：依據欄位名單挑選特徵（E5 核心特徵集使用）。
  2. `imputer` (`TrainingMedianImputer`，選用)：以訓練集中位數填補缺失值。
  3. `drop_const_dup` (`DropConstantAndDuplicate`)：在訓練折 `.fit()` 辨識常數與重複特徵；驗證折僅呼叫 `.transform()` 剔除相同特徵。
  4. `winsorizer` (`Winsorizer`)：在訓練折計算 1% 與 99% 分位數並截斷極值。
  5. `large_value_indicator` (`LargeValueIndicator`，預設關閉)：派生大數值指標特徵。
  6. `scaler` (`QuantileScaler`)：在訓練折學習分位數映射，投影為標準常態分佈。
  7. `corr_filter` (`CorrelationFilter`)：在訓練折計算 Pearson 相關矩陣並貪婪剔除 $|r| > 0.95$ 欄位。
  8. `resampler` (`_get_resampler`，僅限 S2～S6)：**只在 `pipeline.fit()` 時執行合成重採樣；在 `predict_proba()` 時完全跳過**（杜絕驗證集與測試集被人工樣本污染）。
  9. `classifier` (`_get_classifier`)：分類器訓練與推論。

---

### 4. `AdaptiveScaleWeightClassifier`（`bankruptcy/src/pipelines.py:48-110`）
- **做什麼**：專為 Gradient Boosting（XGBoost M4、LightGBM M5）在策略 S1 下設計的動態類別加權包裝器。
- **動態計算公式**：
  $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$
- **三個核心問題**：
  - *它解決什麼問題？* 傳統做法常在全資料集算好比例寫死在設定檔（例如 `scale_pos_weight=30.0`），這在嚴格的資料隔離下屬於跨折洩漏。
  - *沒有它會出什麼錯？* 若在管線外手動傳參，程式碼容易混亂且容易破壞 `clone()` 或 GridSearch 抽象。
  - *有沒有更簡單的做法？* 可以直接用固定常數，但本類別直接在 `fit(X, y)` 當下從傳入的訓練折 `y` 即時統計正負樣本數，做到真正的零洩漏。
- **最小數字例子**：
  - 在全訓練集（5,455 筆）中，$N_{\text{neg}} = 5,279$，$N_{\text{pos}} = 176$，權重動態計算為：
    $$\text{scale\_pos\_weight} = \frac{5,279}{176} \approx 29.994$$
  - 在外層單折訓練集（4,364 筆）中，$N_{\text{neg}} = 4,223$，$N_{\text{pos}} = 141$，權重動態計算為：
    $$\text{scale\_pos\_weight} = \frac{4,223}{141} \approx 29.950$$
  - 在 20 家公司玩具資料中（16 家正常，4 家破產）：
    $$\text{scale\_pos\_weight} = \frac{16}{4} = 4.000$$

---

### 5. 採樣策略 S0～S6 與模型 M0～M5 完整對照表

#### 採樣策略清單（`bankruptcy/src/pipelines.py:112-129`）

| 策略代碼 | 說明 | 演算法與參數 | 最終集成模型 $E_A$ 是否採用 |
|:---:|---|---|:---:|
| **S0** | **原始分佈（無處置）** | 無重採樣、無加權（天然破產率 3.23%） | **✅ 採用（M3、M4、M5 皆以 S0 固化）** |
| **S1** | **代價敏感類別加權** | `class_weight="balanced"` 或動態 `scale_pos_weight` | ❌ 未採用（機率校準惡化） |
| **S2** | SMOTE 輕度過採樣 | `SMOTE(sampling_strategy=0.1, k_neighbors=5)` | ❌ 未採用 |
| **S3** | SMOTE 中度過採樣 | `SMOTE(sampling_strategy=0.3, k_neighbors=5)` | ❌ 未採用（僅 M2/M5 在單獨調參時曾獲選） |
| **S4** | SMOTE 重度過採樣 | `SMOTE(sampling_strategy=0.5, k_neighbors=5)` | ❌ 未採用 |
| **S5** | 邊界 SMOTE | `BorderlineSMOTE(sampling_strategy=0.3, k_neighbors=5)` | ❌ 未採用 |
| **S6** | 混合採樣 | `SMOTETomek(sampling_strategy=0.3)` | ❌ 未採用 |

#### 基準與候選模型清單（`bankruptcy/src/pipelines.py:131-210`）

| 模型代碼 | 分類器類別 | 實作套件 | 預設超參數要點 | 最終集成模型 $E_A$ 角色 |
|:---:|---|---|---|:---:|
| **M0** | `DummyClassifier` | `sklearn.dummy` | `strategy="prior"` | 零技能基準線（Baseline） |
| **M1** | `LogisticRegression` | `sklearn.linear_model` | `max_iter=5000, C=1.0` | 線性模型基準 |
| **M2** | `RandomForestClassifier` | `sklearn.ensemble` | `n_estimators=300, max_features="sqrt"` | Bagging 樹模型 |
| **M3** | `HistGradientBoostingClassifier` | `sklearn.ensemble` | `max_iter=300, min_samples_leaf=20` | **主要對照組＋$E_A$ 成員（權重 1/3）** |
| **M4** | `XGBClassifier` | `xgboost` | `eval_metric="logloss"` | **$E_A$ 成員（權重 1/3）** |
| **M5** | `LGBMClassifier` | `lightgbm` | `verbose=-1` | **$E_A$ 成員（權重 1/3）** |

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **測試集隔離策略** | A. 信任團隊自律<br>B. 物理搬離硬碟<br>C. 權限鎖與 SHA-256 驗證 | **方案 C（程式碼權限鎖）** | `get_test_data()` 預設直接拋出 `PermissionError`，除非傳入 `i_know_this_is_final_evaluation=True`，否則無法讀取測試集，杜絕開發期無意窺視。 | `bankruptcy/src/cv.py:220-234`；`bankruptcy/tests/test_lock.py` |
| **採樣策略抉擇** | A. 全面啟用 SMOTE (S2~S6)<br>B. 類別加權 S1<br>C. 維持原始分佈 S0 | **方案 C（全體維持 S0）** | E1 實驗證實 SMOTE 人工合成樣本破壞了真實財務空間的高維流形，使 Brier 分數大幅惡化（機率嚴重失真），且 AR@0.95 未顯著提升；自然分佈搭配排序集成表現最穩定。 | `outputs/tables/e1_imbalance_comparison.csv`；E3 最終集成配置 |

---

## 機器學習概念白話框

> **白話概念：分層抽樣（Stratified Splitting）與類別不平衡**
> - **什麼是分層抽樣？** 破產公司在全市場只有 3.23%（220 家）。若使用普通隨機抽樣，某些折可能抽到 5% 破產，某些折只抽到 1%，導致不同折難易度天差地遠。分層抽樣強迫每一折的破產率都精確錨定在約 3.23%。
> - **最小例子**：100 間公司中有 4 間破產（4%）。切成 2 折時，分層抽樣保證每一折恰好有 50 間公司、2 間破產（均為 4%），防止資料分佈產生人為偏誤。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「分層交叉驗證」與「類別不平衡」。

---

## 本模組的單元測試
對應單元測試涵蓋：
1. `bankruptcy/tests/test_cv.py`：驗證 `make_holdout_split` 與 `make_cv_splits` 的折數、樣本數、分層正樣本比例與隨機可重現性。
2. `bankruptcy/tests/test_lock.py`：驗證未帶授權標誌呼叫 `get_test_data()` 必然拋出 `PermissionError`。
3. `bankruptcy/tests/test_preprocessing.py:115-135`（`test_adaptive_scale_weight_classifier`）：驗證動態權重計算在訓練折上能正確生效且相容 Pipeline。

---

## 讀碼練習
1. **練習 1**：在 `cv.py` 中，如果使用者在未執行 `save_splits()` 前直接呼叫 `load_splits()`，程式會發生什麼異常？`save_splits()` 如何保證自身具備冪等性？
2. **練習 2**：在 `AdaptiveScaleWeightClassifier.fit()` 中，若訓練集中完全沒有任何正樣本（$N_{\text{pos}} = 0$），程式碼是如何防範 `ZeroDivisionError` 的？
3. **練習 3**：在 `build_pipeline()` 中，為什麼 SMOTE 重採樣步驟必須排在 `CorrelationFilter` 與 `QuantileScaler` 之後，而不能排在第 1 步？

---

## 延伸：想改這裡時要注意什麼
1. **嚴禁讀取或修改 `splits.joblib` 內的 `test_idx`**：測試集已於 2026-10-23 完成一次性評估並永久鎖定。
2. **動態權重不得手動覆寫**：在自訂管線時，應依賴 `AdaptiveScaleWeightClassifier` 動態計算，不可在程式中寫死固定比例。
