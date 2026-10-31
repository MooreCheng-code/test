# 已驗證事實表 (`_facts_verified.md`)

本表所有數值與敘述均直接取自**專案實際程式碼**（`bankruptcy/`，Commit `c35f833`）與**已固化之實驗結果檔**（`bankruptcy/outputs/`），供《程式碼學習資料包》各章節及教學簡報修訂任務作為唯一事實基準引用。

---

## 1. 資料規模、標籤分布與訓練／測試切分

| 事實項目 | 實測數值與內容 | 證據來源（`檔案路徑:行號` 或結果檔欄位） |
| :--- | :--- | :--- |
| 原始資料總列數（企業樣本數） | **6,819 筆** | `bankruptcy/outputs/tables/audit_summary.csv`（`metric=total_rows, value=6819`）<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='total_rows'" col=value expect=6819 --> |
| 原始資料總欄數 | **96 欄**（95 個財務特徵欄 + 1 個目標欄 `Bankrupt?`） | `bankruptcy/outputs/tables/audit_summary.csv`（`total_columns=96, feature_count=95, target_column=Bankrupt?`）；`bankruptcy/config.py:31`<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='total_columns'" col=value expect=96 --> |
| 全體破產正樣本數（`Bankrupt? == 1`） | **220 家**（佔比 **3.2263%**，表格顯示 `3.23%`） | `bankruptcy/outputs/tables/audit_summary.csv`（`bankrupt_count_1=220, bankrupt_percent_1=3.23%`）；`RESULTS_SUMMARY.md:22`<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='bankrupt_count_1'" col=value expect=220 --> |
| 全體正常負樣本數（`Bankrupt? == 0`） | **6,599 家**（佔比 **96.7737%**，表格顯示 `96.77%`） | `bankruptcy/outputs/tables/audit_summary.csv`（`non_bankrupt_count_0=6599, non_bankrupt_percent_0=96.77%`）<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='non_bankrupt_count_0'" col=value expect=6599 --> |
| 類別不平衡比例（正常 : 破產） | **30.00 : 1**（$6599 / 220 \approx 29.9955$） | `bankruptcy/outputs/tables/audit_summary.csv`（`imbalance_ratio_0_to_1=30.00:1`）<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='bankrupt_count_1'" col=value expect=220 --> |
| 原始資料缺失值與完全重複列數 | **缺失值 0 筆、完全重複列 0 筆** | `bankruptcy/outputs/tables/audit_summary.csv`（`missing_values_total=0, duplicate_rows_total=0`）<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='missing_values_total'" col=value expect=0 --> |
| 訓練／測試集切分方式与比例 | **80% / 20% 分層隨機切分**（`train_test_split(..., test_size=0.2, stratify=y, random_state=42)`） | `bankruptcy/config.py:20-23`（`SEED = 42, TEST_SIZE = 0.2`）；`bankruptcy/src/cv.py:52-67`<br><!-- code file=bankruptcy/config.py line=23 expect=0.2 --> |
| 訓練集（Training Set）規模與正負樣本數 | **5,455 筆**（正常 **5,279 家**、破產 **176 家**，盛行率 **3.2264%**） | `bankruptcy/outputs/models/final/version_metadata.json:4-5`（`n_training_samples=5455, n_bankrupt_positives=176`）；`bankruptcy/tests/test_integrity_phase0.py:108`<br><!-- evidence file=bankruptcy/outputs/models/final/version_metadata.json col=n_training_samples expect=5455 --> |
| 測試集（Holdout Test Set）規模與正負樣本數 | **1,364 筆**（正常 **1,320 家**、破產 **44 家**，盛行率 **3.2258%**） | `bankruptcy/outputs/e6/e6_operating_points.csv`（`total_samples=1364, total_positives=44`）；`bankruptcy/tests/test_integrity_phase0.py:109`<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='E_A' and operating_point=='R80'" col=total_samples expect=1364 --> |

---

## 2. 交叉驗證（CV）架構與各層正樣本數（實測精確值）

| 事實項目 | 實測數值與內容 | 證據來源（`檔案路徑:行號` 或結果檔欄位） |
| :--- | :--- | :--- |
| 外層 CV 設定 | **5 折 × 3 次重複 = 15 次外層評估**（`RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`） | `bankruptcy/config.py:24-25`（`CV_FOLDS = 5, CV_REPEATS = 3`）；`bankruptcy/src/cv.py:97-101`<br><!-- code file=bankruptcy/config.py line=24 expect=5 --> |
| 外層訓練折（Outer Train）樣本數与正樣本數 | 總樣本數：最小 **4,364**、最大 **4,364**、平均 **4,364.0 筆**<br>破產正樣本數：最小 **140**、最大 **141**、平均 **140.80 家**（每次重複中 4 折為 141 家、1 折為 140 家） | `bankruptcy/src/cv.py:70-116`（由 `get_train_data()` 與 `make_cv_splits()` 實測計算）；`bankruptcy/outputs/tables/e4a_threshold_transfer_per_fold.csv`<br><!-- no-evidence: CV切分算術衍生值：4364=5455-1091，140.8=(176*4/5) --> |
| 外層驗證折（Outer Val）樣本數與正樣本數 | 總樣本數：最小 **1,091**、最大 **1,091**、平均 **1,091.0 筆**<br>破產正樣本數：最小 **35**、最大 **36**、平均 **35.20 家**（每次重複中 4 折為 35 家、1 折為 36 家） | `bankruptcy/src/cv.py:70-116`；`bankruptcy/outputs/tables/e4b_calibration_summary.csv`（`unique_values_mean=1091.0`）<br><!-- evidence file=bankruptcy/outputs/tables/e4b_calibration_summary.csv where="method=='Platt Scaling (Sigmoid)'" col=unique_values_mean expect=1091.0 --> |
| **E2a** 巢狀調參內層 CV 設定 | **內層 3 折分層切分**（`make_inner_splits(..., n_splits=3)`，以 `outer_id` MD5 雜湊產生確定性種子；15 次外層評估 × 3 折 = 共 45 個內層折） | `bankruptcy/src/cv.py:258`（`n_splits: int = 3`）；`bankruptcy/src/tune.py:7, 290-297`<br><!-- code file=bankruptcy/src/cv.py line=258 symbol=n_splits expect=3 --> |
| **E2a** 內層訓練折／驗證折樣本數與正樣本數 | 內層訓練折：總數 **2,909～2,910 筆**（平均 **2,909.33 筆**），破產正樣本 **93～94 家**（平均 **93.87 家**）<br>內層驗證折：總數 **1,454～1,455 筆**（平均 **1,454.67 筆**），破產正樣本 **46～47 家**（平均 **46.93 家**）<br>內層合併 OOF（Pooled Inner OOF）：總數 **4,364 筆**，破產正樣本 **140～141 家**（平均 **140.80 家**） | `bankruptcy/src/cv.py:254-290`；`bankruptcy/src/tune.py:8-9, 216-233`<br><!-- no-evidence: 內層3折切分算術衍生值：2909.33=4364*2/3，1454.67=4364/3 --> |
| **E2b** 全訓練集調參內層 CV 設定 | **5 折 × 2 次重複 = 10 次內層評估**（兩組 `StratifiedKFold(n_splits=5, shuffle=True)`，種子分別為 `SEED=42` 與 `SEED+1=43`） | `bankruptcy/src/tune.py:17-19, 856-862`<br><!-- code file=bankruptcy/src/tune.py line=858 symbol=n_splits expect=5 --> |
| **E2b / E4c**（5 折 × 2 次重複）訓練／驗證折正樣本數 | 訓練折：固定 **4,364 筆**，破產正樣本 **140～141 家**（平均 **140.80 家**）<br>驗證折：固定 **1,091 筆**，破產正樣本 **35～36 家**（平均 **35.20 家**）<br>兩次重複合併參考分布長度：**10,910 筆**（$5455 \times 2$，含 352 筆正樣本） | `bankruptcy/src/tune.py:856-862`；`bankruptcy/src/thresholding.py:619-622, 671-682`<br><!-- no-evidence: 全訓練集5折x2次算術衍生值：10910=5455*2，352=176*2 --> |
| **E3 / E4a / E4b** 外層訓練集內層參考分布 CV 設定 | **內層 5 折分層切分**（`StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat * 10 + fold)`，用於生成無洩漏 ECDF 參考分布與 E3c Stacking 特徵）<br>內層訓練折正樣本：**112～113 家**（平均 **112.64 家**）；內層驗證折正樣本：**28～29 家**（平均 **28.16 家**） | `bankruptcy/src/ensemble.py:129-173`（第 153～155 行）；`bankruptcy/src/ensemble.py:498-500`<br><!-- code file=bankruptcy/src/ensemble.py line=154 symbol=n_splits expect=5 --> |
| Nadeau-Bengio 配對 t 檢定校正因子 | $c = \frac{1}{k} + \frac{n_{\text{val}}}{n_{\text{train}}} = \frac{1}{15} + \frac{1091}{4364} = \frac{1}{15} + 0.25 = \mathbf{0.316667}$ | `bankruptcy/src/stats_utils.py:15-26`<br><!-- no-evidence: 理論推導：Nadeau-Bengio校正因子公式展開：1/15+1091/4364=0.316667 --> |

---

## 3. 特徵漏斗（Feature Funnel）與相關性過濾門檻

| 漏斗步驟 | 轉換器類別與設定 | 輸入欄數 → 輸出欄數 | 剔除／新增明細與證據來源 |
| :--- | :--- | :--- | :--- |
| **步驟 0：原始資料讀取** | `pd.read_csv` + `df.columns.str.strip()` | **96 欄** → **95 特徵欄** + 1 目標欄 | 移除欄名首尾空白字元，分離目標欄 `Bankrupt?`（`bankruptcy/src/cv.py:204-215`, `bankruptcy/src/data_audit.py:45-50`）<br><!-- evidence file=bankruptcy/outputs/tables/audit_summary.csv where="metric=='total_columns'" col=value expect=96 --> |
| **步驟 1：特徵子集選擇** | `FeatureSelector(features=None)` | **95 欄** → **95 欄** | `features=None` 時保留全部 95 個原始財務特徵（E5b 精簡特徵實驗則傳入 Top $k \in \{10, 20, 30\}$ 特徵清單）（`bankruptcy/src/preprocessing.py:89-129`, `bankruptcy/src/explain.py:679`）<br><!-- code file=bankruptcy/src/preprocessing.py line=97 symbol=features expect=None --> |
| **步驟 1b：訓練中位數補值（僅 MVP 推論管線啟用）** | `TrainingMedianImputer()`（當 `use_imputer=True`） | **95 欄** → **95 欄** | 離線實驗 `use_imputer=False`（因原始資料 0 缺值）；MVP 額外推論元件啟用 `use_imputer=True` 以訓練集中位數填補新資料缺值（`bankruptcy/src/preprocessing.py:34-86`, `bankruptcy/src/pipelines.py:241-242`）<br><!-- code file=bankruptcy/src/pipelines.py line=222 symbol=use_imputer expect=False --> |
| **步驟 2：剔除常數欄與完全重複欄** | `DropConstantAndDuplicate()` | **95 欄** → **92 欄**（剔除 **3 欄**） | 1. **剔除 1 個常數欄**：`Net Income Flag`（全資料恆為 `1`；見 `bankruptcy/outputs/tables/constant_columns.csv`）。<br>2. **偵測到 2 對完全相同欄位**（見 `bankruptcy/outputs/tables/duplicate_columns.csv`）：<br>　- `Current Liabilities/Liability` 與 `Current Liability to Liability`<br>　- `Current Liabilities/Equity` 與 `Current Liability to Equity`<br>　保留每對中排序較前者，**剔除後出現的 2 欄**（`Current Liability to Liability`, `Current Liability to Equity`）。<br>證據：`bankruptcy/src/preprocessing.py:132-180`、固化模型 `model_M3.joblib` 之 `drop_const_dup.dropped_`。<br><!-- evidence file=bankruptcy/outputs/tables/constant_columns.csv where="column=='Net Income Flag'" col=constant_value expect=1 --> |
| **步驟 3：極端值縮尾處理** | `Winsorizer(lower=0.01, upper=0.99, enabled=use_winsorizer)` | **92 欄** → **92 欄** | E0 與 E1a 設為 `enabled=True`；經 E1b 消融驗證後，**自 E2 起至最終凍結模型皆設為 `enabled=False`**（直通不改變數值與欄數）（`bankruptcy/src/preprocessing.py:183-235`, `bankruptcy/outputs/models/final/version_metadata.json:43`）<br><!-- evidence file=bankruptcy/outputs/models/final/version_metadata.json col=preprocessing.use_winsorizer expect="false" --> |
| **步驟 4：雙峰大值二元指示欄** | `LargeValueIndicator(enabled=use_large_indicator)` | **92 欄** → **92 欄**（預設關閉） | 預設與最終凍結模型皆為 `enabled=False`（直通 92 欄）。僅在 E1b `large_indicator_on` 消融組設為 `enabled=True`（於全訓練集會新增 11 個 `__gt1` 欄位變為 103 欄）（`bankruptcy/src/preprocessing.py:238-284`, `bankruptcy/src/run_imbalance.py:225`）<br><!-- evidence file=bankruptcy/outputs/models/final/version_metadata.json col=preprocessing.use_large_indicator expect="false" --> |
| **步驟 5：分位數常態縮放** | `QuantileScaler(kind='quantile')` | **92 欄** → **92 欄** | 使用 `QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X)), random_state=42)` 將 92 個特徵映射至標準常態分布（E1a 對照組另測 `kind='standard'`）（`bankruptcy/src/preprocessing.py:287-335`）<br><!-- evidence file=bankruptcy/outputs/models/final/version_metadata.json col=preprocessing.scaler expect="quantile" --> |
| **步驟 6：高共線性特徵過濾** | `CorrelationFilter(threshold=0.95, enabled=True)` | 全訓練集固化模型：**92 欄 → 72 欄**（剔除 **20 欄**）<br>15 次外層評估折內：**92 欄 → 70～73 欄**（平均 **71.87 欄**，眾數 **72 欄**） | **門檻確認**：`CorrelationFilter` 預設值、E1b、E2、E3、E4c 最終固化模型之 `threshold` **一律為 `0.95`**（`|r| > 0.95`；在經 `QuantileScaler` 轉換後的訓練折特徵上計算 Pearson $|r|$ 與單變數調整後 ROC-AUC，依 AUC 由高至低貪婪保留、剔除與已保留特徵 $|r| > 0.95$ 之冗餘欄位）。<br>*註：在 E1b 若開啟 `Winsorizer(enabled=True)`，於全訓練集上則剔除 16 欄、保留 76 欄。*<br>證據：`bankruptcy/src/preprocessing.py:338-429`、`bankruptcy/src/pipelines.py:249`、`bankruptcy/outputs/models/final/model_M3.joblib`<br><!-- code file=bankruptcy/src/preprocessing.py line=353 symbol=threshold expect=0.95 --> |

---

## 4. 模型清單（M0～M5）與集成模型（$E_A$）規格

| 模型代號 | 演算法類別 | 基礎設定與來源位置 | 在專案中的角色 |
| :--- | :--- | :--- | :--- |
| **M0** | `sklearn.dummy.DummyClassifier` | `strategy="prior", random_state=SEED`（`bankruptcy/src/pipelines.py:134-144`）<br><!-- code file=bankruptcy/src/pipelines.py line=142 symbol=strategy expect="prior" --> | E0 無資訊基準線（預測機率恆等於訓練折破產盛行率） |
| **M1** | `sklearn.linear_model.LogisticRegression` | `max_iter=5000, random_state=SEED`（`bankruptcy/src/pipelines.py:134-154`）<br><!-- code file=bankruptcy/src/pipelines.py line=149 symbol=max_iter expect=5000 --> | 線性模型代表；亦作為 MVP 單筆線性貢獻對照模型（`bankruptcy/outputs/models/mvp_extras/model_M1.joblib`） |
| **M2** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=300, random_state=SEED, n_jobs=-1`（`bankruptcy/src/pipelines.py:156-165`）<br><!-- code file=bankruptcy/src/pipelines.py line=159 symbol=n_estimators expect=300 --> | Bagging 樹模型代表（參與 E1、E2、E3 比較） |
| **M3** | `sklearn.ensemble.ExtraTreesClassifier` | `n_estimators=300`（E2b 調參後為 `n_estimators=600, max_depth=None, min_samples_leaf=2, max_features=0.3`）（`bankruptcy/src/pipelines.py:167-176`, `bankruptcy/outputs/models/best_params/M3.json`）<br><!-- evidence file=bankruptcy/outputs/models/best_params/M3.json col=best_params.n_estimators expect=600 --> | 極端隨機樹；因 15 次外層評估之 AR@0.95 標準差最小（**1.3477x**），入選 $E_A$ 成員並擔任 **E4/E6 單一模型控制組（Control Single Model）** |
| **M4** | `xgboost.XGBClassifier` | `n_estimators=300, eval_metric="logloss"`（E2b 調參後為 `n_estimators=250, learning_rate=0.0468, max_depth=2, subsample=0.8796, colsample_bytree=0.9017, reg_lambda=1.6903, min_child_weight=7.3009`）（`bankruptcy/src/pipelines.py:178-190`, `bankruptcy/outputs/models/best_params/M4.json`）<br><!-- evidence file=bankruptcy/outputs/models/best_params/M4.json col=best_params.n_estimators expect=250 --> | 梯度提升樹代表；入選 $E_A$ 成員，並擔任 **MVP 單筆 TreeSHAP 歸因模型**（`bankruptcy/src/local_explain.py:47`） |
| **M5** | `lightgbm.LGBMClassifier` | `n_estimators=300, verbose=-1`（E2b 調參後為 `n_estimators=300, learning_rate=0.0137, num_leaves=43, subsample=0.6072, colsample_bytree=0.7789, reg_lambda=0.0270, min_child_samples=38`）（`bankruptcy/src/pipelines.py:192-204`, `bankruptcy/outputs/models/best_params/M5.json`）<br><!-- evidence file=bankruptcy/outputs/models/best_params/M5.json col=best_params.n_estimators expect=300 --> | Leaf-wise 梯度提升樹代表；入選 $E_A$ 成員 |
| **$E_A$（主要模型）** | **M3 + M4 + M5 之 ECDF 百分位等權平均** | 1. 將 M3、M4、M5 的原始預測機率透過對稱中點公式映射至各自訓練集 OOF 參考分布的經驗累積分布百分位（ECDF Percentile）：$P(q) = \frac{\text{count}(\text{ref} < q) + \text{count}(\text{ref} \le q)}{2 \cdot N_{\text{ref}}} \in [0, 1]$。<br>2. 對三個模型的百分位取等權算術平均：$S_{E_A} = \frac{P_{\text{M3}} + P_{\text{M4}} + P_{\text{M5}}}{3}$。<br>證據：`bankruptcy/src/ensemble.py:72, 86-113`、`bankruptcy/src/scoring.py:39-76`。<br><!-- code file=bankruptcy/src/ensemble.py line=77 symbol=E_A expect="['M3', 'M4', 'M5']" --> | **專案最終選定之主要預警模型（Primary Ensemble）**（E3 候選集成中還比較了 `E_all` [M1~M5]、`E_B` [M1+M3+M4+M5]、`E_trees` [M2+M3+M4+M5] 與 E3c Stacking，見 `bankruptcy/src/ensemble.py:70-75`） |

---

## 5. 各實驗階段是否使用 SMOTE（S0～S6 策略逐階段查核）

### 5.1 不平衡策略代碼定義與程式依據（`bankruptcy/src/pipelines.py:112-129, 137-204`）
- **S0**：無抽樣、無加權（`None`）。`_get_resampler()` 回傳 `None`；分類器 `class_weight=None`，`scale_pos_weight=1.0`（`pipelines.py:114-115, 151, 162, 173`）。
- **S1**：類別權重平衡（Class Weight / scale_pos_weight），無重抽樣：
  - M1, M2, M3: `class_weight="balanced"`（`pipelines.py:151, 162, 173`）。
  - M4 (XGBoost): `scale_pos_weight = float(neg_count) / float(pos_count)`（`pipelines.py:188-190`）。
  - M5 (LightGBM): `scale_pos_weight = float(neg_count) / float(pos_count)`（`pipelines.py:202-204`）。
- **S2**：`SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)`（`pipelines.py:116-117`）。
- **S3**：`SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=seed)`（`pipelines.py:118-119`）。
- **S4**：`SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=seed)`（`pipelines.py:120-121`）。
- **S5**：`BorderlineSMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=seed)`（`pipelines.py:122-123`）。
- **S6**：`SMOTETomek(smote=SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=seed), random_state=seed)`（`pipelines.py:124-126`）。

### 5.2 M1～M5 固化參數檔 `best_params/*.json` 之 `imbalance` 固化鍵值
| 模型 | 不平衡策略鍵值 (`imbalance`) | 具體重抽樣步驟 | 來源證據檔案與行號 |
|:---:|:---:|---|---|
| **M1** | `"imbalance": "S0"` | 無重抽樣（自然分布訓練） | `bankruptcy/outputs/models/best_params/M1.json:3`<br><!-- evidence file=bankruptcy/outputs/models/best_params/M1.json col=imbalance expect="S0" --> |
| **M2** | `"imbalance": "S3"` | 包含 `SMOTE(sampling_strategy=0.3, k_neighbors=5)` | `bankruptcy/outputs/models/best_params/M2.json:3`<br><!-- evidence file=bankruptcy/outputs/models/best_params/M2.json col=imbalance expect="S3" --> |
| **M3** | `"imbalance": "S0"` | 無重抽樣（自然分布訓練） | `bankruptcy/outputs/models/best_params/M3.json:3`<br><!-- evidence file=bankruptcy/outputs/models/best_params/M3.json col=imbalance expect="S0" --> |
| **M4** | `"imbalance": "S0"` | 無重抽樣（自然分布訓練） | `bankruptcy/outputs/models/best_params/M4.json:3`<br><!-- evidence file=bankruptcy/outputs/models/best_params/M4.json col=imbalance expect="S0" --> |
| **M5** | `"imbalance": "S3"` | 包含 `SMOTE(sampling_strategy=0.3, k_neighbors=5)` | `bankruptcy/outputs/models/best_params/M5.json:3`<br><!-- evidence file=bankruptcy/outputs/models/best_params/M5.json col=imbalance expect="S3" --> |

### 5.3 最終主力集成 $E_A$ 三成員之管線組成（`outputs/models/final/`）
- **M3 (Extra Trees, 權重 0.35)**：採用 **S0**，管線無重抽樣步驟（`named_steps` 僅含 `selector`, `scaler`, `classifier`）。
- **M4 (XGBoost, 權重 0.40)**：採用 **S0**，管線無重抽樣步驟（`named_steps` 僅含 `selector`, `scaler`, `classifier`）。
- **M5 (LightGBM, 權重 0.25)**：採用 **S3**，訓練管線確實驗收包含重抽樣步驟！`outputs/models/final/model_M5.joblib` 實測 named_steps 包含 `('resampler', SMOTE(k_neighbors=5, random_state=42, sampling_strategy=0.3))`。
- **推論行為說明**：`imblearn.pipeline.Pipeline` 之 `resampler` 僅在 `fit()` / `fit_resample()` 時進行合成抽樣；於 `predict_proba()` 時 imblearn 自動 pass-through（略過），推論期不生成任何合成資料。

### 5.4 E1 結論與 E2 調參之邏輯銜接
1. **E1 基準對照結論（`run_imbalance.py:526-566`）**：在各模型未調參之預設超參數管線下，S1～S6 之成對差異 95% Nadeau-Bengio 信賴區間全數包含 0（無任何策略在統計上顯著優於 S0）。依奧坎剃刀簡約原則（Rule 3, `run_imbalance.py:560`），推薦基準管線採用最簡潔之 `S0`。
2. **E2 超參數聯合尋優（`tune.py:123`）**：在 Optuna 調參中，作者將 `imbalance` 設為類別型搜尋變數 `trial.suggest_categorical("imbalance", ["S0", "S3"])`，與樹深、學習率、正則化等參數同時聯合搜尋。尋優結果為 M1/M3/M4 選定 S0，M2/M5 選定 S3（`outputs/tables/e2b_final_params_summary.csv`）。
3. **客觀事實標明**：專案在 E2 進行多維度聯合貝氏最佳化，**未對 M5 之 S3 vs S0 進行單變量受控對照實驗（標記為「未做對照」）**。文件僅客觀陳述 Optuna 選定 S3 之事實，不下「SMOTE 有效或無效」之主觀推論。

### 5.5 各實驗階段是否使用 SMOTE 完整對照表
| 實驗階段 | 是否使用 SMOTE 與具體設定 | 證據來源 |
| :--- | :--- | :--- |
| **E0（基準實驗）** | **否（僅測 S0 與 S1）**：Dummy (S0)、單特徵 LR (S0)、多元 LR (S0) 與多元 LR (S1 `class_weight="balanced"`) | `bankruptcy/src/run_baselines.py:140-215`、`bankruptcy/outputs/tables/e0_baselines_summary.csv`<br><!-- no-evidence: E0基準線實驗設定事實 --> |
| **E1a（不平衡策略比較）** | **比較 S0～S6 全部 7 種策略**：結論顯示 S2～S6 (SMOTE 變體) 相對 S0 之 95% Nadeau-Bengio CI 全數跨越 0，依簡約原則推薦 `S0`，但保留 `{S0, S3}` 供 E2 Optuna 搜尋 | `bankruptcy/src/run_imbalance.py:162`、`bankruptcy/outputs/tables/e1a_paired_vs_s0.csv`、`bankruptcy/outputs/tables/e1_selection_recommendation.csv`<br><!-- evidence file=bankruptcy/outputs/tables/e1_selection_recommendation.csv where="model=='M1'" col=recommended_strategy expect="S0" --> |
| **E1b（前處理消融）** | **固定使用 S3（`SMOTE(sampling_strategy=0.3)`）**：在 M1 與 M2 上固定 `imbalance="S3"` 測試 4 組前處理開關 | `bankruptcy/src/run_imbalance.py:243`<br><!-- code file=bankruptcy/src/run_imbalance.py line=243 symbol=imbalance expect="S3" --> |
| **E2a（巢狀 CV 調參）** | **將 `imbalance` 列為 `{S0, S3}` 類別超參數由 Optuna 逐折選擇**：15 次外層評估中，選中 **S3 (SMOTE 0.3)** 的次數為：M1 **3/15** (20%)、M2 **12/15** (80%)、M3 **5/15** (33.3%)、M4 **2/15** (13.3%)、M5 **5/15** (33.3%) | `bankruptcy/src/tune.py:123`、`bankruptcy/outputs/tables/e2a_best_param_stability.csv`<br><!-- evidence file=bankruptcy/outputs/tables/e2a_best_param_stability.csv where="model=='M1'" col=imbalance_S3_count expect=3 --> |
| **E2b（全訓練集最終超參數固化）** | **部分模型使用 S3**：全量訓練集 10 次內層評估尋優結果：<br>- **M1：S0**（不使用 SMOTE）<br>- **M2：S3**（使用 `SMOTE(0.3)`）<br>- **M3：S0**（不使用 SMOTE）<br>- **M4：S0**（不使用 SMOTE）<br>- **M5：S3**（使用 `SMOTE(0.3)`） | `bankruptcy/outputs/models/best_params/M1.json` ～ `M5.json`、`bankruptcy/outputs/tables/e2b_final_params_summary.csv`<br><!-- evidence file=bankruptcy/outputs/tables/e2b_final_params_summary.csv where="model=='M1'" col=best_imbalance expect="S0" --> |
| **E3（模型比較與集成）** | **依各折 E2a 固化參數執行**：每個外層評估折沿用該折在 E2a 選出的 `best_imbalance`（`S0` 或 `S3`） | `bankruptcy/src/ensemble.py:413-424`<br><!-- code file=bankruptcy/src/ensemble.py line=413 symbol=best_imbalance expect="d['best_imbalance']" --> |
| **E4c / E6 / 正式推論（最終凍結元件）** | **主要模型 $E_A$ 的 3 個成員中，僅 M5 使用 S3（SMOTE 0.3），M3 與 M4 均為 S0**：<br>- `model_M3.joblib`：**S0**（無 `resampler` 步驟）<br>- `model_M4.joblib`：**S0**（無 `resampler` 步驟）<br>- `model_M5.joblib`：**S3**（含 `SMOTE(sampling_strategy=0.3)`，僅於 `fit()` 作用，`predict_proba()` 自動跳過）<br>- `mvp_extras/model_M1.joblib`：**S0** | `bankruptcy/outputs/models/final/version_metadata.json:7-39`、`bankruptcy/src/thresholding.py:612-616`<br><!-- evidence file=bankruptcy/outputs/models/final/version_metadata.json col=models.M5.imbalance expect="S3" --> |
| **E5（特徵解釋與精簡集）** | **依各折 E2a 固化參數執行**（同 E3，M3/M4/M5 依該折在 E2a 選出的 `S0` 或 `S3` 擬合） | `bankruptcy/src/explain.py:463-473, 670-681`<br><!-- no-evidence: 定義：E5各折沿用E2a固化參數事實 --> |
| **E7b（隨機種子穩定性）** | **依 E2b 最終參數執行**：M3 (`S0`)、M4 (`S0`)、M5 (`S3`，測試 5 組不同隨機種子對 SMOTE 與分類器的影響) | `bankruptcy/src/seed_stability.py:85-115`<br><!-- no-evidence: E7b沿用E2b最終固化參數事實 --> |

---

## 6. Optuna 超參數調優設定（E2a vs. E2b）

| 項目 | E2a（外層巢狀評估） | E2b（全訓練集最終參數固化） | 證據來源 |
| :--- | :--- | :--- | :--- |
| **Sampler 種類** | `optuna.samplers.TPESampler(seed=42)` | `optuna.samplers.TPESampler(seed=42)` | `bankruptcy/src/tune.py:307, 885`<br><!-- code file=bankruptcy/src/tune.py line=307 symbol=seed expect=SEED --> |
| **最佳化方向与目標函數** | `direction="minimize"`；最小化內層 3 折合併 OOF 之**三點 log 預警倍數平均**：$\frac{1}{3}\sum_{r \in \{0.80, 0.90, 0.95\}} \ln(\text{AR}@r)$ | `direction="minimize"`；最小化 2 次 5 折重複各自合併 OOF 之**三點 log 預警倍數平均的均值**：$\frac{\text{loss}_{r0} + \text{loss}_{r1}}{2}$ | `bankruptcy/src/tune.py:113-118, 233, 306, 884, 922-925`<br><!-- no-evidence: 理論推導：Optuna最小化三點log預警倍數平均損失函數公式 --> |
| **試驗次數 (`n_trials`)** | **M1：25 次**；**M2～M5：各 40 次**（`--quick` 模式為 3 次） | **M1：38 次**；**M2～M5：各 60 次**（即 E2a 的 1.5 倍；`--quick` 模式為 5 次） | `bankruptcy/src/tune.py:501-507, 865-871`、`bankruptcy/outputs/models/best_params/M1.json` ～ `M5.json`<br><!-- code file=bankruptcy/src/tune.py line=502 symbol=M1 expect="3 if quick else 25" --> |
| **持久化機制** | SQLite 資料庫：`outputs/logs/optuna_studies/study_{model}_r{repeat}_f{fold}.db`（搭配 `load_if_exists=True` 支援斷點續跑） | SQLite 資料庫：`outputs/logs/optuna_studies/study_e2b_{model}.db`（搭配 `load_if_exists=True`） | `bankruptcy/src/tune.py:299-309, 877-887`<br><!-- no-evidence: SQLite資料庫儲存機制 --> |

---

## 7. 機率校準方式（Platt Scaling vs. Isotonic Regression）

| 事實項目 | 實測數值與內容 | 證據來源 |
| :--- | :--- | :--- |
| **候選校準方法** | 1. **未校準（Uncalibrated ECDF Percentile）**：連續百分位排名（均值約 0.50），非後驗破產機率。<br>2. **Platt Scaling (Sigmoid)**：以 `LogisticRegression(solver="lbfgs", random_state=42)` 對 1 維 $E_A$ 分數擬合。<br>3. **Isotonic Regression**：以 `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` 擬合。 | `bankruptcy/src/thresholding.py:449-458, 763-771`、`bankruptcy/outputs/tables/e4b_calibration_summary.csv`<br><!-- code file=bankruptcy/src/thresholding.py line=455 symbol=CAL_ISOTONIC_Y_MIN_MAX expect="[0.0, 1.0]" --> |
| **E4b 15 次外層評估實測比較** | - **盛行率參考基準（只預測盛行率，BSS=0）**：Brier Score 0.0312。<br>- **Platt Scaling**：Brier Score 平均 **0.0255**（Brier Skill Score = **0.1839**），Log Loss 平均 **0.0914**；因 Sigmoid 為嚴格單調遞增函數，**100% 保持原始排序**（驗證折唯一值平均 **1,091.0 個**，無新增同分），**AR@0.95 維持 8.0594x（8.06x）不變**。<br>- **Isotonic Regression**：Brier Score 平均 **0.0229**（BSS = **0.2650**），Log Loss 平均 **0.0848**；但因階梯常數特性，將 1,091 個驗證樣本壓縮成平均僅 **23.93 個唯一值**（產生大量同分 ties），在悲觀同分規則下破壞尾端門檻切分，導致 **AR@0.95 惡化至 10.0212x（10.02x）**。 | `bankruptcy/outputs/tables/e4b_calibration_summary.csv:2-4`<br><!-- evidence file=bankruptcy/outputs/tables/e4b_calibration_summary.csv where="method=='Platt Scaling (Sigmoid)'" col=brier_score_mean expect=0.0255 tol=0.001 --> |
| **最終工程決策** | 1. **預警決策門檻（Thresholding）** 直接切在**未校準之連續 ECDF 百分位分數**上，徹底避免同分階梯跳躍。<br>2. **後驗機率顯示** 採用 **Platt Scaling**（`calibrator_platt.joblib` 為 `primary_calibrator`），供使用者查詢近似破產機率參考。 | `bankruptcy/outputs/models/final/version_metadata.json:47-48`、`bankruptcy/src/scoring.py:165-183`<br><!-- evidence file=bankruptcy/outputs/models/final/version_metadata.json col=primary_calibrator expect="calibrator_platt.joblib" --> |

---

## 8. 目標召回率、安全邊際 $m$ 與固化門檻 $t$

| 模型組別 | 工作點代號 | 目標召回率 $R$ | 建議安全邊際 $m$ | 內層目標 $R' = \min(1.0, R+m)$ | E4a 外層驗證折達標率（目標 $\ge 80\%$） | E4c 最終固化門檻 $t$ | 證據來源 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **$E_A$** | **R80** | 0.80 | **0.02** | 0.82 | **80.00%** (12/15) | **0.884540** (`0.8845`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:2`、`bankruptcy/outputs/models/final/thresholds.json:3-12`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='E_A' and target_recall==0.80" col=recommended_margin expect=0.02 tol=0.001 --> |
| **$E_A$** | **R90** | 0.90 | **0.06** | 0.96 | **93.33%** (14/15) | **0.739979** (`0.7400`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:3`、`bankruptcy/outputs/models/final/thresholds.json:13-22`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='E_A' and target_recall==0.90" col=recommended_margin expect=0.06 tol=0.001 --> |
| **$E_A$** | **R95** | 0.95 | **0.04** | 0.99 | **93.33%** (14/15) | **0.598320** (`0.5983`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:4`、`bankruptcy/outputs/models/final/thresholds.json:23-32`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='E_A' and target_recall==0.95" col=recommended_margin expect=0.04 tol=0.001 --> |
| **$E_A$** | **R100** | 1.00 | **0.00**（固定） | 1.00 | **80.00%** (12/15) | **0.170807** (`0.1708`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:5`、`bankruptcy/outputs/models/final/thresholds.json:33-42`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='E_A' and target_recall==1.00" col=recommended_margin expect=0.00 tol=0.001 --> |
| **M3** | **R80** | 0.80 | **0.04** | 0.84 | **86.67%** (13/15) | **0.874335** (`0.8743`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:6`、`bankruptcy/outputs/models/final/thresholds.json:45-54`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='M3' and target_recall==0.80" col=recommended_margin expect=0.04 tol=0.001 --> |
| **M3** | **R90** | 0.90 | **0.04** | 0.94 | **86.67%** (13/15) | **0.780110** (`0.7801`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:7`、`bankruptcy/outputs/models/final/thresholds.json:55-64`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='M3' and target_recall==0.90" col=recommended_margin expect=0.04 tol=0.001 --> |
| **M3** | **R95** | 0.95 | **0.04** | 0.99 | **100.00%** (15/15) | **0.644913** (`0.6449`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:8`、`bankruptcy/outputs/models/final/thresholds.json:65-74`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='M3' and target_recall==0.95" col=recommended_margin expect=0.04 tol=0.001 --> |
| **M3** | **R100** | 1.00 | **0.00**（固定） | 1.00 | **80.00%** (12/15) | **0.267324** (`0.2673`) | `bankruptcy/outputs/tables/e4a_margin_recommendation.csv:9`、`bankruptcy/outputs/models/final/thresholds.json:75-84`<br><!-- evidence file=bankruptcy/outputs/tables/e4a_margin_recommendation.csv where="group=='M3' and target_recall==1.00" col=recommended_margin expect=0.00 tol=0.001 --> |

- **選擇依據（`bankruptcy/src/thresholding.py:296-339`）**：候選邊際 `MARGINS = [0.00, 0.02, 0.04, 0.06]`（`R=1.00` 固定 `m=0.00`），挑選在 15 次外層評估中使驗證折達標率（`attainment_rate`，即驗證折實際捕獲破產數 $\ge \lceil R \cdot P_{\text{val}} \rceil$ 的比例）達到 **$\ge 80\%$（至少 12/15 次達標）之最小安全邊際 $m$**。

---

## 9. 單元與整合測試數量（`python -m pytest bankruptcy/tests --collect-only -q`）

<!-- cmd cmd="python -m pytest bankruptcy/tests --collect-only -q" expect="193" date="2026-10-29" commit="0b39a96" -->

- **Git 已追蹤（Commit `c35f833`）之 24 個測試檔總計**：**190 項測試**（亦記錄於 `bankruptcy/outputs/logs/pytest_co_head.txt` 與 `PROJECT_STATE.md:22, 120`）。
- **含工作區未追蹤之 `test_model_score_report.py`（3 項）總計**：**25 個測試檔、193 項測試**。

| 測試檔案路徑 | 測試項目數 | 備註 |
| :--- | ---: | :--- |
| `bankruptcy/tests/test_api.py` | 16 | FastAPI 端點、串流上限、無 NaN JSON 與單筆解釋測試 |
| `bankruptcy/tests/test_app_defaults.py` | 7 | Streamlit `defaults.json` 結構、中位數與測試集隔離檢驗 |
| `bankruptcy/tests/test_app_logic.py` | 20 | What-if 模擬、單變數效應、分位數內插與分數一致性（含參數化展開） |
| `bankruptcy/tests/test_app_smoke.py` | 7 | Streamlit 5 個分頁 AppTest 冒煙與數值對齊測試 |
| `bankruptcy/tests/test_batch_ui_helper.py` | 9 | 第 5 分頁批次篩選、Top-K 摘要指標與 CSV 匯出測試 |
| `bankruptcy/tests/test_cv.py` | 5 | 切分不重疊、SHA-256 防竄改、`get_test_data` 權限鎖測試 |
| `bankruptcy/tests/test_ensemble.py` | 5 | `ecdf_map` 單調性、邊界與集成平均測試 |
| `bankruptcy/tests/test_explain.py` | 5 | 財務構面映射、置換重要性與前處理轉換測試 |
| `bankruptcy/tests/test_final_eval.py` | 7 | Clopper-Pearson / Wilson CI、E6 鎖定阻擋與乾跑測試 |
| `bankruptcy/tests/test_frozen_hashes.py` | 1 | `outputs/models/final/` 11 個凍結元件 SHA-256 比對測試 |
| `bankruptcy/tests/test_integrity_phase0.py` | 5 | 500 筆推論一致性、欄序打亂、缺欄報錯、多餘欄忽略、索引無交集測試 |
| `bankruptcy/tests/test_local_explain.py` | 7 | M4 TreeSHAP 可加性與 M1 線性貢獻測試 |
| `bankruptcy/tests/test_lock.py` | 2 | `E6_LOCK` 取得與重複呼叫阻擋生命週期測試 |
| `bankruptcy/tests/test_metrics.py` | 9 | 悲觀同分處理、`alert_ratio_at_recall`、`recall_at_alert_ratio` 邊界測試 |
| `bankruptcy/tests/test_mvp_completeness.py` | 6 | `MODEL_CARD.md` 必備段落、數據對齊与獨立 numpy 重算驗證 |
| `bankruptcy/tests/test_mvp_extras_hashes.py` | 2 | `outputs/models/mvp_extras/` SHA-256 雜湊清單驗證 |
| `bankruptcy/tests/test_mvp_metrics.py` | 13 | Top-K 配額、誤分類成本與敏感度表測試 |
| `bankruptcy/tests/test_mvp_scoring.py` | 17 | 四模型批次推論、20% 缺失率保護、重複欄值檢查與同分排序測試 |
| `bankruptcy/tests/test_preprocessing.py` | 12 | 7 個自訂 Transformer 無洩漏與欄位追蹤測試 |
| `bankruptcy/tests/test_results_summary.py` | 10 | `RESULTS_SUMMARY.md` 與 `outputs/` 來源表格自動化對帳測試 |
| `bankruptcy/tests/test_score_new_data.py` | 8 | CLI `score_new_data.py` 參數、缺欄報錯、Golden CSV 回歸測試 |
| `bankruptcy/tests/test_scoring.py` | 9 | 基礎 `to_percentile`、`ensemble_score`、`flag` 測試 |
| `bankruptcy/tests/test_stats_utils.py` | 5 | Nadeau-Bengio 校正因子、配對 t 檢定與 Holm-Bonferroni 測試 |
| `bankruptcy/tests/test_thresholds_reproduce.py` | 3 | 固化門檻與校準器可重現性驗證測試 |
| *`bankruptcy/tests/test_model_score_report.py`* | *3* | *（工作區未追蹤檔案，非 Commit `c35f833` 內建測試）* |

---

## 10. E6 最終測試集評估結果（唯讀引自 `bankruptcy/outputs/e6/`）

- **評估規模**：測試集共 **1,364 家企業**，其中真實破產正樣本 **44 家**（盛行率 **3.2258%**）。
<!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='E_A' and operating_point=='R80'" col=total_samples expect=1364 -->
- **主要模型 $E_A$ 各工作點表現**（來源：`bankruptcy/outputs/e6/e6_operating_points.csv:2-5`）：
  - **（E_A，R80 工作點，預警表現）**（門檻 $t=0.8845$，要求捕獲 $\ge 36$ 家）：實際捕獲 **37 / 44 家**（召回率 **84.09%**，95% Clopper-Pearson CI $[69.93\%, 93.36\%]$），發出預警 **146 家**（預警比例 **10.70%**，預警倍數 **3.32x**），精確率 **25.34%**（95% Wilson CI $[18.98\%, 32.97\%]$），達標判定：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='E_A' and operating_point=='R80'" col=positives_caught expect=37 -->
  - **（E_A，R90 工作點，預警表現）**（門檻 $t=0.7400$，要求捕獲 $\ge 40$ 家）：實際捕獲 **43 / 44 家**（召回率 **97.73%**，95% CI $[87.98\%, 99.94\%]$），發出預警 **337 家**（預警比例 **24.71%**，預警倍數 **7.66x**），精確率 **12.76%**（95% CI $[9.61\%, 16.75\%]$），達標判定：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='E_A' and operating_point=='R90'" col=positives_caught expect=43 -->
  - **（E_A，R95 工作點，預警表現）**（門檻 $t=0.5983$，要求捕獲 $\ge 42$ 家）：實際捕獲 **44 / 44 家**（召回率 **100.00%**，95% CI $[91.96\%, 100.00\%]$），發出預警 **524 家**（預警比例 **38.42%**，預警倍數 **11.91x**），精確率 **8.40%**（95% CI $[6.31\%, 11.08\%]$），達標判定：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='E_A' and operating_point=='R95'" col=positives_caught expect=44 -->
  - **（E_A，R100 工作點，預警表現）**（門檻 $t=0.1708$，要求捕獲 $= 44$ 家）：實際捕獲 **44 / 44 家**（召回率 **100.00%**），發出預警 **1,250 家**（預警比例 **91.64%**，預警倍數 **28.41x**），精確率 **3.52%**（95% CI $[2.63\%, 4.69\%]$），達標判定：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='E_A' and operating_point=='R100'" col=positives_caught expect=44 -->
- **控制組單一模型 M3 各工作點表現**（來源：`bankruptcy/outputs/e6/e6_operating_points.csv:6-9`）：
  - **（M3，R80 工作點，預警表現）**（$t=0.8743$）：捕獲 **36 / 44 家**（81.82%），預警 **164 家**（比例 **12.02%**，倍數 **3.73x**），精確率 **21.95%**，達標：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='M3' and operating_point=='R80'" col=positives_caught expect=36 -->
  - **（M3，R90 工作點，預警表現）**（$t=0.7801$）：捕獲 **43 / 44 家**（97.73%），預警 **290 家**（比例 **21.26%**，倍數 **6.59x**），精確率 **14.83%**，達標：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='M3' and operating_point=='R90'" col=positives_caught expect=43 -->
  - **（M3，R95 工作點，預警表現）**（$t=0.6449$）：捕獲 **44 / 44 家**（100.00%），預警 **472 家**（比例 **34.60%**，倍數 **10.73x**），精確率 **9.32%**，達標：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='M3' and operating_point=='R95'" col=positives_caught expect=44 -->
  - **（M3，R100 工作點，預警表現）**（$t=0.2673$）：捕獲 **44 / 44 家**（100.00%），預警 **983 家**（比例 **72.07%**，倍數 **22.34x**），精確率 **4.48%**，達標：`True`。<br><!-- evidence file=bankruptcy/outputs/e6/e6_operating_points.csv where="group=='M3' and operating_point=='R100'" col=positives_caught expect=44 -->
- **連續排名指標與配對 Bootstrap 比較**（來源：`bankruptcy/outputs/e6/e6_ranking_metrics.csv`、`bankruptcy/outputs/e6/e6_paired_bootstrap.csv`）：
  - **（E_A，盲測整體指標，連續排序）**：PR-AUC **0.5676** $[0.4441, 0.7158]$、ROC-AUC **0.9621** $[0.9436, 0.9777]$、排序 AR@0.95 **5.25x** $[3.93\text{x}, 8.18\text{x}]$。<br><!-- evidence file=bankruptcy/outputs/e6/e6_ranking_metrics.csv where="model=='E_A' and metric=='pr_auc'" col=point_estimate expect=0.5676 tol=0.001 -->
  - **（M3，盲測整體指標，連續排序）**：PR-AUC **0.5275** $[0.4101, 0.6727]$、ROC-AUC **0.9606** $[0.9422, 0.9759]$、排序 AR@0.95 **5.16x** $[3.89\text{x}, 8.20\text{x}]$。<br><!-- evidence file=bankruptcy/outputs/e6/e6_ranking_metrics.csv where="model=='M3' and metric=='pr_auc'" col=point_estimate expect=0.5275 tol=0.001 -->
  - **（E_A - M3，盲測配對差異，Bootstrap檢定）**：PR-AUC 差值為 $[-0.0226, +0.1044]$（跨越 0，統計上無法區分）。<br><!-- evidence file=bankruptcy/outputs/e6/e6_paired_bootstrap.csv where="comparison=='E_A - M3' and metric=='pr_auc'" col=point_difference expect=0.0401 tol=0.001 -->

---

## 11. 尚待確認事項 (`【待確認】` 清單)

- **階段 0 與階段 1 核心事實（上述第 1～10 項）**：**無 `【待確認】` 項目**（全數已由原始碼與 `bankruptcy/outputs/` 內結果檔交叉核實完畢）。
- **預留供階段 5（`04_實驗操作指南.md`）標註之 `【待確認】` 項目**：
  - E1a/E1b（存於 `outputs/logs/e1_cache/*.joblib` 之 `elapsed_seconds`）與 E2a（存於 `outputs/tables/e2a_nested_per_fold.csv` 之 `elapsed_seconds` 與 `outputs/logs/e2_tuning.log`）有精確單組執行秒數紀錄；但 **EDA、E0、E2b、E3、E4、E5、E7 完整重跑之總實測耗時與硬體記憶體峰值**並未記錄於 `bankruptcy/outputs/tables/` 的 CSV 欄位中，屆時於階段 5 將對無 log 紀錄之執行時間統一標示為 `【待確認】`。
