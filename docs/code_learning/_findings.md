# 程式碼與既有文件審計發現清單 (`_findings.md`)

本檔案記錄在製作《程式碼學習資料包》期間，透過對照**專案實際程式碼（唯一事實來源）**、實驗輸出檔（`bankruptcy/outputs/`）與專案既有文件所發現的：
1. **疑似程式碼問題（Suspected Code Bugs / Edge Cases）**
2. **既有文件與實際程式碼不一致之處（Documentation vs. Code Discrepancies）**
3. **跨階段參數序列化與歷史命名遺留項目（Serialization & Legacy Artifacts）**

> [!NOTE]
> 依據專案硬性規則，本清單內所有發現**僅如實記錄回報，絕不自行修改既有原始碼、既有文件或實驗輸出檔**。

---

## 一、疑似程式碼問題（Suspected Code Bugs / Edge Cases）

### F-01：`freeze_final_components()` 快取命中分支之 `UnboundLocalError`
- **位置**：`bankruptcy/src/thresholding.py:631-636` 與 `bankruptcy/src/thresholding.py:672`
- **嚴重度評級**：**低（Low）**
- **是否影響已固化產物**：**否，完全無影響**。已固化之最終模型（`bankruptcy/outputs/models/final/`）是在快取不存在的初始環境下一次性完整執行產出；正式推論與部署階段僅呼叫 `load_final_components()`（`bankruptcy/src/scoring.py:86`），絕不呼叫 `freeze_final_components()`。
- **觸發條件與原因分析**：
  在 `freeze_final_components()`（E4c 最終元件固化函式）中，第 627～636 行檢查快取檔案是否已存在：
  ```python
  if ref_file.exists() and oof_r0_file.exists() and oof_r1_file.exists():
      oof_r0[m] = np.load(oof_r0_file)
      oof_r1[m] = np.load(oof_r1_file)
      final_references[m] = np.load(ref_file)  # 第 634 行：此時 final_references 尚未定義！
      logger.info(f"[E4c] Loaded cached reference_{m}.npy")
      continue
  ```
  區域變數 `final_references = {}` 直到第 672 行才被初始化。因此，僅在磁碟上已存在 `reference_{m}.npy`、`_tmp_oof_r0_{m}.npy`、`_tmp_oof_r1_{m}.npy` 時，若使用者「再次呼叫」`freeze_final_components()`，才會在第 634 行拋出 `UnboundLocalError: cannot access local variable 'final_references' where it is not associated with a value`。正常流程下無需重跑 E4c，故此問題不影響任何功能。

### F-02：Optuna 派生超參數（M5 `subsample_freq` 與 M1 `solver`/`l1_ratio`）未隨 `best_params` 序列化
- **位置**：`bankruptcy/src/tune.py:129-141`、`bankruptcy/src/tune.py:169`、`bankruptcy/src/tune.py:329-336`、`bankruptcy/src/tune.py:577`、`bankruptcy/src/tune.py:930-946`
- **嚴重度評級**：**中（Medium，訓練／部署一致性瑕疵，影響超參數手冊口徑與選模證據之完全對齊）**
- **是否影響已固化產物**：**已作為既定行為固化**，已於 `PROJECT_STATE.md:117` 與 `bankruptcy/MODEL_CARD.md:145` 列為法定限制。
- **各階段 M5 配置狀態完整對照清單**：

| 階段 | M5 是否含 `subsample_freq=1` | 是否含重抽樣 | 證據（檔案:行號） |
|---|---|---|---|
| E2a 內層搜尋 | 是 (`subsample_freq=1`) | 是（依 trial 動態抽樣 S0 或 S3） | `bankruptcy/src/tune.py:123, 169` |
| E2a 外層重擬合 | 是 (`subsample_freq=1`) | 是（依該折最佳參數，5 折選中 S3 含 SMOTE 0.3） | `bankruptcy/src/tune.py:335-336`, `bankruptcy/src/cv.py:270-285` |
| E2b 搜尋 | 是 (`subsample_freq=1`) | 是（全訓練集 60 trials 尋優選定 S3 含 SMOTE 0.3） | `bankruptcy/src/tune.py:169, 894-905` |
| E3 評估（讀 E2a 快取） | 是 (`subsample_freq=1`) | 是（同 E2a 外層，5 折含 S3 SMOTE，10 折為 S0） | `bankruptcy/src/ensemble.py:406-425` |
| E4c OOF 參考分布與門檻 | 否 (`subsample_freq=0`) | 是（固化載入 `imbalance="S3"`，包含 SMOTE 0.3） | `bankruptcy/src/thresholding.py:610-615, 774-784`, `bankruptcy/outputs/models/best_params/M5.json` |
| E5 | 否 (`subsample_freq=0`) | 是（依各折最佳參數執行，部分折含 S3 SMOTE） | `bankruptcy/src/explain.py:462, 670-681` |
| E6（已部署固化元件） | 否 (`subsample_freq=0`) | 是（`model_M5.joblib` 包含 SMOTE 0.3 訓練步驟，推論時自動略過） | `bankruptcy/outputs/models/final/model_M5.joblib`, `bankruptcy/src/final_eval.py:110-145` |
| E7b | 否 (`subsample_freq=0`) | 是（固定 `imbalance="S3"`，測試 5 組隨機種子對 SMOTE 與 LightGBM 擾動） | `bankruptcy/src/seed_stability.py:63, 107-113` |
| score_new_data／App | 否 (`subsample_freq=0`) | 是（模型物件內含 SMOTE 步驟，推論 `predict_proba` 自動略過） | `bankruptcy/src/scoring.py:277-330`, `bankruptcy/score_new_data.py:80-120`, `bankruptcy/app/app.py:760-800` |

- **關鍵說明與分析**：
  1. **對外引用數字之配置對應**：
     - **E3 引用數字**（$E_A$ 的 AR@0.95 = 8.06 倍、折間標準差 2.12 倍）：對應 **含 `subsample_freq=1` 之配置**（因 E3 直接取用 E2a 外層快取分數）。
     - **E4a 安全邊際**（$E_A$ 建議邊際 0.04）：基於 E2a/E3 外層 15 次評估分布，對應 **含 `subsample_freq=1` 之配置**。
     - **E6 獨立測試集盲測結果**（AR@0.95 = 5.25x [95% CI: 3.93, 8.18]、PR-AUC = 0.5676 [0.3807, 0.7301]）：對應 **`subsample_freq=0` 之配置**（因 E6 評估的是已固化於 `outputs/models/final/model_M5.joblib` 之正式部署元件）。
  2. **E6 評估之有效性**：
     - **E6 評估的就是已部署配置**。E6 程式碼（`final_eval.py:110-145`）透過 `load_final_components()` 直接載入磁碟上的固化模型，該模型實體參數即為 `subsample_freq=0`。因此，**E6 的盲測效能對已部署模型 100% 真實有效，絕無評估脫節**。
  3. **嚴重度重估為中（Medium）之理由**：
     - 雖然已部署模型之 E6 評估無偏且可信，但「選模與調參證據（E2a/E3，啟用列抽樣 bagging）」與「部署版本（`subsample_freq=0`，停用列抽樣 bagging）」的 M5 超參數存在配置差異。這屬於機器學習工程中典型的訓練／部署一致性（Train-Serve Consistency）瑕疵，在面試與程式審計中應評為中（Medium）。
  4. **處置原則**：
     - **不修復、不重跑任何固化流程**。把「選模與調參證據和部署版 M5 配置不完全相同」寫成專案已知限制。
     - 在 `labs/lab_03_boosting_lr_trees.py` 規劃中，把「`subsample_freq` 0 vs 1 對照（只用訓練資料）」標為第一優先實驗項目。

### F-03：`data_audit.py` 與 `preprocessing.py` 中 `QuantileTransformer` 的 `n_quantiles` 設定差異
- **位置**：`bankruptcy/src/data_audit.py:231` 對照 `bankruptcy/src/preprocessing.py:310-315`
- **現象與原因**：
  `QuantileScaler.fit()`（`bankruptcy/src/preprocessing.py:310`）明確設定 `n_quantiles = min(1000, len(X))`，避免當訓練樣本數小於 1,000 時觸發 scikit-learn 警告；而 `audit_near_duplicates()`（`bankruptcy/src/data_audit.py:231`）直接使用 `QuantileTransformer(output_distribution="normal", random_state=SEED)`（預設 `n_quantiles=1000`）。在完整資料集（6,819 筆）上兩者行為一致，但若對小於 1,000 筆的子集呼叫 `audit_near_duplicates()` 則會產生 `UserWarning`。

### F-04：E3 模型集成評估之超參數來源確認（確認無選擇偏誤）
- **位置**：`bankruptcy/src/ensemble.py:406-415`
- **嚴重度評級**：**資訊查核（無問題，確認合規）**
- **現象與查核結果**：
  在 `bankruptcy/src/ensemble.py:406-415` 中：
  ```python
  cache_file = E2_CACHE_DIR / f"{m}_r{rep}_f{f}.joblib"
  d = joblib.load(cache_file)
  val_scores_raw[m] = d["outer_val_scores"]
  best_imbalance = d["best_imbalance"]
  model_params = d.get("model_params", d.get("best_params", {}))
  ```
  程式直接讀取 E2a 各外層折在本地所儲存的 `outer_val_scores` 與該折獨立尋優所得的 `model_params`，且內層 ECDF 參考分佈亦僅在該折的訓練折內部生成。
- **影響評估**：
  證實 E3 的外層評估確實基於外層驗證折隔離，**並未使用包含外層驗證折在內的全訓練集 E2b 參數**。但在同一組 15 次外層折分數上進行最終模型篩選，仍存在輕微的多重比較選擇偏誤（屬一般經驗，未經本專案驗證）；E6 測試集才是完全獨立的最終檢驗。

---

## 二、既有文件與實際程式碼不一致之處（Documentation vs. Code Discrepancies）

### D-01：內層交叉驗證（Inner CV）折數與 `splits.joblib` 儲存內容
- **實際程式碼（唯一事實來源）**：
  1. `bankruptcy/outputs/models/splits.joblib`（由 `bankruptcy/src/cv.py:119-157` `save_splits()` 產生）僅包含 4 個鍵：`metadata`、`train_idx`、`test_idx`、`cv_splits`（外層 5 折 × 3 次重複 = 15 次外層評估之索引）。**並未在 `splits.joblib` 中預存內層切分或 E2b 10 折切分**。
  2. **E2a 巢狀調參內層 CV**：由 `bankruptcy/src/cv.py:254-290` `make_inner_splits(..., n_splits: int = 3)` 與 `bankruptcy/src/tune.py:292-297` 動態產生，實際為 **內層 3 折（3-fold StratifiedKFold）**（每次外層評估做 3 折內層 CV，15 次外層評估共 45 個內層折；合併內層 3 折 OOF 共 140～141 家破產正樣本計算三點 log 預警倍數損失）。
  3. **E2b 全訓練集調參與 E4c 參考分布生成**：在 `bankruptcy/src/tune.py:856-862` 與 `bankruptcy/src/thresholding.py:619-622` 中，對全量訓練集（5,455 筆）動態執行 **5 折 × 2 次重複（共 10 次評估）**。
  4. **E3 外層訓練集參考分布生成**：在 `bankruptcy/src/ensemble.py:153-155` `get_or_generate_reference_distribution()` 中，為了在每個外層訓練折（4,364 筆）上產生無洩漏 ECDF 參考分布與 Stacking 特徵，動態執行 **內層 5 折（5-fold StratifiedKFold）**（15 次外層評估 × 5 折 = 75 個內層折）。
- **不一致之既有文件位置**：
  - `PROJECT_STATE.md:27`：寫為「`cv.py`：固化切分載入、外層 15 折/內層 5 折管理」。
  - `PROJECT_STATE.md:40`：寫為「固化切分：`outputs/models/splits.joblib`（外層 15 折：5 折 × 3 重複；內層 5 折；E2b 10 折；測試集索引）」。
  - `PROJECT_STATE.md:59`：寫為「最小化內層 5 折合併 OOF 後的三點 log 預警倍數平均」（E2a 實為內層 3 折合併 OOF；E2b 為 5 折 × 2 次重複分別計算損失後取平均）。
  - `docs/presentation_outline.md:74`：寫為「15 折外層 Repeated Stratified K-Fold 評估 + 75 折內層調參」（E2a 內層調參實為 15 × 3 = 45 折；75 折為 E3 生成 ECDF 參考分布之折數）。
  - `docs/student_learning_handbook.md:68, 76, 127, 167, 173, 361, 370` 與 `docs/student_presentation_deck.md:217, 221, 596, 600`：多處將 `tune.py` 的內層調參寫為「內層 5 折」。
  - `tools/number_sources.md:39`：寫為 `bankruptcy/src/cv.py` 定義了 `N_OUTER_FOLDS=15, N_INNER_FOLDS=5`（實際程式碼中無此常數名稱，`bankruptcy/config.py:24-25` 定義之常數為 `CV_FOLDS = 5` 與 `CV_REPEATS = 3`）。

### D-02：外層評估方式與每次評估正樣本數（5 折 × 3 次重複 vs. 不重複 15 折）
- **實際程式碼（唯一事實來源）**：
  `bankruptcy/config.py:24-25`（`CV_FOLDS = 5`, `CV_REPEATS = 3`）與 `bankruptcy/src/cv.py:97-101` 使用 `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`，即 **5 折 × 3 次重複 = 15 次外層評估**：
  - 每次外層訓練折：固定 **4,364 筆**（佔訓練集 80%），含破產正樣本 **140～141 家**（平均 **140.8 家**）。
  - 每次外層驗證折：固定 **1,091 筆**（佔訓練集 20%），含破產正樣本 **35～36 家**（平均 **35.2 家**）。
- **不一致之既有文件位置**：
  - `docs/student_learning_handbook.md:68`、`docs/student_presentation_deck.md:596, 600`、`teaching/step_quizzes.md:167`：誤將「5 折 × 3 次重複 = 15 次外層評估」描述為不重複的 15 折切分（「14 折訓練 93.3% 數據 + 1 折驗證 11~12 家破產」）。

### D-03：共線性相關過濾門檻（`|r| > 0.95` vs. `0.98`）
- **實際程式碼（唯一事實來源）**：
  - `bankruptcy/src/preprocessing.py:353`：`CorrelationFilter.__init__(self, threshold: float = 0.95, enabled: bool = True)` 預設門檻為 **0.95**。
  - `bankruptcy/src/pipelines.py:249`：`build_pipeline()` 固定傳入 `CorrelationFilter(threshold=0.95, enabled=use_corr_filter)`。
  - E1b、E2、E3、E4c（含最終凍結模型 `bankruptcy/outputs/models/final/model_M3.joblib`, `model_M4.joblib`, `model_M5.joblib`）實測屬性 `corr_filter.threshold` 皆為 **0.95**。
  - `bankruptcy/src/data_audit.py:159-160`：EDA 高相關欄位對偵測門檻亦為 `corr_val > 0.95`（並分標 `>0.95` 與 `>0.99` 兩階，存於 `bankruptcy/outputs/tables/high_correlation_pairs.csv`）。
- **不一致之既有文件位置**：
  - `docs/student_presentation_deck.md:614`（及未追蹤腳本 `bankruptcy/scripts/student_slides_data.py:1671`）：將共線性過濾門檻寫為 `$r > 0.98$`，與程式碼 `0.95` 不符。

### D-04：M4 (XGBoost) 超參數與調參目標描述差異
- **實際程式碼与結果檔（唯一事實來源）**：
  - 調參目標函數（`bankruptcy/src/tune.py:113-118`）：最小化三點（$R \in \{0.80, 0.90, 0.95\}$）log 預警倍數平均，而非單點 `AR@R90`。
  - M4 最終固化超參數（`bankruptcy/outputs/models/best_params/M4.json`）：`max_depth: 2`, `subsample: 0.879563935600964`, `n_estimators: 250`, `learning_rate: 0.046776815926747296`。
- **不一致之既有文件位置**：
  - `docs/student_learning_handbook.md:76` 與 `docs/student_presentation_deck.md:660`：寫為 `max_depth=3`、`Subsample 0.8`、優化 `AR@R90`。

### F-05：參數登錄表之 JSON 鍵值與動態衍生參數行號定位機制
- **位置**：`tools/build_param_registry.py` 與 `tools/check_code_docs.py:248-261`
- **嚴重度評級**：**實作機制（已在 3a 工具中解決）**
- **現象與解決方案**：
  在建構機器可讀參數登錄表時，`thresholds.json` 與 `defaults.json` 等靜態設定檔無 AST 變數宣告；若行號填 0 則無法通過符號 ±3 行驗證。本專案透過結構化行號搜尋器精確定位各門檻工作點與預設特徵數在 JSON 檔案中的原始行號（如 `thresholds.json:7` 為 `E_A` R80 門檻），實現 100% 符號行號自動核驗。

### F-06：動態樣本權重 `scale_pos_weight` 之來源類型界定
- **位置**：`bankruptcy/src/pipelines.py:53, 72-74`
- **嚴重度評級**：**口徑規範（已符合使用者指令規範）**
- **現象與規範**：
  `AdaptiveScaleWeightClassifier` 在訓練時動態計算 `scale_pos_weight = n_neg / max(1, n_pos)`，嚴格依賴當前訓練折 `y`，故在參數登錄表中將其來源類型標定為 `derived`，目前值標定為「動態：由訓練折計算」，避免初學者誤以為其為 Optuna 靜態搜尋之固定數值。

### F-07：`n_trials_map` 數值串接成因與字典多列分拆（3a-fix 2.1）
- **位置**：`bankruptcy/src/tune.py:501-507`（E2a `n_trials_map_e2a`）與 `bankruptcy/src/tune.py:865-871`（E2b `n_trials_map_e2b`）
- **嚴重度評級**：**低（報告文字呈現錯誤，原始碼無誤）**
- **成因分析**：
  在階段 3a 初期文字報告中，曾出現「外層折 2540 trials、全集 3860 trials」之字串，實為報告將非快速模式（M1=25, M2~M5=40）與（M1=38, M2~M5=60）之多模型字典數值在文字總結時產生串接。在 `tune.py` 原始碼中，兩者皆為結構清楚的 Python 字典：
  - `n_trials_map_e2a`: `{"M1": 3 if quick else 25, "M2": 3 if quick else 40, ...}`
  - `n_trials_map_e2b`: `{"M1": 5 if quick else 38, "M2": 5 if quick else 60, ...}`
- **修正措施**：
  在登錄表 v2 中，將調參試驗數改為「每個模型一列」（M1～M5 各自獨立 10 列），`目前值` 嚴格填入單一整數（E2a: 25, 40, 40, 40, 40; E2b: 38, 60, 60, 60, 60）。並建立獨立單元測試 `tools/test_registry_extractor.py`（測試通過 Ran 3 tests, OK），確保抽取器絕不對字典數值進行字串拼接。

### F-08：人工備註數值與推測性描述之全面審計與清理（3a-fix 2.2）
- **位置**：`docs/code_learning/param_notes.csv` 與 `docs/code_learning/03a_param_registry.csv`
- **嚴重度評級**：**低（文件筆誤修正與客觀化規範）**
- **審計與修訂清單**：
  1. `CFG_MVP_PASS_TOP10_RECALL`：原始值為 0.5（50%），備註筆誤寫為「至少抓出 80% 破產公司」，已對照 `config.py:44` 註解修訂為 50%。
  2. `CFG_COST_FN`：目前值為 20.0，修正先前「10:1」之非專案口徑筆誤，統一為「20:1 假陰性代價」。
  3. `CFG_MVP_PASS_PR_AUC_MULTIPLIER`：目前值為 3.0，修正先前筆誤。
  4. `CFG_DEFAULT_TARGET_RECALL`：目前值為 0.95，修正先前筆誤。
  5. **主觀推測全面清除**：全表逐列檢查，凡未有專案文獻、Git Commit 或實驗輸出佐證之「選擇理由」，一律改寫為「（專案未記載選擇理由）」；一般經驗說法統一補註「（一般經驗，未經本專案驗證）」。
  6. **檢查器防護機制**：於 `tools/check_code_docs.py --registry` 模式新增備註數字查核規則，掃描備註中所有數字與百分比，確保與同列數值及證據完全吻合。

---

## 三、既有已知議題對照（引自 `docs/KNOWN_ISSUES.md`）

1. **凍結模型目錄暫存檔命名遺留**：`bankruptcy/outputs/models/final/` 下存在 `_tmp_oof_r0_M3.npy` 等含有 `_tmp_` 前綴之檔案（由 `bankruptcy/src/thresholding.py:629-630, 668-669` 產生）。因回歸測試與雜湊驗證依賴此路徑，維持現狀不更名。
2. **防禦性預設值**：`docs/KNOWN_ISSUES.md` 所列之 18 處離線實驗腳本 `.get(..., default)`、`getattr(..., default)` 與特徵重要性 `except Exception:` 容錯區塊，皆屬於非推論路徑之低風險防禦性寫法；推論路徑（`bankruptcy/src/scoring.py`、`bankruptcy/score_new_data.py`）則嚴格執行缺欄與異常顯式拋錯。

---

## 四、階段 3b-fix 卡片事實錯誤審計與成因分析（Stage 3b-fix Audit & Fact Corrections）

### F-09：「相關性過濾門檻與開關」卡片數值與路徑錯誤成因分析（3b-fix 2.1 A）
- **錯誤現象**：
  1. 卡片散文先前寫「保留全部 95 個特徵…啟用過濾（0.95）可將特徵由 95 個精簡為 93 個（剔除 2 個冗餘特徵）」，與已驗證特徵漏斗嚴重矛盾。
  2. 卡片寫「對各模型的 AR@0.95 影響極微（變異小於 0.05 倍）」，與 E1b 實際對照實驗數據未符。
  3. 卡片引用路徑 `bankruptcy/outputs/e1b_correlation_filter_experiment.csv`。
- **真實事實來源（唯讀產物）**：
  1. **真實漏斗**：原始 95 特徵 $\to$ 刪除 1 常數欄位（`Net Income Flag`）與 2 實質重複欄位（`Current Liability to Liability`, `Current Liability to Equity`）剩 92 欄位 $\to$ 相關性過濾（門檻 0.95）在全訓練集精簡為 72 欄位（剔除 20 個高相關特徵）；在外層 15 次評估（5 折 × 3 次重複）訓練折內動態保留 70～73 欄位。先前卡片中的「剔除 2 個」實為前置重複欄位剔除，被筆誤混淆為相關性過濾之成果。
  2. **真實數值（`bankruptcy/outputs/tables/e1b_ablation.csv` 與 `e1b_paired_vs_default.csv`）**：
     - M1（Logistic Regression）：預設（72 欄位）AR@0.95 為 10.87 倍（PR-AUC 0.3808），關閉過濾（92 欄位）惡化至 12.00 倍（PR-AUC 0.3616），PR-AUC 顯著下降 0.0192（p=0.0020）。
     - M2（Random Forest）：預設 AR@0.95 為 9.07 倍（PR-AUC 0.4223），關閉過濾微升至 9.18 倍（PR-AUC 0.4068），PR-AUC 下降 0.0155（p=0.0084）。
     - M3（Extra Trees）：預設 AR@0.95 為 8.87 倍，關閉過濾為 8.88 倍。
     - M4（XGBoost）：預設 AR@0.95 為 8.78 倍，關閉過濾為 8.79 倍。
     - M5（LightGBM）：預設 AR@0.95 為 8.94 倍，關閉過濾為 8.94 倍。
     - 關閉過濾對線性模型 M1 具有高度統計顯著惡化影響，對樹模型影響較輕，絕非「影響極微」。
  3. **真實路徑**：真實檔案為 `bankruptcy/outputs/tables/e1b_ablation.csv` 與 `e1b_paired_vs_default.csv`（缺少 `tables/` 子目錄且檔名不同）。
- **為何先前檢查器未能攔截（成因分析）**：
  舊版 `check_code_docs.py` 僅聚焦於 Markdown 標題語法、程式碼符號引用（`file:line` 格式）與登錄表 ID 存在性，**並未對散文內出現的 CSV 檔案路徑進行磁碟存在性比對，亦未對散文陳述中的數字進行資料庫驗證**，導致散文文字與真實 CSV 脫節。
- **補強措施**：
  1. 引入 `FILE_PATH_PATTERN`：全面掃描文檔中所有以 `bankruptcy/` 開頭之路徑，強制要求必須在磁碟真實存在，否則報錯。
  2. 引入結構化證據標籤（evidence tags）：以結構化標籤執行 pandas 精確查詢，並自動核驗陳述中的全部數字。

### F-10：「安全邊際候選集與達標率門檻」卡片數值與候選數錯誤成因分析（3b-fix 2.1 B）
- **錯誤現象**：
  1. 卡片寫「候選集 18 組（6 個 margin × 3 個目標召回率）」，與登錄表 `MARGINS = [0.0, 0.02, 0.04, 0.06]`（4 個值）矛盾。
  2. 卡片寫「$E_A$ 最終推薦安全邊際 $m=0.04$（在 R90 與 R95 達標率皆超過 80%）」，遺漏 R90 與 R95 之細部推薦差異。
  3. 卡片引用路徑 `bankruptcy/outputs/e4a_threshold_margins.csv`（缺少 `tables/` 且檔名錯誤）。
- **真實事實來源（唯讀產物）**：
  1. **真實候選組數**：程式碼 `thresholding.py:67` 定義 `MARGINS = [0.0, 0.02, 0.04, 0.06]`（4 個值）。在 E4a 實驗中，R80/R90/R95 各評估 4 個邊際，R100 固定僅測 $m=0.0$，故實際評估點為 $3 \times 4 + 1 = 13$ 組（參見 `bankruptcy/outputs/tables/e4a_threshold_transfer_summary.csv` 剛好 13 組候選）。先前的「18 組（6 個 margin × 3 個召回率）」為舊版文檔草稿殘留之推測性數字（舊草稿曾假想 $m \in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]$）。
  2. **真實推薦數值（`bankruptcy/outputs/tables/e4a_margin_recommendation.csv`）**：
     - $E_A$ 於 **R90 推薦邊際 $m=0.06$**（外層 CV 達標率達 93.3%，14/15 次外層評估達標；而在 $m=0.04$ 時達標率僅 73.3%，未達 80% 門檻；未加邊際 $m=0.0$ 時僅 53.3%）。
     - $E_A$ 於 **R95 推薦邊際 $m=0.04$**（外層 CV 達標率達 93.3%，14/15 次外層評估達標；在 $m=0.06$ 時達 100.0%，未加邊際 $m=0.0$ 時僅 53.3%）。
  3. **真實路徑**：真實檔案為 `bankruptcy/outputs/tables/e4a_margin_recommendation.csv` 與 `e4a_threshold_transfer_summary.csv`。
- **成因分析與補強**：
  先前的 18 組與 6 個 margin 來自未經程式碼查證的早期文件臆測；透過強化 `check_code_docs.py` 之數值反查與路徑查核，卡片數據已 100% 對齊真實 CSV。

### F-11：「SMOTE 系列重抽樣參數」卡片數值與口徑錯誤成因分析（3b-fix 2.1 C）
- **錯誤現象**：
  卡片散文先前引用 S4 與 S2 數值時出現口徑偏差（如 10.92 vs 10.98，PR-AUC 0.3541 vs 0.3774），且缺少真實路徑前綴。
- **真實事實來源（唯讀產物）**：
  讀取 `bankruptcy/outputs/tables/e1a_imbalance_comparison.csv`，在專案標準管線 `scaler=='quantile'` 下：
  - M1 基準 S0（不重抽樣）：AR@0.95 = 10.3135 倍，PR-AUC = 0.3938。
  - M1 S1（class_weight）：AR@0.95 = 10.9047 倍，PR-AUC = 0.3611。
  - M1 S2（SMOTE 0.1）：AR@0.95 = 10.7353 倍，PR-AUC = 0.3855。
  - M1 S4（SMOTE 0.5）：AR@0.95 = 10.9752 倍，PR-AUC = 0.3774。
  先前卡片寫法疑似取自未標準化或其他 scaler（如 standard）之數值，造成跨行混用。現已統一對齊 `scaler=='quantile'` 並由結構化證據標籤（evidence tags）完成數值鎖定。
---

## 3. 階段 3c-fix 根因分析與防漏機制總結（F1～F5）

### F1 根因分析：E6 盲測數字口徑矛盾與「13.05%」成因
- **現象**：前次報告於 E6 R90 運作點寫下「調查 13.05% 企業，抓出 43/44 且預警倍數 7.66x」，兩者存在嚴重視覺矛盾（$7.6629 \times 44 / 1364 \approx 24.7\%$，且 Precision 12.76% 對應約 337 家預警而非 178 家）。
- **成因追查**：
  1. 查證 `bankruptcy/outputs/e6/e6_operating_points.csv` 第 3 列（$E_A \times R90$）原始數值：
     `alerts_count=337`, `total_samples=1364`, `alert_proportion=0.247067` (**24.71%**), `alert_ratio=7.659091` (**7.66x**), `precision=0.127596` (**12.76%**), `positives_caught=43`, `total_positives=44`。
  2. 經全文交叉比對，「13.05%」之來源確認為前期撰寫時誤取自單折驗證折之 Precision（如 `e4a_threshold_transfer_per_fold.csv` 第 58 列之 precision 0.130597）或混淆了單折之預警比例。
  3. AR@0.95 CI 區間矛盾：`e6_ranking_metrics.csv` 實測為 `point_estimate=5.250000`, `ci_lower_2.5=3.931818`, `ci_upper_97.5=8.181818`。前期報告寫為 `[4.09, 8.18]` 係誤引用內部中間計算表。
  4. 預警倍數（AR）定義澄清：預警倍數之定義為 $\text{AR} = \frac{\text{Alerts}}{P}$（預警家數 ÷ 破產總家數），絕非「所需受測樣本比例」。
- **防漏修復**：全文件統一為 `(E_A, R90, 預警倍數)` 結構，所有數值改以 `<!-- evidence ... -->` 綁定來源 CSV 欄位；更新檢查器，要求所有實測陳述具體數字均須通過標籤比對。

### F2 根因分析：Brier 表基準列標示錯誤
- **現象**：前期表格將 Brier 0.0312、BSS 0.0000 標為「基準（未校準百分位數）」。
- **成因追查**：
  1. 查證 `thresholding.py:513` 程式碼：`brier_bench = p_mean * (1.0 - p_mean) = 0.032264 * (1 - 0.032264) = 0.031223`。
  2. 在統計定義上，Brier Skill Score ($BSS = 1 - \frac{BS}{BS_{\text{bench}}}$) 的基準在定義上是「永遠預測母體盛行率（Prior Probability）」之無資訊模型，其 $BS_{\text{bench}} = \pi(1-\pi)$。
  3. 未校準的 ECDF 分數為順序百分位排名（Percentile Rank, 均值約 0.50），根本不是機率分佈，直接計算 Brier Score 在統計學上毫無意義（標記為 N/A）。
  4. 評估資料流澄清：Platt 與 Isotonic 係於外層訓練折擬合、於外層驗證折評估（15 次外層獨立評估平均）。
- **防漏修復**：修正表格名稱為「盛行率參考基準（只預測盛行率，BSS=0）」，並載明百分位排名不適用 Brier 評估。

### F3 根因分析：`MARGINS` AST 驗證漏洞與常數校正
- **現象**：先前報告與卡片一度寫出 6 個 margin 候選值，而 `--registry` 卻顯示 PASS。
- **成因追查**：
  1. 查證 `bankruptcy/src/thresholding.py:67`：原始程式常數僅有一處宣告，即 `MARGINS = [0.0, 0.02, 0.04, 0.06]`（4 個值）。
  2. 先前檢查器漏洞：先前 `--registry` 僅驗證行號是否存在，未透過 Python AST 在引用行實際求出變數與常數字面值進行逐字/數值比對。
- **防漏修復**：
  1. 於 `tools/check_code_docs.py` 中實作基於 Python AST 之 `extract_python_ast_value`，對所有 `code_constant` 於 AST 節點中提取字面值並與登錄表嚴格核對。
  2. 執行負向測試：刻意將常數改為 `[0.0, 0.02, 0.04, 0.08]`，成功觸發 Value mismatch 攔截（退出碼 1）。

### F4 根因分析：證據標籤覆蓋率不足與檢查器升級
- **現象**：前次 `--evidence` 僅涵蓋 14 個標籤，大量卡片與導讀之具體數字處於未受機器檢驗之真空狀態。
- **成因追查**：先前檢查器僅針對「已有標籤者」進行驗證，對「無標籤之具體數字」僅標為 WARNING，缺乏硬性阻擋機制。
- **防漏修復**：
  1. 強化正則表達式，支援 `12.34x`、百分比、中文單位（`倍`、`家`、`筆`、`折`、`次`）。
  2. 升級檢查器：凡帶【本專案實測】之陳述與 `_facts_verified.md` 表格列，若包含具體數字卻無標籤對應，一律升級為 **ERROR**（退出碼 1）。
  3. 支援 `<!-- code ... -->`（AST 比對程式常數）、`<!-- cmd ... -->`（動態測試統計宣告）與 `<!-- no-evidence: ... -->`（純定義/算術推導豁免）。
  4. 執行負向測試：刻意在陳述中加入未綁定數字 `12.34x`，成功觸發 `Unverified number(s) in claim` 錯誤攔截。

### F5 根因分析：E1 統計口徑與顯著性校正
- **現象**：前次報告 S 系列表格遺漏 S3、S5、S6，且 M1 關閉相關性過濾之「顯著」誤用原始 p 值。
- **成因追查**：
  1. 查證 `e1a_imbalance_comparison.csv`，補齊 S0～S6 全量 7 組策略實測值。
  2. 查證 `e1b_paired_vs_default.csv`：M1 關閉相關性過濾之 Wilcoxon 原始 $p=0.0020$，但在考慮外層折間相關性的 Nadeau-Bengio 修正檢定下 $p=0.2078$（95% CI 跨越 0），且未通過 Holm 多重比較校正（未達統計顯著）。
- **防漏修復**：更正全文件陳述為「未通過多重比較校正，未達統計顯著差異」；全文嚴格區分原始 p 值與 Nadeau-Bengio / Holm 校正後 p 值。

---

## 4. 階段 4a 實作與 F-02 專案內經驗結論

### 4.1 F-02 專案內實證對照結果（lab_03）
- **實驗背景**：專案最佳超參數 `best_params/M5.json` 固化了 `subsample=0.6071695513247772`，但在 `tune.py` 搜尋空間定義中，`subsample_freq` 預設為 0（或依條件指派）。依據 LightGBM 官方規格，若 `subsample_freq=0`，則不啟用 Bagging，`subsample` 實質未生效；必須設為 `subsample_freq=1` 才會在每輪執行隨機子樣本抽樣。
- **實測對照（5 折 CV，3 組隨機種子 [42, 100, 999]）**：
  1. **參數生效檢驗**：透過 `pipe.named_steps["classifier"].get_params()` 檢驗，設定 A 實際生效 `subsample_freq=0`，設定 B 實際生效 `subsample_freq=1`。
  2. **分數逐位比對**：跨 3 組種子共 15 折驗證，**沒有任何一折的分數為逐位相同（`is_bit_identical=False`）**，最大絕對預測分數差為 0.3034。這證實 LightGBM 在 `subsample_freq=1` 時確實啟動了隨機抽樣，打破了完全確定性。
  3. **指標與相關性量級**：
     - 配對 PR-AUC 差異（設定 B 減設定 A 均值）：`-0.00572`（差距僅約千分之五）。
     - 配對 AR@0.95 差異（設定 B 減設定 A 均值）：`-0.0008x`。
     - 兩者預測分數之皮爾森相關係數：`0.983647`。
- **結論與限制**：在專案訓練資料與目前超參數設定下（`num_leaves=43`、`reg_lambda=0.027`、S3 SMOTE 強約束），啟用每輪抽樣帶來的指標變動極微（相關性高達 0.984）。**此結論作為 F-02 在訓練集內的專案經驗證據，嚴格不推廣到部署版 M5 在未見 E6 盲測集上的影響**。

### 4.2 lab_01 至 lab_04 核心發現與專案設計對照
1. **單樹過擬合與集成優勢（lab_01）**：單棵決策樹在深度 4 時達到驗證 PR-AUC 高峰（0.2692），深度為 None 時訓練 PR-AUC 達 1.0000 但驗證 PR-AUC 跌至 0.1066（差距 +0.8934）；相對地，M2、M3、M4 集成模型在各深度均維持在 0.38～0.42，證實集成平均有效消除方差。
2. **森林超參數收斂（lab_02）**：M2 與 M3 樹木數量增加至 200～400 棵時指標收斂且折間標準差維持穩定，`min_samples_leaf` 在 2～5 之間最優，驗證專案 E2b 固化值（M2=10, M3=2）的合理性。
3. **提升法學習率與樹數取捨（lab_03）**：M4 在 `lr=0.05, n_est=150` 時達最高 PR-AUC 0.4526，較高學習率（`0.15`）搭配多樹會提早過擬合。
4. **不平衡重抽樣與機率扭曲（lab_04）**：S1（類別權重）與 S3（SMOTE）使預測機率大幅膨脹（M1 由 0.0327 暴衝至 0.1716，膨脹 5.3 倍；Brier 惡化至 0.0883），而排序指標並未提升，直觀揭示了重抽樣破壞機率校準的機制。


---

## 5. 階段 4a-fix 與階段 4b 實作與驗證結論

### 5.1 G1 根因分析：F-02 深入對照（45 組配對檢定）與結論校準
- **問題**：先前報告僅基於 5 折 x 1 次重複即聲稱「證實影響極其微小」，結論措辭超出統計證據。
- **成因追查與擴大實驗**：
  1. 實作 `--f02-extended` 專屬模式，擴大至 45 組配對（5 折 x 3 次重複 x 3 模型種子 [42, 123, 2025]）。
  2. 實測統計：
     - PR-AUC 平均配對差（設定 B 減設定 A）：`+0.00348`（Std: 0.02421），t-區間 95% CI 為 `[-0.00379, +0.01076]`（涵蓋 0），符號為正比例為 53.3%。
     - AR@0.95 平均配對差：`-0.0861x`（Std: 1.7718），t-區間 95% CI 為 `[-0.6184, +0.4462]`（涵蓋 0），改善比例為 48.9%。
     - 模型種子雜訊參照：設定 B 跨種子之 PR-AUC 雜訊均值為 `0.01817`，AR@0.95 雜訊均值為 `1.0448x`。
  3. **嚴格結論判定**：配對差均值（+0.00348）小於模型種子雜訊（0.01817），且 95% CI 跨越 0。依規則嚴格陳述為**「在本 lab 設定下未偵測到顯著差異（樣本與檢定力有限）」**，嚴禁外推至 E6 盲測集。

### 5.2 G2 根因分析：單樹粗階梯同分機制
- **問題**：單樹深度掃描中 AR@0.95 出現約 31.00x 之失真數值。
- **成因追查**：單棵決策樹葉節點少（例如無深度限制時僅有 0.0 與 1.0 兩個唯一機率值），機率呈粗階梯分佈。在悲觀同分規則（`score >= thr`）下，為了達到 95% 召回率，門檻必須涵蓋包含少數破產樣本的龐大同分葉節點，迫使全折所有樣本（Alerts = N）全數被納入預警名單，使預警倍數退化至母體全體比例 N/P 約 31.00x。在 `test_common.py` 中以 20 筆同分玩具資料驗證此退化現象（`test_pessimistic_tie_breaking_toy_data`）。

### 5.3 G3 根因分析：M1 S3 vs S6 幾何特性與 class_weight 機率膨脹差異
- **M1 S3 vs S6 數值完全相同根因**：S3 是 `SMOTE(0.3)`，S6 是 `SMOTETomek(SMOTE(0.3))`。在專案前處理管線中，先經 `QuantileScaler` 將 76 維特徵縮放至均勻分佈 [0, 1]。在此高維均勻空間中，`TomekLinks` 未能找到互為最近鄰的跨類別邊界對（剔除 0 筆），故 `SMOTETomek` 輸出的重抽樣資料集與 `SMOTE` 100% 逐位相同。此現象與專案正式 E1a 表格（`e1a_imbalance_comparison.csv`）中 M1 的 S3 與 S6 共享完全相同的 `alert_ratio@recall_0.95_mean = 10.865926` 互相印證。
- **class_weight 機率膨脹機制**：
  - M1（Logistic Regression）：`class_weight='balanced'` 等價於在 Log-odds 上加入常數平移 ln(w1/w0) 約 +3.4，全樣本預測機率被線性平移抬高，平均預測機率暴衝 5.3 倍（0.0327 -> 0.1716，Brier 惡化至 0.0883）。
  - M3（ExtraTrees）：`class_weight` 僅調整節點分裂時的 Gini 不純度計算，葉節點機率仍是樣本計數比例，多數葉節點仍為純負樣本（機率 0.0），故平均預測機率僅由 0.0331 升至 0.0402（僅 1.2 倍）。

### 5.4 階段 4b：Lab 05 至 Lab 08 核心實證發現
1. **Lab 05（前處理消融）**：
   - 特徵漏斗折內實測：無過濾 92.0 特徵 -> 門檻 0.95 保留 76.0 特徵 -> 門檻 0.80 保留 56.4 特徵。
   - 縮放器比較：QuantileScaler 使 M1 取得 PR-AUC 0.4064（AR@0.95 9.15x），改為 StandardScaler 後跌至 0.3979（AR@0.95 惡化至 10.67x）；M3 改用 StandardScaler 後 AR@0.95 惡化至 18.38x。
   - 切分雜訊參照：切分種子 42 vs 100 的雜訊為 PR-AUC 0.0037，消融差異落於雜訊範圍內。
2. **Lab 06（召回率權衡與門檻轉移）**：
   - 隨機基準線平坦固定於 N/P = 31.00x。M3 在 R=0.80 時 AR 為 4.22x，但在 R=0.95 時 AR 劇增至 10.09x（膨脹 2.4 倍）。
   - 門檻轉移實驗：R=0.90 下，m=0 達標率僅 33.3%，加 m=0.06 安全緩衝後達標率升至 100.0%（重現正式 E4a 的 Margin 推薦邏輯）。
3. **Lab 07（CV 折數矩陣與階梯化）**：
   - 折內正樣本數與變異：K=3（58.7 正樣本，Std 0.0356）、K=5（35.2 正樣本，Std 0.0564）、K=10（17.6 正樣本，Std 0.0925，AR@0.95 Std 達 8.99x）。
   - 樣本階梯化：保留比例 10%（正樣本約 18 家）時 Clopper-Pearson 95% 寬度達 0.0320（AR@0.95 IQR 達 11.81x）；保留比例 30% 時收斂至 0.0178。
   - 分層平衡單元測試：驗證 K in {3, 5, 10} 各折正樣本差額 <= 1 通過。
4. **Lab 08（種子穩定性與隨機性分解）**：
   - 變異分解：演算法隨機性（條件 1）PR-AUC 跨種子標準差僅 `0.0053`；資料抽樣變異（條件 2）跨切分標準差為 `0.0126`（2.4 倍）；折間標準差高達 `0.0571`（10.8 倍）。
   - 正式 E7b 對照：E7b 中 PR-AUC 的 fold_std（0.0791）為 seed_std（0.0017）的 46.5 倍；seed_std / fold_se 僅 8.3%，證實資料抽樣變異遠大於演算法隨機種子變異。
---

## 6. 階段 4b-fix 根因分析與技術事實確認（H1～H5）

### H1 根因分析：lab_04 M3 數據查證與 TomekLinks 剔除量實測
1. **回報文字誤植成因**：階段 4b 回報 Markdown 表格中，M3 之 S0、S2、S3、S6 曾出現「0.4368、0.4363、9.14x」，經查係手動整理回報時誤複製自 `lab_05` 前處理消融實驗組數據。磁碟上之原始 CSV `labs/outputs/lab_04/lab_04_imbalance_strategies.csv` 與版本庫快照自始至終完全正確無誤：
   - S0: `pr_auc=0.4402`, `ar95=8.54x`, `mean_proba=0.0331`
   - S1: `pr_auc=0.4412`, `ar95=8.77x`, `mean_proba=0.0402`
   - S2: `pr_auc=0.4265`, `ar95=9.55x`, `mean_proba=0.0336`
   - S3: `pr_auc=0.4157`, `ar95=7.95x`, `mean_proba=0.0347`
   - S4: `pr_auc=0.4080`, `ar95=7.87x`, `mean_proba=0.0358`
   - S5: `pr_auc=0.4173`, `ar95=8.47x`, `mean_proba=0.0348`
   - S6: `pr_auc=0.4157`, `ar95=7.94x`, `mean_proba=0.0347`
2. **S3 vs S6 在 M3 上等價機制（實證檢驗）**：經在 5,455 筆訓練集之 5 折交叉驗證中實測 `TomekLinks` 實際剔除筆數：
   - Fold 0: S3 長度 5,491, S6 長度 5,491, 剔除 0 筆
   - Fold 1: S3 長度 5,489, S6 長度 5,489, 剔除 0 筆
   - Fold 2: S3 長度 5,489, S6 長度 5,489, 剔除 0 筆
   - Fold 3: S3 長度 5,489, S6 長度 5,489, 剔除 0 筆
   - Fold 4: S3 長度 5,489, S6 長度 5,489, 剔除 0 筆
   實測證實 TomekLinks 在 QuantileScaler 將 76 維特徵縮放至均勻分佈後，5 折皆未剔除任何樣本（0 筆）。因此 S3 與 S6 之訓練特徵與標籤矩陣完全相同，僅極微小之浮點隨機誤差造成 0.000019 的 PR-AUC 差異，兩者在統計與工程上完全等價。
3. **雜訊標註標準**：依據 `lab_08` 測得 M3 之演算法隨機種子變異標準差為 0.0053，凡指標差異 $\le 0.005$ 者，均嚴格標註為「未超出雜訊」。

### H2 根因分析：lab_08 與 E7b 比較措辭嚴格校準
1. **隨機性量級差異**：E7b 評估對象為最終集成模型 $E_A$（跨 5 模型種子之 PR-AUC Std 僅 0.001700）；而 `lab_08` 條件 1 評估對象為單一模型 M3（跨 5 模型種子之 PR-AUC Std 為 0.005284，約為 $E_A$ 的 3.1 倍）。兩者評估對象不同，不可寫成「印證 E7b」。
2. **嚴格結論邊界**：修正結論為「兩者均獨立顯示演算法隨機種子變異遠小於資料抽樣（折間）變異」。
3. **協定與經驗說明**：E7b 輪換的是模型種子與 SMOTE 種子（42～46），固定外層 15 折切分；「集成較單模型穩定」屬機器學習一般經驗，專案內未進行同架構單模型 vs 集成之種子方差嚴格對照實驗，故文檔中明確標示為一般經驗。

### H3 根因分析：E6 邊際調查 187 家之限制說明與 E4a 補充
1. **單次測試集限制說明**：E6 測試集破產企業僅 44 家（$P=44$），由 R90 升至 R95 調查名單增加 187 家（337 $\to$ 524 家）但僅多抓 1 家破產企業（43 $\to$ 44 家），此屬單次測試集之特定離散觀測。由於破產樣本稀少，極端召回區間的樣本點具有較大偶然性，邊際成本估計之方差極大，不可外推為固定不變之客觀規律。
2. **多折交叉驗證互補（E4a 跨折均值與標準差）**：
   - R80 預警倍數：均值 3.53x（Std: 1.15）
   - R90 預警倍數：均值 5.64x（Std: 1.63）
   - R95 預警倍數：均值 12.63x（Std: 3.52，含 $m=0.04$ 安全邊際）
   證明在連續分佈與多折驗證下，高召回率區間預警倍數確實存在顯著膨脹。
3. **措辭校正**：全文件刪除「飆升」「暴衝」等誇飾動詞，改用客觀技術描述並綁定證據標籤。

### H4 根因分析：保護檔案 git diff 雜湊與 manifest 欄位隔離
1. **保護檔案一致性證明（git diff sha256）**：
   - `git diff -- docs/README.md` SHA-256：`493615c404f487d81efc0d7dcde4038c3538ca14448913d77a8770b6585db233`
   - `git diff -- "開啟指令.txt"` SHA-256：`8bb301ccb776ba3ee5f18b2151d284d1a5058e864827e6c6599f89709adaa2b7`
   兩檔案與階段 0 開始時的工作區狀態完全一致，未發生任何修改、stage 或 commit。
2. **manifest 欄位重構**：
   - 升級 `labs/_common.py:save_manifest`，新增 `git_dirty_labs`（僅檢查 `labs/` 與 `tools/` 未提交變更）、`git_dirty_other_count`（其他工作區未追蹤/修改檔案數量）與 `git_head`，使 Lab 執行記錄不受工作區外部檔案影響。

### H5 根因分析：Lab 模式與 Quick 縮減範圍總表
- 於 `labs/README.md` 增設模式總表，明確定義 quick、standard、full 各自縮小之維度（折數、重複次數、網格點數、樹數、模型數、種子數），並載明「quick 模式結論品質低於 standard，不能拿來下結論，僅用於快速冒煙測試（確認程式與管線可正常運行）」。

---

## 7. 階段 5 實驗操作指南與指令 AST 檢驗架構

### 7.1 指令真實性與 AST 靜態驗證機制（`--commands`）
1. **問題背景**：文檔編寫中若發明不存在的 CLI 參數（例如假想 `--n-trials` 或 `--model-type`），會誤導學習者並造成執行退化。
2. **實作架構**：
   - 於 `tools/check_code_docs.py` 新增 `--commands` 模式，靜態解析 Markdown 內所有 code fence 及 inline 指令。
   - 利用 Python `ast.walk` 掃描目標腳本之 `parser.add_argument(...)` 呼叫，動態提取合法旗標集合。
   - 對指令中所有以 `-` 或 `--` 開頭之選項進行白名單比對；若指令指向不存在之檔案，或傳入未定義之選項，立即中斷並回報 ERROR。
3. **單元測試與負向檢定（`tools/test_commands_checker.py`）**：
   - 6 項單元測試 100% 通過。
   - 負向測試驗證：成功攔截不存在腳本（`fake_non_existent.py`）、未知參數（`--fake-param`）、無 argparse 腳本傳參（`preflight.py --quick`）以及 Markdown 臨時檔案注入測試。

### 7.2 實驗因果鏈與重現路徑設計
1. **因果鏈精確綁定**：Mermaid 流程圖中每條邊均精確對齊實體檔案與程式碼行號（如 `preprocessing.py:132`、`splits.joblib`、`best_params/*.json`、`e3_selection.csv`、`models/final/*.joblib`）。
2. **E6 盲測集受控防護（雙軌方案）**：
   - **方案 A（推薦）**：常態學習與重現僅在 5,455 筆訓練集內部執行 E0～E5 與 E7，以 E4c 固化產物為驗收終點，嚴格不接觸 E6，維護資料集盲測純潔性。
   - **方案 B（沙盒）**：若需演練 E6 流程，必須在全新獨立分支中重新切分 `splits.joblib`，並以 `--dry-run` 進行演練，主分支嚴禁解除 `E6_LOCK`。
3. **常見重跑陷阱**：詳盡解析隨機種子漂移（`SEED=42` 與 `PYTHONHASHSEED`）、磁碟快取殘留（`e1_cache/`、`e2_cache/`、`e5_cache/`）、套件次版本浮點差異、以及 SHA-256 凍結測試防護機制。

---

## 8. 階段 5-fix 查證與修正總結（I1～I8）

### I1. `lab_04` 數據口徑、執行模式與成因查核
1. **磁碟真實數值（唯讀產物比對）**：
   - 查證 `labs/outputs/lab_04/lab_04_imbalance_strategies.csv` 與快照檔 `docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv`：
     - M3 S0：PR-AUC `0.440168`、AR@0.95 `8.536667`、平均預測機率 `0.033133`、Brier 分數 `0.023185`。
     - M3 S3：PR-AUC `0.415741`、AR@0.95 `7.947937`、平均預測機率 `0.063861`、Brier 分數 `0.026952`。
   - 原始磁碟 CSV 與快照 CSV 自始至終完全一致且正確無誤。
2. **回報數值矛盾成因追查**：
   - 先前回報文字手動整理 Markdown 表格時，打字筆誤將 S3 之 `mean_proba_mean` 鍵入為 0.0347（實為 0.0639），將 S0 之 Brier 鍵入為 0.0267（實為 0.0232）。
3. **模式與產生時間確認**：
   - 查證 `labs/outputs/lab_04/run_manifest.json`：執行模式為 `--quick`（5 折 × 1 次重複，種子 42），產生於 `2026-10-29T09:49:41.305944+00:00`。
   - 在文檔與 README 內明確標註「`--quick` 模式實測」，嚴禁不同模式數據混用。

### I2. 「S3 預警倍數最優」措辭校正與雜訊對照
1. **雜訊對照與統計邊界**：
   - M3 演算法隨機種子變異（來自 `lab_08` 實測）：PR-AUC Std 為 `0.0053`，AR@0.95 Std 為 `1.04x`。
   - S3 相較於 S0 之 AR@0.95 差異：`7.95x - 8.54x = -0.59x`，該差異小於種子雜訊（1.04x），處於隨機波動範圍內，在統計上不構成顯著改善。
   - S3 相較於 S0 之 PR-AUC 差異：`0.4157 - 0.4402 = -0.0245`，顯著超出 PR-AUC 種子雜訊（0.0053）達 4.6 倍，代表整體排序效能實質受損。
2. **客觀改寫**：
   - 改寫為：「S3 的 AR@0.95 較低（7.95x vs S0 8.54x），但差異未超出種子雜訊（約 1.04x），且 PR-AUC 低於 S0（0.4157 vs 0.4402，超出雜訊 0.0053），不能據此判定 S3 較好」。
   - 全文掃描文檔，凡未附雜訊比較的「最優」「最佳」「勝出」字眼均已校準為客觀描述。

### I3. `lab_05` M3 StandardScaler AR@0.95=18.38x 數學機制與特徵數查核
1. **每折數值追查**：
   - M3 在 `Scaler=standard` 下 5 折實測數值：
     - Fold 0: PR-AUC 0.5440, AR@0.95 = 5.92x, AR@0.90 = 5.72x
     - Fold 1: PR-AUC 0.3938, **AR@0.95 = 31.17x**, AR@0.90 = 5.09x
     - Fold 2: PR-AUC 0.4330, AR@0.95 = 9.74x, AR@0.90 = 9.00x
     - Fold 3: PR-AUC 0.3073, **AR@0.95 = 31.17x**, AR@0.90 = 8.77x
     - Fold 4: PR-AUC 0.3537, AR@0.95 = 13.89x, AR@0.90 = 6.77x
     - 5 折均值：$(5.92 + 31.17 + 9.74 + 31.17 + 13.89) / 5 = \mathbf{18.38x}$。
2. **數理機制確證**：
   - 樹模型對單調轉換無關，但 StandardScaler 與 QuantileScaler 的轉換非等價單調（StandardScaler 保留長尾偏態，QuantileScaler 轉為常態分佈）。
   - 在 StandardScaler 下，24 個大值特徵嚴重正偏態，ExtraTrees 隨機特徵分割難以在稀疏區域切出有效閾值。在 Fold 1 與 Fold 3 中，最後 2 筆破產正樣本在全部 80 棵樹中均落入純負樣本葉節點，預測機率恰好為 `0.0`。
   - 依據悲觀同分規則（`score >= cutoff`），當達到 95% 召回（需抓 34/35 家）時，截止門檻退化至 0.0，使全折 1,091 筆樣本全被預警，導致 Fold 1 與 Fold 3 的預警倍數精確退化為 $N/P = 1,091 / 35 = \mathbf{31.17\text{x}}$！而在 AR@0.90 下（只需抓 32 家），第 32 筆正樣本分數 $>0.0$，門檻未退化，AR@0.90 均值完全正常（7.07x）。
3. **特徵保留數 76 欄 vs 70～73 欄條件差異**：
   - 啟用截尾（`use_winsorizer=True`，lab 基準）：各折動態保留 **74～77 欄**（均值 75.60 欄，`lab_05` 5 折為 76.0 欄）。截尾使極端值平滑，降低虛假極高共線性，保留較多特徵。
   - 關閉截尾（`use_winsorizer=False`，專案正式管線）：各折動態保留 **70～73 欄**（均值 71.87 欄，全訓練集為 72 欄）。兩者前處理條件不同，數值完全精確對應。

### I4. `lab_06` 數字口徑分離與 Standard 模式重跑
1. **口徑分離**：
   - **純排序指標 $AR@R$（無門檻）**：依經驗累積分布逆推，衡量模型排序能力。
   - **運作門檻轉移實際預警倍數（含邊際 $m$）**：由內層 OOF 導出门檻套至外層，衡量實際調查負擔。
2. **Standard 模式實測重跑（5 折 × 3 內層折）**：
   - 執行 `python labs/lab_06_recall_alert_curve.py --mode standard`（耗時 81.48s），產出 `recall_alert_curve.csv` 與 `threshold_transfer.csv`。
   - 對照 E4a 正式實驗（15 次外層評估）：在 $R=0.95$ 下，$m=0.04$ 均達到 **100.0% 達標率**（實際預警倍數 12.25x vs 12.63x），完全印證 E4a 之推薦安全邊際決策。

### I5. 卡片抽查檔名勘誤
1. 實際檔案清單確認：`docs/code_learning/03_參數手冊/` 下共有 11 個檔案（`01_資料與切分.md`～`11_應用層.md`）。
2. 先前報告提及之「04_特徵工程.md」與「12_營運與成本.md」係手寫回報時誤引用早期大綱草稿檔名。已全面修正 `04_實驗操作指南.md` 中所有引用連結。

### I6. 資源與耗時依據標記
1. 全專案運算依據：**專案未使用 GPU**。全專案所有模型（M1～M5）與前處理元件均運行於 CPU，無 GPU 參數配置（`pipelines.py:126-155`、`tune.py:65-150`）。
2. 耗時來源：E2a 為【專案紀錄】（`e2a_nested_per_fold.csv:elapsed_seconds` 總計 19,143.8 秒 ≈ 5.3 小時）；其餘為本機實測或推測；專案未記載硬體資源紀錄。

### I7. 證據標籤掃描範圍擴展與實測約束
1. `--evidence` 掃描範圍擴展至包含 `labs/README.md`，驗證標籤數由 98 擴充至 **141 個**，0 errors。
2. 檢查器新增硬性約束：文件中每個「本專案實測」陳述均必須包含對應之結構化標籤，否則報錯阻擋。

### I8. 受保護檔案雜湊、行數與 13 階段安全重跑總表
1. **受保護檔案雜湊與行數**：
   - `docs/README.md`：12 新增、8 刪除（共 20 行變更），SHA-256：`493615c404f487d81efc0d7dcde4038c3538ca14448913d77a8770b6585db233`
   - `開啟指令.txt`：11 新增、0 刪除（共 11 行變更），SHA-256：`8bb301ccb776ba3ee5f18b2151d284d1a5058e864827e6c6599f89709adaa2b7`
   - 總計：`2 files changed, 23 insertions(+), 8 deletions(-)`。
2. `04_實驗操作指南.md` 增設 13 階段安全重跑判定總表。
3. `--commands` 32 條指令清單全體驗證通過，標註 E6 與覆寫 output 者。


## 階段 6 發現：動手改程式配方之架構查核、沙盒驗證與檢查器實施

### J1. 配方 1（新增模型 M6）之全域同步節點
1. **涉及模組與精確行號**：
   - `bankruptcy/src/pipelines.py:131-209` (`_get_classifier`)：新增分類器工廠分支。
   - `bankruptcy/src/pipelines.py:264-280` (`get_score`)：確認分類器提供 `predict_proba` 以支援連續機率分數擷取。
   - `bankruptcy/src/tune.py:121-178` (`sample_hyperparameters`)：定義 Optuna 隨機搜尋空間。
   - `bankruptcy/src/tune.py:501, 865` (`n_trials_map`)：分別於 E2a 巢狀 CV 與 E2b 最佳參數固化字典新增配額。
   - `bankruptcy/src/ensemble.py:73, 75` (`SINGLE_MODELS` 與 `ENSEMBLE_CANDIDATES`)：登錄單模型名單與集成組合。
2. **沙盒驗證成果**：在 `scratch/recipes/` 副本上實作 `HistGradientBoostingClassifier`，以 100 筆合成資料擬合，分數輸出介面與形狀均 100% 符合管線契約。
3. **方法論邊界硬性約束**：已固化之 E6 測試集結果與新模型不可直接比較；嚴禁在現有專案中重新解封測試集。

### J2. 配方 2（新增前處理 transformer）之行獨立性與防洩漏檢測
1. **防洩漏規範**：依循 `preprocessing.py:27` 繼承 `BaseEstimator, TransformerMixin`，`fit` 僅校驗特徵數量，`transform` 執行純逐列運算。
2. **沙盒行獨立性檢測（Row-Independence Leakage Detector）**：
   - 乾淨轉換器 `RatioFeatureAdder`：單一樣本 $A$ 獨立 transform 與混入批次 $[A, B, C]$ transform 結果完全一致（`assert_allclose`，差異 0.0）。
   - 錯誤示範 `LeakyMeanScaler`：在 `transform` 內部使用整批均值，造成樣本 $A$ 的輸出受到同批次其他樣本干擾，產生最大 154.67 數值偏移，成功被洩漏檢測器攔截。

### J3. 配方 4（新增評估指標）之悲觀同分一致性
1. **同分處置機制**：實作 `precision_at_top_k`，完全套用專案既有悲觀同分規則（`recall_at_top_k_percent` 相同邏輯，負樣本優先佔用配額）。
2. **20 筆玩具資料手算驗證**：
   - 正負樣本完全分離時（理想排序）：預算 $k=2$，2 筆正樣本全中，精確率為 $1.0$。
   - 門檻存在平手時：預算 $k=2$，門檻處 1 筆正樣本與 1 筆負樣本平手且僅剩 1 個配額，悲觀規則假設負樣本優先被挑選，精確率精確等於 $0.50$，避免隨機打散帶來的數值浮動。

### J4. 配方 6（換成真實世界資料重做）之 9 大檢查清單與核心限制
1. **本專案核心架構限制**：本資料集因去識別化缺乏公司 ID 與財報年度，故專案目前採用 `StratifiedKFold`。若遷移至真實資料，**必須強制採用 `GroupKFold(groups=company_id)` 或滾動時間序切分（Rolling Window Split）**，以消除嚴重的跨時間資料洩漏。
2. **盛行率貝氏敏感度**：真實市場破產率（如 0.5%）遠低於本專案之 3.23%。依 E7a 貝氏推演，相同模型在 0.5% 盛行率下精確率將下降約 6 倍，預警倍數將顯著放大。
3. **外部驗證嚴正聲明**：外部驗證尚未做，目前所有結果只代表這一份資料，不能代表真實世界的表現。

### J5. `--recipes` 檢查器與單元測試
1. `tools/check_code_docs.py` 擴充 `--recipes` 模式，自動檢查 11 項固定模板標題、`檔案:行號` 之 ±3 行符號檢查、以及高風險配方之失效產物說明。
2. `tools/test_recipes_checker.py` 通過 4 項單元測試，包含 3 項負向測試（缺標題、符號/邊界錯誤、高風險缺失效說明）。
## 階段 7 發現（練習題、小測驗與沙盒除錯題庫）

### K1. 練習題結構與答案物理隔離
1. **物理分離原則**：題目（`06_練習題.md`）與測驗（`06_小測驗.md`）嚴格與解答（`06_解答.md`）物理分離，不包含任何解答內容。
2. **禁用詞機器約束**：`check_code_docs.py --exercises` 實作正則掃描，除引用檔名 `06_解答.md` 外，禁止在 `06_練習題.md` 中出現「解答」「答案」「正確」字樣，違者直接判定為 Error。
3. **題項結構完整覆蓋**：讀碼題 10 題覆蓋 8 大核心模組（排除 E6 封存評估）；預測題 10 題對齊已完成之 lab_01～lab_08 快照，引用 18 處 `--evidence` 標籤；除錯題 10 題對應沙盒腳本；小測驗 15 題各附 4 個選項與依據來源。

### K2. 沙盒除錯題庫安全邊界與合成資料量化偏差
1. **沙盒安全四重防線**：
   - 腳本物理隔離於 `labs/bugs/`，不進入 `bankruptcy/`，不被 `pytest bankruptcy/tests` 收集。
   - 100% 採用合成資料（`make_classification`），嚴禁讀取 `data.csv`、不呼叫 `get_train_data()`、不 import 專案資料載入模組。
   - 頂部強制包含 5 行警語標頭，嚴禁放置 `# BUG` 或洩漏答案的註解。
   - 執行輸出表面維持「看似流暢正常」，凸顯實務缺陷之隱蔽性。
2. **合成資料量化偏差實測（量級僅供教學示範，不外推套用至專案）**：
   - **Bug 02（切分前 SMOTE）**：合成樣本跨折滲透引發嚴重資料洩漏，PR-AUC 異常虛增至 0.9994（誠實折內 SMOTE 僅為 0.2143），偏差高達 **+0.7852**。
   - **Bug 03（極端不平衡 Accuracy 陷阱）**：模型對少數類召回率僅 50.00%，但帳面 Accuracy 達 98.33%，造成嚴重判別盲區。
   - **Bug 06（樂觀判定 `>` vs 悲觀同分 `>=`）**：在離散分佈下，樂觀判定宣稱預警倍數僅 0.62x，實測召回率僅 56.25%（目標 90% 嚴重落空）；悲觀同分判定召回率達 93.75%，揭示真實預警倍數為 18.44x。
   - **Bug 09（切分不一致成對檢定）**：未對齊切分下 p-value 為 0.7910（無法檢出差異）；對齊切分消除抽樣雜訊後 p-value 下降至 0.0652。

### K3. `--exercises` 檢查器模式與單元測試
1. **檢查器模式整合**：在 `tools/check_code_docs.py` 擴充 `--exercises` 模式，自動驗證：
   - `06_練習題.md` 無解答關鍵詞、所有模組/實驗連結存在、預測題引用 lab 在 `lab_map.csv` 狀態為「已完成」。
   - `labs/bugs/*.py` 存在必要警語標頭、無專案資料 import、無 `# BUG` 洩漏註解。
2. **單元與負向測試**：在 `tools/test_exercises_checker.py` 實作 7 項測試（含 6 項負向測試：解答字樣洩漏、不存在檔案連結、未完成 lab 引用、缺安全標頭、forbidden token 存取、洩漏答案註解；以及 1 項全量正式文件正向測試），全部通過。


## 階段 7-fix 發現（管線對齊、沙盒量測、配方查核與檢查器強化）

### L1. lab 與專案最終前處理管線對齊與 E1b 決策記載查核
1. **事實查核**：透過 AST 解析與 `best_params/*.json` 查證，專案最終模型（M3、M4、M5）的前處理旗標為 `scaler="quantile"`, `use_winsorizer=False`, `use_corr_filter=True(0.95)`, `use_large_indicator=False`。
2. **E1b 決策依據查核**：
   - 查閱 `bankruptcy/outputs/tables/e1b_ablation.csv` 與 `e1b_paired_vs_default.csv`，確認專案 E1b 僅針對 M1（Logistic Regression）與 M2（Random Forest）評估了 `winsorizer_off`。
   - `PROJECT_STATE.md:68` 雖記錄「Winsorizer 關閉」，但專案歷史文件中**並未記載針對 M3、M4、M5 個別關閉截尾之獨立實證依據**。
3. **lab_05 實測折間敏感度客觀揭露**：
   - 在 `lab_05` 實測中，對照組 C（Quantile + 不截尾，即專案最終管線設定）Fold 3 的預警倍數高達 31.17x（因離散粗階梯同分，門檻被迫包含整批樣本），五折平均 AR@0.95 為 13.20x；而基準組（Quantile + 截尾）為 9.14x。
   - 刪除過去宣稱「兩者完全一致」之措辭，客觀呈現兩者在折間分佈敏感度上的差異。
4. **雙軌管線支援架構**：
   - 在 `labs/_common.py` 實作 AST 解析函式 `get_project_pipeline_flags()`，`lab_01`～`lab_08` 全數新增 `--pipeline {project, teaching}` 命令列引數（預設為 `project`），並在文檔中對齊專案最終管線。

### L2. 沙盒除錯題庫 20 次種子重複實驗量測
1. **Bug 01 題幹重構**：改為有監督特徵選擇洩漏（`SelectKBest(f_classif, k=10)` 在切分前 vs 切分後），單次 PR-AUC 0.6239 vs 0.4314（虛增 +0.1925），20 次均值差 **+0.1305**（100% 正偏差）。移除題幹中洩漏答案的註解。
2. **20 次種子重複實驗量測**：
   - 對 10 支除錯題進行 20 次不同隨機種子重複實驗，在 `06_解答.md` 彙總量測總表與各題詳解，提供客觀數理機制與經驗分佈，避免單次抽樣特異值之誤導。

### L3. 配方 3、6 與檢查器強化
1. **配方 3 修正**：引用位置修正為 `bankruptcy/src/thresholding.py:67` (`MARGINS`)，型別標明為 `list`；固化檔名修正為 `thresholds.json` 與 `thresholds.csv`。
2. **配方 6 修正**：補齊 9 大核心要素（欄位對映、時間結構與公司 ID 隔離限制、盛行率變化與 E7a 玩具計算、特徵漂移與 PSI、scoring.py 20% 缺失率、重建固化產物清單、失效假設、新測試集物理鎖定、外部驗證尚未做之誠實宣告）。
3. **檢查器漏洞修復**：
   - 修復括號內符號比對漏洞（擴充正則支援 `\(\s*`?([a-zA-Z_]\w*)`?`），有效抓出 `config.py:30 (MARGINS)` 誤植。
   - 強化失效清單所有副檔名檔案之實體存在性檢查，未標註 `<!-- planned: -->` 且檔案不存在者嚴格報錯。
   - 在 `tools/test_recipes_checker.py` 新增負向測試 4，確保檢查器能攔截此類漏洞。


## 階段 8-fix 關鍵發現與查證修復紀錄

### 1. L1 面試草稿錯誤類型歸因與修正
- **模型架構誤植**：修正「含神經網絡」與「五種異質等權集成」之不實描述。查證專案程式碼與 `e3_selection.csv`，專案完全沒有神經網絡，$E_A$ 實際由 M3（ExtraTrees）、M4（XGBoost）、M5（LightGBM）經 ECDF 百分位映射後等權平均組成。
- **目標門檻與實際命中混淆**：修正「90% 召回（40/44）」與「95% 召回（42/44）」。40 與 42 係 `ceil(0.90*44)` 與 `ceil(0.95*44)` 之目標門檻門檻值（`required_caught`），實際盲測命中數分別為 43/44（97.73%）與 44/44（100.0%）。
- **區間估計校準**：刪除不合邏輯之 [78.3%, 97.5%] 區間，改為實際 43/44 與 44/44 下之 Clopper-Pearson 95% 雙尾信賴區間（分別為 [88.0%, 99.9%] 與 [92.0%, 100.0%]），並明確註記為補充統計推論。
- **指標混淆與獨立標記**：嚴格區分 M1 基準（PR-AUC 0.3857, ROC-AUC 0.9238，來自 E1a）與 E6 $E_A$ 最終盲測指標（PR-AUC 0.5676, ROC-AUC 0.9621，來自 E6 彙總表）。
- **校準與檢定數理機制**：
  - 機率校準：釐清專案選用 Platt Scaling（保持單調排序性，E4b 驗證折 AR@0.95 維持 8.06x，Brier 0.0255，BSS 18.39%；E6 盲測集 Brier 0.0243，BSS 22.07%）；Isotonic Regression 因階梯同分扭曲極端尾部門檻，使 AR@0.95 惡化至 10.02x 而被捨棄。
  - Nadeau-Bengio 修正檢定：釐清其係修正「方差」（方差放大因子 $c = 1/K + n_{test}/n_{train} = 1/15 + 1/4 = 0.3167$），自由度仍為 $K-1=14$，目的在於校正資料重疊導致的型一錯誤（偽陽性膨脹），非型二錯誤。

### 2. L4 檔案路徑與符號偏離成因
- **歷史草稿與概念殘留**：先前回報中出現 `outputs/e1b/`、`src/data.py`、`src/models.py`、`scripts/e1_baseline_cv.py` 等非真實存在之路徑，係撰寫者早期將概念性目錄或預期結構寫入文檔，未即時與專案實際 `git ls-files` 物理結構對齊。
- **排查與全面校正**：專案實際物理產物位於 `bankruptcy/outputs/tables/`，資料處理與 CV 邏輯收攏於 `src/data_audit.py`、`src/preprocessing.py`、`src/cv.py`，實驗進入點位於 `src/run_baselines.py`、`src/run_imbalance.py` 等。已對所有文檔進行全文檢驗並校準完畢。

### 3. L5 檢查器漏洞成因與加固機制
- **漏洞 1：草稿目錄無標籤數字漏檢**：先前檢查器的 `--evidence` 僅針對正式發布文檔進行具體數字正則掃描，未覆蓋 `_drafts/` 目錄。修補後新增 `--include-drafts` 模式，將草稿納入相同的無標籤具體數字與統計指標審查標準（負向測試攔截 38 處未標註指標）。
- **漏洞 2：文檔路徑缺乏磁碟真實性校驗**：先前正則表達式僅驗證文字路徑格式（如是否帶有 `file:// 通訊協定` 或絕對路徑），未檢查路徑所指之 `.py`、`.csv`、`.json`、`.md` 是否真實存在於檔案系統中。修補後全面對比實體檔案存在性（負向測試成功攔截不存在檔案）。
- **漏洞 3：引用函式名未經 AST 實體符號核驗**：先前文檔出現 `檔案:行號 (函式名)` 時，未檢查該符號是否真實為 AST 中定義的 `def` 或 `class`。修補後引入 AST 語法樹實體符號比對，阻絕憑記憶撰寫之無效符號（負向測試成功攔截未定義符號）。

### 4. F-02 與 lab_05 Project 管線實測真相
- **F-02（M5 subsample_freq）在 Project 管線下之 45 組配對對照**：
  - PR-AUC 配對差值均值：+0.00171（標準差 0.02098，95% CI [-0.00460, +0.00801] 跨越 0）。
  - 對照組種子雜訊：0.01913。配對差值均值（0.00171）遠小於種子雜訊量級，證實在該樣本與模型設定下未檢出差異。
- **lab_05（M3 前處理消融）在 Project 管線下之逐折實測**：
  - 在固定專案正式超參數（`n_estimators=600`, `max_features=0.3`, `min_samples_leaf=2`）之 project 管線下，五折 AR@0.95 分別為 5.25x, 6.49x, 6.00x, 10.11x, 7.63x，五折平均為 7.10x。
  - Fold 3 之 AR@0.95 為 10.11x，並無先前 `--quick` 粗糙階梯下因全數預警所致之 31.17x 退化現象。證實先前的 31.17x 係玩具/快速設定下的粗階梯產物，專案正式管線具備高度穩定性。


## 階段 8-fix2 關鍵發現與查證修復紀錄

### 1. M1 除錯題矛盾成因與重建
- **題名不一致根因**：先前 `labs/outputs/bugs/bugs_summary_table.csv` 使用早期概念性英文命名（如 bug_03 Metric Mismatch、bug_04 Unpruned Trees、bug_05 Wrong Objective in Gradient BO），未與階段 7 定稿之教學大綱（Accuracy 陷阱、驗證折挑門檻、非巢狀調參、測試集盛行率洩漏、切分未對齊）同步。
- **零變異 CI [37.5, 37.5] 根因**：bug_06 原先在固定且確定性的玩具資料上運行，無隨機種子擾動，導致每次重複輸出固定數值 93.75 與 56.25。重建後加入隨機雜訊與同分擾動，20 次重複量測配對差值。
- **數據與執行未對齊根因**：先前 bug_01 與 bug_04 數據來自不同開發階段之殘留快照，非單次批次運行產物。
- **重建方案**：10 支除錯題腳本全面標準化（合成資料、安全標頭、20 個固定隨機種子），輸出至 `labs/outputs/bugs/bug_{01..10}_repeats.csv`，由 `tools/summarize_bug_repeats.py` 彙總為 `docs/code_learning/lab_snapshots/bugs_summary.csv`，`06_解答.md` 全文數字與表格全量引用該 CSV 並綁定標準證據標籤。

### 2. M3 檢查器加固與 5 項負向測試
- **檢查器缺口修補**：先前檢查器對未標註數字放行較寬（僅比對特定前綴）、未檢查 `where` 條件唯一性、未檢查路徑必須為相對路徑且帶有目錄前綴。
- **加固功能**：
  1. 引入全面指標數字提取器（包含小數、百分比、倍數 x、分數 n/m、區間 [a, b]）。
  2. 強化 `<!-- evidence -->` 標籤格式：嚴格驗證 `file` 存在且具目錄前綴、`where` 選出唯一列、`col` 存在且值誤差在 `tol` 內。
  3. 全文與草稿未標註指標數字或表格儲存格嚴格報錯攔截。
- **5 項負向測試實證**（均成功引發 Exit code 1）：
  1. 標籤 `expect` 數值錯誤：攔截（`diff > tol`）。
  2. 標籤欄位不存在：攔截（`Column not found`）。
  3. 檔案路徑缺少目錄前綴：攔截（`missing required directory prefix`）。
  4. 內文未標註指標數字（`9.99x`, `0.0123`, `[-0.0099, +0.0123]`, `43/44`）：攔截（`unverified numbers`）。
  5. 表格儲存格未標註具體數字：攔截（`Table row with concrete numbers lacks evidence/code tag`）。

### 3. M4.4 ExtraTrees 樹數消融實測與同分退化數理機制
- **Fold 3 同分退化實測真相**：
  - 在 Fold 3（35 筆正樣本）中，目標召回率 95% 要求捕獲 $\lceil 0.95 \times 35 \rceil = 34$ 筆正樣本。
  - 當決策樹數量較少（80 棵或 100 棵）時，葉節點機率解析度為 $1/80 = 0.0125$ 或 $1/100 = 0.01$。在極度稀疏的特徵空間中，最後 2 筆正樣本的分數恰好為 `0.000000`。
  - 臨界門檻被迫設為 `0.000000`。依據悲觀同分規則（`score >= cutoff`），Fold 3 全體 1,091 筆樣本全數落入預警，預警倍數拉高至 $1091/35 = 31.17x$（5 折平均 13.20x）。
  - 當樹數擴增至專案固化之 600 棵時，解析度細化至 $1/600 \approx 0.00167$，第 34 筆正樣本分數上升至 `0.008889`（所有正樣本分數皆大於 0.0），臨界門檻維持於非零水準，預警家數收斂至 354 家（AR@0.95 = 10.11x，5 折平均 7.10x），平滑化防範了邊界同分失效。
- **計時核實（M4.5）**：5 折交叉驗證實測耗時為 80 棵樹 6.68 秒、100 棵樹 7.30 秒、600 棵樹 13.70 秒。先前口頭「>180秒」係未經測量之數值，已正式撤回並以實測值取代。

### 4. Q9 F-02 配對層級與外層評估層級統計
- 45 組配對（15 折 × 3 模型種子）：差值均值 +0.001706，標準差 0.020978，95% CI [-0.004596, +0.008008]（涵蓋 0）。
- 15 外層評估（各折 3 種子先平均）：差值均值 +0.001706，標準差 0.011543，95% CI [-0.004687, +0.008098]（涵蓋 0）。
- 兩者信賴區間皆涵蓋 0，且均值遠小於模型種子雜訊（0.01498），結論維持不變。


## 階段 8-fix3 關鍵發現與查證修復紀錄

### 1. N1 證據檢查器漏洞成因與加固機制
- **漏洞 1：舊式 `also=` 規避檢查**：先前的 `also=` 語法內之數值未被檢查器以 `file`、`where`、`col` 嚴格核對，容易導致不相干數值掛在單一標籤下背書。修補後全面禁用 `also=` 語法，一旦出現立即報錯。
- **漏洞 2：同列多欄位與推導數值缺乏精確驗證**：引進 `checks="col1=val1; col2=val2"`（嚴格校驗同一列之多欄位數值）與 `derived="算式=數值"`（使用 AST 安全 eval 驗證同一列欄位透過 `+ - * / ( )` 算術推導之數值）。
- **漏洞 3：`code` 標籤符號範圍漂移**：先前 `code` 標籤未限定符號必須在引用行號近處。修補後引入 AST 符號定位：若 `expect == symbol`，強制檢查目標行號 $\pm 3$ 行內必須出現 `FunctionDef`、`ClassDef` 或變數賦值，確保符號與行號嚴格對齊。
- **6 項負向測試（neg1～neg6）全數攔截**：
  1. `neg1`：標籤指向不存在檔案（`e4b_calibration_comparison.csv`），檢查器回報檔案不存在。
  2. `neg2`：`expect` 型別不符（字串 `'S0'` 對 `'0'`），檢查器回報值失配。
  3. `neg3`：使用廢棄 `also=` 語法，檢查器報錯禁用。
  4. `neg4`：`checks` 內欄位數值錯誤（337 對 999），檢查器回報失配。
  5. `neg5`：`derived` 算式計算結果不符（7.659 對 99.99），檢查器回報推導失配。
  6. `neg6`：AST `symbol` 不在目標行號 $\pm 3$ 行內（不存在之函式名），檢查器報錯攔截。

### 2. N2 面試說明稿數據與分析修正
- **$E_A$ 對 M3 配對比較結論**：`e3b_ensemble_vs_single_paired.csv` 顯示三點損失差值均值為 `-0.0203`，Nadeau-Bengio 95% CI 為 `[-0.0965, +0.0559]`，區間涵蓋 0（$p_{\text{Holm}}=1.0$），在統計上未檢出顯著差異。選用 $E_A$ 係遵循多模型風險分散原則與損失點估計較低之工程決策。
- **相關性過濾維度**：全訓練集（5,455 筆）相關性過濾（門檻 0.95）由 92 欄剔除 20 欄為 72 欄；15 次交叉驗證外層訓練折（4,364 筆）各折動態保留 70～73 欄（均值 71.87 欄）。
- **QuantileScaler 對樹模型分箱之機制說明**：明標為「一般經驗，未經本專案消融驗證」，樹模型對單調轉換理論上具不變性，但在本專案 24 個極度偏態大值特徵下，QuantileScaler 能避免分割門檻落入極端稀疏長尾。
- **S0 與 S3 SMOTE 說法中立化**：在 E1a 初篩中 S0 與 S3 配對差異 95% CI 均涵蓋 0，專案依簡約原則推薦 S0，同時保留 S3 進入 E2 搜尋。最終 M1、M3、M4 固化選用 S0，M2、M5 選用 S3。
- **E2a 與 E2b 試驗次數核實**：E2a 外層每折內層 Optuna 試驗次數為 M1=25、M2～M5=40（`tune.py:502-506`）；E2b 全訓練集調參為各模型 60 次（`tune.py:868`）。
- **ECDF 參考分布來源**：參考分布來自全訓練集 5 折 inner OOF 預測分數，而非單折資料。
- **校準方法選擇機制**：Platt Scaling 具備雙參數（斜率與截距），Brier 分數由無模型盛行率基準 0.0312 降至 0.0255（BSS 0.1839）；Isotonic Regression 測試集 Brier 為 0.0202（低於 Platt 0.0243），但因階梯同分扭曲門檻導致 AR@0.95 惡化至 10.02x，故專案選用 Platt。
- **F-02 並排檢定結論**：Teaching 管線 45 組配對差值均值 +0.00348（95% CI [-0.00379, +0.01076]）；Project 管線均值 +0.00171（95% CI [-0.00460, +0.00801]）。兩者 95% CI 均涵蓋 0，且效應量均小於種子雜訊基準（0.01913）。

### 3. N3 ExtraTrees 樹數消融與 Fold 3 同分退化機制
- **葉節點比例平均機制**：ExtraTrees 的 `predict_proba` 是所有樹葉節點類別比例的均值。
- **樹數對分數空隙之影響**：
  - 80 棵樹時驗證折唯一預測分數僅 150 個，相鄰空隙大；Fold 3 在第 34 筆正樣本處分數恰為 0.0，導致 1,091 家企業（全體驗證集）被全數預警，AR@0.95 退化至 31.17x（即 $N/P$）。
  - 600 棵樹時唯一預測分數擴展至 364 個，Fold 3 預警家數平滑收斂至 354 家（AR@0.95 為 10.11x），5 折均值穩定為 7.09x。
- **實驗快照與文檔整合**：實作 `labs/lab_09_tree_count_ablation.py`，快照落地於 `docs/code_learning/lab_snapshots/lab_09_tree_count_ablation_snapshot.csv` 與 `lab_09_tree_count_folds_snapshot.csv`，並登錄於 `labs/lab_map.csv`。

### 4. N4 除錯題重構與解答校準
- **合成資料與 20 次重複實驗**：除錯題 10 題全面標準化採用合成資料，各題獨立運行 20 個隨機種子，由 `tools/summarize_bug_repeats.py` 彙總為 `docs/code_learning/lab_snapshots/bugs_summary.csv`。
- **`06_解答.md` 引用修復**：解答中所有數字均由該 CSV 帶入並綁定標準證據標籤，無任何手動填寫數值。

### 5. N6 逐句稽核表與 N7 禁用詞排查
- **逐句稽核表 (`09_claims_ledger.csv`)**：全面梳理 `docs/code_learning/_drafts/09_面試說明稿.draft.md`，抽取 129 筆陳述句，統計結果為：數字 99 筆、實驗結論 19 筆、一般經驗 5 筆、程式行為 4 筆、限制 2 筆。驗證狀態為 129 筆已驗證、0 筆待驗證。
- **禁用詞排查**：全文掃描 9 個嚴格禁用詞（依 N7 規範清單），在 `docs/code_learning` 與 `labs` 範圍內出現次數全數為 0。
