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
  2. 卡片寫「對各模型的 AR@0.95 影響極微（變異小於 0.05 倍）」，與 E1b 實際對照實驗數據完全不符。
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
