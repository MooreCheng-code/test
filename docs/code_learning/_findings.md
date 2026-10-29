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
1. **問題背景**：文檔編寫中若發明不存在的 CLI 參數（例如假想 `--n-trials` 或 `--model-type`），會誤導學習者並造成執行崩潰。
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
