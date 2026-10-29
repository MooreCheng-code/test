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
- **嚴重度評級**：**中（Medium，影響超參數手冊口徑與模型重現細節）**
- **是否影響已固化產物**：**已作為既定行為固化**，已於 `PROJECT_STATE.md:117` 與 `bankruptcy/MODEL_CARD.md:145` 列為法定限制。
- **各階段 M5 參數狀態完整對照清單**：
  1. **使用「含 `subsample_freq=1`」之 M5 的階段**：
     - **E2a 內層 3 折目標尋優**（`bankruptcy/src/tune.py:169, 206-214`）：`sample_hyperparameters()` 動態注入 `"subsample_freq": 1 if subsample < 1.0 else 0`，內層管線啟用列抽樣 bagging。
     - **E2a 外層單折重擬合**（`bankruptcy/src/tune.py:335-336, 339-347`）：手動補回 `model_params["subsample_freq"] = 1`，故外層 15 次評估的驗證折分數是在 `subsample_freq=1` 下產出。
     - **E2b 全訓練集 10 次內層尋優**（`bankruptcy/src/tune.py:169, 894-905`）：在尋優過程中同樣動態帶入 `"subsample_freq": 1`。
     - **E3 外層訓練折參考分布生成**（`bankruptcy/src/ensemble.py:414, 417-425`）：從 `outputs/logs/e2_cache/*.joblib` 載入字典，使用 `d.get("model_params", ...)`，保留了 E2a 重建的 `subsample_freq=1`。
  2. **使用「缺少 `subsample_freq`，退回預設值 `0`」之 M5 的階段**：
     - **E2b 最佳參數序列化**（`bankruptcy/src/tune.py:930-946`）：`best_trial.params` 僅包含 Optuna 抽樣的原生參數，因此儲存至 `bankruptcy/outputs/models/best_params/M5.json` 時缺少 `"subsample_freq": 1`。
     - **E4c 全訓練集最終元件固化**（`bankruptcy/src/thresholding.py:610-615, 774-782`）：直接讀取 `M5.json`，故最終凍結之 `model_M5.joblib` 實際參數為 `subsample_freq=0`。
     - **E5 特徵解釋與精簡集外層擬合**（`bankruptcy/src/explain.py:462, 670-681`）：讀取 `bankruptcy/outputs/tables/e2a_nested_per_fold.csv` 的 `best_params` 欄位（`tune.py:577` 存入的是未補正之 `r["best_params"]`），故 E5 全域 SHAP 與精簡特徵重擬合均為 `subsample_freq=0`。
     - **E7b 隨機種子穩定性分析**（`bankruptcy/src/seed_stability.py:63, 107-113`）：直接讀取 `M5.json`，故 5 組種子比較皆在 `subsample_freq=0` 下執行。
     - **正式推論與部署**（`bankruptcy/src/scoring.py`、`bankruptcy/score_new_data.py`、`bankruptcy/src/final_eval.py`）：載入凍結之 `model_M5.joblib`（`subsample_freq=0`）。
- **`subsample` 在缺少 `subsample_freq` 時是否生效**：
  - **一般經驗／LightGBM 官方規格**（未於本專案固化流程重跑驗證，將於 `lab_03` 實測對照）：依據 LightGBM 官方文件，`subsample`（別名 `bagging_fraction`）只有在 `subsample_freq > 0`（別名 `bagging_freq > 0`）時才會啟動列抽樣（每 `subsample_freq` 輪疊代抽樣一次樣本子集）。若 `subsample_freq == 0`（LightGBM 預設值），即使 `subsample = 0.6072`，LightGBM 也會完全忽略該參數，每輪使用 100% 樣本進行擬合。
  - **專案實測證據**：`bankruptcy/outputs/models/final/model_M5.joblib` 之分類器參數確為 `subsample_freq=0, subsample=0.6071695513247772`。
- **教學與實驗規劃**：
  後續在 `labs/lab_03_boosting_lr_trees.py` 中，將特別設計一組 `subsample_freq=0` vs. `1` 的消融對照（僅使用訓練集內部 CV），讓學習者親眼觀察 bagging 開啟前後對 LightGBM 訓練速度、樹深結構與泛化預警倍數的真實影響。

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

---

## 三、既有已知議題對照（引自 `docs/KNOWN_ISSUES.md`）

1. **凍結模型目錄暫存檔命名遺留**：`bankruptcy/outputs/models/final/` 下存在 `_tmp_oof_r0_M3.npy` 等含有 `_tmp_` 前綴之檔案（由 `bankruptcy/src/thresholding.py:629-630, 668-669` 產生）。因回歸測試與雜湊驗證依賴此路徑，維持現狀不更名。
2. **防禦性預設值**：`docs/KNOWN_ISSUES.md` 所列之 18 處離線實驗腳本 `.get(..., default)`、`getattr(..., default)` 與特徵重要性 `except Exception:` 容錯區塊，皆屬於非推論路徑之低風險防禦性寫法；推論路徑（`bankruptcy/src/scoring.py`、`bankruptcy/score_new_data.py`）則嚴格執行缺欄與異常顯式拋錯。
