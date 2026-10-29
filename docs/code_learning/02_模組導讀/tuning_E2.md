# 模組導讀：超參數尋優與調參實驗（tune.py，實驗 E2）

## 一句話用途
使用 Optuna（TPE 貝氏最佳化）以自訂的「三點對數預警倍數損失」為目標，分別在巢狀交叉驗證（E2a，產出無偏泛化估計）與全訓練集（E2b，固化最終超參數）上執行自動化超參數尋優。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「超參數尋優與模型調校層（E2 實驗）」。
- **上游**：接收 `cv.py` 的訓練集切分索引與 `pipelines.py` 的管線生成器。
- **下游**：
  - E2a 產出的外層 OOF 預測儲存於 `outputs/models/oof/`，供 E3 集成模型尋優權重。
  - E2b 固化的最佳超參數儲存於 `outputs/models/best_params/M1.json`～`M5.json`，供 E4c 最終模型擬合、E5 特徵解釋與推論使用。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - `X_train`, `y_train`: 全量訓練集 `(5455, 95)` 與 `(5455,)`。
  - `cv_splits`: 15 個外層切分折。
- **輸出**：
  - 實驗評估報表：`outputs/tables/e2a_nested_tuning_summary.csv`、`outputs/tables/e2a_nested_per_fold.csv`。
  - 最終最佳超參數：`outputs/models/best_params/M1.json` ～ `M5.json`。
  - Optuna 試驗資料庫：`outputs/logs/optuna_studies/study_*.db`（SQLite 格式）。

---

## 核心演算法與設計

### 1. 三點對數預警倍數損失（`bankruptcy/src/tune.py:112-118`）
- **做什麼**：Optuna 試驗尋優的損失函數（Objective Function），量化模型在三個關鍵金融預警工作點上的綜合成本。
- **數學公式**：
  $$\mathcal{L}_{\text{3-pt}} = \frac{1}{3} \sum_{r \in \{0.80, 0.90, 0.95\}} \ln\left(\text{AR}@r\right) = \ln\left( \sqrt[3]{\text{AR}_{80} \cdot \text{AR}_{90} \cdot \text{AR}_{95}} \right)$$
- **三個工作點定義**：
  1. $r = 0.80$（80% 破產召回率）：風控初篩門檻。
  2. $r = 0.90$（90% 破產召回率）：常態警報基準。
  3. $r = 0.95$（95% 破產召回率）：專案核心法定防禦線（底線保證）。
- **為什麼要取對數（Log）？**
  - **尺度對齊（Scale Invariance）**：預警倍數具備乘法特徵（例如 2x, 4x, 8x）。在未取 log 時，$r=0.95$ 的 AR（如 8.0x）遠高於 $r=0.80$ 的 AR（如 2.0x），線性平均會導致模型只討好 $r=0.95$ 而徹底放棄 $r=0.80$ 的表現；取對數後，從 2x 降到 1x（改善 50%）與從 8x 降到 4x（改善 50%）的梯度貢獻完全均等。
  - **等價於幾何平均（Geometric Mean）**：極小化對數平均值數學上等價於極小化三個工作點的幾何平均數，對單一極端崩潰點具備懲罰敏感度。
- **最小數字例子（20 筆玩具資料實測）**：
  在標準 20 家公司玩具資料中（$P=4$）：
  - $R=0.80 \implies \text{need} = \lceil 0.80 \times 4 \rceil = 4 \implies \text{AR} = 10/4 = 2.50\text{x}$
  - $R=0.90 \implies \text{need} = \lceil 0.90 \times 4 \rceil = 4 \implies \text{AR} = 10/4 = 2.50\text{x}$
  - $R=0.95 \implies \text{need} = \lceil 0.95 \times 4 \rceil = 4 \implies \text{AR} = 10/4 = 2.50\text{x}$
  專案實際呼叫 `compute_three_point_log_loss(y_true, scores)` 輸出：
  $$\mathcal{L} = \frac{\ln(2.5) + \ln(2.5) + \ln(2.5)}{3} = \ln(2.5) \approx \mathbf{0.9163}$$

---

### 2. Optuna 搜尋空間與 SQLite 持久化續跑機制

#### 抽樣演算法：TPESampler
專案一律使用 `optuna.samplers.TPESampler(seed=SEED)`（`bankruptcy/src/tune.py:313, 915`）。Tree-structured Parzen Estimator 透過機率密度比值建模高分與低分超參數區域，比隨機搜尋（Random Search）在更少疊代內收斂。

#### 各模型搜尋空間清單（`bankruptcy/src/tune.py:121-178`）

| 模型代碼 | 參數名稱 | 抽樣範圍 | 尺度 / 類型 | 說明 |
|:---:|---|---|:---:|---|
| **全模型** | `imbalance` | `["S0", "S3"]` | 類別 | 是否搭配 SMOTE(0.3) |
| **M1** | `C` | `[1e-3, 10.0]` | 對數連續 (log=True) | 正則化強度倒數 |
| | `penalty` | `["l2", "l1"]` | 類別 | 正則化種類（l1 對映 saga，l2 對映 lbfgs） |
| **M2 / M3** | `n_estimators` | `[300, 800]` | 整數階梯 (step=50) | 樹棵數 |
| | `max_depth` | `[4, 6, 8, 12, None]` | 類別 | 最大樹深 |
| | `min_samples_leaf` | `[1, 10]` | 整數均勻 | 葉節點最小樣本數 |
| | `max_features` | `["sqrt", "log2", 0.3]`| 類別 | 特徵抽樣比例 |
| **M4 (XGB)**| `n_estimators` | `[200, 800]` | 整數階梯 (step=50) | 疊代輪數 |
| | `learning_rate` | `[0.01, 0.2]` | 對數連續 (log=True) | 學習率（步長） |
| | `max_depth` | `[2, 6]` | 整數均勻 | 弱學習器樹深 |
| | `subsample` | `[0.6, 1.0]` | 線性連續 | 樣本列抽樣比例 |
| | `colsample_bytree`| `[0.6, 1.0]` | 線性連續 | 特徵行抽樣比例 |
| | `reg_lambda` | `[1e-2, 10.0]` | 對數連續 (log=True) | L2 正則化係數 |
| | `min_child_weight`| `[1.0, 10.0]` | 線性連續 | 葉節點 Hessian 權重下限 |
| **M5 (LGBM)**| `n_estimators` | `[200, 800]` | 整數階梯 (step=50) | 疊代輪數 |
| | `learning_rate` | `[0.01, 0.2]` | 對數連續 (log=True) | 學習率 |
| | `num_leaves` | `[8, 64]` | 整數均勻 | 最大葉節點數 |
| | `subsample` | `[0.6, 1.0]` | 線性連續 | 樣本列抽樣比例 |
| | `colsample_bytree`| `[0.6, 1.0]` | 線性連續 | 特徵行抽樣比例 |
| | `reg_lambda` | `[1e-2, 10.0]` | 對數連續 (log=True) | L2 正則化係數 |
| | `min_child_samples`| `[5, 50]` | 整數均勻 | 葉節點最小樣本數 |

#### SQLite 斷點續跑機制（`bankruptcy/src/tune.py:307-320`）
```python
study_db_path = STUDIES_DIR / f"study_{setting_id}.db"
storage_url = f"sqlite:///{study_db_path}"
study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    load_if_exists=True,
)
remaining_trials = max(0, n_trials - len(study.trials))
if remaining_trials > 0:
    study.optimize(objective, n_trials=remaining_trials)
```
- 每次 trial 完成後，試驗參數與結果即時寫入 SQLite 資料庫。
- 若程式因逾時或當機中斷，下次啟動時會讀取已完成的 `len(study.trials)`，僅執行剩餘的 `remaining_trials`；若已滿額則直接載入 `study.best_trial`，完全免除重複運算。

---

### 3. E2a 巢狀評估 vs. E2b 全量調參

| 維度 | E2a 巢狀交叉驗證（Nested CV） | E2b 全訓練集最終調參（Final Tuning） |
|---|---|---|
| **回答的核心問題** | **「這套調參演算法在未見資料上的泛化效能究竟如何？」** | **「拿全體訓練集訓練即將上線的最終模型，應鎖定哪組超參數？」** |
| **切分方式** | 外層 5 折 × 3 次重複（15 次外層評估）；每折內層切 3 折 | 全量訓練集（5,455 筆）內部跑 5 折 × 2 次重複（10 次評估） |
| **試驗次數（`n_trials`）** | M1: 25 次；M2～M5: 各 40 次（每折獨立跑，共 2,775 次） | **1.5 倍 E2a 規模**：M1: 38 次；M2～M5: 各 60 次（全專案僅跑 1 輪） |
| **為什麼不會洩漏？** | 每個外層驗證折（1,091 筆）在該折調參過程中**完全被物理屏蔽**，調參過程僅能存取外層訓練折（4,364 筆）；外層驗證分數在調參完成後僅計算一次，保證無偏估計。 | E2b 評估分數具有樂觀偏差，因此其分數**絕不用於宣稱模型效能**；其唯一職責是為最終模型產出固定超參數字典。 |

---

### 4. 已知實作細節：M5 的 `subsample_freq`（F-02 分析）
- **現象說明**：
  在 `sample_hyperparameters()` 中，當抽樣出 `subsample` 時，程式動態注入了 `"subsample_freq": 1 if subsample < 1.0 else 0`（`bankruptcy/src/tune.py:169`）。
  然而，Optuna 的 `study.best_trial.params` 只保存透過 `suggest_*` 註冊的原生參數，因此儲存至 `best_params/M5.json` 時缺少了 `"subsample_freq": 1`。

#### M5 跨階段配置狀態完整對照表
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

#### 關鍵說明與工程洞察
1. **對外引用數字之配置對應**：
   - **E3 引用數字**（$E_A$ 的 AR@0.95 = 8.06 倍、折間標準差 2.12 倍）：對應 **含 `subsample_freq=1` 之配置**（因 E3 直接取用 E2a 外層驗證折快取分數）。
   - **E4a 安全邊際**（$E_A$ 建議邊際 0.04）：基於 E2a/E3 外層 15 次評估分布，對應 **含 `subsample_freq=1` 之配置**。
   - **E6 獨立測試集盲測結果**（AR@0.95 = 5.25x [95% CI: 3.93, 8.18]、PR-AUC = 0.5676 [0.3807, 0.7301]）：對應 **`subsample_freq=0` 之配置**（因 E6 評估的是已固化於 `outputs/models/final/model_M5.joblib` 之正式部署元件）。
2. **E6 評估之有效性**：
   - **E6 評估的就是已部署配置**。E6 程式碼（`final_eval.py:110-145`）透過 `load_final_components()` 直接載入磁碟上的固化模型，該模型實體參數即為 `subsample_freq=0`。因此，**E6 的盲測效能對已部署模型 100% 真實有效，絕無評估脫節**。
3. **F-02 嚴重度評級為中（Medium）**：
   - 雖然已部署模型之 E6 評估無偏且可信，但「選模與調參證據（E2a/E3，啟用列抽樣 bagging）」與「部署版本（`subsample_freq=0`，停用列抽樣 bagging）」的 M5 超參數存在配置差異。這屬於機器學習工程中典型的訓練／部署一致性（Train-Serve Consistency）瑕疵，在面試與程式審計中應如實揭露。
4. **處置原則**：
   - **不修復、不重跑任何固化流程**。把「選模與調參證據和部署版 M5 配置不完全相同」寫成專案已知限制。
   - 在 `labs/lab_03_boosting_lr_trees.py` 規劃中，把「`subsample_freq` 0 vs 1 對照（只用訓練資料）」標為第一優先實驗項目。

---

### 5. 最終固化超參數表（來自 `outputs/models/best_params/*.json`）

| 模型 | 最佳採樣 | 固化超參數列表（依 JSON 原文） | 內層三點損失 | 搜尋試驗數 | 來源檔案與行號 |
|:---:|:---:|---|:---:|:---:|---|
| **M1** | **S0** | `C=0.11616457722286455`, `penalty="l2"` | **1.7450** | 38 | `outputs/models/best_params/M1.json` |
| **M2** | **S3** | `n_estimators=300`, `max_depth=12`, `min_samples_leaf=10`, `max_features="sqrt"` | **1.6800** | 60 | `outputs/models/best_params/M2.json` |
| **M3** | **S0** | `n_estimators=600`, `max_depth=None`, `min_samples_leaf=2`, `max_features=0.3` | **1.6668** | 60 | `outputs/models/best_params/M3.json` |
| **M4** | **S0** | `n_estimators=250`, `learning_rate=0.046777`, `max_depth=2`, `subsample=0.879564`, `colsample_bytree=0.901713`, `reg_lambda=1.690288`, `min_child_weight=7.300913` | **1.6647** | 60 | `outputs/models/best_params/M4.json` |
| **M5** | **S3** | `n_estimators=300`, `learning_rate=0.013733`, `num_leaves=43`, `subsample=0.607170`, `colsample_bytree=0.778890`, `reg_lambda=0.027045`, `min_child_samples=38` | **1.6760** | 60 | `outputs/models/best_params/M5.json` |

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **調參目標函數** | A. 負對數似然（Log Loss）<br>B. 1 - PR-AUC<br>C. 三點對數預警倍數損失 | **方案 C（三點對數預警倍數損失）** | Log Loss 偏向整體機率校準；PR-AUC 包含無意義的低召回區；三點損失精確錨定 80%、90%、95% 召回率下的實際稽核警報倍數。 | `bankruptcy/src/tune.py:112-118`；E2a 實測結果 |
| **調參評估架構** | A. 非巢狀 CV（單純 K 折）<br>B. 巢狀交叉驗證（Nested CV） | **方案 B（Nested CV）** | 非巢狀 CV 將超參數選擇與泛化評估混在同一折，分數嚴重樂觀膨脹；巢狀架構將調參完全限制於內層，外層專職無偏評估。 | `bankruptcy/src/tune.py:270-385`；`outputs/tables/e2a_nested_tuning_summary.csv` |

---

## 機器學習概念白話框

> **白話概念：巢狀交叉驗證（Nested Cross-Validation）**
> - **為什麼要雙層 CV？** 想像一位學生正在準備大考。內層 CV 是「平日模擬考」，用來決定「要挑哪一本參考書（哪一組超參數）」；外層 CV 是「模考聯審」，由完全沒看過模擬考卷的外校老師出題，檢驗「挑選參考書的整個學習策略是否真的有效」。
> - **如果沒有外層會怎樣？** 如果直接拿模擬考挑出的最高分當作大考成績，該分數必然虛高，因為學生只是挑了一本剛好切中模擬考題的書。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「巢狀交叉驗證」與「資料隔離」。

---

## 本模組的單元測試
對應單元測試涵蓋：
1. `bankruptcy/tests/test_results_summary.py`：驗證 E2 調參產物 CSV 格式與指標欄位完整性。
2. `bankruptcy/tests/test_thresholds_reproduce.py`：驗證超參數在載入時能正確還原至模型管線。

---

## 讀碼練習
1. **練習 1**：在 `create_inner_objective()` 中，為什麼內層評估指標採用「池化預測（Pooled inner OOF metrics）」而不是直接平均各內層折的損失？
2. **練習 2**：在 `tune.py` 中，若將某模型的 `n_trials` 從 40 提高到 50，再次執行時 SQLite 會如何運作？它會從頭跑 50 次還是只跑 10 次？
3. **練習 3**：根據 F-02，在 `M5.json` 中 `subsample` 為 0.6072 但缺少 `subsample_freq`。若直接將此 JSON 傳入 LightGBM 官方分類器，依一般經驗，LightGBM 是否會執行列抽樣？為什麼？

---

## 延伸：想改這裡時要注意什麼
1. **禁止在主流程重新呼叫 `run_e2_tuning()`**：這會更新已固化的 OOF 快取與 Optuna 資料庫。
2. **自訂實驗路徑隔離**：欲測試不同搜尋空間或自訂目標時，請在 `labs/` 腳本中自訂獨立的 `study_name` 與 SQLite 路徑。
