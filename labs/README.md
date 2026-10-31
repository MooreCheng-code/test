# 《企業破產預測》互動式參數實驗室 (Labs)

本目錄提供 9 支可獨立執行的參數實驗腳本（`lab_01`～`lab_09` 全數完成），專門設計給學習者以「動手實作（Hands-on）」方式直觀觀察超參數對模型行為、預警指標與泛化差距的影響。

---

## 0. 前處理旗標對照：專案最終模型 vs Labs

### 0.1 專案最終模型（M3、M4、M5）前處理旗標清單（附行號）
在專案原始碼中，最終模型所採納的前處理配置如下：
- `bankruptcy/src/tune.py:897`（E2b 固化參數訓練）：
  `scaler="quantile"`, `use_winsorizer=False`, `use_corr_filter=True`, `use_large_indicator=False`, `use_imputer=False`
- `bankruptcy/src/ensemble.py:159`（E3 集成候選模型評估）：
  `scaler="quantile"`, `use_winsorizer=False`, `use_corr_filter=True`, `use_large_indicator=False`, `use_imputer=False`
- `bankruptcy/src/thresholding.py:642, 656, 776`（E4c 最終模型凍結與儲存）：
  `scaler="quantile"`, `use_winsorizer=False`, `use_corr_filter=True`, `use_large_indicator=False`, `use_imputer=False`
- `bankruptcy/src/pipelines.py:212-218`（`build_pipeline` 函式宣告預設值）：
  `scaler="quantile"`, `use_winsorizer=True`, `use_corr_filter=True`, `use_large_indicator=False`, `use_imputer=False`

### 0.2 Labs（lab_01～lab_09）前處理旗標與管線支援說明
- **各 Lab 對 `--pipeline` 旗標之支援狀況**：
  1. **全體 Lab 完整傳遞支援（`lab_01`～`lab_09`）**：所有 Lab 腳本均支援 `--pipeline {project,teaching}` 旗標並明確傳遞至 `build_pipeline(..., **pipe_flags)`。在 `--pipeline project`（預設）下關閉 Winsorizer（折內保留 70～73 欄，與專案最終管線一致）；在 `--pipeline teaching` 下開啟 Winsorizer（折內保留 74～77 欄，沿用教學預設值）。
  2. **`lab_05`（前處理消融）**：專門針對相關性過濾、截尾開關（Winsorizer on vs off）與各縮放器進行單變量消融對照，以 `--pipeline` 設定值作為基準前處理旗標。
- **旗標對照總表**：
  | 模組 | `scaler` | `use_winsorizer` | `use_corr_filter` | `use_large_indicator` | 與專案最終管線（E2b/E3/E4c）之差異說明 |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | **專案最終模型** | `quantile` | `False` | `True` (0.95) | `False` | 基準正式管線（折內保留 70～73 欄，全訓練集 72 欄）。 |
  | **Labs (lab_01～lab_09)** (`--pipeline project`, 預設) | `quantile` | `False` | `True` (0.95) | `False` | **與專案最終管線一致（無差異）**，折內保留 70～73 欄。 |
  | **Labs (lab_01～lab_09)** (`--pipeline teaching`) | `quantile` | `True` | `True` (0.95) | `False` | 沿用 `build_pipeline` 宣告預設值（開啟 Winsorizer，折內保留 74～77 欄）。 |
  | `lab_05` (前處理消融) | 消融比較 | 消融比較 | 消融比較 | `False` | **專門消融比較**「截尾開 vs 關」與各縮放器之敏感度差異。 |
- **專案關閉截尾之決策依據查核**：
  1. 查核 `bankruptcy/outputs/tables/e1b_ablation.csv` 與 `e1b_paired_vs_default.csv` 顯示：E1b 實際僅對 M1（邏輯迴歸）與 M2（隨機森林）評估過關閉截尾（`winsorizer_off`）：
     - M1: 預警倍數由 10.87x 變為 11.06x（差 +0.20x, 95% CI [-3.54, 3.93], $p_{	ext{NB}}=0.92$）；
     - M2: 預警倍數由 9.07x 變為 8.25x（差 -0.82x, 95% CI [-2.78, 1.14], $p_{	ext{NB}}=0.44$）。
     - M3、M4、M5 在 E1b 中**未進行個別的截尾消融**。
  2. `PROJECT_STATE.md:68` 記載：「消融確認關閉 Winsorizer 對尾端預警無害，保留 CorrelationFilter 可安全去冗餘。**固化前處理：Winsorizer 關、CorrFilter 開、LargeValueIndicator 關、quantile 縮放、imbalance ∈ {S0, S3}**」。專案將 M1/M2 之消融結論直接推廣至所有模型。
  3. **【誠實宣告】專案原始文件中未記載針對 M3、M4、M5 個別關閉截尾的實證依據**。
- **Lab 實測結果客觀呈現（兩者並非「完全一致」）**：
  在 `lab_05` 的消融對照中，M3 在關閉截尾（專案最終管線，Quantile + 不截尾）下，5 折平均 PR-AUC 為 0.4363，但 AR@0.95 為 13.20x（因 Fold 3 發生離散粗階梯同分，單折預警數達 1,091 家、單折 AR=31.17x）；而在開啟截尾（教學基準，Quantile + 截尾）下 5 折平均 AR@0.95 為 9.14x。這說明雖然全域 PR-AUC 趨勢相近，但關閉截尾在特定折的離散同分門檻上具備敏感度，因此依實測數據客觀敘述，不得寫為「完全一致」。
- **lab_05 之 M3 參數來源說明**：
  在 `--standard` 模式下，lab_05 直接載入固化的 `outputs/models/best_params/M3.json`（`n_estimators=600, max_features=0.3, min_samples_leaf=2, max_depth=None`），與專案最終設定完全一致；在 `--quick` 模式下，僅將 `n_estimators` 截斷為 80 棵以確保在 2 分鐘內執行完畢。

---

## 1. 實驗室總覽表

| Lab ID | 實驗主題 | 涵蓋核心參數概念 | 狀態 | 執行指令 | 快速模式耗時 | 產物目錄 | 快照檔案 |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **lab_01** | 決策樹與樹集成深度掃描 | `max_depth`、過擬合軌跡、Train vs Val 差距、粗階梯同分 | **已完成** | `python labs/lab_01_tree_depth.py --quick` | 約 58 秒 | `labs/outputs/lab_01/` | `lab_01_tree_depth_snapshot.csv` |
| **lab_02** | 森林樹數、特徵抽樣與葉約束 | `n_estimators`、`max_features`、`min_samples_leaf` | **已完成** | `python labs/lab_02_forest_hyperparams.py --quick` | 約 55 秒 | `labs/outputs/lab_02/` | `lab_02_forest_hyperparams_snapshot.csv` |
| **lab_03** | 提升法網格與 F-02 實證對照 | `learning_rate`、`n_estimators`、`subsample`、`subsample_freq` | **已完成** | `python labs/lab_03_boosting_f02.py --quick` | 約 52 秒 | `labs/outputs/lab_03/` | `lab_03_f02_extended_snapshot.csv` |
| **lab_04** | 不平衡策略與機率校準偏移 | `S0`～`S6`、SMOTE、`class_weight`、機率扭曲 | **已完成** | `python labs/lab_04_imbalance_strategies.py --quick` | 約 123 秒 | `labs/outputs/lab_04/` | `lab_04_imbalance_strategies_snapshot.csv` |
| **lab_05** | 前處理消融實驗 | 相關性過濾門檻、截尾開關、分位數轉換 vs 標準化 | **已完成** | `python labs/lab_05_preprocessing_ablation.py --quick` | 約 120 秒 | `labs/outputs/lab_05/` | `lab_05_preprocessing_ablation_snapshot.csv` |
| **lab_06** | 召回率權衡與門檻轉移達標 | 目標召回率曲線、隨機猜測基準線、安全邊際 $m$、達標率 | **已完成** | `python labs/lab_06_recall_alert_curve.py --mode quick` | 約 36 秒 | `labs/outputs/lab_06/` | `lab_06_threshold_transfer_snapshot.csv` |
| **lab_07** | CV 折數、重複次數與樣本量效應 | CV 折數 $K$、重複次數 $R$、折內正樣本數、階梯化不確定性 | **已完成** | `python labs/lab_07_cv_folds_repeats.py --mode quick` | 約 185 秒 | `labs/outputs/lab_07/` | `lab_07_cv_matrix_snapshot.csv` |
| **lab_08** | 種子穩定性與隨機性分解 | 演算法隨機種子、切分隨機種子、方差分解 | **已完成** | `python labs/lab_08_seed_stability.py --mode quick` | 約 144 秒 | `labs/outputs/lab_08/` | `lab_08_seed_variance_summary_snapshot.csv` |
| **lab_09** | ExtraTrees 樹數消融與分數空隙退化機制 | `n_estimators`、樹數消融、分數空隙、唯一分數、同分門檻 | **已完成** | `python labs/lab_09_tree_count_ablation.py --pipeline project` | 約 75 秒 | `labs/outputs/lab_09/` | `lab_09_tree_count_ablation_snapshot.csv` |

---

## 2. 環境需求與執行指南

### 2.1 環境與相依套件
本實驗室直接使用專案現有 Python 環境，**無須安裝任何額外套件**：
- Python $\ge 3.10$
- `numpy`、`pandas`、`scipy`、`scikit-learn`、`imbalanced-learn`
- `lightgbm`、`xgboost`、`optuna`
- `matplotlib`（已內建中文字型自動降級相容機制，優先使用 Windows 正黑體，若缺失則自動退回無襯線英文字型，確保繪圖不中斷）

### 2.2 執行方式
每支 Lab 腳本均提供檔頭集中參數字典 `PARAMS`，並支援三種執行模式：
- **快速模式（預設 `--quick` 或 `--mode quick`）**：在訓練集上採用縮減網格或折數，確保每支腳本於 $\le 3$ 分鐘內跑完。
- **標準模式（`--standard` 或 `--mode standard`）**：採用 5 折 $\times$ 1 次重複，使用標準搜尋網格。
- **完整模式（`--full` 或 `--mode full`）**：採用 5 折 $\times$ 3 次重複（5 折 × 3 次重複 = 15 次外層評估），使用完整搜尋網格與專案正式超參數。
- **F-02 專用擴大模式（`--f02-extended`）**：專為 `lab_03` 設計之 45 組配對深入檢定模式（15 次外層評估 $\times$ 3 模型種子）。

```powershell
# 執行單一 Lab（快速模式，均在 3 分鐘內跑完）
python labs/lab_01_tree_depth.py --quick
python labs/lab_02_forest_hyperparams.py --quick
python labs/lab_03_boosting_f02.py --quick
python labs/lab_04_imbalance_strategies.py --quick
python labs/lab_05_preprocessing_ablation.py --quick
python labs/lab_06_recall_alert_curve.py --mode quick
python labs/lab_07_cv_folds_repeats.py --mode quick
python labs/lab_08_seed_stability.py --mode quick

# 執行單元測試驗證守衛機制與分層抽樣
python labs/test_common.py
```

### 2.3 絕對安全規範 (Leak-Free & Read-Only Protection)
1. **僅使用訓練集資料**：資料入口僅限 `labs._common.load_train_data()`，該函式在載入後立即執行守衛斷言（斷言筆數剛好 5,455 筆、正樣本數剛好 176 筆、且總筆數絕不為 6,819 筆）。嚴禁讀取或猜測測試集索引 `splits.joblib`，嚴禁 import `final_eval.py`、`preflight.py`、`lock_utils.py`。
2. **獨立建構 CV 切分**：Lab 於 5,455 筆訓練集上自行以 `StratifiedKFold` 建立驗證切分，不重用專案正式切分檔案。
3. **產物輸出嚴格隔離**：Lab 執行產物（圖表、全量 CSV、執行清冊 `run_manifest.json`）一律寫入 `labs/outputs/<lab_id>/`，並由 `labs/.gitignore` 排除追蹤；版本庫僅收納提交至 `docs/code_learning/lab_snapshots/` 的小型快照 CSV。絕不寫入 `bankruptcy/outputs/`。

### 2.4 Lab 模式與 Quick 縮減範圍總表

每支 Lab 腳本支援多種執行模式，各模式在不同維度上之縮減定義如下表所示：

| 模式名稱 | 參數旗標 | CV 折數與重複次數 | 搜尋網格 / 參數維度 | 樹數量 / 候選模型數 | 適用情境與結論品質宣告 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **快速模式**<br>(`quick`) | `--quick`<br>或 `--mode quick` | 3 折或 5 折 $\times$ 1 次重複 | 網格點數精簡至 2～4 點 | 樹數縮減至 50～100 棵；精簡代表性模型 | **冒煙測試專用**：用於快速驗證腳本與演算法管線能正常執行（每支腳本 $\le 3$ 分鐘）。<br>⚠️ **【重要聲明】quick 模式結論品質低於 standard，不能拿來下正式科學結論，僅用於確認程式可跑。** |
| **標準模式**<br>(`standard`) | `--standard`<br>或 `--mode standard` | 5 折 $\times$ 1 次重複 | 標準網格（涵蓋主要關鍵候選值） | 樹數 100～200 棵；評估完整模型列表 | **教學探索與參數敏感度觀察**：指標趨勢具備代表性，可作為課堂討論與超參數行為分析基礎。 |
| **完整模式**<br>(`full`) | `--full`<br>或 `--mode full` | 5 折 $\times$ 3 次重複<br>（5 折 × 3 次重複 = 15 次外層評估） | 完整細緻網格（與正式實驗對齊） | 樹數 400～600 棵（依固化參數設定） | **高精度重現分析**：消除抽樣隨機性，方差與 E1～E4 正式實驗完全對齊，計算耗時較長。 |
| **F-02 擴充檢定** | `--f02-extended` | 5 折 $\times$ 3 次重複 $\times$ 3 種子<br>（共 45 組配對） | 鎖定特定關鍵超參數組合（A vs B） | LightGBM 固化參數 | **假設驗證專用**：針對單一爭議參數（如 F-02）進行配對統計檢定與雜訊對照。 |

---

## 3. lab_01：決策樹與樹集成深度掃描 (`max_depth`)

- **與專案最終管線的差異：預設 `--pipeline project` 下前處理旗標與專案最終模型一致（無差異）；`--pipeline teaching` 下則開啟 Winsorizer 截尾。**

### 3.1 實驗目的
直觀觀察單棵決策樹（`DecisionTreeClassifier`）與樹集成模型（M2 RandomForest、M3 ExtraTrees、M4 XGBoost）隨著樹深度（`max_depth`）增加時，**訓練折分數與驗證折分數的背離現象（過擬合軌跡）**，理解為何單棵決策樹不可用，而集成模型能藉由平均與正則化約束馴服深度方差。

### 3.2 參數範圍與設定
- 單棵決策樹掃描深度：`[1, 2, 3, 4, 6, 8, 12, None (無限制)]`
- 樹集成比較深度：M2 `[4, 6, 8, 12, None]`、M3 `[4, 6, 8, 12, None]`、M4 `[2, 3, 4, 6]`
- 前處理管線：沿用專案標準 `DropConstantAndDuplicate` $\to$ `Winsorizer` $\to$ `QuantileScaler` $\to$ `CorrelationFilter`

### 3.3 實測執行時間
- 實測時間：約 58 秒（`--quick`）

### 3.4 假設與實際觀察
- **假設**：隨著單樹深度加深，訓練折 PR-AUC 會逼近 1.0（高度擬合訓練資料），但驗證折 PR-AUC 會在深度 3～4 達到頂點後大幅下滑，產生巨大的過擬合差距；而隨機森林與極端隨機樹即使深度為 None，驗證折 PR-AUC 仍能維持在 0.38 以上。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='4'" col=val_pr_auc_mean expect=0.256910 tolerance=0.001 -->
  - 單棵決策樹在深度 4 時達到驗證 PR-AUC 最高峰 0.2569（訓練 PR-AUC 0.5341，差距 +0.2772）。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='None (無限制)'" col=train_pr_auc_mean expect=1.000000 tolerance=0.0001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='None (無限制)'" col=val_pr_auc_mean expect=0.119414 tolerance=0.001 -->
  - 當單樹深度無限制（`None`）時，訓練折 PR-AUC 達到 1.0000（完全擬合），但驗證折 PR-AUC 崩跌至 0.1194，過擬合差距高達 +0.8806。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='4'" col=val_ar95_mean expect=30.981429 tolerance=0.01 -->
  - **單樹粗階梯同分機制**：單棵樹的驗證折預警倍數（AR@0.95）始終高達約 30.98 倍（接近全折樣本比例 $N/P \approx 31.00\text{x}$）。這是因為單樹葉節點少（例如無深度限制時僅有 0.0 與 1.0 兩個唯一機率值），機率呈粗階梯分佈。在悲觀同分規則（`score >= thr`）下，為了達到 95% 召回率，門檻被迫納入龐大的同分葉節點，使全折樣本幾乎全數被納入預警名單（$Alerts = N$），導致預警倍數退化至 $N/P$。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M2' and max_depth=='12'" col=val_pr_auc_mean expect=0.399925 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M3' and max_depth=='12'" col=val_pr_auc_mean expect=0.437545 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M4' and max_depth=='2'" col=val_pr_auc_mean expect=0.402685 tolerance=0.001 -->
  - 相對地，集成模型展現出極強的泛化保護：M2 在深度 12 時 PR-AUC 為 0.3999（AR@0.95 約 13.10 倍）；M3 在深度 12 時 PR-AUC 為 0.4375（AR@0.95 約 8.10 倍）；M4 在深度 2 時 PR-AUC 達 0.4033（AR@0.95 約 8.46 倍）。

### 3.5 解讀與正式實驗對照
- **解讀**：單樹深度增加時，方差（Variance）顯著擴大；而隨機森林與極端隨機樹透過「Bagging（樣本自助抽樣）+ Random Subspace（特徵隨機抽樣）」將多棵高方差的深樹平均，有效抵消了方差。
- **正式實驗對照**：專案 E2b 固化參數中，M4（XGBoost）選用深度為 2（極淺樹以避免 Boosting 累加過擬合），而 M2 選用深度為 12、M3 選用深度為 `None`（深樹配合低樣本葉節點，由 600 棵樹的集成平均消除方差）。本實驗重現了此設計取捨。

---

## 4. lab_02：森林樹數、特徵抽樣與葉約束 (`M2`, `M3`)

- **與專案最終管線的差異：預設 `--pipeline project` 下前處理旗標與專案最終模型一致（無差異）；`--pipeline teaching` 下則開啟 Winsorizer 截尾。**

### 4.1 實驗目的
以隨機森林（M2）與極端隨機樹（M3）為對象，進行單變量超參數掃描：
1. 樹木數量（`n_estimators`）：驗證樹數增加是否能持續降低折間標準差，以及邊際效益遞減點。
2. 特徵抽樣比例（`max_features`）：比較 `sqrt`、`log2` 與固定比例 `0.3`。
3. 葉節點最小樣本數（`min_samples_leaf`）：比較葉節點正則化約束對預警倍數的影響。

### 4.2 參數範圍與設定
- 掃描參數：`n_estimators ∈ [100, 200, 400]`、`max_features ∈ ['sqrt', 'log2', 0.3]`、`min_samples_leaf ∈ [1, 2, 5, 10]`
- 基準參數：M2 採用固化 `max_depth=12, imbalance='S3'`；M3 採用固化 `max_depth=None, imbalance='S0'`

### 4.3 實測執行時間
- 實測時間：約 55 秒（`--quick`）

### 4.4 假設與實際觀察
- **假設**：增加樹數量不會導致過擬合，只會使折間方差收斂並線性增加訓練時間；較小的 `max_features` 能降低樹間相關性，使 ExtraTrees 表現更平穩。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='n_estimators' and param_value=='100'" col=pr_auc_mean expect=0.397349 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='n_estimators' and param_value=='400'" col=pr_auc_mean expect=0.407519 tolerance=0.001 -->
  - M2 樹數由 100 增至 400 時，PR-AUC 由 0.3973 微幅提升至 0.4075，AR@0.95 由 9.14 倍降至 8.46 倍，折間標準差保持穩定（約 0.076～0.083），但每折擬合時間由 1.85 秒增至 3.26 秒。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M3' and param_name=='max_features' and param_value=='log2'" col=pr_auc_mean expect=0.454269 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M3' and param_name=='max_features' and param_value=='0.3'" col=pr_auc_mean expect=0.440134 tolerance=0.001 -->
  - M3 在 `max_features='log2'` 時取得最高的 PR-AUC 0.4543（AR@0.95 約 8.02 倍），優於固定比例 0.3 的 0.4401。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='min_samples_leaf' and param_value=='5'" col=ar95_mean expect=8.183968 tolerance=0.01 -->
  - M2 在 `min_samples_leaf=5` 時取得較低的預警倍數（8.18 倍，但需注意在種子雜訊範圍內），過大的葉樣本約束（`min_samples_leaf=10`）反而使 AR@0.95 惡化至 9.41 倍。

### 4.5 解讀與正式實驗對照
- 樹數增加具有明確的邊際遞減效應（100 $\to$ 200 提升顯著，200 $\to$ 400 趨於平緩），正式實驗選擇 300～600 棵樹是在「穩定性」與「訓練資源」間的平衡點。

---

## 5. lab_03：提升法網格與 F-02 深入對照 (`M4`, `M5`)

- **與專案最終管線的差異：預設 `--pipeline project` 下前處理旗標與專案最終模型一致（無差異，M5 關閉 Winsorizer）；`--pipeline teaching` 下則開啟 Winsorizer 截尾。**

### 5.1 實驗目的
1. **網格掃描**：探索 XGBoost（M4）與 LightGBM（M5）在 `learning_rate` $\times$ `n_estimators` 網格下的互補消長關係。
2. **F-02 擴大實證對照（45 組配對檢定）**：以專案 M5 固化參數（含 `subsample=0.607169...` 與 S3 SMOTE）為基底，固定 5 折 × 3 次重複 = 15 次外層評估切分（5 折 $\times$ 3 次重複），跨 3 組模型種子（42, 123, 2025）嚴格對照設定 A（`subsample_freq=0`）與設定 B（`subsample_freq=1`）。

### 5.2 參數範圍與設定
- 網格：`learning_rate ∈ [0.01, 0.05, 0.15]`，`n_estimators ∈ [40, 80]`（快速模式）
- F-02 基底：讀取 `best_params/M5.json`，45 組配對（15 次外層評估 $\times$ 3 模型種子）。

### 5.3 實測執行時間與版本
- 實測時間：快速模式約 52 秒；`--f02-extended` 模式約 441 秒
- LightGBM 執行版本：`4.7.0`

### 5.4 假設與實際觀察
- **假設**：
  - 當 `subsample_freq=0` 時 LightGBM 預設不執行樣本抽樣；當 `subsample_freq=1` 時啟動每輪抽樣，兩者預測分數絕非逐位相同。
  - 在配對差異上，若信賴區間涵蓋 0 且差異小於模型種子雜訊，則在統計上判定為「未偵測到顯著差異（樣本與檢定力有限）」。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_extended_snapshot.csv where="model_seed==42 and fold_idx==0" col=is_bit_identical expect=False -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_extended_snapshot.csv where="model_seed==42 and fold_idx==0" col=max_abs_score_diff expect=0.294649 tolerance=0.001 -->
  - **F-02 生效確認**：在全體 45 組配對中，逐位完全相同比例為 **0.0%（`is_bit_identical=False`）**，最大絕對預測分數差達 0.3063，皮爾森相關係數 0.9839，斯皮爾曼秩相關 0.9556。這證實 `subsample_freq=1` 確實啟動了隨機子樣本抽樣。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_extended_snapshot.csv where="model_seed==42 and fold_idx==0" col=pr_auc_diff_B_minus_A expect=0.031603 tolerance=0.001 -->
  - **統計推論與信賴區間（45 組配對）**：
    - PR-AUC 平均配對差異（B 減 A）：`+0.00348`（Std: 0.02421），$t$-檢定 95% CI 為 `[-0.00379, +0.01076]`（涵蓋 0），符號為正比例為 53.3%。
    - AR@0.95 平均配對差異：`-0.0861x`（Std: 1.7718），$t$-檢定 95% CI 為 `[-0.6184, +0.4462]`（涵蓋 0），改善比例為 48.9%。
    - AR@0.90 平均配對差異：`-0.3599x`（Std: 0.8986），$t$-檢定 95% CI 為 `[-0.6298, -0.0899]`。
    - Brier 平均配對差異：`+0.000920`，95% CI 為 `[+0.000704, +0.001137]`。
  - **模型種子雜訊參照**：在同一切分下，僅改變設定 B 的模型隨機種子所造成的 PR-AUC 雜訊為 `0.01817`，AR@0.95 雜訊為 `1.0448x`。

### 5.5 解讀與規則判定
- **嚴格判定結論**：在 45 組配對檢定中，PR-AUC 均值差（+0.00348）小於模型種子雜訊（0.01817），且 95% 信賴區間涵蓋 0。**依統計推論原則判定為「在本 lab 設定下未偵測到顯著差異（樣本與檢定力有限）」**。
- **邊界宣告**：此結論僅適用於本 Lab 之交叉驗證環境，**嚴格不外推到部署版 M5 在未見 E6 盲測集上的影響**。

---

## 6. lab_04：不平衡策略與機率校準偏移 (`M1`, `M3`)

- **與專案最終管線的差異：預設 `--pipeline project` 下前處理旗標與專案最終模型一致（無差異）；`--pipeline teaching` 下則開啟 Winsorizer 截尾。**

### 6.1 實驗目的
固定模型（M1 與 M3），在訓練折內嚴格執行 S0～S6 全體 7 組策略，觀察排序指標與機率校準偏移。

### 6.2 參數範圍與設定
- 策略清單：`S0`（不處理）、`S1`（類別權重）、`S2`（SMOTE 0.1）、`S3`（SMOTE 0.3）、`S4`（SMOTE 0.5）、`S5`（BorderlineSMOTE 0.3）、`S6`（SMOTETomek 0.3）

### 6.3 實測執行時間
- 實測時間：約 123 秒（`--quick`）

### 6.4 假設與實際觀察
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S0'" col=mean_proba_mean expect=0.032544 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S1'" col=mean_proba_mean expect=0.171719 tolerance=0.001 -->
  - **M1 的 class_weight 機率膨脹機制**：M1 在 S0 下平均預測機率為 0.0325（與真實盛行率 0.0323 一致）；採用 S1（`class_weight='balanced'`）時大幅上升至 0.1717（膨脹 5.3 倍，Brier 惡化至 0.0893）。這是因為線性模型中 class_weight 等價於在 Log-odds 上平移 $\ln(w_1/w_0) \approx +3.4$。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M3' and strategy=='S0'" col=mean_proba_mean expect=0.033400 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M3' and strategy=='S1'" col=mean_proba_mean expect=0.039128 tolerance=0.001 -->
  - **M3 的 class_weight 膨脹緩和**：M3 在 S0 下為 0.0334，S1 下僅微升至 0.0391（僅 1.2 倍）。因為樹模型中 class_weight 僅改變分裂時的不純度計算，葉節點仍按樣本比例計算，多數葉節點仍為純負樣本（機率 0.0）。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S3'" col=pr_auc_mean expect=0.387373 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S6'" col=pr_auc_mean expect=0.387230 tolerance=0.001 -->
  - **M1 S3 vs S6 數值在不同管線下的表現與 Tomek 剔除實測**：
    - 在 **teaching 管線（Winsorizer 啟用）** 下：QuantileScaler 將 76 維特徵縮放至均勻分佈且無極端長尾值，TomekLinks 跨 5 折每折剔除 0 筆樣本，故數值完全相同（PR-AUC 均為 0.3883，AR@0.95 均為 11.70x）。專案正式 E1a 表格（`e1a_imbalance_comparison.csv`，採 teaching 口徑）中 M1 的 S3 與 S6 亦共享完全相同的 `alert_ratio@recall_0.95_mean = 10.865926`。
    - 在 **project 管線（Winsorizer 關閉，預設）** 下：極端值使第 2 折 TomekLinks 實際剔除了 2 筆邊界樣本（5,489 筆降至 5,487 筆，5 折平均每折剔除 0.4 筆），S3 與 S6 數值微幅不同（S3 PR-AUC 0.3874、AR@0.95 8.95x 對比 S6 PR-AUC 0.3872、AR@0.95 9.00x，PR-AUC 差距 0.000143，AR 差距 0.0457x）。

#### M3（ExtraTrees）全策略實測對照表（`--quick --pipeline project` 模式，5 折 × 1 次重複，註：正式 E1a 無 M3，數據為 Lab 04 實測）：

> [!NOTE]
> **執行模式標註**：本表數據來自 `--quick --pipeline project` 模式（5 折 × 1 次重複，種子 42）。快照檔案與磁碟原始 CSV 內容完全一致。

| 策略代號 | 策略名稱與實作方法 | 驗證 PR-AUC (Mean) | 驗證 AR@0.95 (Mean) | 平均預測機率 (Mean Proba) | Brier 分數 | 與 S0 差距及雜訊判定（PR-AUC 雜訊 Std: 0.0053，AR@0.95 雜訊 Std: 1.04x） |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **S0** | 基準線（不進行任何重抽樣） | **0.4341** | **13.20x** | 0.0334 | 0.0233 | 基準線（最精簡無介入） |
| **S1** | `class_weight='balanced'` | **0.4067** | **13.63x** | 0.0391 | 0.0240 | PR-AUC 差 `-0.0274`（超出雜訊）；AR@0.95 差 `+0.43x`（未超出雜訊） |
| **S2** | SMOTE (`sampling_strategy=0.1`) | **0.4181** | **8.23x** | 0.0476 | 0.0244 | PR-AUC 差 `-0.0160`（超出雜訊）；AR@0.95 改善至 8.23x（差 `-4.97x`） |
| **S3** | SMOTE (`sampling_strategy=0.3`) | **0.4075** | **7.37x** | 0.0643 | 0.0273 | PR-AUC 差 `-0.0266`（超出雜訊）；AR@0.95 降至 7.37x（差 `-5.83x`） |
| **S4** | SMOTE (`sampling_strategy=0.5`) | **0.4057** | **9.47x** | 0.0717 | 0.0293 | PR-AUC 差 `-0.0284`（超出雜訊），過多合成樣本導致邊際精度下降，機率失真加劇 |
| **S5** | BorderlineSMOTE (`sampling_strategy=0.3`) | **0.4142** | **8.38x** | 0.0547 | 0.0256 | PR-AUC 差 `-0.0199`（超出雜訊），AR@0.95 為 8.38x |
| **S6** | SMOTETomek (`sampling_strategy=0.3`) | **0.4051** | **7.30x** | 0.0642 | 0.0273 | 與 S3 微幅差異（PR-AUC 差 `-0.0024`，Tomek 在第 2 折剔除 2 筆樣本） |

> [!NOTE]
> **數據查證與歷史成因說明**：
> 1. **回報誤植成因**：先前 Markdown 手動整理表格時，曾誤將 S3 之 `mean_proba_mean` 寫為 0.0347（磁碟 CSV 實為 0.063861）、S0 Brier 寫為 0.0267（磁碟 CSV 實為 0.023185）。磁碟上之原始 CSV `labs/outputs/lab_04/lab_04_imbalance_strategies.csv` 與快照 CSV 本身自始至終完全一致且正確無誤。
> 2. **S3 vs S6 等價成立條件**：在 teaching 管線（Winsorizer 啟用）下 TomekLinks 剔除 0 筆樣本故數值完全相同；在 project 管線下每折剔除 0.4 筆（第 2 折剔除 2 筆，其餘 0 筆），數值微幅不同（M1 PR-AUC 差距 0.000143，M3 PR-AUC 差距 0.002660）。
> 3. **雜訊標註標準**：依據 `lab_08` 測得之 M3 演算法隨機種子標準差（PR-AUC Std: 0.0053，AR@0.95 Std: 1.04x），凡差異未超出雜訊者，嚴格標註為「未超出雜訊」，不得隨意使用「基準」或「勝出」等定性字眼。

---

## 7. lab_05：前處理消融實驗 (`M1`, `M3`)

- **與專案最終管線的差異：本實驗為前處理消融專項，明確對照「截尾開 vs 關（Winsorizer on/off）」與各縮放器之敏感度差異，基準設定依 `--pipeline` 參數可切換為 project 或 teaching。**

### 7.1 實驗目的
以線性模型（M1）與樹集成（M3）為對象，在訓練折內消融三大前處理步驟：
1. 相關性過濾門檻（`corr_threshold ∈ [None, 0.95, 0.80]`）。
2. 極端值截尾（`Winsorizer` 開 vs 關）。
3. 特徵縮放器（`QuantileScaler` vs `StandardScaler`）。
並引入切分種子雜訊參照（基準 seed=42 vs 雜訊 seed=100）。

### 7.2 參數範圍與設定
- 模型：M1（LogisticRegression）、M3（ExtraTreesClassifier）
- 5 折交叉驗證，折內特徵選取與縮放嚴格透過 Pipeline。

### 7.3 實測執行時間
- 實測時間：約 119.63 秒（`--quick`）

### 7.4 假設與實際觀察
- **假設**：
  - 相關性過濾對線性模型（M1）至關重要（去除多重共線性），但對樹集成（M3）影響較小甚至可能損失資訊。
  - QuantileScaler 能將長尾分佈攤平成均勻分佈，使距離計算與分割更穩定；改為 StandardScaler 會使極端值影響未經截尾的特徵。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and use_corr_filter==False" col=retained_features_mean expect=92.0 tolerance=0.1 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and corr_threshold==0.95 and ablation_group=='corr_filter'" col=retained_features_mean expect=71.6 tolerance=0.1 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and corr_threshold==0.8 and ablation_group=='corr_filter'" col=retained_features_mean expect=46.0 tolerance=0.1 -->
  - **特徵漏斗折內實測與條件差異**：
    - 在啟用截尾（`use_winsorizer=True`，lab 基準）下：無過濾保留 92.0 欄；門檻 0.95 時各折動態保留 **75～77 欄**（平均 76.0 欄）；門檻 0.80 時平均保留 56.4 欄。
    - 在關閉截尾（`use_winsorizer=False`）下：門檻 0.95 各折動態保留 **70～73 欄**（平均 71.6 欄，min=70, max=73）。未經截尾之極端長尾值放大了成對 Pearson 相關係數，導致更多特徵超過 0.95 被剔除。專案正式管線在 E1b 結論後關閉截尾（`use_winsorizer=False`），因此正式外層 15 次評估之保留數為 70～73 欄（全訓練集為 72 欄）。兩者前處理條件不同，數值完全精確對應。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='Scaler=quantile'" col=pr_auc_mean expect=0.406385 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='Scaler=standard'" col=pr_auc_mean expect=0.352656 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M3' and setting_name=='Scaler=standard'" col=ar95_mean expect=9.080794 tolerance=0.01 -->
  - **縮放器影響與 M3 AR@0.95 退化機制（有證據的事實、推測與對照驗證分離）**：
    - M1 在 QuantileScaler 下 PR-AUC 為 0.4064（AR@0.95 為 9.15x），改為 StandardScaler 後 PR-AUC 跌至 0.3979（AR@0.95 升至 10.67x）。
    - M3 改用 StandardScaler 後 AR@0.95 均值由 9.14x 劇增至 **18.38x**（折間標準差 10.75x）。
    - **1. 有證據的事實（逐折數據與正樣本分數查驗）**：
      - 逐折實測數值：Fold 0 為 5.92x、Fold 1 為 31.17x、Fold 2 為 9.74x、Fold 3 為 31.17x、Fold 4 為 13.89x，5 折平均即為 $(5.92 + 31.17 + 9.74 + 31.17 + 13.89) / 5 = \mathbf{18.38\text{x}}$。
      - 門檻與正樣本分數查驗（Fold 1 與 Fold 3）：
        - **Fold 1**（$P=35, N=1,091$，需抓取 $k=\lceil 0.95 \times 35 \rceil = 34$ 家）：最後 3 個正樣本的預測分數分別為 `[0.0000, 0.0000, 0.00625]`。第 34 筆正樣本分數恰為 `0.0000`，使截止門檻退化至 `0.0000`。依專案悲觀同分規則（`score >= cutoff`），該折全體 1,091 筆樣本全被預警，AR@0.95 退化至 $N/P = 1,091 / 35 = \mathbf{31.17\text{x}}$。
        - **Fold 3**（$P=35, N=1,091$）：最後 3 個正樣本的預測分數分別為 `[0.0000, 0.0000, 0.0125]`。第 34 筆正樣本分數亦為 `0.0000`，截止門檻退化至 `0.0000`，AR@0.95 同樣退化至 $1,091 / 35 = \mathbf{31.17\text{x}}$。
        - AR@0.90（需抓 32 家）：第 32 筆正樣本分數 $>0.0000$（Fold 1 為 0.0063, Fold 3 為 0.0125），門檻未退化，AR@0.90 均值維持正常的 7.07x。
    - **2. 推測的歸因（標註未對照驗證）**：
      - 「StandardScaler 保留長尾極端偏態，隨機特徵選取在稀疏極端區難以切出有效分割點，導致少數極端破產樣本落入純負葉節點」：**【合理推測，未做個別特徵分割點對照驗證】**。
    - **3. 驗證性補充（對照組實驗數據）**：
      - **對照組 A（StandardScaler + 關閉截尾 `use_winsorizer=False`）**：Fold 1 最後 3 筆正樣本分數為 `[0.0000, 0.0167, 0.0208]`（僅 1 筆為 0，cutoff=0.0167，AR=8.57x）；Fold 3 為 `[0.0063, 0.0104, 0.0125]`（0 筆為 0，cutoff=0.0104，AR=9.86x），無門檻退化，5 折 AR 均值為 **8.16x**。
      - **對照組 B（QuantileScaler + 開啟截尾 `use_winsorizer=True`，即 lab 基準）**：Fold 1 為 `[0.0000, 0.0125, 0.0188]`（cutoff=0.0125，AR=9.89x）；Fold 3 為 `[0.0063, 0.0063, 0.0104]`（cutoff=0.0063，AR=12.31x），亦無門檻退化，5 折 AR 均值為 **9.14x**。
      - **對照組 C（QuantileScaler + 關閉截尾 `use_winsorizer=False`，即專案正式管線）**：Fold 1 為 6.89x；Fold 3 出現 2 筆正樣本分數為 0.0000，觸發單折 31.17x 退化，5 折 AR 均值為 **13.20x**。
  - **切分雜訊參照**：M1 在更換切分種子（seed=100）後之 PR-AUC 差異為 0.0037，AR@0.95 差異為 0.75x。消融實驗中門檻 0.95 vs 0.80 之差異落於切分雜訊範圍內。

### 7.5 解讀與正式實驗對照
- 正式實驗 E1b 中，M1 關閉相關性過濾在考慮折間相關性的 Nadeau-Bengio 檢定下 $p=0.2078$（95% CI 跨越 0），未達顯著。Lab 05 同樣觀察到門檻 0.95 與無過濾之 PR-AUC 均值相近（0.4064 vs 0.4065），但過濾顯著精簡特徵（92 $\to$ 76），提高管線推論效率。

---

## 8. lab_06：召回率與預警率權衡曲線與門檻轉移實驗 (`M3`)

- **前處理配置說明：本實驗腳本未傳遞 `--pipeline` 旗標至底層 `build_pipeline`，維持使用教學管線（開啟 Winsorizer 截尾），專注於門檻與安全邊際在驗證折轉移之核心課題。**

### 8.1 實驗目的
1. 繪製 Recall 在 $[0.80, 1.00]$ 區間之 Recall–AR 權衡曲線，並標定隨機猜測基準線。
2. 模擬 E4a 門檻轉移協定：在訓練折內以 3 折 OOF 呼叫純計算函式 `derive_operating_threshold` 導出门檻，套用到外層驗證折，檢驗各安全邊際 $m \in [0.0, 0.02, 0.04, 0.06]$ 下的達標率。
3. 20 筆資料玩具示範：說明 $\lceil R \times P \rceil$ 階梯跳躍機制。

### 8.2 參數範圍與設定
- 模型：M3（ExtraTrees，固化參數）
- 評估網格：$R \in [0.80, 0.90, 0.95]$，Margin $m \in [0.0, 0.02, 0.04, 0.06]$

### 8.3 實測執行時間
- `--mode quick`（3 外層折 × 2 內層折）：約 36.17 秒
- `--mode standard`（5 外層折 × 3 內層折）：約 81.48 秒

### 8.4 假設與實際觀察（口徑分離：純排序 AR@R vs 門檻轉移實際預警倍數）
- **指標定義口徑嚴格分離**：
  1. **純排序指標 $AR@R$（無門檻）**：依真實標籤與驗證集預測機率之經驗累積分佈（ECDF）逆推，衡量模型單純排序能力。
  2. **門檻轉移實際預警倍數（含安全邊際 $m$）**：由內層 OOF 分數決定運作門檻（扣除邊際 $m$），再施加於外層測試折，衡量實際營運調查負擔。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_recall_alert_curve_snapshot.csv where="target_recall==0.80" col=random_guess_ar expect=30.994318 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_recall_alert_curve_snapshot.csv where="target_recall==0.80" col=alert_ratio_mean expect=3.198730 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_recall_alert_curve_snapshot.csv where="target_recall==0.95" col=alert_ratio_mean expect=7.887778 tolerance=0.01 -->
  - **純排序指標 $AR@R$ 與隨機基準線**：隨機猜測基準線平坦固定於 $N/P = 5,455 / 176 \approx 31.00\text{x}$。M3 在 $R=0.80$ 時純排序 AR 僅 3.20x（相當於隨機猜測的 10.3%）；升至 $R=0.95$ 時純排序 AR 上升至 7.89x。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.95 and margin==0.0" col=alert_ratio_mean expect=7.642540 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.90 and margin==0.0" col=attainment_rate expect=0.600000 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.90 and margin==0.06" col=attainment_rate expect=0.600000 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.95 and margin==0.04" col=attainment_rate expect=1.000000 tolerance=0.01 -->
  - **門檻轉移實際預警倍數與安全邊際 $m$ 緩衝效應**：
    - 在 $R=0.95$ 下，若不加安全邊際（$m=0.00$），門檻轉移實際預警倍數為 7.64x（達標率僅 60.0%）；加入 $m=0.04$ 安全邊際後，達標率提升至 100.0%（實際預警倍數為 12.25x）。
    - 在 $R=0.90$ 下，固定 $m=0.00$ 時轉移達標率為 60.0%（實際預警倍數 5.53x）；加入安全緩衝 $m=0.06$ 後實際預警倍數為 8.56x。

### 8.5 Standard 模式與正式 E4a 實驗並排對照（5 折 × 3 內層折）

為避免僅依 `--quick`（3 折）下結論，以 `--standard`（5 外層折 × 3 內層折）重跑之實測結果與專案正式 E4a（15 次外層評估）並排對比如下：

| 目標召回率 $R$ | 安全邊際 $m$ | Lab 06 Standard 達標率 | Lab 06 Standard 實際預警倍數 | E4a 正式達標率 | E4a 正式實際預警倍數 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.80** | 0.00 | 60.0% | 3.46x | 60.0% | 3.53x |
| **0.80** | 0.02 | 80.0% | 3.96x | 73.3% | 3.87x |
| **0.80** | 0.04 | 80.0% | 4.36x | 86.7% | 4.32x |
| **0.80** | 0.06 | 80.0% | 4.76x | 86.7% | 4.87x |
| **0.90** | 0.00 | 60.0% | 5.53x | 53.3% | 5.64x |
| **0.90** | 0.02 | 60.0% | 5.96x | 66.7% | 6.55x |
| **0.90** | 0.04 | 60.0% | 7.15x | 86.7% | 7.59x |
| **0.90** | 0.06 | 60.0% | 8.56x | 93.3% | 8.83x |
| **0.95** | 0.00 | 60.0% | 7.64x | 66.7% | 7.96x |
| **0.95** | 0.02 | 60.0% | 8.93x | 66.7% | 9.20x |
| **0.95** | 0.04 | **100.0%** | **12.25x** | **100.0%** | **12.63x** |
| **0.95** | 0.06 | **100.0%** | **24.00x** | **100.0%** | **23.74x** |

**對照結論**：Lab 06 在 standard 模式下與正式 E4a 高度吻合：在 $R=0.95$ 下，$m=0.04$ 均精確達到 100% 達標率（實際預警倍數 12.25x vs 12.63x）；$m=0.06$ 均達到 24x 的高預警負擔。推薦邊際規則在兩者完全一致。

---

## 9. lab_07：交叉驗證折數與重複次數效應 (`M1`)

- **前處理配置說明：本實驗腳本未傳遞 `--pipeline` 旗標至底層 `build_pipeline`，維持使用教學管線（開啟 Winsorizer 截尾），專注於交叉驗證折數與樣本量效應之核心課題。**

### 9.1 實驗目的
1. 探索 $K \in \{3, 5, 10\} \times R \in \{1, 3\}$ 交叉驗證矩陣之折內正樣本數、評估標準差與計算時間。
2. 訓練集內部樣本稀少度實驗：分層子抽樣保留比例 $\{0.1, 0.2, 0.3\}$，評估正樣本稀少帶來的評估變異放大與 Clopper-Pearson 95% 寬度。
3. 單元測試：驗證分層切分在各折中的正樣本數最大差額 $\le 1$。

### 9.2 參數範圍與設定
- 模型：M1（LogisticRegression，固化參數）
- 子抽樣比例：10%（~18 家破產）、20%（~35 家破產）、30%（~53 家破產），各 20 次隨機分層評估。

### 9.3 實測執行時間
- 實測時間：約 185.47 秒（`--mode quick`）

### 9.4 假設與實際觀察
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_07_cv_matrix_snapshot.csv where="n_splits_K==3 and n_repeats_R==1" col=val_positives_mean expect=58.666667 tolerance=0.1 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_07_cv_matrix_snapshot.csv where="n_splits_K==5 and n_repeats_R==1" col=val_positives_mean expect=35.200000 tolerance=0.1 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_07_cv_matrix_snapshot.csv where="n_splits_K==10 and n_repeats_R==1" col=val_positives_mean expect=17.600000 tolerance=0.1 -->
  - **折內正樣本數與變異**：$K=3$ 時每折約 58.7 家正樣本，PR-AUC 折間標準差為 0.0356；$K=5$ 時每折 35.2 家正樣本（min 35, max 36），PR-AUC 標準差為 0.0564；$K=10$ 時每折僅 17.6 家正樣本，PR-AUC 標準差擴大至 0.0925，AR@0.95 標準差擴大至 8.99x。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_07_sample_size_ladder_snapshot.csv where="subsample_ratio==0.1" col=clopper_pearson_95ci_width_mean expect=0.031950 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_07_sample_size_ladder_snapshot.csv where="subsample_ratio==0.2" col=clopper_pearson_95ci_width_mean expect=0.021891 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_07_sample_size_ladder_snapshot.csv where="subsample_ratio==0.3" col=clopper_pearson_95ci_width_mean expect=0.017834 tolerance=0.001 -->
  - **樣本稀少階梯化**：當子抽樣保留比例為 10%（正樣本僅約 18 家）時，Clopper-Pearson 95% 信賴區間寬度為 0.0320（AR@0.95 IQR 達 11.81x）；保留比例提升至 30%（正樣本約 53 家）時，信賴區間寬度收斂至 0.0178（AR@0.95 IQR 降至 4.15x）。
  - **分層平衡單元測試**：在 $K \in \{3, 5, 10\}$ 下，各折破產企業數最大與最小差額均嚴格 $\le 1$（單元測試 100% 通過）。

### 9.5 解讀與正式實驗對照
- 專案在 `config.py` 選擇 5 折 $\times$ 3 次重複（每折正樣本 35～36 家）是權衡折衷方案：若採用 10 折，每折正樣本過少（17～18 家），評估方差過大；若採用 3 折，訓練資料利用率僅 66.7%。5 折能平衡折內樣本充足性（35 家足以支持 95% 召回率之階梯劃分）與訓練集覆蓋率（80%）。

---

## 10. lab_08：種子穩定性與隨機性分解 (`M3`)

- **前處理配置說明：本實驗腳本未傳遞 `--pipeline` 旗標至底層 `build_pipeline`，維持使用教學管線（開啟 Winsorizer 截尾），專注於隨機種子穩定性與方差分解之核心課題。**

### 10.1 實驗目的
將模型評估的變異來源進行正交分解，比較三種實驗條件：
1. **條件 1【演算法隨機性】**：固定切分種子（42），改變模型隨機種子（42, 7, 123, 2025, 31337）。
2. **條件 2【資料抽樣變異】**：固定模型種子（42），改變切分隨機種子（42, 7, 123, 2025, 31337）。
3. **條件 3【複合變異】**：同時改變切分種子與模型種子。

### 10.2 參數範圍與設定
- 模型：M3（ExtraTrees，固化參數）
- 5 組隨機種子：`[42, 7, 123, 2025, 31337]`

### 10.3 實測執行時間
- 實測時間：約 144.30 秒（`--mode quick`）

### 10.4 假設與實際觀察
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_08_seed_variance_summary_snapshot.csv where="condition=='1_algorithm_randomness_only'" col=pr_auc_across_runs_std expect=0.005284 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_08_seed_variance_summary_snapshot.csv where="condition=='2_data_split_variation_only'" col=pr_auc_across_runs_std expect=0.012583 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_08_seed_variance_summary_snapshot.csv where="condition=='1_algorithm_randomness_only'" col=pr_auc_within_fold_std_mean expect=0.057119 tolerance=0.001 -->
  - **隨機性正交分解**：
    - 演算法隨機性（條件 1）：PR-AUC 跨種子標準差僅 `0.0053`。
    - 資料抽樣變異（條件 2）：PR-AUC 跨切分標準差為 `0.0126`（約為演算法隨機性的 2.4 倍）。
    - 折間標準差（Within-fold Std）：高達 `0.0571`（為演算法隨機性的 10.8 倍）。

---

## 11. 專門分析小節

### 11.1 演算法隨機性 vs 資料抽樣變異之相對大小

在機器學習工程中，初學者常混淆「演算法隨機性（Algorithm Randomness）」與「資料抽樣變異（Data Sampling Variation）」。本專案透過 `lab_08` 與正式實驗 `E7b`（`e7b_seed_stability.csv`）提供了嚴謹的實證對照：

| 評估維度 | 評估對象與設定 | 指標 | 演算法隨機種子變異 (`seed_std`) | 資料折間抽樣變異 (`fold_std`) | 折間標準誤差 (`fold_se`) | 種子與折 SE 比率 (`ratio_seed_to_fold_se`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **正式 E7b** | **$E_A$ 集成模型**<br>（15 外層折，輪換模型與 SMOTE 種子） | **PR-AUC** | **0.001700** | **0.079081** | 0.020419 | **0.0833 (8.3%)** |
| **正式 E7b** | **$E_A$ 集成模型**<br>（15 外層折，輪換模型與 SMOTE 種子） | **AR@0.95** | **0.147909** | **2.128618** | 0.549607 | **0.2691 (26.9%)** |
| **正式 E7b** | **$E_A$ 集成模型**<br>（15 外層折，輪換模型與 SMOTE 種子） | **Log Loss** | **0.014713** | **0.240029** | 0.061975 | **0.2374 (23.7%)** |
| **Lab 08** | **M3 單一模型**<br>（5 折，固定切分輪換模型種子） | **PR-AUC** | **0.005284** | **0.057119** | - | **約 9.2%** |

#### 關鍵工程洞察與嚴格邊界：
1. **單模型與集成的隨機性量級差異**：M3 單一模型的種子標準差（0.0053）約為 $E_A$ 集成模型種子標準差（0.0017）的 **3.1 倍**。兩者不可直接等同，但**兩者均獨立顯示演算法隨機種子變異遠小於資料抽樣（折間）變異**。
2. **資料抽樣變異佔據主導地位**：在正式 E7b 中，PR-AUC 的折間標準差（0.0791）是演算法種子標準差（0.0017）的 **46.5 倍**；AR@0.95 的折間標準差（2.13x）是種子標準差（0.15x）的 **14.4 倍**。
3. **種子變異遠小於折間標準誤差**：`seed_std` 與 `fold_se`（$\text{fold\_std} / \sqrt{15}$）的比率僅 8.3%（PR-AUC）與 26.9%（AR@0.95），說明固定外層切分下，改變隨機種子引起的擾動，遠小於資料切分帶來的抽樣方差。
4. **協定與經驗邊界宣告**：
   - E7b 固定 5 折 × 3 次重複 = 15 次外層評估切分，輪換的是模型隨機種子（42～46）與 SMOTE 隨機種子（42～46），其評估本質是檢驗固化集成管線在固定抽樣下的數值收斂性。
   - 「集成模型較單一模型在隨機種子上更穩定」屬於機器學習社群之**一般經驗**，本專案未針對同架構單模型與集成進行完全受控的種子方差對比實驗，故不以此作為正式實證結論。

---

### 11.2 高召回率極端區間之非線性膨脹

在破產預測等極端不平衡場景中，業務目標往往要求捕獲極高比例的破產企業（例如 90% 或 95%）。然而，隨之而來的代價是**預警名單規模的非線性劇增**。

本專案透過 `lab_06`、正式實驗 `E4a`（`e4a_threshold_transfer_summary.csv`）與盲測集 `E6`（`e6_operating_points.csv`）揭示了此非線性趨勢：

| 實驗來源 | 目標召回率 ($R$) | 捕獲破產家數 | 調查預警名單數 | 預警比例 | 預警倍數 ($\text{AR} = \text{Alerts}/P$) | 精確率 (Precision) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **正式 E6 (盲測)** | **R80** | 36 / 44 | 146 家 | 10.70% | **3.32x** | 24.66% |
| **正式 E6 (盲測)** | **R90** | 43 / 44 | 337 家 | 24.71% | **7.66x** | 12.76% |
| **正式 E6 (盲測)** | **R95** | 44 / 44 | 524 家 | 38.42% | **11.91x** | 8.40% |
| **正式 E4a (M3 跨折均值)** | **R80** | ~28 / 35 | - | - | **3.53x** (Std: 1.15) | ~24.0% |
| **正式 E4a (M3 跨折均值)** | **R90** | ~32 / 35 | - | - | **5.64x** (Std: 1.63) | ~17.0% |
| **正式 E4a (M3 跨折均值)** | **R95** | ~33 / 35 | - | - | **12.63x** (Std: 3.52，含 $m=0.04$) | ~7.5% |
| **隨機基準線** | **任一 $R$** | $R \times P$ | $R \times N$ | $R$ | **31.00x ($N/P$)** | 3.23% |

#### 關鍵業務與技術洞察：
1. **邊際捕獲成本急遽上升（單次測試集觀察）**：在 E6 盲測集上，由 R80 升至 R90（多抓 7 家破產企業），需多調查 191 家企業（平均每多抓 1 家需調查 27.3 家）；而由 R90 升至 R95（僅多抓 1 家破產企業），需多調查 187 家企業。
   > [!IMPORTANT]
   > **單次盲測集限制說明**：上述「多調查 187 家僅多抓 1 家」為單次測試集（$P=44$ 家破產企業）之特定離散觀測。由於破產樣本稀少，極端召回區間的樣本點具有較大偶然性，邊際成本估計之方差極大，**不可外推為固定不變之客觀規律**。
2. **多折交叉驗證之連續印證**：為避免單次測試集偏差，對照 E4a 跨折（15 次外層評估）之預警倍數：R80 為 3.53x（Std: 1.15）、R90 為 5.64x（Std: 1.63），而 R95（含安全邊際）為 12.63x（Std: 3.52）。多折均值證實了由 90% 邁向 95% 召回率時，排查名單規模呈倍數擴大的本質特性。
3. **預警倍數相對於隨機基準線**：
   - 在 R80 下，模型預警倍數（3.32x）僅為隨機猜測（31.00x）的 **10.7%**，排查效率顯著。
   - 在 R95 下，為了追求極高捕獲率（44/44 家），預警倍數攀升至 11.91x（調查全體企業之 38.4%），精確率降至 8.40%。
4. **安全邊際 $m$ 的必要性**：在 E4a 門檻轉移實驗中，若不加邊際（$m=0$），在驗證折上的召回率達標率僅 53%～66%；加入 $m=0.04$ 安全邊際後，達標率達 86%～100%，但代價是預警倍數由 7.96x 膨脹至 12.63x。這具體體現了風險合規與審計成本之間的本質取捨。


---

## 12. lab_09：ExtraTrees (M3) 樹數消融與分數空隙退化機制

- **前處理配置說明：採用專案正式管線（`--pipeline project`，Quantile 縮放、關閉 Winsorizer 截尾、開啟 CorrelationFilter 0.95）。**

### 12.1 實驗目的與退化機制
在專案模型設計中，ExtraTrees (M3) 最終超參數固化為 `n_estimators=600`。本實驗針對 `n_estimators ∈ {80, 100, 300, 600}` 進行五折受控單變量消融，深入分析樹數量對預測分數解析度與門檻同分行為的影響。

- **【重要觀念澄清】預測機率非投票，而是葉節點類別比例平均**：
  在 `scikit-learn` 的 `ExtraTreesClassifier` 中，`predict_proba` 是所有樹葉節點類別比例的均值，並非整數投票；因此「解析度為 $1/n$」的說法並不精確。
- **分數空隙與同分退化機制**：
  每棵樹的輸出為葉節點類別比例，樹數越少，兩個相鄰分數之間的空隙越大，模型產生的「唯一預測分數個數」越少。當正樣本在邊界分數聚集時，較少的樹數會導致門檻處發生大量樣本同分。專案實施保守判定（`score >= cutoff` Flag 全部同分樣本），一旦同分樣本包含大量正常企業，預警倍數將急遽退化。

### 12.2 五折實測數據對照表（快照快取：`lab_09_tree_count_ablation_snapshot.csv`）

| 樹數 (`n_estimators`) | 5 折平均唯一分數個數 | 5 折平均門檻 $t$ | 5 折平均預警家數 | 5 折平均 AR@0.95 | Fold 3 預警家數 | Fold 3 AR@0.95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **80 棵** | 150.4 個 | 0.4907 | 462.2 家 | **13.20x** | 1,091 家 | **31.17x** (退化) |
| **100 棵** | 179.6 個 | 0.5376 | 321.4 家 | **9.18x** | 370 家 | **10.57x** |
| **300 棵** | 277.6 個 | 0.5796 | 258.4 家 | **7.38x** | 363 家 | **10.37x** |
| **600 棵（專案固化）** | **364.2 個** | 0.5960 | **248.2 家** | **7.09x** | **354 家** | **10.11x** (平滑穩定) |

### 12.3 核心結論
1. **唯一分數個數佐證解析度提升**：隨著樹數從 80 棵增至 600 棵，驗證折唯一預測分數由 150 個提升至 364 個（增幅 142%），有效縮小了相鄰分數間的空隙。
2. **消融證實 Fold 3 退化之根因**：在 80 棵樹時，Fold 3 於門檻處發生粗同分，使 1,091 家企業（佔全體驗證集 100%）被同時判定預警，AR@0.95 退化為 31.17x（即 $N/P$ 全數預警）。而提升至專案正式之 600 棵樹後，Fold 3 預警家數平滑收斂至 354 家（AR@0.95 為 10.11x），5 折平均 AR 穩定達到 7.09x。
3. **專案參數決策支持**：實證確認了專案將樹數固化為 600 棵之必要性，徹底避免了因粗同分導致的預警倍數退化現象。
