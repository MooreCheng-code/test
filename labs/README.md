# 《企業破產預測》互動式參數實驗室 (Labs)

本目錄提供 8 支可獨立執行的參數實驗腳本（階段 4a 完成 `lab_01`～`lab_04`；階段 4b 規劃 `lab_05`～`lab_08`），專門設計給學習者以「動手實作（Hands-on）」方式直觀觀察超參數對模型行為、預警指標與泛化差距的影響。

---

## 1. 實驗室總覽表

| Lab ID | 實驗主題 | 涵蓋核心參數概念 | 狀態 | 執行指令 | 快速模式耗時 | 產物目錄 | 快照檔案 |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **lab_01** | 決策樹與樹集成深度掃描 | `max_depth`、過擬合軌跡、Train vs Val 差距 | **已完成** | `python labs/lab_01_tree_depth.py --quick` | 約 184 秒 | `labs/outputs/lab_01/` | `lab_01_tree_depth_snapshot.csv` |
| **lab_02** | 森林樹數、特徵抽樣與葉約束 | `n_estimators`、`max_features`、`min_samples_leaf` | **已完成** | `python labs/lab_02_forest_hyperparams.py --quick` | 約 214 秒 | `labs/outputs/lab_02/` | `lab_02_forest_hyperparams_snapshot.csv` |
| **lab_03** | 提升法網格與 F-02 實證對照 | `learning_rate`、`n_estimators`、`subsample`、`subsample_freq` | **已完成** | `python labs/lab_03_boosting_f02.py --quick` | 約 302 秒 | `labs/outputs/lab_03/` | `lab_03_f02_comparison_snapshot.csv` |
| **lab_04** | 不平衡策略與機率校準偏移 | `S0`～`S6`、SMOTE、`class_weight`、機率扭曲 | **已完成** | `python labs/lab_04_imbalance_strategies.py --quick` | 約 123 秒 | `labs/outputs/lab_04/` | `lab_04_imbalance_strategies_snapshot.csv` |
| **lab_05** | 前處理消融實驗 | 相關性過濾門檻、截尾開關、分位數轉換 | 規劃中 | `python labs/lab_05_preprocessing_ablation.py --quick` | 待測 | `labs/outputs/lab_05/` | 階段 4b 產出 |
| **lab_06** | 門檻轉移、安全邊際與損失函數 | 目標召回率曲線、安全邊際 $m$、達標率 | 規劃中 | `python labs/lab_06_recall_alert_curve.py --quick` | 待測 | `labs/outputs/lab_06/` | 階段 4b 產出 |
| **lab_07** | CV 折數、重複次數與樣本量效應 | CV 折數 $K$、重複次數 $R$、折內正樣本數 | 規劃中 | `python labs/lab_07_cv_folds_repeats.py --quick` | 待測 | `labs/outputs/lab_07/` | 階段 4b 產出 |
| **lab_08** | 模型種子與切分種子穩定性 | 模型種子、切分種子、折間變異 vs 種子變異 | 規劃中 | `python labs/lab_08_seed_stability.py --quick` | 待測 | `labs/outputs/lab_08/` | 階段 4b 產出 |

---

## 2. 環境需求與執行指南

### 2.1 環境與相依套件
本實驗室直接使用專案現有 Python 環境，**完全不需安裝任何額外套件**：
- Python $\ge 3.10$
- `numpy`、`pandas`、`scipy`、`scikit-learn`、`imbalanced-learn`
- `lightgbm`、`xgboost`、`optuna`
- `matplotlib`（已內建中文字型自動降級相容機制，優先使用 Windows 正黑體，若缺失則自動退回無襯線英文字型，確保繪圖不中斷）

### 2.2 執行方式
每支 Lab 腳本均提供檔頭集中參數字典 `PARAMS`，並支援兩種執行模式：
- **快速模式（預設 `--quick`）**：在訓練集上採用 5 折 $\times$ 1 次重複，使用縮減網格，確保約 2～3 分鐘內跑完。
- **完整模式（`--full`）**：採用 5 折 $\times$ 3 次重複（共 15 折），使用完整搜尋網格與專案正式超參數。

```powershell
# 執行單一 Lab（快速模式，約 2~3 分鐘）
python labs/lab_01_tree_depth.py --quick
python labs/lab_02_forest_hyperparams.py --quick
python labs/lab_03_boosting_f02.py --quick
python labs/lab_04_imbalance_strategies.py --quick

# 執行單元測試驗證守衛機制
python labs/test_common.py
```

### 2.3 絕對安全規範 (Leak-Free & Read-Only Protection)
1. **僅使用訓練集資料**：資料入口僅限 `labs._common.load_train_data()`，該函式在載入後立即執行守衛斷言（斷言筆數剛好 5,455 筆、正樣本數剛好 176 筆、且總筆數絕不為 6,819 筆）。嚴禁讀取或猜測測試集索引 `splits.joblib`，嚴禁 import `final_eval.py`、`preflight.py`、`lock_utils.py`。
2. **獨立建構 CV 切分**：Lab 於 5,455 筆訓練集上自行以 `StratifiedKFold` 建立驗證切分，不重用專案正式切分檔案。
3. **產物輸出嚴格隔離**：Lab 執行產物（圖表、全量 CSV、執行清冊 `run_manifest.json`）一律寫入 `labs/outputs/<lab_id>/`，並由 `labs/.gitignore` 排除追蹤；版本庫僅收納提交至 `docs/code_learning/lab_snapshots/` 的小型快照 CSV。絕不寫入 `bankruptcy/outputs/`。

---

## 3. lab_01：決策樹與樹集成深度掃描 (`max_depth`)

### 3.1 實驗目的
直觀觀察單棵決策樹（`DecisionTreeClassifier`）與樹集成模型（M2 RandomForest、M3 ExtraTrees、M4 XGBoost）隨著樹深度（`max_depth`）增加時，**訓練折分數與驗證折分數的背離現象（過擬合軌跡）**，理解為何單棵決策樹不可用，而集成模型能藉由平均與正則化約束馴服深度方差。

### 3.2 參數範圍與設定
- 單棵決策樹掃描深度：`[1, 2, 3, 4, 6, 8, 12, None (無限制)]`
- 樹集成比較深度：M2 `[4, 6, 8, 12, None]`、M3 `[4, 6, 8, 12, None]`、M4 `[2, 3, 4, 6]`
- 前處理管線：沿用專案標準 `DropConstantAndDuplicate` $\to$ `Winsorizer` $\to$ `QuantileScaler` $\to$ `CorrelationFilter`

### 3.3 實測執行時間
- 實測時間：約 183.78 秒（`--quick`，5 折 $\times$ 1 次重複）

### 3.4 假設與實際觀察
- **假設**：隨著單樹深度加深，訓練折 PR-AUC 會逼近 1.0（完美背誦訓練資料），但驗證折 PR-AUC 會在深度 3～4 達到頂點後大幅下滑，產生巨大的過擬合差距；而隨機森林與極端隨機樹即使深度為 None，驗證折 PR-AUC 仍能維持在 0.38 以上。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='4'" col=val_pr_auc_mean expect=0.269229 tolerance=0.001 -->
  - 單棵決策樹在深度 4 時達到驗證 PR-AUC 最高峰 0.2692（訓練 PR-AUC 0.5220，差距 +0.2527）。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='None (無限制)'" col=train_pr_auc_mean expect=1.000000 tolerance=0.0001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='None (無限制)'" col=val_pr_auc_mean expect=0.106602 tolerance=0.001 -->
  - 當單樹深度無限制（`None`）時，訓練折 PR-AUC 達到 1.0000（完美擬合），但驗證折 PR-AUC 崩跌至 0.1066，過擬合差距高達 +0.8934。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='4'" col=val_ar95_mean expect=30.957792 tolerance=0.01 -->
  - 單棵樹的驗證折預警倍數（AR@0.95）始終高達 30.96 倍左右，這是因為單樹葉節點預測機率為粗顆粒階梯，無法對 95% 召回率進行精細排序。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M2' and max_depth=='12'" col=val_pr_auc_mean expect=0.392631 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M3' and max_depth=='12'" col=val_pr_auc_mean expect=0.413149 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M4' and max_depth=='2'" col=val_pr_auc_mean expect=0.402685 tolerance=0.001 -->
  - 相對地，集成模型展現出極強的泛化保護：M2 在深度 12 時 PR-AUC 為 0.3926（AR@0.95 約 9.46 倍）；M3 在深度 12 時 PR-AUC 為 0.4131（AR@0.95 約 9.38 倍）；M4 在深度 2 時 PR-AUC 達 0.4027（AR@0.95 約 10.30 倍）。

### 3.5 解讀與正式實驗對照
- **解讀**：單樹深度增加時，方差（Variance）急遽暴增；而隨機森林與極端隨機樹透過「Bagging（樣本自助抽樣）+ Random Subspace（特徵隨機抽樣）」將多棵高方差的深樹平均，有效抵消了方差。
- **正式實驗對照**：專案 E2b 固化參數中，M4（XGBoost）最佳深度選為 2（極淺樹以避免 Boosting 累加過擬合），而 M2 最佳深度選為 12、M3 最佳深度選為 `None`（深樹配合低樣本葉節點，由 600 棵樹的集成平均消除方差）。本實驗完美重現了此設計取捨。

---

## 4. lab_02：森林樹數、特徵抽樣與葉約束 (`M2`, `M3`)

### 4.1 實驗目的
以隨機森林（M2）與極端隨機樹（M3）為對象，進行單變量超參數掃描：
1. 樹木數量（`n_estimators`）：驗證樹數增加是否能持續降低折間標準差，以及邊際效益遞減點。
2. 特徵抽樣比例（`max_features`）：比較 `sqrt`、`log2` 與固定比例 `0.3`。
3. 葉節點最小樣本數（`min_samples_leaf`）：比較葉節點正則化約束對預警倍數的影響。

### 4.2 參數範圍與設定
- 掃描參數：`n_estimators ∈ [100, 200, 400]`、`max_features ∈ ['sqrt', 'log2', 0.3]`、`min_samples_leaf ∈ [1, 2, 5, 10]`
- 基準參數：M2 採用固化 `max_depth=12, imbalance='S3'`；M3 採用固化 `max_depth=None, imbalance='S0'`

### 4.3 實測執行時間
- 實測時間：約 213.72 秒（`--quick`，5 折 $\times$ 1 次重複）

### 4.4 假設與實際觀察
- **假設**：增加樹數量不會導致過擬合，只會使折間方差收斂並線性增加訓練時間；較小的 `max_features` 能降低樹間相關性，使 ExtraTrees 表現更平穩。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='n_estimators' and param_value=='100'" col=pr_auc_mean expect=0.397349 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='n_estimators' and param_value=='400'" col=pr_auc_mean expect=0.407519 tolerance=0.001 -->
  - M2 樹數由 100 增至 400 時，PR-AUC 由 0.3973 微幅提升至 0.4075，AR@0.95 由 9.14 倍降至 8.46 倍，折間標準差保持穩定（約 0.076～0.083），但每折擬合時間由 1.85 秒增至 3.26 秒。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M3' and param_name=='max_features' and param_value=='log2'" col=pr_auc_mean expect=0.454269 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M3' and param_name=='max_features' and param_value=='0.3'" col=pr_auc_mean expect=0.440134 tolerance=0.001 -->
  - M3 在 `max_features='log2'` 時取得最高的 PR-AUC 0.4543（AR@0.95 約 8.02 倍），優於固定比例 0.3 的 0.4401。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='min_samples_leaf' and param_value=='5'" col=ar95_mean expect=8.181818 tolerance=0.01 -->
  - M2 在 `min_samples_leaf=5` 時預警倍數最優（8.18 倍），過大的葉樣本約束（`min_samples_leaf=10`）反而使 AR@0.95 惡化至 9.41 倍。

### 4.5 解讀與正式實驗對照
- 樹數增加具有明確的邊際遞減效應（100 $\to$ 200 提升顯著，200 $\to$ 400 趨於平緩），正式實驗選擇 300～600 棵樹是在「穩定性」與「訓練資源」間的平衡點。

---

## 5. lab_03：提升法網格與 F-02 實證對照 (`M4`, `M5`)

### 5.1 實驗目的
1. **網格掃描**：探索 XGBoost（M4）與 LightGBM（M5）在 `learning_rate` $\times$ `n_estimators` 網格下的互補消長關係。
2. **F-02 實證對照（本批最關鍵）**：以專案 M5 固化參數（含 `subsample=0.607169...` 與 S3 SMOTE）為基底，固定切分，改變 `subsample_freq ∈ {0, 1}`（並以 `subsample=1.0` 作為全樣本參考），跨 3 組隨機種子嚴格檢驗：**LightGBM 在 `subsample_freq=0` 時是否未啟用 bagging，而在 `subsample_freq=1` 時才真正生效？兩者分數的逐位差異與相關性量級為何？**

### 5.2 參數範圍與設定
- 網格：`learning_rate ∈ [0.01, 0.05, 0.15]`，`n_estimators ∈ [40, 80, 150]`（快速模式）
- F-02 基底：讀取 `best_params/M5.json`，模型種子 `[42, 100, 999]`，5 折交叉驗證。

### 5.3 實測執行時間與版本
- 實測時間：約 301.83 秒（`--quick`）
- LightGBM 執行版本：`4.7.0`

### 5.4 假設與實際觀察
- **假設**：
  - 提升法中較小的學習率需要搭配較多的樹（如 `lr=0.01` 搭配 `n_est=150`），而較大的學習率若搭配過多樹會提早過擬合。
  - 對於 F-02，當 `subsample_freq=0` 時 LightGBM 預設不執行樣本抽樣；當 `subsample_freq=1` 時 LightGBM 啟動每輪抽樣，因此兩者預測分數**絕不會逐位完全相同**，但由於超參數其他約束極強，兩者分數相關性預期仍高達 0.98 以上。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_comparison_snapshot.csv where="model_seed==42 and fold_idx==0" col=is_bit_identical expect=False -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_comparison_snapshot.csv where="model_seed==42 and fold_idx==0" col=max_abs_score_diff expect=0.234968 tolerance=0.001 -->
  - **F-02 生效確認**：透過 `get_params()` 檢驗，設定 A 實際傳入 `subsample_freq=0`，設定 B 實際傳入 `subsample_freq=1`。在全體 15 折（3 種子 $\times$ 5 折）中，**沒有任何一折的分數為逐位相同（`is_bit_identical=False`）**，最大絕對分數差達 0.3034。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_comparison_snapshot.csv where="model_seed==42 and fold_idx==0" col=pr_auc_diff_B_minus_A expect=0.009941 tolerance=0.001 -->
  - **指標變異幅度**：跨折平均配對 PR-AUC 差異為 `-0.00572`（設定 B 減設定 A），AR@0.95 平均差異為 `-0.0008` 倍，預測分數之皮爾森相關係數高達 `0.983647`。
  - **網格掃描觀察**：M4 在 `lr=0.05, n_est=150` 時達到 PR-AUC 最高峰 0.4526（AR@0.95 達 7.48 倍）；當 `lr=0.15, n_est=150` 時 PR-AUC 跌至 0.3963（提早過擬合）。

### 5.5 解讀與正式實驗對照
- **F-02 專案內經驗總結**：本實證研究證實，當 `subsample_freq=0` 時 LightGBM 確實未啟用 Bagging；但即使修正為 `subsample_freq=1`，所帶來的分數變異量級極小（PR-AUC 均值差約 0.005，相關性 0.984），這說明 M5 的強大表現主要源自 `num_leaves`、`reg_lambda` 與 S3 SMOTE 的綜合約束。**此結論作為 F-02 的專案內經驗證據，不推廣為 E6 盲測的影響推論**。

---

## 6. lab_04：不平衡策略與機率校準偏移 (`M1`, `M3`)

### 6.1 實驗目的
固定模型（M1 羅吉斯迴歸與 M3 極端隨機樹），在訓練折內嚴格執行 S0～S6 不平衡重抽樣，同時觀察兩大維度：
1. **排序指標維度**：PR-AUC、ROC-AUC、AR@0.95。
2. **機率校準維度（核心觀察點）**：驗證折上的平均預測機率（`mean_proba`）對比真實破產盛行率（約 3.23%）、以及 Brier 分數。

### 6.2 參數範圍與設定
- 策略清單：`S0`（不處理）、`S1`（類別權重）、`S2`（SMOTE 0.1）、`S3`（SMOTE 0.3）、`S4`（SMOTE 0.5）、`S5`（BorderlineSMOTE 0.3）、`S6`（SMOTETomek 0.3）
- 資料保證：重抽樣嚴格透過 `imblearn.pipeline.Pipeline` 在訓練折內執行，驗證折絕無重抽樣污染。

### 6.3 實測執行時間
- 實測時間：約 123.46 秒（`--quick`，5 折 $\times$ 1 次重複）

### 6.4 假設與實際觀察
- **假設**：
  - 重抽樣會人為拔高正樣本在訓練集的比例，導致輸出之未校準機率嚴重偏高（偏離真實盛行率 0.0323），Brier 分數大幅惡化。
  - 對於線性模型（M1），機率偏移尤其劇烈；但對於以單調排序為主的指標（PR-AUC、AR@0.95），重抽樣的影響可能不如預期顯著，甚至使線性模型的邊界產生失真。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S0'" col=mean_proba_mean expect=0.032688 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S0'" col=brier_mean expect=0.024847 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S1'" col=mean_proba_mean expect=0.171587 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S1'" col=brier_mean expect=0.088339 tolerance=0.001 -->
  - **M1 的機率扭曲極其顯著**：S0 基準下，M1 平均預測機率為 0.0327，與真實盛行率 0.0323 完美貼合（偏移 1.0 倍，Brier=0.0248）；但當採用 S1（`class_weight='balanced'`）時，平均預測機率暴衝至 0.1716（膨脹達 5.3 倍，Brier 惡化至 0.0883）。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S3'" col=mean_proba_mean expect=0.106822 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S3'" col=pr_auc_mean expect=0.388301 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S0'" col=pr_auc_mean expect=0.406385 tolerance=0.001 -->
  - M1 在 S3（SMOTE 0.3）時，平均預測機率膨脹至 0.1068（3.3 倍），而 PR-AUC 反而由 S0 的 0.4064 降至 0.3883，AR@0.95 由 9.15 倍惡化至 11.70 倍。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M3' and strategy=='S0'" col=mean_proba_mean expect=0.033100 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M3' and strategy=='S3'" col=mean_proba_mean expect=0.063901 tolerance=0.001 -->
  - **M3 的機率扭曲相對緩和但依然存在**：M3 在 S0 時平均預測機率為 0.0331（Brier=0.0232）；在 S3 時機率膨脹至 0.0639（2.0 倍，Brier 升至 0.0270）。

### 6.5 解讀與正式實驗對照
- **解讀**：重抽樣方法改變了訓練折的先驗機率分佈，分類器直接輸出的值不再是代表母體真實發生率的「後驗機率」，而是人為膨脹後的評分。這解釋了為什麼專案在後續階段必須透過 ECDF 映射與 Platt Scaling 進行機率校準。
- **正式實驗對照**：對照正式 E1a 表格（`e1a_imbalance_comparison.csv`），正式實驗中 M1 在 S0 的 PR-AUC 為 0.3938，S3 為 0.3808；M3 在 S0 的 PR-AUC 為 0.4284，S3 為 0.4239。Lab 觀察到的趨勢（S0 排序能力均不亞於甚至略優於 SMOTE）與專案正式結論完全吻合。

---

## 7. 階段 4b 預告（待放行後實作）
- **lab_05**：前處理消融實驗（相關性過濾門檻、截尾開關、分位數轉換與標準化對比）。
- **lab_06**：門檻轉移、安全邊際 $m$ 與目標召回率曲線取捨。
- **lab_07**：交叉驗證折數（$K \in \{3, 5, 10\}$）與重複次數（$R \in \{1, 3\}$）變異分析。
- **lab_08**：模型隨機種子與切分種子之穩定性實驗。
