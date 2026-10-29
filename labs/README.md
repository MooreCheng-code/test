# 《企業破產預測》互動式參數實驗室 (Labs)

本目錄提供 8 支可獨立執行的參數實驗腳本（`lab_01`～`lab_08` 全數完成），專門設計給學習者以「動手實作（Hands-on）」方式直觀觀察超參數對模型行為、預警指標與泛化差距的影響。

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

---

## 2. 環境需求與執行指南

### 2.1 環境與相依套件
本實驗室直接使用專案現有 Python 環境，**完全不需安裝任何額外套件**：
- Python $\ge 3.10$
- `numpy`、`pandas`、`scipy`、`scikit-learn`、`imbalanced-learn`
- `lightgbm`、`xgboost`、`optuna`
- `matplotlib`（已內建中文字型自動降級相容機制，優先使用 Windows 正黑體，若缺失則自動退回無襯線英文字型，確保繪圖不中斷）

### 2.2 執行方式
每支 Lab 腳本均提供檔頭集中參數字典 `PARAMS`，並支援三種執行模式：
- **快速模式（預設 `--quick` 或 `--mode quick`）**：在訓練集上採用縮減網格或折數，確保每支腳本於 $\le 3$ 分鐘內跑完。
- **標準模式（`--standard` 或 `--mode standard`）**：採用 5 折 $\times$ 1 次重複，使用標準搜尋網格。
- **完整模式（`--full` 或 `--mode full`）**：採用 5 折 $\times$ 3 次重複（共 15 折），使用完整搜尋網格與專案正式超參數。
- **F-02 專用擴大模式（`--f02-extended`）**：專為 `lab_03` 設計之 45 組配對深入檢定模式（15 折 $\times$ 3 模型種子）。

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
| **完整模式**<br>(`full`) | `--full`<br>或 `--mode full` | 5 折 $\times$ 3 次重複<br>（共 15 折外層評估） | 完整細緻網格（與正式實驗對齊） | 樹數 400～600 棵（依固化參數設定） | **高精度重現分析**：消除抽樣隨機性，方差與 E1～E4 正式實驗完全對齊，計算耗時較長。 |
| **F-02 擴充檢定** | `--f02-extended` | 5 折 $\times$ 3 次重複 $\times$ 3 種子<br>（共 45 組配對） | 鎖定特定關鍵超參數組合（A vs B） | LightGBM 固化參數 | **假設驗證專用**：針對單一爭議參數（如 F-02）進行配對統計檢定與雜訊對照。 |

---

## 3. lab_01：決策樹與樹集成深度掃描 (`max_depth`)

### 3.1 實驗目的
直觀觀察單棵決策樹（`DecisionTreeClassifier`）與樹集成模型（M2 RandomForest、M3 ExtraTrees、M4 XGBoost）隨著樹深度（`max_depth`）增加時，**訓練折分數與驗證折分數的背離現象（過擬合軌跡）**，理解為何單棵決策樹不可用，而集成模型能藉由平均與正則化約束馴服深度方差。

### 3.2 參數範圍與設定
- 單棵決策樹掃描深度：`[1, 2, 3, 4, 6, 8, 12, None (無限制)]`
- 樹集成比較深度：M2 `[4, 6, 8, 12, None]`、M3 `[4, 6, 8, 12, None]`、M4 `[2, 3, 4, 6]`
- 前處理管線：沿用專案標準 `DropConstantAndDuplicate` $\to$ `Winsorizer` $\to$ `QuantileScaler` $\to$ `CorrelationFilter`

### 3.3 實測執行時間
- 實測時間：約 58 秒（`--quick`）

### 3.4 假設與實際觀察
- **假設**：隨著單樹深度加深，訓練折 PR-AUC 會逼近 1.0（完美背誦訓練資料），但驗證折 PR-AUC 會在深度 3～4 達到頂點後大幅下滑，產生巨大的過擬合差距；而隨機森林與極端隨機樹即使深度為 None，驗證折 PR-AUC 仍能維持在 0.38 以上。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='4'" col=val_pr_auc_mean expect=0.269229 tolerance=0.001 -->
  - 單棵決策樹在深度 4 時達到驗證 PR-AUC 最高峰 0.2692（訓練 PR-AUC 0.5220，差距 +0.2527）。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='None (無限制)'" col=train_pr_auc_mean expect=1.000000 tolerance=0.0001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='None (無限制)'" col=val_pr_auc_mean expect=0.106602 tolerance=0.001 -->
  - 當單樹深度無限制（`None`）時，訓練折 PR-AUC 達到 1.0000（完美擬合），但驗證折 PR-AUC 崩跌至 0.1066，過擬合差距高達 +0.8934。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='DT' and max_depth=='4'" col=val_ar95_mean expect=30.957792 tolerance=0.01 -->
  - **單樹粗階梯同分機制**：單棵樹的驗證折預警倍數（AR@0.95）始終高達約 30.96 倍（接近全折樣本比例 $N/P \approx 31.00\text{x}$）。這是因為單樹葉節點少（例如無深度限制時僅有 0.0 與 1.0 兩個唯一機率值），機率呈粗階梯分佈。在悲觀同分規則（`score >= thr`）下，為了達到 95% 召回率，門檻被迫納入龐大的同分葉節點，使全折樣本幾乎全數被納入預警名單（$Alerts = N$），導致預警倍數退化至 $N/P$。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M2' and max_depth=='12'" col=val_pr_auc_mean expect=0.392631 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M3' and max_depth=='12'" col=val_pr_auc_mean expect=0.413149 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_01_tree_depth_snapshot.csv where="model_name=='M4' and max_depth=='2'" col=val_pr_auc_mean expect=0.402685 tolerance=0.001 -->
  - 相對地，集成模型展現出極強的泛化保護：M2 在深度 12 時 PR-AUC 為 0.3926（AR@0.95 約 9.46 倍）；M3 在深度 12 時 PR-AUC 為 0.4131（AR@0.95 約 9.38 倍）；M4 在深度 2 時 PR-AUC 達 0.4027（AR@0.95 約 10.30 倍）。

### 3.5 解讀與正式實驗對照
- **解讀**：單樹深度增加時，方差（Variance）急遽暴增；而隨機森林與極端隨機樹透過「Bagging（樣本自助抽樣）+ Random Subspace（特徵隨機抽樣）」將多棵高方差的深樹平均，有效抵消了方差。
- **正式實驗對照**：專案 E2b 固化參數中，M4（XGBoost）最佳深度選為 2（極淺樹以避免 Boosting 累加過擬合），而 M2 最佳深度選為 12、M3 最佳深度選為 `None`（深樹配合低樣本葉節點，由 600 棵樹的集成平均消除方差）。本實驗重現了此設計取捨。

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
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_02_forest_hyperparams_snapshot.csv where="model_code=='M2' and param_name=='min_samples_leaf' and param_value=='5'" col=ar95_mean expect=8.181818 tolerance=0.01 -->
  - M2 在 `min_samples_leaf=5` 時預警倍數最優（8.18 倍），過大的葉樣本約束（`min_samples_leaf=10`）反而使 AR@0.95 惡化至 9.41 倍。

### 4.5 解讀與正式實驗對照
- 樹數增加具有明確的邊際遞減效應（100 $\to$ 200 提升顯著，200 $\to$ 400 趨於平緩），正式實驗選擇 300～600 棵樹是在「穩定性」與「訓練資源」間的平衡點。

---

## 5. lab_03：提升法網格與 F-02 深入對照 (`M4`, `M5`)

### 5.1 實驗目的
1. **網格掃描**：探索 XGBoost（M4）與 LightGBM（M5）在 `learning_rate` $\times$ `n_estimators` 網格下的互補消長關係。
2. **F-02 擴大實證對照（45 組配對檢定）**：以專案 M5 固化參數（含 `subsample=0.607169...` 與 S3 SMOTE）為基底，固定外層 15 折切分（5 折 $\times$ 3 次重複），跨 3 組模型種子（42, 123, 2025）嚴格對照設定 A（`subsample_freq=0`）與設定 B（`subsample_freq=1`）。

### 5.2 參數範圍與設定
- 網格：`learning_rate ∈ [0.01, 0.05, 0.15]`，`n_estimators ∈ [40, 80]`（快速模式）
- F-02 基底：讀取 `best_params/M5.json`，45 組配對（15 折 $\times$ 3 模型種子）。

### 5.3 實測執行時間與版本
- 實測時間：快速模式約 52 秒；`--f02-extended` 模式約 441 秒
- LightGBM 執行版本：`4.7.0`

### 5.4 假設與實際觀察
- **假設**：
  - 當 `subsample_freq=0` 時 LightGBM 預設不執行樣本抽樣；當 `subsample_freq=1` 時啟動每輪抽樣，兩者預測分數絕非逐位相同。
  - 在配對差異上，若信賴區間涵蓋 0 且差異小於模型種子雜訊，則在統計上判定為「未偵測到顯著差異（樣本與檢定力有限）」。
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_extended_snapshot.csv where="model_seed==42 and fold_idx==0" col=is_bit_identical expect=False -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_extended_snapshot.csv where="model_seed==42 and fold_idx==0" col=max_abs_score_diff expect=0.234968 tolerance=0.001 -->
  - **F-02 生效確認**：在全體 45 組配對中，逐位完全相同比例為 **0.0%（`is_bit_identical=False`）**，最大絕對預測分數差達 0.3063，皮爾森相關係數 0.9839，斯皮爾曼秩相關 0.9556。這證實 `subsample_freq=1` 確實啟動了隨機子樣本抽樣。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_03_f02_extended_snapshot.csv where="model_seed==42 and fold_idx==0" col=pr_auc_diff_B_minus_A expect=0.009941 tolerance=0.001 -->
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

### 6.1 實驗目的
固定模型（M1 與 M3），在訓練折內嚴格執行 S0～S6 全體 7 組策略，觀察排序指標與機率校準偏移。

### 6.2 參數範圍與設定
- 策略清單：`S0`（不處理）、`S1`（類別權重）、`S2`（SMOTE 0.1）、`S3`（SMOTE 0.3）、`S4`（SMOTE 0.5）、`S5`（BorderlineSMOTE 0.3）、`S6`（SMOTETomek 0.3）

### 6.3 實測執行時間
- 實測時間：約 123 秒（`--quick`）

### 6.4 假設與實際觀察
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S0'" col=mean_proba_mean expect=0.032688 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S1'" col=mean_proba_mean expect=0.171587 tolerance=0.001 -->
  - **M1 的 class_weight 機率膨脹機制**：M1 在 S0 下平均預測機率為 0.0327（與真實盛行率 0.0323 一致）；採用 S1（`class_weight='balanced'`）時大幅上升至 0.1716（膨脹 5.3 倍，Brier 惡化至 0.0883）。這是因為線性模型中 class_weight 等價於在 Log-odds 上平移 $\ln(w_1/w_0) \approx +3.4$。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M3' and strategy=='S0'" col=mean_proba_mean expect=0.033100 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M3' and strategy=='S1'" col=mean_proba_mean expect=0.040200 tolerance=0.001 -->
  - **M3 的 class_weight 膨脹緩和**：M3 在 S0 下為 0.0331，S1 下僅微升至 0.0402（僅 1.2 倍）。因為樹模型中 class_weight 僅改變分裂時的不純度計算，葉節點仍按樣本比例計算，多數葉節點仍為純負樣本（機率 0.0）。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S3'" col=pr_auc_mean expect=0.388301 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_04_imbalance_strategies_snapshot.csv where="model_code=='M1' and strategy=='S6'" col=pr_auc_mean expect=0.388301 tolerance=0.001 -->
  - **M1 S3 vs S6 數值完全相同根因**：S3 為 `SMOTE(0.3)`，S6 為 `SMOTETomek(SMOTE(0.3))`。在 QuantileScaler 將 76 維特徵縮放至均勻分佈後，高維空間中幾無跨類別最近鄰邊界對，TomekLinks 剔除 0 筆樣本，故 S6 輸出與 S3 完全相同。此現象與專案正式 E1a 表格（`e1a_imbalance_comparison.csv`）中 M1 的 S3 與 S6 共享完全相同的 `alert_ratio@recall_0.95_mean = 10.865926` 一致。

#### M3（ExtraTrees）全策略實測對照表：

| 策略代號 | 策略名稱與實作方法 | 驗證 PR-AUC (Mean) | 驗證 AR@0.95 (Mean) | 平均預測機率 (Mean Proba) | 與 S0 差距及雜訊判定（M3 種子雜訊 Std: 0.0053） |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **S0** | 基準線（不進行任何重抽樣） | **0.4402** | **8.54x** | 0.0331 | 基準線（最精簡無介入） |
| **S1** | `class_weight='balanced'` | **0.4412** | **8.77x** | 0.0402 | PR-AUC 差 `+0.0010`（**未超出雜訊**，差異在種子變異內） |
| **S2** | SMOTE (`sampling_strategy=0.1`) | **0.4265** | **9.55x** | 0.0336 | PR-AUC 差 `-0.0137`，AR@0.95 惡化至 9.55x |
| **S3** | SMOTE (`sampling_strategy=0.3`) | **0.4157** | **7.95x** | 0.0347 | PR-AUC 差 `-0.0245`，但預警倍數收斂至最優 7.95x |
| **S4** | SMOTE (`sampling_strategy=0.5`) | **0.4080** | **7.87x** | 0.0358 | PR-AUC 差 `-0.0322`，過多合成樣本導致邊際精度下降 |
| **S5** | BorderlineSMOTE (`sampling_strategy=0.3`) | **0.4173** | **8.47x** | 0.0348 | 與 S3 差異在雜訊內（PR-AUC 差 `+0.0016`） |
| **S6** | SMOTETomek (`sampling_strategy=0.3`) | **0.4157** | **7.94x** | 0.0347 | 與 S3 實質等價（PR-AUC 差 `0.000019`，**未超出雜訊**） |

> [!NOTE]
> **數據查證與歷史成因說明**：
> 1. **回報誤植成因**：階段 4b 回報文字中，M3 之 S0、S2、S3、S6 曾出現「0.4368、0.4363、9.14x」，經查係撰寫報告整理 Markdown 時誤複製自 `lab_05` 前處理消融實驗的對照組數據。磁碟上之原始 CSV `labs/outputs/lab_04/lab_04_imbalance_strategies.csv` 與快照 CSV 本身自始至終完全正確無誤。
> 2. **S3 vs S6 在 M3 上等價機制**：經單獨驗證 5 個訓練折中 TomekLinks 實際剔除筆數，5 折剔除樣本數全數為 0 筆（Fold 0～4 均為 0 筆）。因此 S6（SMOTETomek）與 S3（SMOTE）產生的訓練矩陣完全相同，僅極微小的隨機浮點誤差造成 0.000019 的 PR-AUC 差異，兩者在工程與統計上完全等價。
> 3. **雜訊標註標準**：依據 `lab_08` 測得之 M3 演算法隨機種子標準差（PR-AUC Std: 0.0053），凡與基準線差異絕對值 $\le 0.005$ 者，均嚴格標註為「未超出雜訊」，避免將隨機抖動誤讀為策略效果。

---

## 7. lab_05：前處理消融實驗 (`M1`, `M3`)

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
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='無過濾'" col=retained_features_mean expect=92.0 tolerance=0.1 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='門檻=0.95'" col=retained_features_mean expect=76.0 tolerance=0.1 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='門檻=0.8'" col=retained_features_mean expect=56.4 tolerance=0.1 -->
  - **特徵漏斗折內實測**：無過濾時保留 92.0 個特徵；門檻 0.95 時平均保留 76.0 個特徵（剔除 16 個高度相關特徵）；門檻 0.80 時平均保留 56.4 個特徵。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='Scaler=quantile'" col=pr_auc_mean expect=0.406385 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M1' and setting_name=='Scaler=standard'" col=pr_auc_mean expect=0.397893 tolerance=0.001 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_05_preprocessing_ablation_snapshot.csv where="model_code=='M3' and setting_name=='Scaler=standard'" col=ar95_mean expect=18.376840 tolerance=0.01 -->
  - **縮放器影響**：M1 在 QuantileScaler 下 PR-AUC 為 0.4064（AR@0.95 為 9.15x），改為 StandardScaler 後 PR-AUC 跌至 0.3979（AR@0.95 升至 10.67x）；M3 改用 StandardScaler 後 AR@0.95 嚴重惡化至 18.38x。
  - **切分雜訊參照**：M1 在更換切分種子（seed=100）後之 PR-AUC 差異為 0.0037，AR@0.95 差異為 0.75x。消融實驗中門檻 0.95 vs 0.80 之差異落於切分雜訊範圍內。

### 7.5 解讀與正式實驗對照
- 正式實驗 E1b 中，M1 關閉相關性過濾在考慮折間相關性的 Nadeau-Bengio 檢定下 $p=0.2078$（95% CI 跨越 0），未達顯著。Lab 05 同樣觀察到門檻 0.95 與無過濾之 PR-AUC 均值相近（0.4064 vs 0.4065），但過濾顯著精簡特徵（92 $\to$ 76），提高管線推論效率。

---

## 8. lab_06：召回率與預警率權衡曲線與門檻轉移實驗 (`M3`)

### 8.1 實驗目的
1. 繪製 Recall 在 $[0.80, 1.00]$ 區間之 Recall–AR 權衡曲線，並標定隨機猜測基準線。
2. 模擬 E4a 門檻轉移協定：在訓練折內以 3 折 OOF 呼叫純計算函式 `derive_operating_threshold` 導出门檻，套用到外層驗證折，檢驗各安全邊際 $m \in [0.0, 0.02, 0.04, 0.06]$ 下的達標率。
3. 20 筆資料玩具示範：說明 $\lceil R \times P \rceil$ 階梯跳躍機制。

### 8.2 參數範圍與設定
- 模型：M3（ExtraTrees，固化參數）
- 評估網格：$R \in [0.80, 0.90, 0.95]$，Margin $m \in [0.0, 0.02, 0.04, 0.06]$

### 8.3 實測執行時間
- 實測時間：約 36.17 秒（`--mode quick`）

### 8.4 假設與實際觀察
- **實際觀察**：
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_recall_alert_curve_snapshot.csv where="target_recall==0.80" col=random_guess_ar expect=30.994318 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_recall_alert_curve_snapshot.csv where="target_recall==0.80" col=alert_ratio_mean expect=4.217417 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_recall_alert_curve_snapshot.csv where="target_recall==0.95" col=alert_ratio_mean expect=10.094292 tolerance=0.01 -->
  - **隨機猜測基準線與非線性膨脹**：隨機猜測基準線平坦固定於 $N/P = 5,455 / 176 \approx 31.00\text{x}$。M3 在 $R=0.80$ 時 AR 僅 4.22x（相當於隨機猜測的 13.6%）；但升至 $R=0.95$ 時 AR 劇增至 10.09x（膨脹達 2.4 倍）；當要求 $R=1.00$ 時 AR 逼近 23.74x。
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.90 and margin==0.0" col=attainment_rate expect=0.333333 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.90 and margin==0.06" col=attainment_rate expect=1.000000 tolerance=0.01 -->
  <!-- evidence file=docs/code_learning/lab_snapshots/lab_06_threshold_transfer_snapshot.csv where="target_recall==0.95 and margin==0.04" col=attainment_rate expect=1.000000 tolerance=0.01 -->
  - **安全邊際 $m$ 緩衝效應**：
    - 在 $R=0.90$ 下，固定 $m=0.00$ 時轉移達標率僅 33.3%；加入安全緩衝 $m=0.06$ 後達標率升至 100.0%（實際召回率均值 0.9658）。
    - 在 $R=0.95$ 下，$m=0.04$ 即可確保達標率達 100.0%（實際召回率均值 0.9887）。

### 8.5 解讀與正式實驗對照
- 正式實驗 E4a（`e4a_margin_recommendation.csv`）中，M3 在 $R=0.90$ 與 $R=0.95$ 的推薦安全邊際均為 $m=0.04$（達標率達 80% 以上）。Lab 06 成功重現了未加邊際（$m=0$）時驗證達標率不足、而適度加入邊際能克服分佈轉移的關鍵機制。

---

## 9. lab_07：交叉驗證折數與重複次數效應 (`M1`)

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
- 專案在 `config.py` 選擇 5 折 $\times$ 3 次重複（每折正樣本 35～36 家）是最佳實踐：若採用 10 折，每折正樣本過少（17～18 家），評估方差過大；若採用 3 折，訓練資料利用率僅 66.7%。5 折能平衡折內樣本充足性（35 家足以支持 95% 召回率之階梯劃分）與訓練集覆蓋率（80%）。

---

## 10. lab_08：種子穩定性與隨機性分解 (`M3`)

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
   - E7b 固定外層 15 折切分，輪換的是模型隨機種子（42～46）與 SMOTE 隨機種子（42～46），其評估本質是檢驗固化集成管線在固定抽樣下的數值收斂性。
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
