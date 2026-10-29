# 模組導讀：全域設定檔（config.py）

## 一句話用途
集中管理全專案唯一的路徑錨點、隨機種子、外層驗證折數、業務成本比與評估門檻，作為全系統的「單一事實來源」（Single Source of Truth）。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 最頂層的「全域環境與基礎設施層」。
- **上游**：無（純常數定義模組）。
- **下游**：全專案 22 個核心模組（包含 `src/data_audit.py`、`src/cv.py`、`src/preprocessing.py`、`src/tune.py`、`src/ensemble.py`、`src/thresholding.py`、`src/final_eval.py`、`app/` 等）均直接匯入本檔常數。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：無動態參數輸入；在模組被 import 時讀取主機檔案系統路徑 `Path(__file__)`。
- **輸出**：匯出 20 個常數變數，供下游模組調用。
- **副作用**：在模組載入時，自動建立 outputs 下的 4 個目錄（`tables/`、`figures/`、`models/`、`logs/`）。

## 核心常數與定義

### 1. 目錄路徑與自動建檔（`bankruptcy/config.py:6-17`）
- **做什麼**：以當前檔案為基準計算 repo 根目錄與各輸出資料夾，並確保輸出資料夾必然存在。
- **常數定義**：
  | 常數名 | 型別 | 預設值 | 意義與角色 |
  |---|---|---|---|
  | `BASE_DIR` | `Path` | `bankruptcy/` | 專案套件根目錄 |
  | `DATA_PATH` | `Path` | `bankruptcy/data/data.csv` | 原始資料路徑 |
  | `OUTPUT_DIR` | `Path` | `bankruptcy/outputs/` | 實驗輸出主目錄 |
  | `TABLES_DIR` | `Path` | `bankruptcy/outputs/tables/` | 表格 CSV 輸出目錄 |
  | `FIGURES_DIR` | `Path` | `bankruptcy/outputs/figures/` | 圖表 PNG 輸出目錄 |
  | `MODELS_DIR` | `Path` | `bankruptcy/outputs/models/` | 模型 `.joblib` 輸出目錄 |
  | `LOGS_DIR` | `Path` | `bankruptcy/outputs/logs/` | 實驗日誌與快取目錄 |
- **三個核心問題**：
  - *它解決什麼問題？* 避免不同作業系統（Windows 反斜線 vs. Linux 斜線）或不同工作目錄執行腳本時引發的 `FileNotFoundError`。
  - *沒有它會出什麼錯？* 若使用者在 repo 根目錄或子目錄下執行腳本，相對路徑 `./outputs` 會噴出找不到目錄或將檔案散落在各處。
  - *有沒有更簡單的做法，為什麼沒選？* 可以用字串硬編碼路徑（例如 `"/custom/path/to/data.csv"`），但此做法無法跨機器運行且破壞可移植性。
- **重點程式碼（`bankruptcy/config.py:6-17`）**：
  ```python
  # Paths (relative to this config file's directory)
  BASE_DIR = Path(__file__).resolve().parent
  DATA_DIR = BASE_DIR / "data"
  DATA_PATH = DATA_DIR / "data.csv"
  OUTPUT_DIR = BASE_DIR / "outputs"
  TABLES_DIR = OUTPUT_DIR / "tables"
  FIGURES_DIR = OUTPUT_DIR / "figures"
  MODELS_DIR = OUTPUT_DIR / "models"
  LOGS_DIR = OUTPUT_DIR / "logs"

  # Ensure output directories exist
  for directory in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
      directory.mkdir(parents=True, exist_ok=True)
  ```
- **逐段解說**：
  - 第 6 行利用 `Path(__file__).resolve().parent` 錨定 `bankruptcy/` 目錄，確保不論從哪裡啟動 Python，路徑計算基準均一致。
  - 第 16～17 行使用 `mkdir(parents=True, exist_ok=True)` 具冪等性（Idempotent），已存在時不拋出異常，保證初次 clone 專案的使用者執行任何腳本時目錄即刻就緒。
- **常見誤用或陷阱**：
  在 `labs/` 腳本中**切勿直接匯入 `OUTPUT_DIR` 來存檔**！Lab 的實驗產物一律必須寫入 `labs/outputs/`，以防止覆寫已固化的正式實驗結果。

---

### 2. 隨機種子與切分參數（`bankruptcy/config.py:20-25`）
- **做什麼**：固化全專案所有涉及隨機抽樣的亂數起點，並指定外層資料劃分結構。
- **常數定義**：
  | 常數名 | 型別 | 預設值 | 意義與角色 |
  |---|---|---|---|
  | `SEED` | `int` | `42` | 全域隨機種子 |
  | `TEST_SIZE` | `float` | `0.2` | 最終測試集切分比例（20% = 1,364 筆） |
  | `CV_FOLDS` | `int` | `5` | 外層交叉驗證折數（每折佔訓練集 20% = 1,091 筆） |
  | `CV_REPEATS` | `int` | `3` | 外層交叉驗證重複次數（3 次不同種子打散） |
- **三個核心問題**：
  - *它解決什麼問題？* 確保任何團隊成員或評審在不同機器上執行時，切出的資料集與隨機數列 100% 相同。
  - *沒有它會出什麼錯？* 未固定種子將導致每次切出的破產樣本不同，指標產生隨機漂移，無法比對不同模型的真實優劣。
  - *有沒有更簡單的做法，為什麼沒選？* 可以只跑單次 5 折 CV，但因為破產樣本極少（訓練集僅 176 家，單折僅 35～36 家），單次切分的折變異過大，專案選擇「5 折 × 3 次重複 = 15 次外層評估」以平滑切分雜訊。
- **重點程式碼（`bankruptcy/config.py:20-25`）**：
  ```python
  # Random Seed
  SEED = 42

  # Data Split & Cross-Validation
  TEST_SIZE = 0.2
  CV_FOLDS = 5
  CV_REPEATS = 3  # Can be adjusted to 5 if computation time allows
  ```
- **逐段解說**：
  - 第 20 行固定 `SEED = 42`，被傳遞給 `StratifiedKFold`、`train_test_split`、Optuna sampler 與各機器學習分類器。
  - 第 24～25 行定義 5 折與 3 次重複。必須注意：統計術語稱作「**5 折 × 3 次重複 = 15 次外層評估**」，絕不能稱為「15 折」（15 折意味著每折訓練 93.3% 驗證 6.7%，而本專案每折始終是訓練 80% 驗證 20%）。

---

### 3. 業務評估指標與成本常數（`bankruptcy/config.py:28-44`）
- **做什麼**：定義不平衡分類下的警報倍數目標、成本不對稱權重與上線檢驗標準。
- **常數定義**：
  | 常數名 | 型別 | 預設值 | 意義與角色 |
  |---|---|---|---|
  | `TARGET_RECALLS` | `list[float]` | `[0.80, 0.90, 0.95, 1.00]` | 評估預警成本的召回率階梯 |
  | `DEFAULT_TARGET_RECALL` | `float` | `0.95` | 專案核心業務檢驗點（95% 破產召回率） |
  | `ALERT_RATIOS` | `list[float]` | `[1.5, 3, 5, 10]` | 固定警報預算下的召回率檢驗點 |
  | `COST_FN` | `float` | `20.0` | 漏報一間破產公司的損失權重（偽陰性損失） |
  | `COST_FP` | `float` | `1.0` | 誤報一間正常公司的調查成本（偽陽性成本） |
  | `COST_FN_SENSITIVITY_LIST` | `list[float]` | `[10.0, 20.0, 50.0]` | E7a 成本敏感度分析的不同 FN 損失假設 |
  | `TOP_K_PERCENT` | `float` | `0.10` | 高危險群前 10% 預警名額 |
  | `MIN_BATCH_SIZE_FOR_TOP_K` | `int` | `25` | 批次名額切分所需的最低樣本數門檻 |
- **三個核心問題**：
  - *它解決什麼問題？* 在破產預測中，漏報（FN）的代價（壞帳呆帳）遠大於誤報（FP，派員審核成本），常規的 Accuracy 失去意義。
  - *沒有它會出什麼錯？* 模型會偏向「全猜不破產」，Accuracy 高達 96.77%，但實際召回率為 0，商業上一文不值。
  - *有沒有更簡單的做法，為什麼沒選？* 可以用預設門檻 0.5 計算 F1-score，但預設門檻無法反映 20:1 的真實成本傾斜，本專案選擇基於業務情境自訂 `COST_FN=20.0` 與 95% 召回率門檻。

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **外層評估架構** | A. 單次 80/20 切分<br>B. 單次 5 折 CV<br>C. 5 折 × 3 次重複 | **方案 C（5 折 × 3 次重複 = 15 次外層評估）** | 破產為極端不平衡資料（正面樣本僅 220 筆，訓練集僅 176 筆），單次切分驗證折僅 35～36 筆，指標變異極大；3 次重複打散可提供足夠樣本點進行 Nadeau-Bengio 假說檢定。 | `bankruptcy/config.py:24-25`；`bankruptcy/src/cv.py:97-101`；`outputs/tables/e1_baseline_cv15.csv` |
| **業務成本比** | A. 1:1 對稱成本<br>B. 5:1<br>C. 20:1（基準）與 10~50 敏感度 | **方案 C（COST_FN=20.0, COST_FP=1.0）** | 依據放貸審核實務，放貸違約損失本金利息，而誤報僅耗費徵信人員 1～2 小時核對財報；另設 `COST_FN_SENSITIVITY_LIST` 確保決策魯棒性。 | `bankruptcy/config.py:38-40`；E7a 實驗結果 `outputs/tables/e7a_prevalence_sensitivity.csv` |

---

## 機器學習概念白話框

> **白話概念：單一事實來源（Single Source of Truth, SSOT）**
> - **什麼是 SSOT？** 在軟體工程與資料科學中，當某個常數（例如隨機種子 42、成本比例 20:1）在多個檔案中出現時，如果散落在各處複製貼上，日後一旦要修改就會漏改一部分，導致不同模組間溝通雞同鴨講。
> - **最小例子**：若 A 腳本寫 `SEED=42` 切出第 1 批資料，B 腳本在調參時寫 `SEED=123` 切出不同驗證集，兩個模組所評估的對象根本不同。全專案統一 `from bankruptcy.config import SEED` 即可徹底杜絕這種不一致。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「單一事實來源」與「可重現性」。

---

## 本模組的單元測試
本模組為純常數設定，無專屬獨立測試檔，但作為基礎設施，其常數被全體單元測試直接引用：
1. `bankruptcy/tests/test_integrity_phase0.py`：驗證資料欄位規格與全域一致性。
2. `bankruptcy/tests/test_preprocessing.py:35`：驗證前處理管線是否遵守 `config.SEED`。
3. `bankruptcy/tests/test_metrics.py`：驗證指標計算函式是否與 `config.DEFAULT_TARGET_RECALL` 行為相符。

---

## 讀碼練習
1. **練習 1**：在 `bankruptcy/config.py` 中，如果把 `CV_REPEATS` 從 3 改為 5，外層總共會產出幾次外層評估？這是否意味著執行「25 折 CV」？（請以精確術語回答）
2. **練習 2**：若某位工程師在執行推論時將 `DATA_PATH` 直接以字串 `"data/data.csv"` 取代 `BASE_DIR / "data" / "data.csv"`，當他在專案外的目錄（例如桌面）執行 `python bankruptcy/score_new_data.py` 時會發生什麼錯誤？為什麼？
3. **練習 3**：`DEFAULT_TARGET_RECALL = 0.95` 與 `TOP_K_PERCENT = 0.10` 分別代表什麼業務意涵？兩者在何種情況下可能產生衝突？

---

## 延伸：想改這裡時要注意什麼
1. **不得在進行中覆寫正式產物**：修改 `config.py` 中的任何常數（例如 `SEED` 或 `CV_FOLDS`）會直接導致既有的 `outputs/` 快取失效。
2. **Lab 練習的隔離守則**：若欲實驗不同的 `CV_REPEATS` 或 `SEED`，請在 `labs/` 腳本中自訂區域常數，**絕不要直接修改 `bankruptcy/config.py`**。
