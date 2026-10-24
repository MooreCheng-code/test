# Streamlit 互動頁面：實作規劃

> 放在專案的 `docs/streamlit_plan.md`。實作者（AI）請先讀完整份，**一次只做一個步驟，做完貼出實際輸出並停下來等我確認**。
> 本規劃由專案外部撰寫，沒有看過程式碼；第 1 節列出的檔案、函式與欄位名稱都是「假設」，請先逐項驗證並回報不一致之處，再開始實作。

---

## 0. 目標與範圍

做一個 Streamlit 頁面，讓使用者輸入主要財務比率，看到：

1. 凍結模型（E_A：M3 + M4 + M5 的百分位平均）給出的風險百分位、各工作點是否預警、校準機率。
2. 「為什麼」：哪些輸入推高或拉低風險、輸入值在訓練資料中的相對位置、調整單一特徵的假設分析。
3. 模型如何運作的白話說明、各工作點的成本，以及限制與免責。

用途：學習與履歷作品集。**不是**真實的信用或破產判斷工具。

非目標：不重新訓練模型、不修改凍結元件、不使用測試資料、不做使用者登入或資料庫、第一版不做 SHAP。

---

## 1. 前置檢查（第 1 節檢查結果與更正紀錄）

依步驟 A 前置檢查實際執行結果，更正與確認如下：

1. 環境：Python 版本為 3.14.4；`pip install streamlit` 安裝成功；`streamlit.__version__` 為 1.66.1。
2. `src/scoring.py` 中 `load_final_components` 簽名為 `(final_dir: Path | str) -> dict[str, Any]`；`score_dataset` 簽名為 `(X: pd.DataFrame, components: dict[str, Any], model_names: list[str] = ['M3', 'M4', 'M5'], include_threshold_flags: bool = True) -> pd.DataFrame`。回傳欄位包含：`score_E_A`, `score_M3`, `calibrated_prob_platt`, `calibrated_prob_isotonic`, 各模型 raw/pct scores，以及各工作點 flag（`flag_R80`, `flag_R90`, `flag_R95`, `flag_R100`，依據原始碼第 188-191 行確認對應 E_A 門檻）。
3. `outputs/models/final/thresholds.json` 的門檻鍵名為 **`threshold`**（而非 `final_threshold`）；`outputs/tables/operating_points_for_report.csv` 的欄位包含 `fixed_threshold`、`actual_recall_mean`、`alert_ratio_mean`、`alert_proportion_mean`、`precision_mean`、`attainment_rate`。
4. `outputs/models/final/core_features_extended.txt` 的 18 個核心特徵**全數**在凍結 pipeline 實際送入分類器的 72 個特徵之中，沒有任何一個被相關性過濾剔除，全部可作為使用者輸入。模型實際使用 72 個特徵：18 個由使用者輸入、其餘 54 個用訓練中位數補（`fill_columns`）；另外 23 個原始欄位模型不使用（`unused_columns`），但因 FeatureSelector 仍要求 95 個欄位齊全，所以也要補值。「補值說明」只列影響模型的 54 個。
5. 凍結 pipeline 需要的原始欄位數為 95 欄，順序在三模型與訓練集中完全一致；缺欄位時由 `feature_selector` 拋出 `ValueError`。
6. `outputs/tables/operating_points_for_report.csv` 包含各工作點的固定門檻、預警比例、CV 召回、預警倍數、精準率與達標率。
7. 訓練資料取得函式為 `cv.get_train_data()`，回傳 5455 筆訓練資料，與 1364 筆測試資料完全隔離（交集為空集合）；`cv.get_test_data()` 預設拋出 `PermissionError` 防護。
8. 專案根目錄為 `bankruptcy/`，所有程式碼路徑相對於它。`app/` 與 `scripts/` 內需把專案根目錄加入 `sys.path` 後使用 `from src import ...`。

---

## 2. 鐵則（違反就停下來問我）

1. 不得修改 `outputs/models/final/` 的任何檔案，不得更動 `scoring.py` 的評分邏輯。
2. App 與其產生預設值的腳本不得讀取或使用測試資料：不得呼叫 `cv.get_test_data`，不得讀取 `outputs/e6/` 下的任何檔案。預設值、範圍、範例都只能由訓練資料產生。
3. 頁面上所有數字（門檻、各工作點成本、盛行率等）都由檔案讀取，不得寫死在程式或文字中；要有測試驗證。
4. 不得使用「顯著」「證明」等沒有檢定支持的措辭；特徵的財務構面說明一律標註「依欄位名稱，定義待確認」。
5. 所有隨機性固定種子；任何與本規劃不同的設計決定，先說明理由。

---

## 3. 資料與設定檔

### 3.1 `scripts/make_app_defaults.py` → `app/assets/defaults.json`

只用訓練資料（`cv.get_train_data()`），輸出：

- `raw_columns`：模型需要的 95 個原始欄位名稱與順序（取自凍結 pipeline 的 `feature_selector.selected_features_`，並與訓練資料欄位比對一致）。
- `model_columns`：72 個送入分類器的特徵（`corr_filter.selected_features_`）。
- `fill_columns`：`model_columns` 中不屬於 18 個輸入的 54 個欄位。
- `unused_columns`：95 個中模型不使用的 23 個欄位。
- `medians`：95 個欄位各自的訓練中位數（用來補沒有輸入的欄位）。
- `inputs`：18 個輸入特徵各自的 `min`、`p01`、`p05`、`p25`、`p50`、`p75`、`p95`、`p99`、`max`，以及 `tier`（`main` 或 `advanced`）與其依據（排名與平均重要性）。
- `presets`：
  - `normal_profile`：18 個輸入特徵取「訓練資料中非破產公司」的中位數。
  - `bankrupt_profile`：18 個輸入特徵取「訓練資料中破產公司」的中位數。
  - 兩者都標註為「訓練資料彙總出來的虛擬輪廓，不是任何一家真實公司」，頁面上要標示。
- `meta`：訓練資料列數與破產比例（由訓練標籤計算，不寫死）、產生時間、訓練索引雜湊（需與 `outputs/models/final/index_hashes.json` 或 `splits.joblib` 的訓練索引雜湊一致，並在腳本中 assert）。

### 3.2 輸入特徵選擇與分級（Tier）

使用 `core_features_extended.txt` 的 18 個特徵，全數在 72 個模型特徵內。主要輸入與進階輸入的分法由腳本計算決定，不寫死：
- 讀取 `outputs/tables/e5a_permutation_importance.csv`，計算這 18 個特徵在 M3、M4、M5 的平均重要性（`pr_auc_drop_mean`）。
- 排名前 8 為「主要輸入」（`main`，預設展開），其餘 10 個為「進階輸入」（`advanced`，收合）。依資料中的欄位名稱分組（槓桿與償債、獲利、流動性、其他），分組標籤標註「AI 推測，待確認」。

### 3.3 `app/assets/feature_notes.json`

每個輸入特徵一行簡短說明與單位提示（可由欄位名稱推得，無法確定的標註「定義待確認」）。

---

## 4. 頁面設計（繁體中文，四個分頁）

### 分頁 1「輸入與結果」

- 輸入區：`st.number_input`，範圍與預設用 `defaults.json`（範圍取 p01～p99 並略放寬；預設用中位數）。按鈕：套用「典型正常公司」「訓練中破產公司的典型輪廓」「重設」。
- 輸入超出 [p01, p99]：黃色警告「超出訓練資料的常見範圍」；超出 [min, max]：紅色警告「超出訓練資料範圍，結果不可靠」。
- 「補值說明」：expander，列出未輸入而用訓練中位數補上的欄位（只需列影響模型的 54 個 `fill_columns`），並加註說明：「用中位數補其他特徵可能與使用者調整的輸入在財務上不完全一致（例如相關的獲利指標），頁面需明確揭露；『相似公司（k 近鄰）補值』列為第二階段選項，第一版不做」。
- 結果區（按下「計算」後）：
  1. 風險百分位，用白話表達：「高於訓練資料中 X% 的公司」（`score_E_A`）。
  2. 表格：R80／R90／R95／R100 四個工作點，每列顯示：是否預警（對應 E_A 門檻的 `flag_R80`～`flag_R100`）、門檻（`fixed_threshold`）、該工作點在交叉驗證中的預警比例（`alert_proportion_mean`）、預期召回（`actual_recall_mean`）、預警倍數（`alert_ratio_mean`）、精準率（`precision_mean`）與達標率（`attainment_rate`）（取自 `operating_points_for_report.csv`）。
  3. Platt 校準機率（`calibrated_prob_platt`），旁邊註明「反映訓練資料約 3.2% 的破產率，不代表真實世界機率」。盛行率數字由資料計算，不寫死。

### 分頁 2「為什麼」

1. **單項影響**：對每個輸入特徵 i，計算「只把特徵 i 換成訓練中位數、其他不變」後的風險百分位變化 Δ（以一次批次推論完成）。用條形圖顯示（正值＝該特徵目前的值讓風險較高）。註明這描述的是模型行為，不是因果。
2. **相對位置**：每個輸入值落在訓練資料分布的第幾百分位，用表格或色條呈現。
3. **假設分析**：選一個特徵，在 [p01, p99] 掃 40 點，其餘固定，畫風險百分位曲線，並畫出四個工作點的門檻水平線。
4. 第二階段（可選，第一版不做）：SHAP 單筆解釋。

### 分頁 3「模型如何運作」

白話說明：資料、前處理、三個模型、百分位平均、門檻與安全邊際。附一張各工作點成本表（預警比例、預期召回、預警倍數，讀自 CSV），並說明盛行率越低、審查成本越高（可附從 `e7a_prevalence_sensitivity.csv` 讀取的重點列）。

### 分頁 4「限制與免責」

示範用途；資料來源與破產定義待確認；特徵定義依欄位名稱、須以資料來源為準；盛行率與真實世界不同；凍結門檻僅適用於與訓練資料相近的資料；沒有年份與公司識別碼，可能存在同公司跨年度的樂觀偏差；有少數破產公司的財報看起來與正常公司相似，模型抓不到。

---

## 5. 檔案結構

專案根目錄為 `bankruptcy/`，所有路徑相對於此目錄：

```
app/
  app.py            # 只做介面，不放計算邏輯
  logic.py          # 純函式：補值組成完整一列、評分、單項影響、假設分析、相對位置
  ui_text.py        # 所有說明文字
  assets/
    defaults.json
    feature_notes.json
scripts/make_app_defaults.py
tests/test_app_defaults.py
tests/test_app_logic.py
tests/test_app_smoke.py
requirements-app.txt
README_APP.md
../docs/streamlit_plan.md
```

`logic.py` 預計函式（名稱可調整）：`build_full_row(user_inputs, defaults)`、`score_rows(df)`（呼叫 `scoring.score_dataset`）、`single_feature_effects(user_inputs, defaults)`、`what_if_curve(feature, user_inputs, defaults, n=40)`、`position_table(user_inputs, defaults)`。

---

## 6. 效能與部署注意

- `st.cache_resource` 載入凍結元件，只載一次；`defaults.json` 用 `st.cache_data`。
- 推論時 ExtraTrees 的 `n_jobs=1` 已在 `scoring.py` 處理，不要在 App 內另設。
- 假設分析與單項影響都用批次推論（一次呼叫），目標：每次互動 < 3 秒；超過請回報並提出優化方案。
- 套件版本：產生 `requirements-app.txt`，固定 scikit-learn、xgboost、lightgbm、pandas、numpy、joblib、streamlit 的版本，與目前環境一致。模型檔與版本綁定，換環境前請先驗證可載入。

---

## 7. 測試計畫與完成標準

| 編號 | 測試 | 通過條件 |
|---|---|---|
| T1 | `build_full_row` | 產生 95 欄、順序與訓練一致、未輸入欄位等於訓練中位數 |
| T2 | 與評分一致 | App 的風險百分位、flag、機率與直接呼叫 `score_dataset` 逐位元一致（最大差 0） |
| T3 | 單項影響 | 把某特徵換成中位數、其他不變的 Δ，與手動呼叫的結果一致；輸入全為中位數時所有 Δ 為 0 |
| T4 | 假設分析 | 曲線在掃描點上與逐點呼叫 `score_dataset` 一致 |
| T5 | 數字來源 | 頁面顯示的門檻等於 `thresholds.json`；成本表等於 `operating_points_for_report.csv`；程式中沒有寫死的門檻或百分比（用掃描測試） |
| T6 | 測試資料隔離 | `app/` 與 `scripts/make_app_defaults.py` 沒有 `get_test_data`、`e6` 路徑、`test_idx` 字串（靜態掃描測試） |
| T7 | 輸入警告 | 超出 [p01, p99]、超出 [min, max] 時分別出現警告 |
| T8 | 冒煙測試 | 使用 `streamlit.testing.v1.AppTest`：App 載入無例外；套用預設範例並按「計算」後有結果；四個分頁都能渲染 |
| T9 | 既有測試 | `pytest -q` 全部通過，凍結檔案雜湊不變 |

---

## 8. 任務分解與模型分工

目前可用 **Gemini 4 Argon**（每週額度有限）與 **Gemini 3.8 Flash**（額度寬鬆）。原則：計算正確性與審查用 Argon，機械性工作與文字用 Flash。每步結束都要貼出實際輸出，並停下來等我確認。

| 步驟 | 內容 | 建議模型 | 完成標準 |
|---|---|---|---|
| A | 第 1 節前置檢查（只回報） | Flash | 7 項檢查的實際輸出 |
| B | `make_app_defaults.py` 與 `defaults.json`、T6 掃描測試 | Flash | JSON 內容符合 3.1；只用訓練資料；T6 通過 |
| C | `logic.py` 與 T1～T4 | Argon | 測試全過；與 `score_dataset` 最大差 0 |
| D | `app.py` 骨架與 `ui_text.py`（分頁 1、3、4） | Flash | 本機啟動、可輸入、可看到結果；T5、T7 |
| E | 分頁 2（單項影響、相對位置、假設分析） | Flash 起草，Argon 審查 | 圖表正確；T3、T4 對應畫面一致 |
| F | AppTest 冒煙測試、效能測量 | Flash | T8 通過；互動 < 3 秒 |
| G | 最終審查：全部測試、凍結檔案雜湊、無測試資料存取、數字來源 | Argon | T1～T9 全過，並貼出審查輸出 |
| H | `README_APP.md`、`requirements-app.txt`、截圖腳本或說明 | Flash | 另一台環境可依 README 啟動（至少在乾淨 venv 驗證） |

開始前請建立分支：`git checkout -b feature/streamlit-app`；每個步驟完成後 commit；全部完成後打 tag `app-v1`。

---

## 9. 作品集化

- 截圖三張：輸入與結果、單項影響圖、假設分析曲線。再做一段約 15 秒的操作動畫（可用螢幕錄影轉 GIF）。
- README 的 App 章節：一句話介紹、截圖、如何啟動（`streamlit run app/app.py`）、頁面各區塊的說明、限制。
- **公開到 GitHub 之前先確認**：資料集的授權與是否可以公開 `data.csv`；模型檔大小與是否需要 Git LFS；若資料不能公開，改成「README 說明如何自行取得資料並重現」，並確認 App 在沒有 `data.csv` 時能用已產生的 `defaults.json` 與模型檔運作。
- 履歷寫法的重點：嚴謹的驗證設計（巢狀交叉驗證、一次性測試集、洩漏防護）、誠實的統計說明、以業務成本（預警倍數、盛行率敏感度）為核心的評估，再加上這個可互動的示範。

---

## 10. 風險與待決定

1. Python 3.14 能否安裝 Streamlit 與相關套件；若不行，要不要另建與模型檔相容的環境。
2. 18 個輸入中有幾個確實影響模型（第 1 節第 4 點），這會決定表單大小。
3. 只輸入部分特徵、其餘用中位數補，結果可能不代表一家真實公司；頁面上的說明要夠清楚。
4. 資料授權與能否公開。
5. 第二階段是否加入 SHAP 單筆解釋。
