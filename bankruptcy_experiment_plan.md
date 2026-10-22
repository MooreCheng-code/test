# 破產預測專題｜實驗規劃與程式實作規格

> 本文件是給 AI 寫程式用的完整規格。請依「第 12 節 實作順序」一個模組一個模組實作，每完成一個模組就先執行、貼出結果，再進下一個。
> **鐵則**：所有前處理、特徵選擇、重抽樣（SMOTE）、閾值選擇都只能在訓練資料上 fit，絕不能用到驗證／測試資料（見第 10 節）。

---

## 1. 專案目標

- 資料：`data.csv`，6,819 筆、95 個財務特徵、目標欄位 `Bankrupt?`（1 = 破產，220 筆，約 3.2%）。
- 任務：二元分類，預測公司是否破產。
- **業務目標：盡可能找出全部破產公司，寧可錯殺（接受大量誤報）。**
- 本階段範圍：只用 `data.csv` 完成「前處理 → 不平衡處理 → 訓練 → 驗證 → 解釋」的完整流程。
- **不在本階段範圍**：真實世界資料的外部驗證與遷移學習（之後會用真實資料把整個流程重做一次，所以程式要保持可重用，見第 13 節）。

### 1.1 關於「全部找出」的現實限制（務必寫進報告）

用未調參的 Random Forest 做 5 折 CV 的粗略基線（僅供參考，用來檢查你的實作結果是否合理）：

| 目標召回率 | 需要預警的家數 | 預警數 / 實際破產數 | 佔全部公司比例 |
|---|---|---|---|
| 80% | 約 757 | 約 3.4 倍 | 約 11% |
| 90% | 約 1,349 | 約 6.1 倍 | 約 20% |
| 95% | 約 1,828 | 約 8.3 倍 | 約 27% |
| 100% | 約 5,939 | 約 27 倍 | 約 87% |

結論：**要 100% 召回，幾乎得把全部公司都標成高風險，等於沒有篩選價值。** 因此本專題不把「100%」當唯一目標，而是：

1. 畫出完整的「召回率 vs 預警成本」曲線，讓決策者選擇工作點。
2. 預設工作點：**召回率 ≥ 95%**，並同時報告 90%、100% 的成本。
3. 模型的好壞，用「達到目標召回率所需的預警倍數」來比較（越低越好）。

---

## 2. 環境與套件

```
python >= 3.10
numpy, pandas, scipy, scikit-learn, imbalanced-learn
xgboost, lightgbm, optuna, shap, matplotlib, seaborn, joblib, pyyaml
```

建議專案結構：

```
bankruptcy/
├── data/data.csv
├── config.py                 # 全域設定（種子、路徑、目標召回率等）
├── src/
│   ├── data_audit.py         # 資料檢查
│   ├── preprocessing.py      # 自訂 transformer（去常數/重複、winsorize、相關性過濾）
│   ├── metrics.py            # 自訂評估函數（見第 7 節）
│   ├── pipelines.py          # 組裝 imblearn Pipeline
│   ├── cv.py                 # 切分與交叉驗證工具
│   ├── run_baselines.py      # E0
│   ├── run_imbalance.py      # E1
│   ├── run_tuning.py         # E2
│   ├── run_ensemble.py       # E3
│   ├── threshold.py          # E4
│   ├── explain.py            # E5
│   ├── final_eval.py         # E6
│   └── sensitivity.py        # E7
└── outputs/{tables,figures,models,logs}
```

`config.py` 至少包含：

```python
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 3            # 可視運算時間調為 5
TARGET_RECALLS = [0.80, 0.90, 0.95, 1.00]
ALERT_RATIOS = [1.5, 3, 5, 10]    # 用來報告「在這個預警倍數下召回多少」
DEFAULT_TARGET_RECALL = 0.95
TARGET_COL = "Bankrupt?"
```

---

## 3. 資料審查（`data_audit.py`）

讀入資料後**先 `df.columns = df.columns.str.strip()`**（原始欄名開頭有空白）。輸出以下結果並存成表格／圖：

1. 形狀、目標分布、缺值數（預期為 0）、重複列數（預期為 0）。
2. 已知需處理的項目（請程式實際驗證，不要假設）：
   - `Net Income Flag` 為常數（全部為 1）→ 移除。
   - 完全重複欄位：`Current Liabilities/Liability` = `Current Liability to Liability`；`Current Liabilities/Equity` = `Current Liability to Equity` → 各保留一個。
   - 約 24 個欄位最大值達 10⁹～10¹⁰（其餘多在 0～1）→ 疑似截斷或編碼值，需做穩健處理。
   - 相關係數 > 0.95 的欄位約 23 組（> 0.99 約 12 組）→ 共線性。
   - 二元旗標欄位：`Liability-Assets Flag`（只有約 0.1% 為 1）。
3. 各特徵對目標的單變量訊號：每個特徵的 ROC-AUC（以特徵值當分數）與 Mann-Whitney U 檢定，輸出前 20 名。
4. **近重複樣本檢查（洩漏風險）**：資料沒有公司 ID／年份，同一家公司可能出現多筆。用標準化後的最近鄰距離，統計「距離極小的樣本對」數量，以及這些樣本對的標籤是否一致，結果寫入報告限制。

---

## 4. 資料切分（`cv.py`）

1. **外層留出測試集**：`train_test_split(test_size=0.2, stratify=y, random_state=SEED)`。測試集只在 E6 使用一次，之前任何實驗都不能碰。
   - 注意：測試集只有約 44 個破產樣本，每多漏一家，召回率就掉約 2.3%，結果會很不穩定，所以主要結論以 CV 為準，測試集只作最終確認並附信賴區間。
2. **訓練集內的交叉驗證**：`RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)`。
   - 每折的驗證集約 35 個破產樣本。
3. **超參數調整**使用**巢狀 CV**：外層 5 折評估、內層 3 折調參（`n_trials` 控制在 30～50）。若運算時間太長，可改為「內層調參 + 外層單次 5 折」，但必須在報告註明。
4. 所有切分都以固定種子產生並存檔（折的索引），方便重現與不同模型使用**完全相同**的切分（配對比較需要）。

---

## 5. 前處理（`preprocessing.py`）

所有步驟都做成 sklearn 相容的 transformer（`fit` / `transform`），放進 Pipeline，**只在訓練折 fit**。

| 順序 | 步驟 | 說明 |
|---|---|---|
| 1 | `DropConstantAndDuplicate` | 移除常數欄位與完全重複欄位（fit 時記錄欄位清單） |
| 2 | `Winsorizer` | 以訓練折的 1%／99% 分位數截尾（只對連續欄位，旗標欄位不處理） |
| 3 | `QuantileTransformer(output_distribution="normal")` 或 `StandardScaler` | 作為前處理選項之一（見 E1），兩者都試，先預設用 Quantile |
| 4 | `CorrelationFilter(threshold=0.95)` | 依訓練折相關矩陣，高度相關的欄位只留一個（選擇與目標單變量 AUC 較高者）。此步驟設為**可開關**的選項 |
| 5 | SMOTE 或其他重抽樣 | 見第 6 節，**放在縮放之後**，因為 SMOTE 用距離做插值 |
| 6 | 分類器 | 見第 8 節 |

說明：樹模型對單調轉換不敏感，但為了讓 SMOTE 在所有模型上行為一致，所有模型共用同一套前處理，不需要為樹模型另外去掉縮放。

---

## 6. 不平衡處理策略（E1 的比較對象）

**必須使用 `imblearn.pipeline.Pipeline`，讓 SMOTE 只在每個訓練折內執行，驗證折保持原始分布。** 先對整份資料做 SMOTE 再切分是嚴重的資料洩漏，會讓成績虛高，不可以這樣做。

| 編號 | 策略 | 備註 |
|---|---|---|
| S0 | 不處理 | 基準 |
| S1 | `class_weight="balanced"`（模型支援者） | XGBoost／LightGBM 用 `scale_pos_weight` |
| S2 | `SMOTE(sampling_strategy=0.1)` | 少數類補到多數類的 10% |
| S3 | `SMOTE(sampling_strategy=0.3)` | |
| S4 | `SMOTE(sampling_strategy=0.5)` | |
| S5 | `BorderlineSMOTE(sampling_strategy=0.3)` | 只在邊界區域合成 |
| S6 | `SMOTETomek(sampling_strategy=0.3)` | 合成後再清理邊界雜訊 |

- `k_neighbors` 預設 5；若某折正樣本太少導致錯誤，才下調。
- 專題已決定使用 SMOTE，因此 E1 的目的是「**證明 SMOTE 有沒有幫助、哪個設定最好**」，結果要誠實呈現，即使 SMOTE 沒有勝過 `class_weight` 也要寫出來。
- SMOTE 會改變訓練時的類別比例，使預測機率偏高。本專題的主要指標是排序型（見第 7 節），不受影響；但若要報告機率值，需另外校正（見 E4）。

---

## 7. 評估指標（`metrics.py`）

因為目標是「找出全部破產」，主要指標是**排序型**的，只看模型分數的排名，不依賴特定閾值。

### 7.1 主要指標：達到目標召回率所需的預警倍數

```python
import numpy as np

def alert_ratio_at_recall(y_true, score, target_recall):
    """達到 target_recall 時，預警家數 / 實際破產家數。越低越好。
    同分樣本一併計入（保守估計），避免樹模型分數相同造成低估。"""
    y_true = np.asarray(y_true); score = np.asarray(score)
    P = y_true.sum()
    need = int(np.ceil(target_recall * P))
    pos_scores = np.sort(score[y_true == 1])[::-1]
    thr = pos_scores[need - 1]              # 第 need 個正樣本的分數
    flagged = (score >= thr).sum()          # 分數 >= thr 全部算預警
    return flagged / P

def recall_at_alert_ratio(y_true, score, ratio):
    """預警數固定為 ratio * P 時的召回率。越高越好。"""
    y_true = np.asarray(y_true); score = np.asarray(score)
    P = y_true.sum()
    k = int(ratio * P)
    thr = np.sort(score)[::-1][k - 1]
    flagged = score >= thr
    return y_true[flagged].sum() / P
```

### 7.2 要輸出的完整指標

| 類別 | 指標 |
|---|---|
| 主要 | `alert_ratio@recall` 於 TARGET_RECALLS（0.80／0.90／0.95／1.00） |
| 主要 | `recall@alert_ratio` 於 ALERT_RATIOS（1.5／3／5／10） |
| 輔助 | PR-AUC（`average_precision_score`）、ROC-AUC |
| 輔助（給定閾值時） | Recall、Precision、F2-score、MCC、混淆矩陣 |
| 機率品質 | Brier score、校準曲線（只在 E4） |

### 7.3 計算方式

- 每折各算一次（用於配對檢定與標準差）。
- 另外把同一次 repeat 的 5 折驗證預測合併（out-of-fold, OOF），再算一次「合併版」指標（樣本數較多，估計較穩定）。
- 因為正樣本少，**「召回率 100%」的結果非常容易受單一個難抓的樣本影響**，要同時報告 95%，並說明最難抓的那幾家的特徵（見 E5）。

### 7.4 調參目標函數

- 超參數調整用 **PR-AUC**（連續、穩定）當目標；最終模型選擇再看 `alert_ratio@recall 0.95` 與 `0.90`。
- 不要直接用 `alert_ratio@recall 1.0` 當調參目標，因為它只取決於最難抓的那一個樣本，波動太大。

---

## 8. 模型與超參數搜尋空間

| 編號 | 模型 | 主要搜尋範圍 |
|---|---|---|
| M0 | `DummyClassifier(strategy="prior")` | 基準，不調參 |
| M1 | Logistic Regression（L2、L1、ElasticNet） | `C`: 1e-3～10（對數）、`solver="saga"`（L1/EN）、`max_iter=5000` |
| M2 | Random Forest | `n_estimators`: 300～800、`max_depth`: 4～None、`min_samples_leaf`: 1～10、`max_features`: sqrt/log2/0.3 |
| M3 | ExtraTrees | 同 M2 |
| M4 | XGBoost | `n_estimators`: 200～800、`learning_rate`: 0.01～0.2、`max_depth`: 2～6、`subsample`/`colsample_bytree`: 0.6～1.0、`reg_lambda`、`scale_pos_weight` |
| M5 | LightGBM | `num_leaves`: 8～64、`min_child_samples`: 5～50、其餘同上 |
| M6 | SVM（RBF，`probability=False`，用 `decision_function` 當分數） | `C`、`gamma` |
| M7 | MLP（sklearn `MLPClassifier`） | 隱藏層 (64,)/(128,64)、`alpha`、`early_stopping` |

- 所有模型都要設定 `random_state=SEED`。
- 樹模型使用 `n_jobs=-1`。
- 需要「分數」的指標統一取 `predict_proba[:, 1]`（SVM 用 `decision_function`）。

---

## 9. 實驗清單（依序執行）

### E0 基準（`run_baselines.py`）

- Dummy、單一最強特徵（直接用該特徵當分數）、LR（預設參數 + 標準化，無重抽樣）。
- 目的：確認流程沒有洩漏，並給後續實驗一條底線。**預期：AUC 約 0.9 以上；若 AUC 異常接近 1.0，立刻檢查洩漏。**

### E1 不平衡策略比較（`run_imbalance.py`）

- 因子：S0～S6 × 前處理（Quantile／Standard）× 模型 M1、M2、M4（預設或輕量參數）。
- 評估：重複 5 折 CV，輸出主要指標的平均與標準差。
- 輸出：一張表（策略 × 模型）與一張圖（每種策略的 `alert_ratio@recall 0.95` 箱型圖）。
- **決定 E2 要用的前處理與重抽樣設定**（若多組差距在一個標準差內，視為無顯著差異，優先選簡單的）。

### E2 調參（`run_tuning.py`）

- 對 M1～M7，使用 E1 選出的前處理與重抽樣設定，Optuna 搜尋（`TPESampler(seed=SEED)`），目標為內層 CV 的 PR-AUC。
- 使用巢狀 CV 取得**不樂觀偏誤**的外層成績。
- 輸出：每個模型的最佳參數、外層 CV 成績表、各模型的 OOF 預測（存 `.npy`，E3、E4 要用）。

### E3 模型比較與集成（`run_ensemble.py`）

- 配對比較：在**相同的折**上，用 Wilcoxon signed-rank 檢定比較模型的 PR-AUC 與 `alert_ratio@recall 0.95`（多重比較用 Holm 校正）。
- 集成：
  1. Soft Voting（對各模型分數先做**排名正規化**再平均，因為不同模型分數尺度不同）。
  2. Stacking（`StackingClassifier`，meta-learner 用 LR；內層 CV 一定要分層）。
- 判斷集成是否真的比最佳單一模型好（差距要大於一個標準差才算有意義）。

### E4 閾值與機率校正（`threshold.py`）

1. 對最佳模型，用**訓練集的 OOF 預測**，為每個目標召回率（0.90／0.95／1.00）找出對應閾值：
   - 閾值 = 讓 OOF 召回率達標的最高分數門檻。
   - 為了「寧可錯殺」，可再加一個**安全邊際**：例如目標 0.95 時，取 OOF 達到 0.97 的閾值。
2. 繪製：召回率 vs 預警倍數曲線、PR 曲線、Lift／Gain 曲線。
3. 機率校正（選做）：對有用 SMOTE 的模型，在**未經重抽樣的 OOF 預測**上做 Platt（sigmoid）或 Isotonic 校正，報告校正前後的 Brier score 與校準曲線。
4. 閾值只能由訓練集決定；用在測試集時不可再調整。

### E5 特徵重要性與可解釋性（`explain.py`）

- 方法：LR 係數（標準化後）、樹模型內建重要性、**Permutation Importance**（以 PR-AUC 為評分、在驗證折上計算）、SHAP（TreeExplainer，取樣 500 筆以內以控制時間）。
- **穩定性**：在各折、各 repeat 重算重要性，報告每個特徵進入前 20 名的頻率。
- **精簡特徵集**：用重要性前 10／20／30 個特徵重訓最佳模型，比較與全特徵的成績差距，並存下精簡特徵清單，**這份清單將用於之後的真實世界資料重做**（真實資料可能算不出全部 95 個比率）。
- **難抓樣本分析**：找出 OOF 預測分數最低的 10 家破產公司（也就是最後才會被抓到的），描述它們的特徵與一般破產公司有什麼不同。

### E6 最終測試集評估（`final_eval.py`，只執行一次）

1. 用整個訓練集，以 E2 的最佳參數與 E4 的閾值，訓練最終模型。
2. 在測試集上評估，報告：
   - 在各閾值下的 Recall、Precision、預警倍數、混淆矩陣。
   - 所有主要指標。
3. **Bootstrap 信賴區間**：對測試集重抽樣 2,000 次，報告主要指標的 95% 信賴區間（因為只有約 44 個破產樣本，區間會很寬，要在報告中明講）。
4. 與 CV 結果比較：若測試集成績明顯差於 CV，要討論可能原因（運氣、近重複樣本洩漏造成 CV 偏樂觀等）。
5. 儲存最終模型 `outputs/models/final_pipeline.joblib`。

### E7 敏感度分析（`sensitivity.py`）

- **盛行率敏感度**：資料集 3.2% 的破產率不等於真實世界。用 OOF 預測，對負樣本做重抽樣（模擬破產率 1%、3%、5%），報告每種情況下「達到 95% 召回所需的預警數佔全部公司的比例」與精確率。
- **隨機種子穩定性**：以 5 個不同種子重跑 E1 的最佳設定，報告主要指標的變動範圍。

---

## 10. 資料洩漏與常見錯誤檢查表（寫程式時逐項確認）

- [ ] 所有 fit（含 winsorize 分位數、scaler、相關性過濾、SMOTE）都只用訓練折。
- [ ] SMOTE 在 `imblearn.pipeline.Pipeline` 內，驗證折沒有被重抽樣。
- [ ] 特徵選擇（含相關性過濾、重要性篩選）若在 CV 外做，會洩漏；若要用於選特徵再評估，必須放進 Pipeline 或巢狀 CV。
- [ ] 閾值只由訓練集／OOF 決定，不在測試集調整。
- [ ] 測試集在 E6 之前從未被讀取用於任何決策。
- [ ] 不同模型使用相同的 CV 折，才能做配對檢定。
- [ ] 所有隨機性都有固定種子，且種子寫在 config。
- [ ] 欄位名稱有 strip；`Bankrupt?` 不可混進特徵。
- [ ] 沒有用準確率當指標。
- [ ] 結果表格附標準差或信賴區間，不只報告最好的一次。

---

## 11. 輸出物

`outputs/tables/`
- `audit_summary.csv`、`univariate_auc.csv`
- `e1_imbalance_comparison.csv`、`e2_tuned_results.csv`、`e3_pairwise_tests.csv`
- `e4_thresholds.csv`、`e6_test_results.csv`、`e7_sensitivity.csv`
- `feature_importance_stability.csv`、`core_features.txt`（精簡特徵清單）

`outputs/figures/`
- 目標分布、相關性熱圖、單變量 AUC 前 20 名
- E1 策略比較箱型圖
- 各模型 PR 曲線、召回率 vs 預警倍數曲線、Lift／Gain 曲線
- SHAP summary plot、Permutation Importance 圖
- 校準曲線（若做 E4 校正）

`outputs/models/`
- `final_pipeline.joblib`、各折索引、各模型 OOF 預測

`outputs/logs/`
- 每次實驗的設定、種子、套件版本、執行時間

---

## 12. 實作順序（給 AI 的工作流程）

1. 建立專案結構與 `config.py`，確認 `data.csv` 能讀入。
2. 實作 `data_audit.py`，執行並貼出結果（確認第 3 節的各項預期）。
3. 實作 `metrics.py` 並寫**單元測試**：用人工構造的分數驗證 `alert_ratio_at_recall`、`recall_at_alert_ratio` 的結果（例如完美排序時倍數應為 1.0）。
4. 實作 `preprocessing.py`、`pipelines.py`、`cv.py`，用最小例子（LR + SMOTE）跑通一次 5 折 CV。
5. 執行 E0，確認沒有洩漏（AUC 約 0.9 上下、不是 0.99）。
6. 依序實作並執行 E1 → E2 → E3 → E4 → E5 → E6 → E7。每個實驗完成後存檔，再進下一個。
7. 最後產出整體結果摘要（一份 `README_results.md`），包含主要表格、圖與限制說明。

給 AI 的提醒：
- 長時間的實驗（E2、E3）請提供 `--quick` 模式（減少折數與 trials），先確認程式能跑，再跑完整版。
- 每個腳本要能獨立執行，並把結果存檔，不要只印在終端機。
- 任何與本規格不同的設計決定，請在程式註解與結果摘要中說明理由。

---

## 13. 為「之後用真實世界資料重做」預留的設計

本階段不做外部驗證，但請讓程式滿足以下條件，之後重做才不用重寫：

1. **特徵清單由設定檔決定**：`config.py` 中有 `FEATURES = None`（None 表示用全部），之後可改成精簡特徵清單（`core_features.txt`）。
2. **資料路徑與目標欄位可設定**：不寫死 `data.csv`，所有腳本接受 `--data` 與 `--target` 參數。
3. **Pipeline 完整封裝並可儲存／載入**：前處理與模型在同一個 Pipeline 內，載入後能直接對新資料 `predict_proba`。
4. **提供 `score_new_data.py`**：讀入新 CSV，驗證欄位名稱與訓練時一致（缺欄位時明確報錯），輸出每筆的風險分數與是否預警。
5. **記錄訓練時的特徵分布摘要**（均值、分位數），之後可與真實資料比較，檢查分布偏移。
6. 目前資料集沒有公司名稱、產業與年份欄位；之後取得真實資料時，若有這些欄位，應保留並用於分群分析與時間切分。

---

## 14. 完成標準

- [ ] E0～E7 全部可重現地執行完畢（固定種子，重跑結果一致）。
- [ ] 清楚呈現：不同重抽樣策略、不同模型之間的比較，附統計檢定。
- [ ] 提供「召回率 vs 預警倍數」曲線，以及 90%／95%／100% 召回所需的預警倍數。
- [ ] 測試集結果附 bootstrap 信賴區間，並誠實討論其不確定性。
- [ ] 提供精簡特徵清單與可解釋性分析。
- [ ] 限制與風險（近重複樣本、無時間欄位、盛行率差異、100% 召回的成本）都寫進報告。
