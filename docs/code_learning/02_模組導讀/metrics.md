# 模組導讀：評估指標與統計檢定（metrics.py 與 stats_utils.py）

## 一句話用途
提供面向不平衡破產預警的自訂評估指標（`AR@Recall`、`Recall@AR`、`Brier Score`、業務成本損失）以及針對重複交叉驗證（5 折 × 3 次重複）的統計檢定工具（Nadeau-Bengio 修正檢定、Holm-Bonferroni 多重比較校正）。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「評估與指標層」。
- **上游**：接收各模型預測的分數（連續風險評分或預測機率 `y_score`）與真實標籤 `y_true`。
- **下游**：
  - `bankruptcy/src/cv.py`、`bankruptcy/src/tune.py`（E2 目標尋優）、`bankruptcy/src/ensemble.py`（E3 融合權重尋優）、`bankruptcy/src/thresholding.py`（E4 決策門檻校準）直接依賴本模組指標作為最佳化標的。
  - `bankruptcy/src/final_eval.py` 與 `app/` 用於產出營運成本分析與高危險群警報名額。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - `y_true`: 一維二元陣列（0 或 1），長度為 $N$。
  - `score`: 一維連續型分數陣列，長度為 $N$，數值越大代表破產風險越高。
  - `diffs`: 長度為 15 的一維浮點數陣列，代表 15 次外層評估中兩模型的指標差值。
- **輸出**：
  - 浮點數標量指標（例如 `ar_95 = 2.45`、`brier = 0.031`、`cost = 450.0`）。
  - 統計檢定元組 `(mean_diff, ci_lower, ci_upper, p_value)`。

---

## 核心函式與類別

### 1. `alert_ratio_at_recall()`（`bankruptcy/src/metrics.py:108-142`）
- **做什麼**：計算「為了抓出目標比例（`target_recall`）的破產公司，系統必須發出多少倍實際破產數量的警報（Alert Ratio）」。數值越低越好（1.0 代表完美排序，零誤報）。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `y_true` | `ArrayLike` | （必填） | 真實標籤（0 或 1） |
  | `score` | `ArrayLike` | （必填） | 預測風險分數（越高越危險） |
  | `target_recall` | `float` | （必填） | 目標召回率，必須在 $(0, 1]$ 區間內 |
- **回傳**：`float`，代表警報倍數 $\text{AR} = \frac{\text{被標記為高危險的公司總數}}{\text{實際破產公司總數 } P}$。
- **三個核心問題**：
  - *它解決什麼問題？* 傳統 Precision 與 Recall 隨門檻變動，無法直接告訴風控團隊「需要派幾位稽核員去調查」。Alert Ratio 將預警成本具象化為破產公司數量的倍數（例如 3x 代表抓出 10 間破產需調查 30 間公司）。
  - *沒有它會出什麼錯？* 模型可能只抓出 1 間破產公司就宣稱 Precision 100%，卻放任其他 95% 破產公司倒閉；或者達到 95% Recall 卻把全市場 90% 公司都列為警報，導致風控團隊癱瘓。
  - *有沒有更簡單的做法，為什麼沒選？* 可以用 PR-AUC，但 PR-AUC 是全區間積分，包含不可行的低召回率區域；金融風控要求「召回率底線（如 95%）」，因此本專案直接針對特定召回率下的警報成本最佳化。
- **重點程式碼（`bankruptcy/src/metrics.py:129-142`）**：
  ```python
  # Number of positive samples required to achieve target recall
  need = int(np.ceil(target_recall * P))
  need = max(1, min(need, P))

  # Identify the threshold score (the need-th highest score among positive samples)
  pos_scores = np.sort(s_arr[y_arr == 1])[::-1]
  thr = pos_scores[need - 1]

  # Flag all samples with score >= thr (conservative tie-breaking)
  flagged = int((s_arr >= thr).sum())
  return float(flagged / P)
  ```
- **逐段解說**：
  - 第 130 行：使用無條件進位 $\lceil \text{target\_recall} \times P \rceil$ 算出「必須抓到的破產公司個數（`need`）」。
  - 第 134～135 行：將所有真實破產公司的分數由大到小排序，第 `need` 個破產公司的分數即為達成該召回率所需的「門檻分數 `thr`」。
  - 第 138 行：**保守同分處理（Conservative Tie-breaking）**——只要樣本分數大於等於 `thr` 就一律發出警報（`s_arr >= thr`）。這防止決策樹模型產生粗糙同分時低估警報成本。

---

### 2. `recall_at_alert_ratio()`（`bankruptcy/src/metrics.py:145-200`）
- **做什麼**：當風控團隊的調查預算被嚴格限制在 $k = \lfloor \text{ratio} \times P \rfloor$ 間公司時，模型最多能抓到多少比例的破產公司。數值越高越好（1.0 代表 100% 破產全數攔截）。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `y_true` | `ArrayLike` | （必填） | 真實標籤（0 或 1） |
  | `score` | `ArrayLike` | （必填） | 預測風險分數（越高越危險） |
  | `ratio` | `float` | （必填） | 警報預算倍數（必須 $> 0$） |
- **回傳**：`float`，固定名額預算下的召回率（Recall）。
- **重點程式碼（`bankruptcy/src/metrics.py:175-199`）**：
  ```python
  # Compute alert budget k
  k = int(np.floor(ratio * P))
  if k <= 0: return 0.0
  if k >= N: return 1.0

  sorted_scores = np.sort(s_arr)[::-1]
  thr = sorted_scores[k - 1]

  strictly_greater = s_arr > thr
  tp_strictly_greater = int((strictly_greater & (y_arr == 1)).sum())
  rem = k - int(strictly_greater.sum())

  tied_mask = s_arr == thr
  tied_negatives = int((tied_mask & (y_arr == 0)).sum())

  # Pessimistic allocation: tied negatives consume remaining slots first
  tp_from_ties = max(0, rem - tied_negatives)
  total_tp = tp_strictly_greater + tp_from_ties
  return float(total_tp / P)
  ```
- **逐段解說**：
  - 第 175 行：預算以無條件捨去 $\lfloor \text{ratio} \times P \rfloor$ 計算名額上限 $k$。
  - 第 182～184 行：嚴格大於門檻分數（`score > thr`）的樣本享有絕對優先權，先計入真陽性（TP）。
  - 第 193～196 行：**悲觀同分處理（Pessimistic Allocation）**——對於剛好卡在門檻分數上的同分樣本群，在最壞情況下假設「正常公司（負樣本）先搶佔剩餘配額」，只有剩餘配額超過同分負樣本時，才允許將同分正樣本算入 TP。這保證了在面臨粗糙機率輸出時，回報的召回率具有最壞情況保證（Worst-case guarantee）。

---

### 3. `nadeau_bengio_paired_test()`（`bankruptcy/src/stats_utils.py:29-77`）
- **做什麼**：針對「重複 $K$ 折交叉驗證」計算校正後的配對 t 檢定。校正因重複抽樣所違反的「獨立同分佈（i.i.d.）」假設，防止第一型錯誤（偽陽性）膨脹。
- **參數**：
  | 參數名 | 型別 | 預設值 | 意義 |
  |---|---|---|---|
  | `diffs` | `Sequence[float]` | （必填） | 各折配對指標差值（策略 - 基準），長度為 15 |
  | `n_splits` | `int` | `5` | 每輪折數（預設 5） |
  | `n_repeats` | `int` | `3` | 重複次數（預設 3） |
  | `alpha` | `float` | `0.05` | 顯著水準（95% 信賴區間） |
- **回傳**：`tuple[float, float, float, float]`，即 `(mean_diff, ci_lower, ci_upper, p_value)`。
- **重點程式碼（`bankruptcy/src/stats_utils.py:15-26, 56-62`）**：
  ```python
  def nadeau_bengio_correction_factor(k: int, n_splits: int = 5) -> float:
      # c = 1/k + (n_test / n_train) = 1/k + 1/(n_splits - 1)
      return (1.0 / k) + (1.0 / (n_splits - 1))

  # 在 nadeau_bengio_paired_test 內部：
  s2 = float(np.var(d, ddof=1))
  c = nadeau_bengio_correction_factor(k=k, n_splits=n_splits)
  var_corrected = c * s2
  se_corrected = np.sqrt(max(0.0, var_corrected))
  ```
- **逐段解說**：
  - 重複交叉驗證中，不同輪次的訓練折有高達 60%～80% 的樣本重疊，標準配對 t 檢定會嚴重低估標準誤（Standard Error），產生極度樂觀的虛假顯著性（p-value 奇低）。
  - Nadeau-Bengio (2003) 證明校正係數為 $c = \frac{1}{k} + \frac{n_{\text{test}}}{n_{\text{train}}} = \frac{1}{k} + \frac{1}{n_{\text{splits}} - 1}$。
  - 在本專案中，$n_{\text{splits}} = 5$，$k = 15$，校正係數為 $c = \frac{1}{15} + \frac{1}{4} = 0.0667 + 0.25 = 0.3167$。這將方差放大了數倍，提供嚴謹保守的統計結論。

---

## 最小數字例子：20 家公司手算與程式實測比對

為讓學習者能跨教材對照，此處統一使用標準教學玩具資料（共 20 家公司，其中 4 家真正破產，標註 ★）：

| 公司 | 分數 | 真實標籤 | 排序名次 |
|:---:|:---:|:---:|:---:|
| **A★** | 0.95 | 破產 (1) | 1 |
| **B** | 0.90 | 正常 (0) | 2 |
| **C★** | 0.85 | 破產 (1) | 3 |
| **D** | 0.80 | 正常 (0) | 4 |
| **E** | 0.70 | 正常 (0) | 5 |
| **F★** | 0.65 | 破產 (1) | 6 |
| **G** | 0.55 | 正常 (0) | 7 |
| **H** | 0.50 | 正常 (0) | 8 |
| **I** | 0.45 | 正常 (0) | 9 |
| **J★** | 0.40 | 破產 (1) | 10 |
| **K**～**T** | 0.35～0.02 | 均為正常 (0) | 11～20 |

總樣本數 $N = 20$，破產數 $P = 4$。

### 1. 實測 `alert_ratio_at_recall`（目標召回率下的警報倍數）
專案實際呼叫 `alert_ratio_at_recall(y_true, y_score, target_recall=R)` 輸出如下：

| 目標召回率 $R$ | 公式 $\lceil R \times P \rceil$ | 需要抓出破產數 (`need`) | 鎖定門檻分數 (`thr`) | 觸發警報公司 (`score >= thr`) | 警報總數 (`flagged`) | 警報倍數 (`AR`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.25** | $\lceil 0.25 \times 4 \rceil$ | **1 家** | 0.95 | A★ | 1 家 | **0.25x** ($1/4$) |
| **0.50** | $\lceil 0.50 \times 4 \rceil$ | **2 家** | 0.85 | A★, B, C★ | 3 家 | **0.75x** ($3/4$) |
| **0.75** | $\lceil 0.75 \times 4 \rceil$ | **3 家** | 0.65 | A★, B, C★, D, E, F★ | 6 家 | **1.50x** ($6/4$) |
| **1.00** | $\lceil 1.00 \times 4 \rceil$ | **4 家** | 0.40 | A★～J★ (前 10 家) | 10 家 | **2.50x** ($10/4$) |

> **深入理解 $\lceil R \times P \rceil$ 階梯跳躍**：
> 當 $R = 0.51$ 時，$\lceil 0.51 \times 4 \rceil = 3$，所需破產數立即從 2 級跳至 3，門檻分數瞬間從 0.85 降為 0.65，警報倍數立刻從 0.75x 跳升至 1.50x。這種不連續階梯現象正是離散樣本下的數學必然特性。

### 2. 實測 `recall_at_alert_ratio`（固定預算下的召回率）
專案實際呼叫 `recall_at_alert_ratio(y_true, y_score, ratio=AR)` 輸出如下：

| 預算倍數 (`AR`) | 公式 $\lfloor \text{AR} \times P \rfloor$ | 調查名額 ($k$) | 截斷門檻分數 (`thr`) | 納入調查清單名單 | 抓出破產數 (TP) | 實際召回率 (Recall) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.25x** | $\lfloor 0.25 \times 4 \rfloor$ | 1 家 | 0.95 | A★ | 1 | **0.25** ($1/4$) |
| **0.50x** | $\lfloor 0.50 \times 4 \rfloor$ | 2 家 | 0.90 | A★, B | 1 | **0.25** ($1/4$) |
| **0.75x** | $\lfloor 0.75 \times 4 \rfloor$ | 3 家 | 0.85 | A★, B, C★ | 2 | **0.50** ($2/4$) |
| **1.00x** | $\lfloor 1.00 \times 4 \rfloor$ | 4 家 | 0.80 | A★, B, C★, D | 2 | **0.50** ($2/4$) |
| **1.50x** | $\lfloor 1.50 \times 4 \rfloor$ | 6 家 | 0.65 | A★～F★ | 3 | **0.75** ($3/4$) |
| **2.00x** | $\lfloor 2.00 \times 4 \rfloor$ | 8 家 | 0.50 | A★～H | 3 | **0.75** ($3/4$) |
| **2.50x** | $\lfloor 2.50 \times 4 \rfloor$ | 10 家 | 0.40 | A★～J★ | 4 | **1.00** ($4/4$) |

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **不平衡指標標的** | A. F1-Score<br>B. PR-AUC<br>C. AR@Recall=0.95 與 Recall@AR | **方案 C（自訂警報倍數）** | PR-AUC 無法反映固定高召回率（95%）底線下的真實邊際調查成本；AR@0.95 具備直觀的商業意義（需要耗費幾倍資源）。 | `bankruptcy/src/metrics.py:108-200`；E1～E7 實驗表格 |
| **重複 CV 統計檢定** | A. 傳統 Paired t-test<br>B. 獨立兩樣本 t-test<br>C. Nadeau-Bengio 修正檢定 | **方案 C（Nadeau-Bengio 修正檢定）** | 重複 K 折 CV 的測試折並非完全獨立，傳統 t-test 會低估標準誤產生大量偽顯著；Nadeau-Bengio 引入 $1/k + n_{te}/n_{tr}$ 修正項提供嚴格的第一型錯誤控制。 | `bankruptcy/src/stats_utils.py:15-77`；E1/E3 顯著性比對表格 |

---

## 機器學習概念白話框

> **白話概念：保守同分（Conservative）vs 悲觀同分（Pessimistic）**
> - **什麼是同分問題？** 決策樹（如 RandomForest、LightGBM）預測出的機率常是一小塊一小塊的階梯值，可能出現 5 家公司分數都是 0.80。
> - **為什麼評估要區分「保守」與「悲觀」？**
>   1. 計算「預警成本（AR）」時，專案採**保守策略（全數發警報）**：只要有人分數為 0.80，系統不能投機取巧只查其中 1 人，必須 5 人全查，這樣才能如實反映「最重成本」。
>   2. 計算「預算限制下的召回率（Recall）」時，專案採**悲觀策略（壞事優先）**：如果預算只夠查 2 個人，系統假設最倒楣的情況——抽到的 2 個人全是正常公司（負樣本），不給模型虛假的幸運分數。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「悲觀同分處理」與「最壞情況保證」。

---

## 本模組的單元測試
對應單元測試涵蓋 14 項專屬測試：
1. `bankruptcy/tests/test_metrics.py`：
   - `test_perfect_ranking`：驗證完美排序下 $\text{AR} = 1.0$。
   - `test_completely_tied_scores`：驗證所有樣本分數相等時的保守與悲觀同分處理。
   - `test_boundary_crossing_ties`：驗證同分剛好跨越門檻邊界時的數值穩定性。
   - `test_inverse_relationship_consistency`：驗證 `AR@Recall` 與 `Recall@AR` 之間的單調遞增反向關係。
2. `bankruptcy/tests/test_stats_utils.py`：
   - `test_nadeau_bengio_correction_factor`：驗證校正因子計算精確度。
   - `test_holm_bonferroni_properties`：驗證多重檢定校正後 p-value 之單調性保證。

---

## 讀碼練習
1. **練習 1**：在 20 家公司的玩具資料中，若將公司 B 的分數從 0.90 提升到 0.95（與 A★ 相同），呼叫 `alert_ratio_at_recall(y_true, y_score, target_recall=0.25)` 時，回傳的 AR 會是多少？為什麼？
2. **練習 2**：在 `bankruptcy/src/metrics.py:130` 中，為什麼計算 `need` 必須使用 `np.ceil(target_recall * P)` 而不能使用 `round()`？若 $P=4, \text{target\_recall}=0.1$，`round` 與 `ceil` 的結果分別是多少？
3. **練習 3**：在 5 折 × 3 次重複評估中，若指標差值的變異數為 $s^2 = 0.04$，傳統配對 t 檢定的標準誤為 $\sqrt{s^2 / 15} \approx 0.0516$；而在 Nadeau-Bengio 修正下，校正後的標準誤是多少？這對 p-value 會產生什麼影響？

---

## 延伸：想改這裡時要注意什麼
1. **不得更改同分裁決邏輯**：隨意將 `s_arr >= thr` 改為 `s_arr > thr` 會嚴重低估樹模型的警報成本，並導致既有單元測試失敗。
2. **統計檢定之樣本長度限制**：傳入 `nadeau_bengio_paired_test` 的差值陣列長度必須與專案外層評估折次（5 折 × 3 次重複 = 15 次外層評估）完全對齊。
