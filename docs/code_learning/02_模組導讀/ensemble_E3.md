# 模組導讀：模型集成與評估比對（ensemble.py，實驗 E3）

## 一句話用途
使用經驗累積分布函數（ECDF）百分位數映射將多個模型之異質風險分數投影至均勻刻度，並以等權平均融合構建主要集成模型 $E_A$（M3+M4+M5），同時進行成對假說檢定與難抓破產樣本診斷。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「模型集成與評估比對層（E3 實驗）」。
- **上游**：讀取 E2a 在外層 15 次評估中所儲存的單一模型 OOF 預測矩陣（`outputs/models/oof/e2_M*.npy`）與 E2a 折結果表。
- **下游**：
  - 選出專案主力模型 $E_A$ 與對照組 M3，輸出集成評估結果。
  - 將 $E_A$ 候選清單傳入 E4 執行跨折門檻轉移（E4a）、機率校準（E4b）與最終固化（E4c）。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - 各單一模型的外層 OOF 分數陣列，形狀為 `(3, 5455)`（3 次重複 × 5,455 訓練樣本）。
  - `y_train`: 訓練集真實標籤，長度 5,455。
- **輸出**：
  - 模型成對統計比對表：`outputs/tables/e3a_pairwise_models.csv`、`outputs/tables/e3a_friedman_ranks.csv`。
  - 集成模型評估摘要表：`outputs/tables/e3b_ensemble_summary.csv`、`outputs/tables/e3b_ensemble_vs_single_paired.csv`。
  - Stacking 探索摘要表：`outputs/tables/e3c_stacking_summary.csv`。
  - 難抓破產樣本清單：`outputs/tables/e3d_hard_positives.csv`。
  - 最終模型決策卡：`outputs/tables/e3_selection.csv`。

---

## 核心函式與演算法

### 1. `ecdf_map()` 分位數映射（`bankruptcy/src/ensemble.py:80-99`）
- **做什麼**：將測試或查詢樣本的原始預測分數（`query_scores`），依據參考分佈（`reference_scores`）的經驗累積分布函數（ECDF），映射為 `[0, 1]` 之間的經驗百分位數。
- **數學原理（對稱中點處理同分）**：
  $$P(q) = \frac{\text{count}(\text{ref} < q) + \text{count}(\text{ref} \le q)}{2 \cdot N_{\text{ref}}}$$
  在遇上完全同分的值時，取嚴格小於與小於等於排名的對稱中點，保證單調性且數值嚴格落在 `[0, 1]` 區間內。
- **三個核心問題**：
  - *它解決什麼問題？* 不同模型產出的原始分數分佈極端不同（例如 Logistic Regression 輸出機率在 0～1 且集中在 0.03；XGBoost 輸出 logit 可能在 -6 到 +2）。直接將未對齊的分數相加會導致某個量綱偏大的模型完全壟斷集成結果。
  - *沒有它會出什麼錯？* 若直接取平均，梯度提升樹的分數變異數遠大於線性模型，等權平均實質上退化為單一模型的決策。
  - *有沒有更簡單的做法，為什麼沒選？* 可以用 Min-Max 歸一化，但極端離群值會把 Min-Max 壓縮到無效；ECDF 依排名轉換，天然具備對離群值的不變性（Rank-based Invariance）。
- **重點程式碼（`bankruptcy/src/ensemble.py:80-99`）**：
  ```python
  def ecdf_map(reference_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
      ref_sorted = np.sort(np.asarray(reference_scores, dtype=float))
      n_ref = len(ref_sorted)
      if n_ref == 0:
          raise ValueError("reference_scores cannot be empty.")
      query = np.asarray(query_scores, dtype=float)
      left = np.searchsorted(ref_sorted, query, side="left")
      right = np.searchsorted(ref_sorted, query, side="right")
      return (left + right) / (2.0 * n_ref)
  ```
- **最小數字例子（5 個參考數值）**：
  - 設定參考分佈陣列：`ref = np.array([0.1, 0.2, 0.4, 0.4, 0.8])`（長度 $N_{\text{ref}} = 5$）。
  - 對查詢樣本 `query = [0.05, 0.2, 0.4, 0.9]` 計算 ECDF：
    1. 查詢 $0.05$（小於所有參考值）：$\text{left}=0, \text{right}=0 \implies \frac{0 + 0}{2 \times 5} = \mathbf{0.0}$
    2. 查詢 $0.2$（落在第 2 名）：$\text{left}=1, \text{right}=2 \implies \frac{1 + 2}{10} = \mathbf{0.3}$
    3. 查詢 $0.4$（同分 2 次，佔據索引 2 與 3）：$\text{left}=2, \text{right}=4 \implies \frac{2 + 4}{10} = \mathbf{0.6}$（對稱中點）
    4. 查詢 $0.9$（大於所有參考值）：$\text{left}=5, \text{right}=5 \implies \frac{5 + 5}{10} = \mathbf{1.0}$

---

### 2. $E_A$ 的組成成員與融合方式
- **成員清單**：$E_A$ 整合三種底層架構互異的 Gradient Boosting 模型：
  1. `M3`: `HistGradientBoostingClassifier`（scikit-learn 內建直方圖梯度提升）
  2. `M4`: `XGBClassifier`（XGBoost 極限梯度提升）
  3. `M5`: `LGBMClassifier`（LightGBM 輕量梯度提升）
- **融合公式（等權平均）**：
  $$S_{E_A}(x) = \frac{1}{3} \cdot P_{\text{ECDF}, M3}(x) + \frac{1}{3} \cdot P_{\text{ECDF}, M4}(x) + \frac{1}{3} \cdot P_{\text{ECDF}, M5}(x)$$
  每個模型先查閱自身在訓練集上的 ECDF 參考分佈取得百分位數，再取算術平均。

---

### 3. E3 實驗子項目分工與回答的問題

| 實驗代碼 | 實驗名稱 | 回答的核心問題 | 產出檔案依據 |
|:---:|---|---|---|
| **E3a** | 單一模型成對統計比較 | 5 個調優後的單一模型（M1～M5）之間，是否有任何一個模型在統計上顯著優於其他模型？ | `outputs/tables/e3a_pairwise_models.csv`<br>`outputs/tables/e3a_friedman_ranks.csv` |
| **E3b** | ECDF 百分位數集成比較 | 將多個模型經 ECDF 映射後等權平均（$E_{\text{all}}, E_A, E_B, E_{\text{trees}}$），能否取得比單一模型更低的三點損失或更穩定的預警倍數？ | `outputs/tables/e3b_ensemble_summary.csv`<br>`outputs/tables/e3b_ensemble_vs_single_paired.csv` |
| **E3c** | Stacking 後設學習器探索 | 使用 Logistic Regression 或 Ridge 作為 Meta-learner 學習動態權重，是否優於簡單的 ECDF 等權平均？ | `outputs/tables/e3c_stacking_summary.csv` |
| **E3d** | 難抓破產樣本診斷 | 在訓練集 176 家破產公司中，是否存在所有模型在 15 次外層評估中皆難以辨識的「頑固破產樣本」？其財務特徵為何？ | `outputs/tables/e3d_hard_positives.csv` |

---

### 4. 決策規則 `apply_e3_selection_rule()`（`bankruptcy/src/ensemble.py:1122-1230`）
專案依據預先註冊的嚴謹三階段決策樹挑選主力模型：
- **規則 1（顯著勝出）**：若領先者（Leader，15 次外層評估平均三點損失最低者）在 Nadeau-Bengio 配對檢定中顯著優於次名（Runner-up，95% CI 上限 $< 0$ 且 $p < 0.05$），直接選取領先者。
- **規則 2（不顯著時選穩定度）**：若領先者與次名的差異無統計顯著性（95% CI 包含 0），比較兩者外層評估 AR@0.95 的標準差（`ar95_fold_std`），**選取標準差較小（預警成本波動較穩定）者**。
- **規則 3（全勝單一模型）**：若領先者顯著優於全體 5 個單一模型，直接確立為主力模型。

---

### 5. 專案實際數據與 CSV 欄位依據（絕無編造）

所有結論均自 `outputs/tables/` 固化表格核對：

### 數字來源與評估協定對照表

| 評估對象與數值 | 來源檔案與欄位 | 評估協定（切分與參數來源） | 是否可與其他模型直接比大小？ | 角色與決策脈絡 |
|---|---|---|:---:|---|
| **M3 AR@0.95 折標準差**<br>**1.347676** (~1.35x) | `outputs/tables/e2a_nested_summary.csv`<br>欄位 `ar95_fold_std` (第 3 列) | 5 折 × 3 次重複（15 次外層評估）；<br>超參數來自 E2a 各折獨立尋優 | 僅能與單一模型比波動度 | 單一模型中折間波動最小；但平均三點損失為 1.6547，排在第 6 名 |
| **$E_A$ AR@0.95 折標準差**<br>**2.117742** (~2.12x) | `outputs/tables/e3_selection.csv`<br>欄位 `leader_ar95_std` | 同上（15 次外層評估）；M3/M4/M5 各折以 ECDF 百分位融合；參數來自 E2a 各折 | **可與 $E_{\text{trees}}$ 比較** | **Leader**（平均三點損失 1.6344 全專案最低，排第 1 名） |
| **$E_{\text{trees}}$ AR@0.95 折標準差**<br>**2.303133** (~2.30x) | `outputs/tables/e3_selection.csv`<br>欄位 `runner_up_ar95_std` | 同上（15 次外層評估）；M2/M3/M4/M5 各折以 ECDF 百分位融合；參數來自 E2a 各折 | **可與 $E_A$ 比較** | **Runner-up**（平均三點損失 1.6496，排第 2 名） |

> **重要決策釐清：為什麼 M3 標準差（1.35x）看起來較低，最終卻選 $E_A$（2.12x）？**
> 1. **初選門檻是「三點對數損失」**：在 `apply_e3_selection_rule`（`bankruptcy/src/ensemble.py:1122`）中，所有候選模型首先按平均三點損失排序。$E_A$ 以 **1.6344** 奪得第 1 名（Leader），$E_{\text{trees}}$ 以 **1.6496** 奪得第 2 名（Runner-up），而 M3 損失為 **1.6547** 僅排在第 6 名（見 `outputs/tables/e3d_metric_alignment.csv`）。
> 2. **Rule 2 比較的對象是「Leader vs. Runner-up」**：當 $E_A$ 與 $E_{\text{trees}}$ 的損失差異未達統計顯著時，Rule 2 比較的是 **$E_A$ 的 2.12x** 與 **$E_{\text{trees}}$ 的 2.30x**，判定 $E_A$ 的折間穩定度優於次名。**該規則根本未將 M3 納入標準差比對**，因為 M3 在第一關損失排序時已非前兩名。
> 3. **模型多樣性優勢**：$E_A$ 融合三大異質樹模型（M3 直方圖、M4 深度 2 極淺樹、M5 葉節點導向樹），相比單一演算法 M3 具備更強的結構多樣性與抗過擬合能力。

### E3 使用之超參數來源查核（程式碼事實）
- **查核位置**：`bankruptcy/src/ensemble.py:406-415`
  ```python
  cache_file = E2_CACHE_DIR / f"{m}_r{rep}_f{f}.joblib"
  d = joblib.load(cache_file)
  val_scores_raw[m] = d["outer_val_scores"]
  best_imbalance = d["best_imbalance"]
  model_params = d.get("model_params", d.get("best_params", {}))
  ```
- **結論**：E3 外層評估使用的 M3、M4、M5 超參數，**完全來自 E2a 各外層折自己調出的最佳參數**，且外層驗證折分數是直接讀取該折在 E2a 中完成的無偏預測；內層 ECDF 參考分佈亦僅在該折的訓練折內部以 5 折 CV 生成。因此，**E3 的評估嚴格遵循巢狀交叉驗證防禦，不存在使用全訓練集 E2b 參數所導致的選擇偏誤**。

---

### 各項具體結論與檔案欄位比對

1. **單一模型成對比較（E3a）**：
   - 來源：`outputs/tables/e3a_pairwise_models.csv` 之 `p_value_holm` 欄位。
   - 實測結論：所有 10 組單一模型兩兩比對之 Holm 校正後 p 值**全部為 1.0000**（無顯著差異）。
   - 來源：`outputs/tables/e3a_friedman_ranks.csv` 之 `mean_rank` 欄位。
   - 實測結論：M3 平均排名為 **2.5333**（單一模型第 1 名，Friedman $p = 0.5512$）。
2. **M3 折間標準差最小 1.35x 的來源**：
   - 來源：`outputs/tables/e2a_nested_summary.csv` 之 `ar95_fold_std` 欄位，M3 列數值為 **`1.347676`**（約 1.35x）。
   - 對比其他模型：M1 為 3.72x、M2 為 3.93x、M4 為 2.81x、M5 為 2.86x；M3 是所有單一模型中折間預警成本最穩定的模型。
3. **集成模型評估與決策輸出（`outputs/tables/e3_selection.csv` 原文）**：
   - **領先者（Leader）**：`E_A`，15 次外層平均三點損失為 **1.634437**（`leader_loss_mean`），AR@0.95 折標準差為 **2.117742**（`leader_ar95_std`）。
   - **次名（Runner-up）**：`E_trees`，平均三點損失為 **1.649579**（`runner_up_loss_mean`），AR@0.95 折標準差為 **2.303133**（`runner_up_ar95_std`）。
   - **成對統計檢定**：差值均值為 `-0.015142`，Nadeau-Bengio 95% CI 為 `[-0.043678, 0.013393]`（包含 0），$p_{\text{NB}} = 0.274163$。
   - **觸發決策路徑**：觸發 **Rule 2**——兩者損失無統計顯著差異，因 $E_A$ 的 AR@0.95 折標準差更小（$2.12 \le 2.30$），由 **$E_A$ 獲選為專案主要模型**。
4. **為什麼仍選 $E_A$ 而不單用 M3？**
   - 雖然 $E_A$ 與 M3 的差異未達統計顯著（`diff=-0.0203`, CI `[-0.0965, 0.0559]`, $p=0.5771$），但 $E_A$ 取得了全專案最低的平均三點損失（1.6344 vs M3 1.6547）。
   - 更重要的是：$E_A$ 同時融合了直方圖（M3）、深度為 2 的極淺樹（M4）與葉節點導向樹（M5），在面對未見市場變動時，比單一演算法更具抗過擬合與結構多樣性（Ensemble Diversity）防禦力。
5. **難抓破產樣本診斷（E3d）**：
   - 來源：`outputs/tables/e3d_hard_positives.csv`。
   - 實測結論：在訓練集 176 家破產公司中，**僅有 1 家公司（原始資料列索引 2246）在 15 次評估中全數被所有模型漏報**；其財務特徵顯示資產負債率處於正常安全區，屬於隱蔽型突發破產。

---

## 統計檢定假設與一般經驗說明

> [!NOTE]
> **統計檢定樣本數與假設說明（一般經驗，未經本專案特別驗證）**
> - **樣本數限制**：本專案的外層評估樣本數為 $K = 15$（來自 5 折 × 3 次重複）。在統計學中，$N=15$ 的檢定力（Statistical Power）相對有限，難以檢出微小效應（Small Effect Size）。
> - **Nadeau-Bengio 修正假設**：Nadeau-Bengio 校正最初是針對「隨機重複子抽樣（Repeated Subsampling）」所推導。本專案採用的「5 折 × 3 次重複」屬於「重複 K 折（Repeated K-Fold）」，兩者的資料重疊機制不完全相同。因此，專案將此校正視為**保守的風險控制參考依據**，而不作為絕對的真理裁判。

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **集成融合方式** | A. 原始預測機率直接平均<br>B. Stacking 後設學習器<br>C. ECDF 百分位數等權平均 | **方案 C（ECDF 百分位數等權平均）** | 各模型原始機率量綱不一致；Stacking（E3c）在極小樣本下存在後設過擬合風險；ECDF 嚴格保序、消除量綱、無可調參數，泛化最穩定。 | `bankruptcy/src/ensemble.py:80-99`；`outputs/tables/e3c_stacking_summary.csv` |
| **主要模型定案** | A. 單一冠軍 M3<br>B. 全模型集成 $E_{\text{all}}$<br>C. 三大梯度提升集成 $E_A$ | **方案 C（$E_A$: M3+M4+M5）** | $E_A$ 達成最低三點損失（1.6344），且 AR@0.95 波動度優於 $E_{\text{trees}}$（2.12 vs 2.30），同時整合不同 Boosting 機制。 | `bankruptcy/src/ensemble.py:1122`；`outputs/tables/e3_selection.csv` |

---

## 機器學習概念白話框

> **白話概念：經驗累積分布函數（ECDF）與百分位數映射**
> - **為什麼要換算成百分位？** 假設甲老師改卷極嚴，全班最高分只有 60 分；乙老師改卷極鬆，全班最低分是 80 分。如果直接把兩科分數相加，乙老師的分數會完全主導總成績。
> - **ECDF 做了什麼？** 它不看絕對分數，只看「你在這科贏過全班多少人」。如果甲老師給你 60 分（全班第 1 名，贏過 100% 人），你的 ECDF 分數就是 1.0；乙老師給你 85 分（排在第 50%），ECDF 就是 0.5。兩科換算成百分位後再平均，才是公正的等權綜合評估。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「ECDF 百分位數融合」與「模型多樣性」。

---

## 本模組的單元測試
對應單元測試涵蓋：
1. `bankruptcy/tests/test_ensemble.py`：驗證 `ecdf_map` 之單調性、同分中點對稱性與值域嚴格落於 `[0, 1]`。
2. `bankruptcy/tests/test_results_summary.py`：驗證 E3 產出的各項統計比較 CSV 與選擇規則欄位完整性。

---

## 讀碼練習
1. **練習 1**：在 `ecdf_map` 中，如果查詢分數大於參考陣列中的所有數值，回傳的百分位數是多少？它是如何透過 `np.searchsorted(..., side="left")` 與 `side="right"` 計算出來的？
2. **練習 2**：在 `apply_e3_selection_rule` 中，為什麼 Rule 2 要在兩者差異不顯著時，選擇 `ar95_fold_std` 較小的模型？這在金融風控業務上有何意義？
3. **練習 3**：在 E3d 中發現的「第 2246 號破產公司」在 15 次外層評估中皆被漏報。對於這種隱蔽型破產公司，單純增加模型複雜度是否能夠解決問題？（提示：思考特徵空間覆蓋度與標籤雜訊）

---

## 延伸：想改這裡時要注意什麼
1. **不得在生產環境重新擬合 ECDF 參考分佈**：正式部署應直接讀取 E4c 固化之 `reference_M*.npy`，不可在推論時動態重新計算參考分佈。
2. **保護 OOF 快取目錄**：`outputs/models/oof/` 內檔案為固化快取，手動刪除將導致 E3 腳本必須從頭耗費數小時重新運算。
