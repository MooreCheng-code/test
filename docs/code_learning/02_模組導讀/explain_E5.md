# 模組導讀：特徵可解釋性與精簡特徵集（explain.py 與 local_explain.py，實驗 E5）

## 一句話用途
綜合置換重要性、家族群組置換重要性與 TreeSHAP 跨模型剖析破產預測的核心財務驅動因子，並透過精簡特徵集消融評估（E5b）篩選出 12 個核心財務特徵與 18 個擴大核心特徵。

## 它在整體流程的位置（連回 01 地圖的節點）
- **地圖節點**：位於 [01_專案地圖與資料流.md](../01_專案地圖與資料流.md) 的「模型可解釋性與特徵精簡層（E5 實驗）」。
- **上游**：讀取 E2a 各折外層重擬合模型與特徵管線。
- **下游**：
  - E5b 精簡集決策輸出 `outputs/tables/e5b_core_features.csv` 與 `outputs/models/final/core_features.txt`。
  - `local_explain.py` 提供單筆企業推論時的 SHAP 瀑布圖與前三大破產財務主因歸因，直接驅動 Web 儀表板（`app/`）。

## 輸入 / 輸出（型別、形狀、範例）
- **輸入**：
  - 各折訓練好的 Pipeline 模型。
  - 全訓練集清洗後特徵矩陣 `(5455, 72)` 與標籤 `(5455,)`。
  - 單筆企業財報字典（95 個財務特徵數值）。
- **輸出**：
  - 全域重要性表格：`outputs/tables/e5a_shap_importance.csv`、`outputs/tables/e5a_group_importance.csv`、`outputs/tables/e5a_permutation_importance.csv`。
  - 精簡特徵集消融表：`outputs/tables/e5b_compact_summary.csv`、`outputs/tables/e5b_compact_vs_full_paired.csv`。
  - 核心特徵清單：`outputs/models/final/core_features.txt`（12 欄）、`core_features_extended.txt`（18 欄）。
  - 單筆歸因字典：包含 `base_value`、72 個特徵的邊際貢獻（attribution）、正負向前三大財務推手。

---

## 核心演算法與設計

### 1. 單特徵置換 vs. 家族（群組）置換重要性
- **做什麼**：衡量特徵被隨機打散（Permutation）破壞資訊後，模型 PR-AUC 或預警倍數的下降幅度。
- **為什麼高度共線性特徵必須「按家族分組打散」？**
  - **單特徵置換的盲點缺陷**：若特徵 $X_1$（例如「負債比率」）與特徵 $X_2$（「負債對淨值比」）高度相關（$|r| > 0.90$）。當只隨機打散 $X_1$ 時，模型依然能透過未被打散的 $X_2$ 獲得幾乎相同的資訊，導致計算出的 $X_1$ 重要性趨近於 0！這會造成「會計實務上最重要的負債指標，在單特徵重要性上看起來毫無用處」的荒謬結論。
  - **家族置換（Group Permutation）解法**：依據會計維度將強相關特徵綁定為同一個「家族（Family）」。在打散時，**將該家族內的所有特徵同步一起打散**，徹底阻斷代理特徵（Proxy Feature）的資訊洩漏，真實揭露該財務維度的核心影響力。
- **最小數字例子**：
  - 假設公司破產完全取決於負債規模，有兩個高相關指標 $X_1, X_2$。
  - 模型預測：$Y \approx 0.5 X_1 + 0.5 X_2$。
  - 單獨打散 $X_1$：模型 PR-AUC 僅微跌 0.01（$X_2$ 撐住大局）。
  - 單獨打散 $X_2$：模型 PR-AUC 亦僅微跌 0.01。
  - **群組同步打散 $(X_1, X_2)$**：模型 PR-AUC 重挫 0.35！證實負債家族具有決定性權重。
- **本專案實測結論**：
  在 `outputs/tables/e5a_group_importance.csv` 中，**Family 38（負債比率家族）**在同步打散時導致全模型預警能力產生最大跌幅，100% 進入前 20 大重要維度。

---

### 2. TreeSHAP 與線性貢獻（`bankruptcy/src/local_explain.py:180-240`）
- **做什麼**：以合作賽局理論（Shapley Values）量化各特徵對「單一企業預測分數相對於全體平均背景值」的加性邊際貢獻。
- **模型支援矩陣**：
  1. `M4` (XGBoost)、`M3` (HistGradientBoosting)、`M5` (LightGBM)：使用 `shap.TreeExplainer` 演算法，在多項式時間內精確計算樹節點條件期望值。
  2. `M1` (Logistic Regression)：使用線性對數勝算分解 $\text{attribution}_i = w_i \cdot x_i$。
- **加性公式（Additivity Property）**：
  $$\text{Pred\_Margin}(x) = \text{Base\_Value} + \sum_{i=1}^{72} \phi_i(x)$$
  在樹模型中，此加性等式嚴格成立於**對數勝算空間（Log-odds / Margin Space）**。
- **單位重要提醒（一般經驗，未經本專案重跑驗證）**：
  > [!WARNING]
  > 樹模型 TreeSHAP 輸出的數值單位是**對數勝算（Log-odds / Margin），絕非「機率百分點」**！
  > 例如某特徵的 SHAP 值為 $+0.85$，代表它將模型的 Margin 往破產方向推升了 0.85 單位（勝算比增加了 $e^{0.85} \approx 2.34$ 倍），而不是「讓破產機率增加了 85%」。在向業務或評審報告時，切勿混淆為機率百分點。

---

### 3. E5 實驗子項目分工與結論（附 CSV 依據）

#### E5a：跨模型重要性比對與會計維度收斂
- **回答問題**：不同演算法（直方圖樹 M3、極淺樹 M4、葉節點樹 M5）所看到的關鍵財務維度是否具備一致性？
- **實測證據**：`outputs/tables/e5a_shap_importance.csv`。
- **結論**：M3、M4、M5 在前 20 名 SHAP 特徵中的**交集高達 13 個特徵**。不論演算法如何變換，現金流穩定度（如 `Net Income to Total Assets`）、負債槓桿（如 `Debt ratio %`）與營運週轉能力始終穩居全域影響力前列。

#### E5b：精簡特徵集消融評估（Feature Selection Ablation）
- **回答問題**：在金融徵信實務上，若無法蒐集齊全 95 個財務欄位，只使用 Top 10、Top 20 或 Top 30 核心指標，模型表現會打折多少？
- **實測數據表格（來自 `outputs/tables/e5b_compact_summary.csv` 與 `e5b_compact_vs_full_paired.csv`）**：

| 特徵子集規模 | 15 次外層評估平均三點損失 (5 折 × 3 次重複) | AR@0.95 均值 | PR-AUC 均值 | 相對全特徵之配對 t 檢定 Holm p-value | 統計結論與業務建議 |
|:---:|:---:|:---:|:---:|:---:|---|
| **Full (72 欄)** | **1.6344** | **8.06x** | **0.4598** | （基準 Baseline） | 完整模型表現選定 |
| **Top 30** | 1.6391 | 8.12x | 0.4552 | $p_{\text{Holm}} = 0.4485$（未達顯著） | 表現與全特徵無統計顯著差異 |
| **Top 20** | 1.6425 | 8.21x | 0.4510 | $p_{\text{Holm}} = 0.1423$（未達顯著） | **安全精簡集**（未達顯著衰退） |
| **Top 10** | 1.7102 | 9.45x | 0.4182 | **$p_{\text{Holm}} = 0.0251$（顯著劣化）** | 資訊量不足，預警倍數顯著惡化 |

- **結論依據**：
  依據 `e5b_compact_vs_full_paired.csv`，當特徵壓減至 10 欄時，$p_{\text{Holm}} = 0.0251 < 0.05$，證實過度精簡會破壞風險防線；而 Top 20 則能維持無顯著差異的強大預警力。專案進一步篩選出具備獨立會計含義的 **12 個核心特徵（`core_features.txt`）** 與 **18 個擴大核心特徵（`core_features_extended.txt`）**。

---

## 設計決策卡

| 決策項目 | 候選方案 | 專案選擇 | 理由 | 證據 |
|---|---|---|---|---|
| **財務特徵歸因法** | A. 單純看決策樹內建 Importance<br>B. 單特徵置換<br>C. 群組置換 + TreeSHAP | **方案 C（群組置換 + TreeSHAP）** | 內建 Importance 偏好高基數連續特徵；單特徵置換被共線性矇蔽；群組置換暴露真實家族影響力，TreeSHAP 具備賽局理論嚴格加性確保。 | `bankruptcy/src/explain.py:250-380`；`outputs/tables/e5a_group_importance.csv` |
| **精簡特徵集規格** | A. 極限壓至 Top 10<br>B. 鎖定 Top 20（核心 12/18 欄）<br>C. 維持 72 全特徵 | **方案 B（Top 20 / 核心 12 欄）** | Top 10 顯著劣化（$p=0.0251$）；Top 20 與全特徵無顯著差異（$p=0.1423$），同時大幅降低真實企業徵信時的資料填報負擔。 | `outputs/tables/e5b_compact_vs_full_paired.csv`；`outputs/models/final/core_features.txt` |

---

## 機器學習概念白話框

> **白話概念：SHAP 值與邊際貢獻（Shapley Values）**
> - **什麼是 Shapley 值？** 想像 3 位球員組隊打籃球贏得比賽。勝利不是平均分配，而是看「如果沒有張三，球隊少得幾分；如果加入李四，球隊多得幾分」。SHAP 遍歷所有特徵組合，精確計算出每個特徵在各種排列組合下的邊際得分。
> - **瀑布圖在看什麼？** 背景破產基準值如果是 $-3.5$（對數勝算）。公司 A 負債比率高（貢獻 $+1.2$），流動資金低（貢獻 $+0.8$），營業利益好（貢獻 $-0.4$）。所有特徵加總後變成 $-1.9$，轉換為破產機率。每個柱子長度就是該特徵的推動力。
> - **名詞連動**：參見 [08_術語表](../08_術語表.md) 之「TreeSHAP」與「加性特徵歸因」。

---

## 本模組的單元測試
對應單元測試涵蓋：
1. `bankruptcy/tests/test_explain.py`：驗證 `compute_permutation_importance`、`compute_group_permutation_importance` 與 `TreeExplainer` 在特徵置換下數值不變形。
2. `bankruptcy/tests/test_local_explain.py`：驗證 `LocalExplainer` 在單筆推論時加性等式（$\text{Base} + \sum \phi_i = \text{Margin}$）誤差小於 $10^{-5}$，且缺欄率超過 20% 時拋出 `ValueError`。

---

## 讀碼練習
1. **練習 1**：在 `local_explain.py` 中，如果單筆推論資料缺失了 25 個特徵（缺失率 $25/95 \approx 26.3\%$），程式碼會如何處理？為什麼要設定 20% 的硬性上限？
2. **練習 2**：在 `explain.py` 中，為什麼置換特徵時要在管線的 `QuantileScaler` 與 `CorrelationFilter` 之後才進行置換，而不是在原始 DataFrame 上打散欄位？
3. **練習 3**：根據 E5b 實驗，若某金融科技新創公司為了簡化表單，堅持只讓貸款申請人填寫 5 個財務欄位，依據本專案的研究結論，其破產預警模型會面臨什麼風險？

---

## 延伸：想改這裡時要注意什麼
1. **TreeSHAP 記憶體開銷**：對 5,455 筆樣本計算全量 TreeSHAP 極耗記憶體，離線實驗採分批計算；線上推論僅針對單筆（$N=1$）計算，確保毫秒級響應。
2. **特徵名稱對齊鐵律**：`local_explain.py` 嚴格依賴 `feature_manifest.json` 所固化的原始 95 欄順序，任何修改前處理管線的行為必須同步更新 manifest。
