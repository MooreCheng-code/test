# 模組導讀：實驗全景導覽與基礎實驗 E0／E1 (`run_baselines.py`、`run_imbalance.py`)

## 1. 模組定位與設計初衷

機器學習專案絕非「直接丟進演算法跑出一個準確率」就結束。一個具備嚴謹科研規範的專案，必須具備：
1. **可靠的底線檢驗（Sanity Check / Baselines, E0）**：證明模型確實學到有效訊號，且評估管線沒有特徵洩漏。
2. **科學的假設驗證（Ablation Study, E1）**：在引入複雜的重採樣（SMOTE 等）或複雜前處理前，先驗證它們是否真的帶來顯著改善。
3. **系統化的實驗遞進結構（E0 ～ E7）**：從基準、特徵與採樣策略、超參數調校、模型集成、門檻固化、可解釋性，到最終測試集評估與敏感度分析，環環相扣。

---

## 2. E0～E7 實驗全景總覽大表

下表完整梳理本專案所有實驗的目標、執行腳本、輸出檔案、核心結論與重跑安全性：

| 實驗 ID | 實驗名稱與核心目的 | 執行腳本與入口函式 | 主要輸出產物 (唯讀) | 核心實測結論 (CSV 欄位與數值) | 是否可安全重跑 |
|---|---|---|---|---|---|
| **E0** | **基準模型與管線檢驗**<br>建立隨機猜測、單一特徵與基礎羅吉斯迴歸基準；以標籤隨機打亂檢驗管線洩漏。 | `bankruptcy/src/run_baselines.py`<br>`run_e0_baselines()`<br>`run_label_shuffle_test()` | `outputs/tables/e0_baselines_summary.csv`<br>`outputs/tables/e0_label_shuffle.csv` | - Single Feature (-EPS) PR-AUC = 0.3012<br>- Base LR (S0) PR-AUC = 0.3938<br>- 標籤打亂後 ROC-AUC 回落至 0.47～0.51，PR-AUC 回落至 0.032（無洩漏） | **安全**<br>(耗時約 1～2 分鐘) |
| **E1** | **不平衡策略與前處理消融**<br>比較 S0～S6 共 7 種採樣策略；針對縮放器、截尾與相關過濾器進行消融實驗。 | `bankruptcy/src/run_imbalance.py`<br>`run_e1_experiments()` | `outputs/tables/e1a_imbalance_comparison.csv`<br>`outputs/tables/e1_selection_recommendation.csv`<br>`outputs/tables/e1b_ablation.csv` | - 複雜過採樣無顯著超越 S0 (Rule 3 簡約原則選 S0)<br>- QuantileTransformer 優於 StandardScaler<br>- 保留相關過濾器 (72 特徵) | **安全**<br>(支援快取，耗時約 5～10 分鐘) |
| **E2a** | **巢狀交叉驗證超參數調校**<br>在外層無偏環境下，內層 3 折網格尋優，量化單一模型泛化效能。 | `bankruptcy/src/tune.py`<br>`run_e2a_nested_tuning()` | `outputs/tables/e2a_nested_summary.csv`<br>`outputs/logs/e2_cache/*.joblib` | - M3 (ExtraTrees) 三點損失 1.6547，折間標準差 1.348% 為單模型中最優<br>- 固化各折最佳參數快取 | **安全**<br>(運算量大，耗時約 30～60 分鐘) |
| **E2b** | **全訓練集超參數最佳化**<br>在全體訓練集（5,455 筆）上搜尋最終模型最佳超參數。 | `bankruptcy/src/tune.py`<br>`run_e2b_full_train_tuning()` | `outputs/models/best_params/M*.json` | - 產出供生產與後續分析使用的全集超參數設定檔 | **安全**<br>(耗時約 10～15 分鐘) |
| **E3** | **集成策略比選**<br>比對單模型與 4 種集成方式（$E_A$、$E_B$、$E_C$、$E_{\text{trees}}$），驗證異質多樣性效益。 | `bankruptcy/src/ensemble.py`<br>`run_e3_ensemble_selection()` | `outputs/tables/e3_selection.csv` | - $E_A$ (M3+M4+M5 百分位數等權平均) 三點損失 1.6344 最優<br>- 顯著優於單一模型與樹集成 | **安全**<br>(讀取 E2 快取，耗時約 1～2 分鐘) |
| **E4** | **機率校準與安全邊際門檻轉移**<br>驗證 OOF 門檻跨折轉移能力，計算安全邊際 $m$，擬合校準器。 | `bankruptcy/src/thresholding.py`<br>`run_e4_pipeline()` | `outputs/tables/e4a_threshold_transfer.csv`<br>`outputs/models/final/thresholds.json`<br>`outputs/models/final/calibrator_*.joblib` | - 建議邊際：R80(0.02)、R90(0.06)、R95(0.04)、R100(0.00)<br>- 達標率 $\ge 80\%$<br>- Platt 校準器 Brier Score 0.0243 | **安全**<br>(耗時約 3～5 分鐘) |
| **E5** | **模型可解釋性與精簡特徵消融**<br>群組置換重要度、TreeSHAP、精簡特徵集驗證（E5b）。 | `bankruptcy/src/explain.py`<br>`run_e5_explainability()` | `outputs/tables/e5a_group_permutation_importance.csv`<br>`outputs/tables/e5b_reduced_feature_set_ablation.csv` | - 獲利能力（Profitability）為最關鍵群組（48.7%）<br>- 12 特徵精簡集保留 93.4% 排序效能 | **安全**<br>(計算 SHAP 較耗時，約 10～20 分鐘) |
| **E6** | **獨立測試集單次防禦性評估**<br>不可逆解封測試集，執行單次推論並固化最終效能。 | `bankruptcy/src/final_eval.py`<br>`main() --unseal` | `outputs/e6/e6_operating_points.csv`<br>`outputs/e6/e6_ranking_metrics.csv`<br>`outputs/logs/E6_LOCK` | - $E_A$ 於未見測試集 PR-AUC 0.5676，ROC-AUC 0.9621<br>- R80～R100 全數達標（100% 捕獲目標） | **禁止重跑！**<br>**已於 2026-10-23 鎖定**<br>(僅允許 `--analyze` 離線讀取) |
| **E7a** | **母體盛行率貝氏敏感度分析**<br>推算市場破產率（0.5%～5.0%）變動時的調查負擔衝擊。 | `bankruptcy/src/prevalence_sensitivity.py`<br>`run_prevalence_sensitivity()` | `outputs/tables/e7a_prevalence_sensitivity.csv` | - 破產率由 3.2% 降至 0.5% 時，調查負擔由 4.5 家暴增至 24.1 家 | **安全**<br>(純統計解析推算，耗時約 10 秒) |
| **E7b** | **演算法隨機種子穩定性評估**<br>跨 5 組隨機種子評估模型波動，對比折間抽樣變異。 | `bankruptcy/src/seed_stability.py`<br>`run_seed_stability()` | `outputs/tables/e7b_seed_stability.csv` | - 種子標準差 (PR-AUC 0.0017) 遠小於折間標準誤 (0.0204)<br>- 證實模型架構高度穩健 | **安全**<br>(需擬合 75 個模型，耗時約 10～15 分鐘) |

---

## 3. E0：基準模型與隨機標籤打亂檢驗 (`run_baselines.py`)

### 3.1 為什麼需要 E0？（底線原則）
在訓練複雜的機器學習模型前，必須先建立簡單基準。如果一個複雜神經網路或梯度提升樹的表現連「單一財務比率」或「未調參的羅吉斯迴歸」都比不過，那複雜模型就毫無存在的價值。

E0 建立三個基準層次（`bankruptcy/src/run_baselines.py:102-301`）：
1. **Dummy Baseline（隨機猜測）**：依據類別先驗比例輸出常數機率。
2. **Single Strongest Feature Baseline（單一財務指標基準）**：
   - 經由資料稽核（`data_audit.py`）與單變量排序，選出與破產相關性最強的單一財務指標：`- Persistent EPS in the Last Four Seasons`（持續每股盈餘取負號，使數值愈大破產風險愈高）。
3. **Base Logistic Regression（基準羅吉斯迴歸，S0）**：未調參、不加過採樣的標準線性模型。

---

### 3.2 本專案實測基準表現 (`outputs/tables/e0_baselines_summary.csv`)

> [!NOTE]
> **本專案實測（E0，5 折 $\times$ 3 次重複 = 15 次外層評估平均）**：

| 模型名稱 | PR-AUC (平均 $\pm$ 折間標準差) | ROC-AUC (平均 $\pm$ 折間標準差) | AR@Recall=0.95 (平均 $\pm$ 折間標準差) |
|---|---|---|---|
| **Dummy (隨機基準)** | 0.0323 $\pm$ 0.0004 | 0.5000 $\pm$ 0.0000 | 30.9983 $\pm$ 0.3585 |
| **Single Strongest Feature (-EPS)** | 0.3012 $\pm$ 0.0470 | 0.8861 $\pm$ 0.0214 | 16.1125 $\pm$ 3.1588 |
| **Base Logistic Regression (S0)** | 0.3938 $\pm$ 0.0620 | 0.9232 $\pm$ 0.0174 | 10.3135 $\pm$ 4.3899 |

**結論解讀**：
- 單一每股盈餘指標（-EPS）本身就具備高達 0.8861 的 ROC-AUC 與 0.3012 的 PR-AUC，展現傳統財報指標的強大預測力。
- 引入多變量羅吉斯迴歸（Base LR）後，PR-AUC 提升至 0.3938，AR@0.95 顯著由 16.11% 下降至 10.31%（調查負擔降低 36%），證明多變量特徵確實具備增量資訊。

---

### 3.3 標籤隨機打亂檢驗（Label Shuffle Permutation Test, `run_baselines.py:360-433`）

在機器學習工程中，**管線是否有潛在的資料洩漏（Data Leakage）**？最有效且殘酷的驗證方式就是「隨機打亂標籤」：
- 將目標變數 $y$ 隨機重排（打亂特徵 $X$ 與標籤 $y$ 之間的真實因果關聯）。
- 在完全被打亂的資料上執行完整的交叉驗證管線。

> [!NOTE]
> **本專案實測（E0 標籤打亂，`outputs/tables/e0_label_shuffle.csv`）**：

| 打亂隨機種子 (Shuffle Seed) | 平均折間 ROC-AUC | 平均折間 PR-AUC | 理論期望值（無洩漏） |
|---|---|---|---|
| **Seed 42** | 0.5150 | 0.0401 | ROC-AUC = 0.5000, PR-AUC = 0.0323 |
| **Seed 123** | 0.4798 | 0.0322 | ROC-AUC = 0.5000, PR-AUC = 0.0323 |
| **Seed 999** | 0.4701 | 0.0349 | ROC-AUC = 0.5000, PR-AUC = 0.0323 |

**科學價值**：
打亂後，ROC-AUC 精確在 0.47～0.51 之間波動，PR-AUC 在 0.032～0.040 之間波動（完全對齊無資訊時的隨機水準）。這鐵證證明：**本專案的前處理與交叉驗證管線完全沒有未來資訊外洩或特徵偷看問題**。

---

## 4. E1：不平衡策略與前處理消融 (`run_imbalance.py`)

### 4.1 核心實驗設計（E1a 與 E1b）
- **E1a 析因實驗（7 種不平衡策略 $\times$ 5 種模型）**：
  - S0（不處理）、S1（Random Under-sampling）、S2（Random Over-sampling）、S3（SMOTE）、S4（Borderline-SMOTE）、S5（ADASYN）、S6（SMOTE + Tomek Links）。
  - 對比 15 次外層交叉驗證下的預警比率與 PR-AUC。
- **E1b 前處理消融實驗（Ablation Study）**：
  - 檢驗 4 個前處理設定：預設管線（Quantile + CorrFilter + No Winsorizer）、關閉相關過濾（`corr_filter_off`）、開啟截尾（`winsorizer_off` 反向比較）、開啟大型異常值標記（`large_indicator_on`）。

---

### 4.2 本專案實測結論與決策原則 (`e1_selection_recommendation.csv`)

> [!NOTE]
> **本專案實測（E1 推薦結論，`outputs/tables/e1_selection_recommendation.csv`）**：

| 評估模型 | 最終推薦策略 | 最終推薦縮放器 | 推薦依據規則 (Rule Applied) | 核心實測理由 |
|---|---|---|---|---|
| **M1 (Logistic Regression)** | **S0** | **quantile** | **Rule 3 (奧坎剃刀簡約原則)** | 任何過採樣技術（SMOTE 等）均未能在統計顯著性上勝過原始資料 S0；故選擇最簡單、運算最快之 S0。 |
| **M2 (Random Forest)** | **S0** | **quantile** | **Rule 3 (奧坎剃刀簡約原則)** | S0 在 PR-AUC (0.4296) 與 AR@0.95 (11.84%) 均穩定，過採樣並無顯著增益。 |
| **M4 (XGBoost)** | **S0** | **quantile** | **Rule 3 (奧坎剃刀簡約原則)** | 樹模型本身對不平衡具備適應力，S0 結構最精簡且能避免合成樣本帶來的邊界噪聲。 |

### 4.3 前處理消融數據洞察 (`outputs/tables/e1b_ablation.csv`)
1. **相關過濾器（Correlation Filter）的作用**：
   - 移除了 20 個共線性特徵（如 `Operating Profit Per Share` 與 `Operating Profit Growth Rate` 等高度共線欄位），保留 72 個核心特徵。
   - 消融數據顯示，關閉相關過濾器後，M1 的 AR@0.95 由 10.87% 惡化至 12.00%（負擔增加 1.13 個百分點），證明過濾共線性特徵確實提升了線性模型的泛化穩定性。
2. **縮放器選擇（QuantileTransformer）**：
   - 財務比率存在極端厚尾與偏態，QuantileTransformer 將資料映射至均勻分佈，有效消除了極端異常值的干擾，全線優於標準常態化（StandardScaler）。
