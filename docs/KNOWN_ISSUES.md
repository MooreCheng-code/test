# 已知議題與靜默預設值審計清單 (Known Issues & Silent Defaults)

本清單記錄程式碼庫中經由全面靜默預設值掃描（`.get(..., default)`、`getattr(..., default)`、裸 `except:`、`fillna`、`errors="ignore"`）確認但暫不重構的次要項目。
高風險項目（`scoring.py` 校準器與門檻拋出例外、`final_eval.py` metadata 缺漏與 `splits.joblib` 讀取失敗）已於 tag `pre-app` 嚴格重構並補齊單元測試。其餘項目評估為低風險或標準慣用寫法，標記為「低優先，未處理」。

---

## 審計清單

| 檔案路徑 | 行號 | 程式碼片段 | 原本用途 | 風險等級 | 狀態 | 備註說明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/data_audit.py` | 61 | `n_bankrupt = target_counts.get(1, 0)` | 統計破產正樣本數 | 低 | 低優先，未處理 | 資料集必含 0 與 1，若真為空回傳 0 語義正確。 |
| `src/data_audit.py` | 62 | `n_non_bankrupt = target_counts.get(0, 0)` | 統計未破產負樣本數 | 低 | 低優先，未處理 | 同上，僅用於 EDA 資料審計摘要顯示。 |
| `src/ensemble.py` | 414 | `d.get("model_params", d.get("best_params", {}))` | 向下相容參數字典命名 | 低 | 低優先，未處理 | 僅用於離線訓練實驗載入舊版實驗結果。 |
| `src/explain.py` | 647 | `all_features_frequency.get(feat, 0) + 1` | 特徵頻率累加計數 | 極低 | 低優先，未處理 | 標準 Python 計數字典慣用法（等價 `Counter`）。 |
| `src/pipelines.py` | 75 | `getattr(self.estimator, "scale_pos_weight", 1.0)` | 讀取模型正樣本加權 | 低 | 低優先，未處理 | 僅用於支援該參數之模型，未支援者預設 1.0 屬正常設定。 |
| `src/pipelines.py` | 99 | `getattr(self, 'estimator_', self.estimator)` | 錯誤訊息格式化 | 極低 | 低優先，未處理 | 僅用於拋出未擬合例外時顯示底層類別名稱。 |
| `src/pipelines.py` | 107 | `getattr(self, 'estimator_', self.estimator)` | 錯誤訊息格式化 | 極低 | 低優先，未處理 | 同上，用於顯示無係數之底層模型名稱。 |
| `src/run_baselines.py` | 87 | `except Exception:` | 特徵重要性萃取防護 | 低 | 低優先，未處理 | Dummy 或特定無重要性屬性之模型，略過不影響評估指標計算。 |
| `src/run_imbalance.py` | 95 | `except Exception:` | 特徵重要性萃取防護 | 低 | 低優先，未處理 | 同上，E1 不平衡實驗中安全萃取特徵重要性。 |
| `src/tune.py` | 330 | `penalty = model_params.get("penalty", "l2")` | 懲罰項預設值 | 低 | 低優先，未處理 | 僅用於 Optuna 搜尋空間配置解析。 |
| `src/tune.py` | 335 | `subsample = model_params.get("subsample", 1.0)` | 子樣本比例預設值 | 低 | 低優先，未處理 | 僅用於 Optuna 搜尋空間配置解析。 |
| `src/tune.py` | 373 | `best_trial.user_attrs.get("fold_loss_std", 0.0)` | 萃取內層折標準差 | 低 | 低優先，未處理 | 僅用於 Optuna 調優日誌輸出。 |
| `src/tune.py` | 545 | `matched_roc = r.get("matched_roc", np.nan)` | 報表欄位防護填補 | 低 | 低優先，未處理 | 允許未執行該基準之模型在彙整表格填入 NaN。 |
| `src/tune.py` | 546 | `auc_jump = r.get("auc_jump", np.nan)` | 報表欄位防護填補 | 低 | 低優先，未處理 | 同上。 |
| `src/tune.py` | 547 | `single_feat_auc = r.get("single_feat_auc", np.nan)` | 報表欄位防護填補 | 低 | 低優先，未處理 | 同上。 |
| `src/tune.py` | 610 | `except Exception:` | Optuna study 物件清理 | 低 | 低優先，未處理 | 清理先前 study 物件的保護區塊，失敗不影響實驗進行。 |
| `src/tune.py` | 1081 | `color = colors.get(m, "black")` | 圖表繪圖顏色映射 | 極低 | 低優先，未處理 | 視覺化調色防護，若模型未定義顏色則預設黑色。 |
| `outputs/models/final/` | N/A | `_tmp_oof_*.npy` | 凍結模型外層折 OOF 預測暫存 | 低 | 已知命名遺留，維持現狀 | 檔名含有 `_tmp_` 但被回歸測試依賴，屬於歷史遺留命名；暫不更名以保持模型目錄檔案雜湊與 preflight 檢查穩定。未來重訓時應更名為標準命名。 |

---

## 處理準則與後續建議

1. **推論路徑零靜默原則**：所有進入生產或推論之路徑（`src/scoring.py`、`score_new_data.py`）嚴格禁止靜默填補，必須拋出明確例外（如 `KeyError`、`FileNotFoundError`、`ValueError`）。
2. **研究調優路徑容錯原則**：離線調優與視覺化工具（`tune.py`、`explain.py`、`run_baselines.py`）允許保留無害的防禦性預設值，若未來重構為正式模組時再行收緊。
3. **凍結目錄檔案命名遺留**：`outputs/models/final/_tmp_oof_*.npy` 檔名含有 `_tmp_`，係初期實驗輸出命名遺留。由於既有回歸測試直接引用該檔名，且模型目錄已於 preflight 凍結 SHA256 雜湊，為確保環境可重現性與雜湊一致性，目前不變更檔名。未來重訓或模型版號升級時應統一重構為正式命名（如 `oof_preds_*.npy`）。
