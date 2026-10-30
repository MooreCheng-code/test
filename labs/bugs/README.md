# 沙盒除錯題庫 (Bug-Hunt Sandbox)

本目錄收錄 10 支刻意植入常見機器學習與資料科學實作錯誤的練習腳本，專門供學習者進行除錯偵錯演練。

> [!WARNING]
> **教學警語與安全隔離**
> 1. 本目錄內的所有腳本均為**刻意包含實作錯誤**的反面教材，**嚴禁作為正式專案開發之參考代碼**。
> 2. 所有腳本**僅使用合成資料**（`sklearn.datasets.make_classification` 或隨機陣列），**絕不讀取專案資料**（不讀取 `data.csv`、不呼叫 `get_train_data()`、不引用任何專案資料載入模組）。
> 3. 本目錄被嚴格隔離於正式專案原始碼 `bankruptcy/` 之外，不參與專案單元測試（不被 `pytest bankruptcy/tests` 收集），亦不在 `lab_map.csv` 註冊。
> 4. 題目描述請參閱 [`docs/code_learning/06_練習題.md`](../docs/code_learning/06_練習題.md)，除錯詳解與修正代碼請參閱 [`docs/code_learning/06_解答.md`](../docs/code_learning/06_解答.md)。

---

## 1. 腳本清單與演練主題

| 腳本檔名 | 演練主題 | 表面症狀說明 |
| :--- | :--- | :--- |
| [`bug_01_preprocessing_leakage.py`](bug_01_preprocessing_leakage.py) | 前處理管線特徵轉換範圍 | 驗證集特徵尺度與分位數轉換看似流暢，但存在潛在資訊洩漏 |
| [`bug_02_resampling_leakage.py`](bug_02_resampling_leakage.py) | 不平衡樣本過採樣時機 | 模型交叉驗證 PR-AUC 異常虛高至 0.99 以上 |
| [`bug_03_accuracy_trap.py`](bug_03_accuracy_trap.py) | 極度不平衡資料之評估指標選取 | 測試集準確率高達 98%，看似判定精準，實則存在評估盲區 |
| [`bug_04_val_threshold_selection.py`](bug_04_val_threshold_selection.py) | 操作門檻挑選與泛化評估 | 驗證集門檻選擇與同集評估未分離，欠缺泛化安全邊際 |
| [`bug_05_no_nested_tuning.py`](bug_05_no_nested_tuning.py) | 超參數搜尋與泛化性能報告 | 報出的最佳超參數驗證分數高估了對未見資料的泛化能力 |
| [`bug_06_optimistic_tie_breaking.py`](bug_06_optimistic_tie_breaking.py) | 預測分數同分判定與預警統計 | 門檻判定邏輯導致部分臨界樣本未納入，預警覆蓋與召回計算失真 |
| [`bug_07_missing_random_state.py`](bug_07_missing_random_state.py) | 隨機性控制與實驗對照有效性 | 兩次執行指標不同，被誤判為模型參數調優帶來的效益 |
| [`bug_08_target_leakage_distribution.py`](bug_08_target_leakage_distribution.py) | 決策門檻調校與目標分佈資訊 | 動態調整門檻時誤用了未來/測試階段真實分佈資訊 |
| [`bug_09_unmatched_splits_comparison.py`](bug_09_unmatched_splits_comparison.py) | 跨模型性能比較之統計檢定前提 | 成對檢定使用的樣本折切分不一致，違反成對檢定基本假設 |
| [`bug_10_uncalibrated_score_pooling.py`](bug_10_uncalibrated_score_pooling.py) | 跨折預測機率直接池化排序 | 跨折模型輸出之原始預測機率尺度未對齊，全域指標排序失真 |

---

## 2. 執行方式

學習者可於專案根目錄下直接執行任一腳本觀察其終端機輸出與症狀：

```bash
# 範例：執行第 1 題前處理除錯題
python labs/bugs/bug_01_preprocessing_leakage.py

# 範例：執行第 2 題重採樣除錯題
python labs/bugs/bug_02_resampling_leakage.py
```

---

## 3. 除錯目標

請學習者在閱讀各腳本代碼時，思考以下三個問題：
1. **錯誤究竟藏在第幾行？**
2. **這個錯誤為何能讓程式碼「看似正常執行」，卻在數理邏輯上造成了嚴重的評估偏差？**
3. **若在正式破產預測專案中發生此類錯誤，專案管線（如 `bankruptcy/cv.py`、`bankruptcy/pipelines.py`、`bankruptcy/thresholding.py`）是如何設計來徹底阻絕它的？**
