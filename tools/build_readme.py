#!/usr/bin/env python3
"""Builds docs/code_learning/README.md dynamically from filesystem counts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "code_learning"
LABS_DIR = REPO_ROOT / "labs"


def build_readme() -> str:
    # 1. Compute dynamic counts from filesystem
    n_module_guides = len(list((DOCS_DIR / "02_模組導讀").glob("*.md")))
    n_param_manuals = len(list((DOCS_DIR / "03_參數手冊").glob("*.md")))
    n_lab_scripts = len(list(LABS_DIR.glob("lab_[0-9][0-9]_*.py")))
    n_bug_scripts = len(list((LABS_DIR / "bugs").glob("bug_[0-9][0-9]_*.py")))
    n_wip_bugs = len(list((LABS_DIR / "bugs" / "_wip").glob("bug_[0-9][0-9]_*.py")))
    n_total_md = len(list(DOCS_DIR.rglob("*.md")))

    content = f"""# 《企業破產預測專案》程式碼學習資料包：一頁式索引 (`README.md`)

> [!NOTE]
> 本資料包為《企業破產預測專案》（Bankruptcy Prediction Project）之全套程式碼導讀、參數手冊、數理統計推導與動手實驗環境。詳細學習規劃與環境建置請參閱 [00_讀我_學習路線.md](00_讀我_學習路線.md)。

---

## 1. 第一天 90 分鐘上手路徑

若初次接觸本專案且時間有限，建議依序進行以下 90 分鐘核心演練：

1. **掌握專案定位與架構**：
   - 研讀 [00_讀我_學習路線.md](00_讀我_學習路線.md)，理解高召回風險預警定位。
   - 研讀 [01_專案地圖與資料流.md](01_專案地圖與資料流.md)，掌握防洩漏管線與資料切分邊界。
2. **核心評估指標與玩具例子演練**：
   - 精讀 [02_模組導讀/metrics.md](02_模組導讀/metrics.md)，跟著 20 家公司玩具例子親手計算 AR@0.95 與悲觀同分規則。
3. **動手執行快速實驗**：
   - 於終端機執行 Lab 04 不平衡策略快速實驗：
     ```bash
     python labs/lab_04_imbalance_strategies.py --quick
     ```
   - 對照 [labs/README.md](../../labs/README.md) 的 Lab 04 段落，核對重抽樣如何扭曲機率分佈並破壞校準。
4. **沙盒除錯初體驗**：
   - 閱讀 [06_練習題.md](06_練習題.md) 前兩題（前處理洩漏與過採樣時機陷阱）。
   - 實地查看與執行位於 [labs/bugs/](../../labs/bugs/) 的沙盒腳本：
     ```bash
     python labs/bugs/bug_01_preprocessing_leakage.py
     python labs/bugs/bug_02_resampling_leakage.py
     ```

---

## 2. 警語：不要執行危險專案指令

> [!WARNING]
> **嚴禁任意執行正式專案實驗腳本**
> - 請參閱 [04_實驗操作指南.md 的危險指令總表](04_實驗操作指南.md#24-危險指令總表danger-command-table)。
> - `bankruptcy/src/` 下之正式實驗指令（如 `final_eval.py`、`thresholding.py`）會覆寫 `bankruptcy/outputs/` 已固化之實驗數據，或觸發測試集鎖定異常。
> - 學習與演練時，請僅執行 `labs/` 探索腳本與 `labs/bugs/` 沙盒除錯題。

---

## 3. 建議閱讀順序（三大路徑）

依據讀者的學習目標，本資料包規劃三條建議閱讀路徑：

- **路徑 A：面試速成衝刺（閱讀時間估計，非實測：約 2～3 小時）**
  1. [00_讀我_學習路線.md](00_讀我_學習路線.md)：掌握專案定位、資料規模與核心成果。
  2. [01_專案地圖與資料流.md](01_專案地圖與資料流.md)：理解管線防洩漏架構與模組職責。
  3. [03c_速查表.md](03c_速查表.md)：速記核心超參數配置、檢定結論與指令。
  4. [_drafts/09_面試說明稿.md](_drafts/09_面試說明稿.md)：演練第一人稱口語技術講稿與 10 大面試核心 Q&A。
  5. [10_已知限制.md](10_已知限制.md)：掌握 F-01～F-04 缺陷分析與小樣本統計邊界。
- **路徑 B：機器學習工程實踐（閱讀時間估計，非實測：約 6～8 小時）**
  1. [00_讀我_學習路線.md](00_讀我_學習路線.md) → [01_專案地圖與資料流.md](01_專案地圖與資料流.md)。
  2. [02_模組導讀/](02_模組導讀/config.md)：深入共 {n_module_guides} 篇原始碼剖析（資料稽核、前處理、CV 切分、調參、集成、校準與推論）。
  3. [03_參數手冊.md](03_參數手冊.md) 與 [03_參數手冊/](03_參數手冊/01_資料與切分.md)：精讀 {n_param_manuals} 份主題參數手冊（收錄 37 張模型卡與 115 項參數配置規格）。
  4. [04_實驗操作指南.md](04_實驗操作指南.md)：理解實驗管線依賴與重現流程。
  5. [05_動手改程式配方.md](05_動手改程式配方.md)：演練 7 大安全擴充配方與沙盒隔離機制。
  6. [10_已知限制.md](10_已知限制.md)：了解工業級部署之邊界與外推風險。
- **路徑 C：動手除錯與實驗（閱讀時間估計，非實測：約 4～5 小時）**
  1. [06_練習題.md](06_練習題.md)：動手實作 {n_bug_scripts} 道沙盒除錯題（腳本位於 [labs/bugs/](../../labs/bugs/)，另有 {n_wip_bugs} 道題目存於 `_wip` 備查）。
  2. [06_解答.md](06_解答.md)：比對修復程式碼與 20 次重複實驗配對差異統計。
  3. [06_小測驗.md](06_小測驗.md)：自我檢驗 15 題核心觀念選擇題。
  4. [labs/README.md](../../labs/README.md)：執行 {n_lab_scripts} 支獨立動手實驗腳本（樹深、超參數、提升法、不平衡策略、門檻轉移等）。
  5. [07_常見錯誤與除錯.md](07_常見錯誤與除錯.md) 與 [08_術語表.md](08_術語表.md)：掌握常見除錯排查與 48 個專案核心術語。

---

## 4. 全文件目錄與定位總結

全套資料包共收錄 {n_total_md} 份 Markdown 文件，結構與定位總結如下：

| 編號 | 檔案路徑 | 定位與一句話總結 |
| :--- | :--- | :--- |
| **00** | [00_讀我_學習路線.md](00_讀我_學習路線.md) | 全套資料包之入口指南，涵蓋讀者定位、相依環境建置步驟與學習路徑地圖。 |
| **01** | [01_專案地圖與資料流.md](01_專案地圖與資料流.md) | 系統性拆解專案模組架構、職責分工、防洩漏資料流與核心設定檔。 |
| **02** | [02_模組導讀/](02_模組導讀/config.md) | 共 {n_module_guides} 篇模組導讀，逐一剖析核心原始碼之函式職責、防洩漏設計與調用關係。 |
| **03** | [03_參數手冊.md](03_參數手冊.md) / [目錄](03_參數手冊/01_資料與切分.md) | 包含 {n_param_manuals} 份主題手冊，系統化收錄 37 張標準模型卡與 115 項參數配置規格。 |
| **03c**| [03c_速查表.md](03c_速查表.md) | 一站式速查核心參數、實驗結論、防洩漏規範與終端機常用執行指令。 |
| **04** | [04_實驗操作指南.md](04_實驗操作指南.md) | 完整解析實驗因果鏈、重現順序、危險指令總表與重跑避坑指南。 |
| **05** | [05_動手改程式配方.md](05_動手改程式配方.md) | 7 道安全修改配方（新增模型、轉換器、自訂指標與推論端點），附沙盒驗證。 |
| **06** | [06_練習題.md](06_練習題.md) | {n_bug_scripts} 道工業級除錯實戰題目，涵蓋前處理洩漏、未對齊切分與不平衡採樣陷阱。 |
| **06解**| [06_解答.md](06_解答.md) | {n_bug_scripts} 道除錯題之完整成因分析、修正程式碼與 20 次重複實驗配對差異統計。 |
| **06測**| [06_小測驗.md](06_小測驗.md) | 15 題專案核心觀念單選題，自我檢驗對非平衡學習與評估架構之理解。 |
| **07** | [07_常見錯誤與除錯.md](07_常見錯誤與除錯.md) | 涵蓋常見報錯排查（如 PermissionError、形狀不匹配、記憶體溢出）與防護對策。 |
| **08** | [08_術語表.md](08_術語表.md) | 專案專用術語、統計評估指標、符號約定與數學公式定義大全（48 個詞條）。 |
| **09** | [_drafts/09_面試說明稿.md](_drafts/09_面試說明稿.md) | 第一人稱口語技術講稿與 10 題深度 Q&A，所有量測數字由程式自輸出 CSV/JSON 帶入。 |
| **10** | [10_已知限制.md](10_已知限制.md) | 彙總 F-01～F-04 架構缺陷、未序列化參數邊界、小樣本統計檢定力與外推限制。 |

---

## 5. 驗證工具與執行指令速查

本資料包配備自動化代碼與文件一致性檢查器，確保文件引用之代碼符號、輸出數據與檔案路徑吻合一致：

```bash
# 1. 嚴格證據、數值與相對連結檢查（驗證所有表格標籤、推導數值與連結存在性）
python tools/check_code_docs.py --strict

# 2. 參數卡片覆蓋率檢查（檢查 37 張模型卡與 115 個 ID）
python tools/check_code_docs.py --cards

# 3. 完整性語法、命令與術語檢查
python tools/check_code_docs.py --commands
python tools/check_code_docs.py --recipes
python tools/check_code_docs.py --exercises
python tools/check_code_docs.py --glossary
python tools/check_code_docs.py --registry

# 4. 面試講稿自動渲染（由磁碟 CSV / JSON 自動帶入數據）
python tools/render_interview.py
python -m pytest tools/test_render_interview.py -v
```
"""
    return content


def main():
    readme_path = DOCS_DIR / "README.md"
    readme_content = build_readme()
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Successfully generated {readme_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
