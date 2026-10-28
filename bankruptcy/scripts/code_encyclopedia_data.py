"""
bankruptcy/scripts/code_encyclopedia_data.py

Comprehensive Python Codebase Encyclopedia for the Bankruptcy Prediction Project.
Designed specifically for Python & ML learners to understand:
1. WHY the code is designed this way (underlying engineering & statistical purpose)
2. WHAT each code block does line-by-line (with beginner analogies)
3. PYTHON SYNTAX & IDIOMS (OOP, NumPy vectorization, Scikit-Learn contracts, defensive coding)
4. DATA SHAPE & TYPE TRACING (how DataFrames, Series, and ndarrays transform)
"""

from __future__ import annotations
from typing import Any

# ==============================================================================
# 8 Beginner vs. Senior Python Idiom Comparisons (Bad vs. Good)
# ==============================================================================
PYTHON_IDIOMS_COMPARISON: list[dict[str, Any]] = [
    {
        "id": "idiom-1",
        "title": "1. 路徑管理：字串硬拼接 vs. pathlib.Path 物件導向路徑",
        "tag": "檔案與跨平台 I/O",
        "analogy": "就像寄信時手寫死『C槽某資料夾』，換台電腦就迷路；改用 GPS 相對定位（Path(__file__)），整包專案搬到哪都能自動找到家。",
        "bad_code": """# ❌ 新手常見寫法：用字串相加 + 相對工作目錄
import os
data_path = "data\\\\data.csv"  # 在 Mac/Linux 斜線方向會錯！
# 且只要從不同資料夾執行 python，相對路徑就會找不到檔案 (FileNotFoundError)
if not os.path.exists("outputs/tables"):
    os.makedirs("outputs/tables")""",
        "good_code": """# ✅ 本專案寫法 (config.py)：以當前檔案位置為基準 + 運算子多載 (/)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)""",
        "why_better": (
            "1. `__file__` 代表目前這支 `.py` 檔自己的位置，`.resolve().parent` 取得它所在的絕對目錄，不管你在哪裡下 `python` 指令都不會迷路。\n"
            "2. `Path` 物件把除法符號 `/` 改寫（Operator Overloading）成「自動依作業系統拼接路徑」，Windows 自動用 `\\`、Linux 自動用 `/`。\n"
            "3. `mkdir(parents=True, exist_ok=True)` 一行搞定「連同上層目錄一起建，且若已存在不報錯（冪等性 Idempotency）」。"
        ),
    },
    {
        "id": "idiom-2",
        "title": "2. 轉換器設計：寫散落的函式 vs. 繼承 Scikit-Learn BaseEstimator",
        "tag": "OOP 與 Scikit-Learn 規範",
        "analogy": "自己寫普通函式就像自製非標準插頭，插不進 Scikit-Learn 的延長線（Pipeline）；繼承 BaseEstimator 與 TransformerMixin 就像裝上國際標準三孔插頭。",
        "bad_code": """# ❌ 新手常見寫法：寫一個同時算分位數又改資料的普通函式
def clip_outliers(df):
    for col in df.columns:
        low = df[col].quantile(0.01)   # 糟糕！在測試集呼叫時會偷算測試集的分位數！
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(low, high)
    return df""",
        "good_code": """# ✅ 本專案寫法 (preprocessing.py)：嚴格分離 fit(學參數) 與 transform(套參數)
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99, enabled: bool = True):
        self.lower = lower
        self.upper = upper
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        # 只在訓練折學習上下界，存進結尾有底線的屬性 (_)
        self.lower_bounds_ = {c: float(X[c].quantile(self.lower)) for c in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 驗證折與測試折只讀取 self.lower_bounds_，絕不重新計算！
        ...""",
        "why_better": (
            "1. **底層防洩漏保證**：Scikit-Learn 的 `Pipeline` 在交叉驗證時，對訓練折呼叫 `fit_transform()`，對驗證折只呼叫 `transform()`。把「學習（fit）」與「轉換（transform）」拆成兩個方法，從架構上保證驗證折永遠不會重算分位數！\n"
            "2. **免費獲得超能力**：繼承 `BaseEstimator` 後自動擁有 `get_params()` 與 `set_params()`（讓 Optuna 可以直接調參、讓 `clone()` 可以深度複製）；繼承 `TransformerMixin` 後只要寫好 `fit` 和 `transform`，就自動免費獲得 `fit_transform()` 方法！"
        ),
    },
    {
        "id": "idiom-3",
        "title": "3. 複製模型物件：直接賦值 (=) vs. sklearn.base.clone()",
        "tag": "OOP 與 Scikit-Learn 規範",
        "analogy": "直接用 `=` 賦值就像兩個人共用同一本考卷，第二折作答時把第一折的答案塗掉了；`clone()` 則是拿一張全新的空白考卷（只保留題目設定，清空作答紀錄）。",
        "bad_code": """# ❌ 新手常見寫法：在 Wrapper 或迴圈中直接共用同一個 estimator 物件
class MyWrapper:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        # 直接修改了外部傳進來的原始物件！跨折時狀態會互相污染
        self.estimator.set_params(scale_pos_weight=30.0)
        self.estimator.fit(X, y)""",
        "good_code": """# ✅ 本專案寫法 (pipelines.py)：fit 時使用 clone() 建立乾淨的未訓練副本
from sklearn.base import clone

class AdaptiveScaleWeightClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, **fit_params):
        # 複製一份擁有相同超參數、但完全未訓練 (unfitted) 的新物件
        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(scale_pos_weight=scale_weight)
        self.estimator_.fit(X, y, **fit_params)
        return self""",
        "why_better": (
            "在 Python 中，`a = b` 只是複製「記憶體位址標籤（Reference）」，指的還是同一個模型物件。"
            "`sklearn.base.clone(estimator)` 是專門設計來「只複製模型的超參數設定（`__init__` 參數），但丟棄所有已經 `fit` 出來的權重與樹結構」的標準工具，確保 15 折交叉驗證每一折都從零開始乾淨訓練。"
        ),
    },
    {
        "id": "idiom-4",
        "title": "4. 百分位排名計算：雙層 for 迴圈 O(M×N) vs. np.searchsorted 二分搜尋 O(M log N)",
        "tag": "NumPy 向量化",
        "analogy": "要在 5,455 人成績單中查自己的排名，新手做法是從第 1 個人逐一問到第 5,455 人；`searchsorted` 則是先把成績單排好序，直接翻正中間比大小（二分搜尋法），13 個步驟內必找到位置！",
        "bad_code": """# ❌ 新手常見寫法：Python 雙層迴圈逐一比大小 (極慢！)
percentiles = []
for q in query_scores:
    less_count = 0
    le_count = 0
    for r in reference_scores:
        if r < q: less_count += 1
        if r <= q: le_count += 1
    percentiles.append((less_count + le_count) / (2 * len(reference_scores)))""",
        "good_code": """# ✅ 本專案寫法 (scoring.py)：C 語言底層二分搜尋向量化 (快 500 倍以上)
ref_sorted = np.sort(np.asarray(reference_scores, dtype=float))
n_ref = len(ref_sorted)
left = np.searchsorted(ref_sorted, query, side="left")   # 嚴格小於 q 的個數
right = np.searchsorted(ref_sorted, query, side="right") # 小於等於 q 的個數
percentiles = (left + right) / (2.0 * n_ref)""",
        "why_better": (
            "1. 時間複雜度從 $O(M \\times N)$ 降為 $O(M \\log N)$，且整個陣列在 C 語言底層一次算完，完全避開 Python `for` 迴圈的直譯器開銷。\n"
            "2. 巧妙利用 `side='left'`（插入在相同值最左邊 $\\implies$ 索引值剛好等於 $< q$ 的個數）與 `side='right'`（插入在相同值最右邊 $\\implies$ 索引值剛好等於 $\\le q$ 的個數），兩行程式碼完美處理所有平手同分（Ties）情況！"
        ),
    },
    {
        "id": "idiom-5",
        "title": "5. Python 型別檢查大坑：為什麼 isinstance(True, int) 會是 True？",
        "tag": "防禦性設計與型別陷阱",
        "analogy": "在 Python 族譜裡，布林值（bool）其實是整數（int）的親生兒子（True 就是 1，False 就是 0）。如果只檢查『是不是整數』，布林值會蒙混過關！",
        "bad_code": """# ❌ 新手常見寫法：只檢查 isinstance(val, (int, float))
def validate_num(val):
    if not isinstance(val, (int, float)):
        raise ValueError("必須是數字！")
    return float(val)

# 災難發生：validate_num(True) 不會報錯，還會回傳 1.0！""",
        "good_code": """# ✅ 本專案寫法 (app/logic.py)：先攔截 bool 與 None，再檢查數值與 NaN/Inf
def _validate_numeric_value(feat: str, val: Any) -> float:
    if isinstance(val, bool) or val is None or not isinstance(val, (int, float, np.number)):
        raise ValueError(f"Invalid non-numeric value for {feat!r}: {val!r}")
    fval = float(val)
    if np.isnan(fval) or np.isinf(fval):
        raise ValueError(f"NaN or Inf is not allowed for {feat!r}: {val!r}")
    return fval""",
        "why_better": (
            "1. 在 Python 底層設計中，`issubclass(bool, int)` 為 `True`。因此驗證數值輸入時，**永遠要先寫 `if isinstance(val, bool): raise ...`**。\n"
            "2. 同時考慮了 `np.number`（如 `np.float64`, `np.int32`），並攔截了浮點數黑洞 `NaN`（Not a Number）與 `Inf`（無限大），因為 `isinstance(float('nan'), float)` 也是 `True`！"
        ),
    },
    {
        "id": "idiom-6",
        "title": "6. 排序後還原順序：手動記錄字典 vs. np.argsort 反向索引賦值",
        "tag": "NumPy 向量化",
        "analogy": "全班按身高排隊量完衣服尺寸後，怎麼一秒鐘讓大家坐回原本的學號座位？只要拿著當初排隊的『座號對照表（order）』，直接 `out[order] = 新結果` 就全員歸位了！",
        "bad_code": """# ❌ 新手常見寫法：轉成 tuple 列表排完序，再用迴圈慢慢找回原本位置
pairs = [(p, idx) for idx, p in enumerate(p_values)]
pairs.sort(key=lambda x: x[0])
# 算完校正值後還要再 sort 一次 index 才能還原...""",
        "good_code": """# ✅ 本專案寫法 (stats_utils.py)：argsort 取得索引排列 + 一步還原
order = np.argsort(p_arr)      # 取得由小到大排序的「原始位置索引」
sorted_p = p_arr[order]        # 依序排列 p 值進行 Holm-Bonferroni 計算
...
out = np.empty(m, dtype=float)
out[order] = adjusted          # 魔法一行：用花式索引 (Fancy Indexing) 直接歸位！""",
        "why_better": (
            "`np.argsort(p_arr)` 不會直接回傳排好序的值，而是回傳「排第 1 小的元素原本在第幾格、排第 2 小的原本在第幾格」。"
            "當我們在排序空間算完 `adjusted` 後，寫 `out[order] = adjusted`，NumPy 就會自動把排第 $i$ 的校正結果塞回原本的 `order[i]` 位置，優雅且零迴圈錯誤。"
        ),
    },
    {
        "id": "idiom-7",
        "title": "7. 防止誤觸禁區：寫在 README 口頭警告 vs. 門衛函式長參數鎖",
        "tag": "防禦性設計與型別陷阱",
        "analogy": "核彈發射鈕不能只貼一張『請勿誤觸』貼紙，必須設計成『必須插入鑰匙並親手輸入確認碼』才能打開保護蓋。",
        "bad_code": """# ❌ 新手常見寫法：直接提供 get_test_data() 或全域變數 X_test
# 寫在文件說「大家不要在調參時用 X_test 喔」，結果寫程式自動補全時不小心就讀到了
X_test, y_test = load_test_data()""",
        "good_code": """# ✅ 本專案寫法 (src/cv.py & lock_utils.py)：故意設計超長確認參數 + 實體鎖檔
def get_test_data(i_know_this_is_final_evaluation: bool = False):
    if not i_know_this_is_final_evaluation:
        raise RuntimeError(
            "存取遭拒！測試集嚴格封存至 E6，一般實驗階段禁止讀取。"
        )
    acquire_e6_lock()  # 寫入 outputs/logs/E6_LOCK，第二次呼叫直接拋出異常！
    ...""",
        "why_better": (
            "這在軟體工程中稱為 **Intentional Friction（刻意製造摩擦力的防呆設計）**。"
            "任何腳本只要沒明確寫出 `i_know_this_is_final_evaluation=True` 就會立刻崩潰報錯；且一旦解封就會在硬碟寫下含程式碼 SHA-256 指紋的 `E6_LOCK` 檔案，從物理上杜絕重複刷測試集！"
        ),
    },
    {
        "id": "idiom-8",
        "title": "8. 多條件貪婪排序：巢狀 if-else 比大小 vs. Tuple Key Lambda 排序",
        "tag": "Python 進階語法與演算法",
        "analogy": "成績單排序時，規定『總分高者優先；若總分同分，座號小者優先』。在 Python 只要給一個 `(-總分, 座號)` 的小括號元組（Tuple），就能一行排完！",
        "bad_code": """# ❌ 新手常見寫法：自己寫複雜的比較函式或排兩次出錯
# 想先按 auc 由大到小排，同分時按原本欄位順序由小到大排""",
        "good_code": """# ✅ 本專案寫法 (preprocessing.py - CorrelationFilter)：
col_order = {c: i for i, c in enumerate(X.columns)}
ranked_cols = sorted(
    X.columns,
    key=lambda c: (-auc_dict[c], col_order[c])
)""",
        "why_better": (
            "Python 在比較兩個 Tuple `(a1, b1)` 與 `(a2, b2)` 時，天生就會「先比第一個元素 `a1 vs a2`；只有當 `a1 == a2` 完全平手時，才會繼續比第二個元素 `b1 vs b2`」。"
            "加上負號 `-auc_dict[c]` 就把預設的升序（由小到大）變成了降序（AUC 越大越前面），而 `col_order[c]` 維持升序（越早出現越前面），確保結果 100% 確定性（Deterministic）！"
        ),
    },
]


# ==============================================================================
# 12 Core Modules Encyclopedia (Part 1: M01 to M06)
# ==============================================================================
CODE_MODULES_PART1: list[dict[str, Any]] = [
    {
        "id": "M01",
        "code": "config.py",
        "title": "M01. 全域設定與單一真相中心 (config.py)",
        "file_path": "bankruptcy/config.py",
        "category": "一、基礎建設與防洩漏資料層",
        "analogy": "整棟大樓的「中央控制室與建築藍圖」——所有房間在哪裡、亂數種子幾號、目標召回率定多少，全部集中在這張表單，絕不散落在各個房間裡。",
        "purpose": (
            "消除程式碼中的「魔術數字（Magic Numbers）」與「硬編碼路徑（Hardcoded Paths）」。"
            "當未來要將隨機種子從 42 改為其他值，或更換資料集路徑時，只需改這一個檔案，全專案 20 幾支程式就會同步生效。"
        ),
        "calls_and_called_by": "被全專案所有 `src/*.py`、`scripts/*.py` 與 `tests/*.py` 匯入使用（零外部專案依賴的最底層模組）。",
        "blocks": [
            {
                "block_title": "區塊 1：跨平台動態路徑錨點與自動建目錄",
                "tags": ["檔案與跨平台 I/O", "防禦性設計與型別陷阱"],
                "code": """from pathlib import Path

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
    directory.mkdir(parents=True, exist_ok=True)""",
                "what_it_does": (
                    "1. 自動抓出 `config.py` 所在的資料夾（`bankruptcy/`）作為專案根目錄 `BASE_DIR`。\n"
                    "2. 依序定義資料檔 `data.csv` 以及報表（`tables`）、圖表（`figures`）、模型（`models`）、日誌（`logs`）四個輸出資料夾的完整路徑。\n"
                    "3. 在任何程式 `import config` 的瞬間，自動檢查這四個輸出資料夾是否存在，若不存在就自動建立。"
                ),
                "why_written_this_way": (
                    "初學者常在儲存圖表或 CSV 時遇到 `FileNotFoundError: No such file or directory: 'outputs/tables/xxx.csv'`，因為忘了先建資料夾。"
                    "把 `directory.mkdir(parents=True, exist_ok=True)` 寫在 `config.py` 中，保證只要匯入設定檔，所有輸出目錄就一定已經準備好！"
                ),
                "syntax_points": (
                    "• `Path(__file__).resolve().parent`：`__file__` 是 Python 內建變數（當前檔案路徑），`.resolve()` 轉為絕對路徑並解析捷徑，`.parent` 取上一層目錄。\n"
                    "• `BASE_DIR / 'data'`：`Path` 物件支援用 `/` 連接子目錄，比 `os.path.join()` 簡潔易讀。\n"
                    "• `parents=True`：若上層 `outputs/` 還沒建，會連同上層一起建；`exist_ok=True`：若資料夾已存在則直接略過不報錯。"
                ),
                "shape_and_type": "所有路徑變數皆為 `pathlib.WindowsPath`（在 Mac/Linux 則自動為 `PosixPath`）物件，而非普通 `str`。",
            },
            {
                "block_title": "區塊 2：實驗常數與評估目標集中定義",
                "tags": ["Scikit-Learn 規範"],
                "code": """# Random Seed
SEED = 42

# Data Split & Cross-Validation
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 3

# Evaluation Metrics Parameters
TARGET_RECALLS = [0.80, 0.90, 0.95, 1.00]
ALERT_RATIOS = [1.5, 3, 5, 10]
DEFAULT_TARGET_RECALL = 0.95
TARGET_COL = "Bankrupt?"
FEATURES = None""",
                "what_it_does": (
                    "定義全專案共用的隨機亂數種子 `SEED = 42`、測試集切分比例 20%、外層 5 折 × 3 重複（共 15 折），"
                    "以及評估時統一觀察的四個目標召回率 `[0.80, 0.90, 0.95, 1.00]` 與目標欄位名稱 `'Bankrupt?'`。"
                ),
                "why_written_this_way": (
                    "Python PEP 8 規範要求「全大寫底線命名（UPPER_CASE）」代表**全域常數（Constants）**，提醒所有開發者這些數值在程式執行期間不應被動態竄改。"
                    "這確保了 E0、E1、E2、E3、E4、E5 在完全相同的 `SEED=42` 與相同的召回率切點下公平對決。"
                ),
                "syntax_points": (
                    "• `SEED = 42`：固定隨機性（Reproducibility），使 `train_test_split`、`SMOTE`、`ExtraTrees` 每次執行結果 100% 相同。\n"
                    "• `FEATURES = None`：以 `None` 作為預設哨兵值（Sentinel Value），代表「預設使用全部特徵」。"
                ),
                "shape_and_type": "`SEED: int`, `TEST_SIZE: float`, `TARGET_RECALLS: list[float]`, `TARGET_COL: str`。",
            },
        ],
    },
    {
        "id": "M02",
        "code": "src/cv.py",
        "title": "M02. 交叉驗證切分、SHA-256 指紋與測試集門衛 (src/cv.py)",
        "file_path": "bankruptcy/src/cv.py",
        "category": "一、基礎建設與防洩漏資料層",
        "analogy": "銀行的「金庫管理員與分發中心」——負責把 6,819 筆資料分成 80% 訓練區與 20% 封存金庫，並幫資料蓋上防偽數位指紋（SHA-256），誰想偷開金庫就會觸發警報。",
        "purpose": (
            "1. 產生並固化全專案唯一的資料切分檔 `outputs/models/splits.joblib`（確保所有模型都在完全相同的 15 個外層折與內層折上訓練與驗證）。\n"
            "2. 透過 SHA-256 雜湊碼驗證資料有沒有被意外洗牌或增刪。\n"
            "3. 嚴格隔離 20% 最終測試集（1,364 筆）。"
        ),
        "calls_and_called_by": "匯入 `config.py`、`metrics.py`；被 E0～E7 所有實驗腳本呼叫以取得 `get_train_data()` 與 `load_splits()`。",
        "blocks": [
            {
                "block_title": "區塊 1：目標標籤陣列之 SHA-256 數位指紋 (compute_y_hash)",
                "tags": ["雜湊與檔案鎖", "NumPy 向量化"],
                "code": """def compute_y_hash(y: Union[np.ndarray, pd.Series, Sequence]) -> str:
    \"\"\"Compute SHA-256 hash of target label array.\"\"\"
    y_bytes = np.ascontiguousarray(np.asarray(y).ravel()).tobytes()
    return hashlib.sha256(y_bytes).hexdigest()""",
                "what_it_does": (
                    "將傳入的破產標籤序列 `y`（不論是 Pandas Series、Python List 或 NumPy Array）轉成一維連續記憶體二進位位元組（bytes），"
                    "並計算出長度 64 字元的 SHA-256 十六進位防偽指紋字串。"
                ),
                "why_written_this_way": (
                    "如果有人不小心在 Excel 打開 `data.csv` 重新排序了某幾列，總筆數雖然還是 6,819 筆，但原本固化的 `train_idx` 就會指到錯誤的公司！"
                    "有了 `compute_y_hash(y)`，只要有一筆標籤順序改變，產生的 SHA-256 就會完全不同，`load_splits()` 就會立刻攔截並報錯。"
                ),
                "syntax_points": (
                    "• `np.asarray(y).ravel()`：將輸入轉為 NumPy 陣列並攤平為一維 `(N,)`（不複製記憶體視圖）。\n"
                    "• `np.ascontiguousarray(...)`：確保陣列在電腦記憶體中是**連續排列（C-contiguous）**的，這樣呼叫 `.tobytes()` 轉成二進位時才不會受切片步長影響。\n"
                    "• `hashlib.sha256(y_bytes).hexdigest()`：標準密碼學雜湊函式，輸出如 `'ca3b5205...'` 的唯一字串。"
                ),
                "shape_and_type": "輸入：`Sequence | pd.Series | np.ndarray` 長度 $N$ ➔ 內部：`bytes` ➔ 輸出：`str` (64 hex chars)。",
            },
            {
                "block_title": "區塊 2：外層 15 折重複分層切分與原始列號映射 (make_cv_splits)",
                "tags": ["NumPy 向量化", "Scikit-Learn 規範"],
                "code": """rskf = RepeatedStratifiedKFold(
    n_splits=CV_FOLDS,
    n_repeats=CV_REPEATS,
    random_state=SEED,
)

splits = []
for split_idx, (sub_train_idx, sub_val_idx) in enumerate(rskf.split(np.zeros(n_train), y_arr)):
    repeat_num = split_idx // CV_FOLDS
    fold_num = split_idx % CV_FOLDS

    splits.append({
        "repeat": int(repeat_num),
        "fold": int(fold_num),
        "split_id": f"r{repeat_num}_f{fold_num}",
        "train_idx": train_orig_indices[sub_train_idx],
        "val_idx": train_orig_indices[sub_val_idx],
    })""",
                "what_it_does": (
                    "在 5,455 筆訓練集上進行「5 折 × 3 次重複 = 15 折」的分層抽樣（確保每一折的破產比例都精確維持 3.23%），"
                    "並把每一折的編號（如 `r0_f0` ~ `r2_f4`）與對應的列索引存成字典清單。"
                ),
                "why_written_this_way": (
                    "注意兩個極聰明的細節：\n"
                    "1. `rskf.split(np.zeros(n_train), y_arr)`：因為分層切分只需要知道「總共有幾筆」跟「每筆的標籤 `y_arr`」，根本不需要讀入龐大的特徵矩陣 `X`，傳入 `np.zeros(n_train)` 既省記憶體又快！\n"
                    "2. `repeat_num = split_idx // 5`（整數商）與 `fold_num = split_idx % 5`（餘數）：將 `0~14` 的單一編號優雅拆解為「第幾個 Repeat (0~2)」與「第幾個 Fold (0~4)」。"
                ),
                "syntax_points": (
                    "• `enumerate(...)`：在迴圈中同時取得「目前是第幾次迭代 `split_idx`」與「切分出的 `(sub_train_idx, sub_val_idx)`」。\n"
                    "• `train_orig_indices[sub_train_idx]`：NumPy 花式索引（Fancy Indexing），將訓練集內部的相對位置 `0..5454` 轉回原始 `data.csv` 的列號。"
                ),
                "shape_and_type": "輸出：`list[dict]`，長度為 15；每個 dict 內的 `train_idx` 為 `int64 ndarray (4364,)`，`val_idx` 為 `int64 ndarray (1091,)`。",
            },
        ],
    },
    {
        "id": "M03",
        "code": "src/lock_utils.py",
        "title": "M03. E6 測試集一次性硬體鎖與程式碼雜湊 (src/lock_utils.py)",
        "file_path": "bankruptcy/src/lock_utils.py",
        "category": "一、基礎建設與防洩漏資料層",
        "analogy": "考場的「彌封封條與防偽鋼印」——一旦拆封考卷（執行 E6 測試集評估），系統立即蓋上當下時間與全體程式碼指紋的封條，禁止塗改考卷重考。",
        "purpose": (
            "機器學習最大的學術與工程不端行為就是「看到測試集結果不理想，偷偷改一點模型或門檻再跑一次測試集」。"
            "`lock_utils.py` 實作了物理級別的單次執行鎖 `outputs/logs/E6_LOCK`。"
        ),
        "calls_and_called_by": "被 `src/final_eval.py` 與 `tests/test_lock.py` 呼叫。",
        "blocks": [
            {
                "block_title": "區塊 1：全專案原始碼 SHA-256 雜湊計算 (compute_codebase_hash)",
                "tags": ["雜湊與檔案鎖", "檔案與跨平台 I/O"],
                "code": """def compute_codebase_hash(src_dir: Path | str | None = None) -> str:
    \"\"\"Computes SHA-256 hash of all python files in src directory.\"\"\"
    if src_dir is None:
        src_dir = Path(__file__).resolve().parent
    src_path = Path(src_dir)

    hasher = hashlib.sha256()
    for py_file in sorted(src_path.glob("*.py")):
        hasher.update(py_file.name.encode("utf-8"))
        hasher.update(py_file.read_bytes())
    return hasher.hexdigest()""",
                "what_it_does": (
                    "掃描 `src/` 資料夾底下所有的 `.py` 程式碼檔案，按檔名排序後，將每個檔案的「檔案名稱」與「完整檔案內容位元組」餵進同一個 SHA-256 雜湊器，算出代表當前程式碼版本的唯一指紋。"
                ),
                "why_written_this_way": (
                    "1. **為什麼 `src_path.glob('*.py')` 外面一定要包 `sorted(...)`？** 因為不同作業系統或硬碟檔案系統在列出資料夾檔案時，順序是不固定的！如果不先排序，同一份程式碼在兩台電腦算出的 SHA-256 會不一樣。\n"
                    "2. **為什麼連 `py_file.name` 也要 `update`？** 防止把 A 檔案內容搬到 B 檔案時雜湊碰撞，確保檔名與內容同時受到保護。"
                ),
                "syntax_points": (
                    "• `Path | str | None = None`：Python 3.10+ 的新型聯集型別標註（Union Type Hint），等同於舊版的 `Union[Path, str, None]`。\n"
                    "• `hasher.update(bytes)`：增量式更新雜湊器狀態，最後呼叫 `.hexdigest()` 一次性產出結果。"
                ),
                "shape_and_type": "輸入：`Path | str | None` ➔ 輸出：`str` (64 字元 SHA-256 十六進位指紋)。",
            },
            {
                "block_title": "區塊 2：原子性鎖檔檢查與建立 (acquire_e6_lock)",
                "tags": ["雜湊與檔案鎖", "防禦性設計與型別陷阱"],
                "code": """if path.exists():
    content = path.read_text(encoding="utf-8")
    raise RuntimeError(
        f"E6_LOCK file already exists at {path}!\\n"
        "The holdout test set has already been evaluated. Duplicate evaluation is strictly prohibited.\\n"
        f"Lock details:\\n{content}"
    )

now = datetime.datetime.now().isoformat()
code_hash = compute_codebase_hash()
...
path.write_text(lock_payload, encoding="utf-8")""",
                "what_it_does": (
                    "在解封測試集前先檢查 `outputs/logs/E6_LOCK` 是否存在：若已存在，立刻拋出 `RuntimeError` 並印出上次鎖定的時間與雜湊碼，強制終止程式；"
                    "若不存在，則記錄當前 ISO 時間戳記與程式碼雜湊並寫入鎖檔。"
                ),
                "why_written_this_way": (
                    "採用「先檢查、立即寫鎖、再放行（Check-and-Lock）」模式，搭配 `pytest` 中的 `test_lock.py` 與 `test_frozen_hashes.py`，確保最終報告中的測試集成績具備 100% 可稽核的公信力。"
                ),
                "syntax_points": (
                    "• `path.read_text(encoding='utf-8')` / `path.write_text(..., encoding='utf-8')`：`pathlib.Path` 內建的檔案讀寫捷徑，會自動處理 `open()` 與 `close()`，比傳統 `with open(...) as f:` 更精簡！"
                ),
                "shape_and_type": "回傳：`dict[str, str]` 包含 `'locked_at'`, `'code_hash'`, `'lock_path'`。",
            },
        ],
    },
    {
        "id": "M04",
        "code": "src/preprocessing.py",
        "title": "M04. 自訂無洩漏前處理轉換器 (src/preprocessing.py)",
        "file_path": "bankruptcy/src/preprocessing.py",
        "category": "二、管線轉換與演算法核心層",
        "analogy": "工廠的「6 道自動化原料淨化工作站」——每個工作站都在訓練樣本上調校好濾網規格（fit），之後新原料進來時只照既有濾網過濾（transform），絕不偷改濾網。",
        "purpose": (
            "實作完全相容於 Scikit-Learn 與 imbalanced-learn Pipeline 的 6 個自訂轉換器（`FeatureSelector`, `DropConstantAndDuplicate`, `Winsorizer`, `LargeValueIndicator`, `QuantileScaler`, `CorrelationFilter`），"
            "並且全程保留 `pandas.DataFrame` 的欄位名稱（Column Names），解決原生 Scikit-Learn 轉換器常把 DataFrame 變成無欄名 NumPy Array 的痛點！"
        ),
        "calls_and_called_by": "匯入 `sklearn.base`、`sklearn.preprocessing`；被 `src/pipelines.py` 組裝進模型管線。",
        "blocks": [
            {
                "block_title": "區塊 1：Scikit-Learn 自訂轉換器標準樣板 (FeatureSelector & DropConstantAndDuplicate)",
                "tags": ["OOP 與繼承", "Scikit-Learn 規範"],
                "code": """class DropConstantAndDuplicate(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cols = list(X.columns)
        # 1. Constant columns (nunique <= 1)
        constant_cols = [c for c in cols if X[c].nunique(dropna=False) <= 1]
        # 2. Duplicate columns
        non_const = [c for c in cols if c not in constant_cols]
        dup_cols, seen = [], []
        for c1 in non_const:
            if any(X[c1].equals(X[c2]) for c2 in seen):
                dup_cols.append(c1)
            else:
                seen.append(c1)

        self.dropped_ = constant_cols + dup_cols
        self.selected_features_ = [c for c in cols if c not in self.dropped_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.selected_features_, dtype=object)""",
                "what_it_does": (
                    "在 `fit()` 階段掃描訓練折 DataFrame：找出所有數值都一樣的常數欄（`nunique <= 1`）與數值完全一模一樣的重複欄（`X[c1].equals(X[c2])`），"
                    "將保留下來的欄位清單存進 `self.selected_features_`；在 `transform()` 階段直接切片保留這些欄位。"
                ),
                "why_written_this_way": (
                    "**為什麼 `fit(self, X, y=None)` 就算沒用到 `y`，參數列也一定要寫 `y=None`？**\n"
                    "這是 Scikit-Learn `Pipeline` 的底層契約！當你呼叫 `pipeline.fit(X_train, y_train)` 時，Pipeline 會把 `X_train` 和 `y_train` 同時傳給管線中的**每一個步驟**。"
                    "如果你的 `fit` 只寫 `def fit(self, X):`，Pipeline 傳入 `y` 時就會拋出 `TypeError: fit() takes 2 positional arguments but 3 were given`！"
                ),
                "syntax_points": (
                    "• **結尾底線命名慣例（Trailing Underscore `self.selected_features_`）**：在 Scikit-Learn 規範中，凡是在 `fit()` 期間才從資料學出來的屬性，名稱結尾一律加底線 `_`。Scikit-Learn 的 `check_is_fitted()` 就是靠檢查有沒有結尾底線屬性來判斷模型訓練過了沒！\n"
                    "• `.copy()`：切片後呼叫 `.copy()`，防止後續步驟修改 DataFrame 時觸發 Pandas 著名的 `SettingWithCopyWarning`。"
                ),
                "shape_and_type": "輸入：`pd.DataFrame (N, 95)` ➔ `fit()` 學習 92 個保留欄名 ➔ `transform()` 輸出：`pd.DataFrame (N, 92)`。",
            },
            {
                "block_title": "區塊 2：保留欄名的分位數常態化轉換器 (QuantileScaler)",
                "tags": ["OOP 與繼承", "Scikit-Learn 規範", "NumPy 向量化"],
                "code": """class QuantileScaler(BaseEstimator, TransformerMixin):
    def __init__(self, kind: str = "quantile"):
        self.kind = kind

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)

        if self.kind == "quantile":
            n_quantiles = min(1000, len(X))
            self.scaler_ = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=SEED,
            )
        elif self.kind == "standard":
            self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vals = self.scaler_.transform(X)
        return pd.DataFrame(vals, columns=self.feature_names_in_, index=X.index)""",
                "what_it_does": (
                    "在內部包裝 Scikit-Learn 原生的 `QuantileTransformer`（或 `StandardScaler`），"
                    "並在 `transform()` 將原生轉換器輸出的純 NumPy 矩陣 `vals` 重新包裝回帶有原始欄位名稱 `self.feature_names_in_` 與原始列索引 `X.index` 的 `pd.DataFrame`。"
                ),
                "why_written_this_way": (
                    "1. 原生 `QuantileTransformer.transform(X)` 會把 `pd.DataFrame` 降級成沒有欄位名稱的 `np.ndarray`，這會導致下一層 `CorrelationFilter` 或最後的 `SHAP` 無法得知哪一欄是哪個財務指標！用 `pd.DataFrame(vals, columns=..., index=...)` 完美保留所有 Metadata。\n"
                    "2. `n_quantiles = min(1000, len(X))`：防禦性設計！如果今天在小樣本單元測試（例如只有 20 筆資料）呼叫 `QuantileTransformer(n_quantiles=1000)`，Scikit-Learn 會發出 UserWarning 甚至異常；加上 `min(1000, len(X))` 讓它在任何樣本數下都穩健運作。"
                ),
                "syntax_points": (
                    "• 組合模式（Composition over Inheritance）：不直接繼承 `QuantileTransformer`，而是把它作為內部屬性 `self.scaler_`，讓我們可以用同一個 `kind` 參數自由切換 `'quantile'` 與 `'standard'` 進行 E1 消融實驗！"
                ),
                "shape_and_type": "輸入：`pd.DataFrame (N, 92)` ➔ 內部 `vals: np.ndarray (N, 92)` ➔ 輸出：`pd.DataFrame (N, 92)`（數值呈標準常態分布 $\\mathcal{N}(0,1)$）。",
            },
            {
                "block_title": "區塊 3：監督式共線性貪婪過濾器 (CorrelationFilter)",
                "tags": ["Python 進階語法與演算法", "Scikit-Learn 規範"],
                "code": """# 1. Pearson correlation matrix on training fold X
corr = X.corr(method="pearson").abs()

# 2. Univariate adjusted AUC for each feature against training fold y
auc_dict = {}
for c in X.columns:
    raw_auc = roc_auc_score(y_arr, X[c].values)
    auc_dict[c] = float(max(raw_auc, 1.0 - raw_auc))

# 3. Greedy maximum-predictive feature selection
col_order = {c: i for i, c in enumerate(X.columns)}
ranked_cols = sorted(X.columns, key=lambda c: (-auc_dict[c], col_order[c]))

dropped = set()
selected = []
for c in ranked_cols:
    if c in dropped:
        continue
    selected.append(c)
    for other in ranked_cols:
        if other != c and other not in dropped and other not in selected:
            if corr.loc[c, other] > self.threshold:
                dropped.add(other)""",
                "what_it_does": (
                    "1. 計算訓練折 92 個特徵兩兩之間的絕對皮爾森相關係數矩陣 `corr`。\n"
                    "2. 計算每個特徵單獨預測破產 `y_arr` 的調整後 ROC-AUC（把負相關特徵如獲利能力的 AUC 0.11 自動翻轉為 `1 - 0.11 = 0.89`）。\n"
                    "3. 按單變數 AUC 由高到低排序，依序選入 `selected`，並把所有跟它高度共線（`|r| > 0.95`）的後段班特徵丟進 `dropped` 集合剔除。"
                ),
                "why_written_this_way": (
                    "1. **為什麼 `dropped = set()` 用集合（set）而不是列表（list）？** 因為在迴圈裡要頻繁檢查 `if c in dropped:`。在 Python 中，`list` 的查找時間是 $O(N)$（從頭掃描），而 `set` 底層是雜湊表（Hash Table），查找與新增（`.add()`）的時間複雜度是 **$O(1)$ 瞬間完成**！\n"
                    "2. **為什麼最後還要寫 `[c for c in X.columns if c in selected]`？** 因為 `selected` 是按 AUC 排的順序，最後用列表生成式按 `X.columns` 過濾一次，就能在剔除冗餘欄位的同時，保持原本特徵在資料表中的先後順序不變。"
                ),
                "syntax_points": (
                    "• `max(raw_auc, 1.0 - raw_auc)`：因為好指標（如 EPS）數值越高越不容易破產（`raw_auc < 0.5`），壞指標（如負債比）數值越高越容易破產（`raw_auc > 0.5`）。取 `max(auc, 1-auc)` 才能公平比較兩者的「區別力強弱」。\n"
                    "• `set()` 與 `.add()`：演算法中記錄「已走訪 / 已剔除節點」的標準高效資料結構。"
                ),
                "shape_and_type": "輸入：`X: pd.DataFrame (N, 92)`, `y: ndarray (N,)` ➔ `corr: DataFrame (92, 92)` ➔ 輸出：`pd.DataFrame (N, 72)`。",
            },
        ],
    },
    {
        "id": "M05",
        "code": "src/pipelines.py",
        "title": "M05. 動態權重包覆器與 Imblearn 管線組裝工廠 (src/pipelines.py)",
        "file_path": "bankruptcy/src/pipelines.py",
        "category": "二、管線轉換與演算法核心層",
        "analogy": "汽車工廠的「總裝配線（Assembly Line）」——把前處理工作站、SMOTE 合成器與分類器引擎按固定順序串成一條龍，並為梯度提升樹裝上自動感應正負樣本比例的變速箱。",
        "purpose": (
            "提供統一的工廠函式 `build_pipeline()`，將 6 層前處理、7 種不平衡策略（S0~S6）與 6 種模型（M0~M5）無縫組裝為單一 `imblearn.pipeline.Pipeline` 物件。"
        ),
        "calls_and_called_by": "匯入 `src/preprocessing.py`；被 `run_baselines.py`、`run_imbalance.py`、`tune.py`、`ensemble.py`、`thresholding.py`、`explain.py` 呼叫。",
        "blocks": [
            {
                "block_title": "區塊 1：零洩漏動態類別權重包覆器與 @property 代理 (AdaptiveScaleWeightClassifier)",
                "tags": ["OOP 與繼承", "Scikit-Learn 規範", "Python 進階語法與演算法"],
                "code": """class AdaptiveScaleWeightClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, enabled: bool = True):
        self.estimator = estimator
        self.enabled = enabled

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        if self.enabled:
            y_arr = np.asarray(y).ravel()
            n_neg = int((y_arr == 0).sum())
            n_pos = int((y_arr == 1).sum())
            scale_weight = float(n_neg / max(1, n_pos))
            self.scale_pos_weight_ = scale_weight
            self.estimator_.set_params(scale_pos_weight=scale_weight)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    @property
    def feature_importances_(self):
        if hasattr(self, "estimator_") and hasattr(self.estimator_, "feature_importances_"):
            return self.estimator_.feature_importances_
        raise AttributeError("Underlying estimator has no feature_importances_.")""",
                "what_it_does": (
                    "包覆 XGBoost (M4) 與 LightGBM (M5)，在每一個訓練折呼叫 `fit(X, y)` 的當下，動態計算該折內部的負正樣本比 `n_neg / n_pos`（約 30.0），"
                    "自動設定給底層模型的 `scale_pos_weight` 參數，並透過 `@property` 將底層模型的 `feature_importances_` 透明轉發給外部。"
                ),
                "why_written_this_way": (
                    "1. Scikit-Learn 的 `RandomForest` 支援 `class_weight='balanced'`（會自動在每折內算比例），但 `XGBClassifier` 的 `scale_pos_weight` 只接受一個寫死的浮點數！如果我們在進入 CV 前用全資料算好傳進去，就違反了「只在訓練折內學參數」的鐵則；寫成這個 Wrapper 類別就能在每折 `fit()` 瞬間動態算比例！\n"
                    "2. `max(1, n_pos)`：防禦除以零錯誤（ZeroDivisionError）。\n"
                    "3. `self.classes_ = self.estimator_.classes_`：Scikit-Learn 分類器契約規定，訓練完成的分類器必須具備 `classes_` 屬性。"
                ),
                "syntax_points": (
                    "• `@property` 裝飾器：把一個方法偽裝成「屬性（Attribute）」。外部程式寫 `clf.feature_importances_`（不用加小括號 `()`）時，Python 會自動執行這個函式去向內部的 `self.estimator_` 拿特徵重要性！\n"
                    "• `**fit_params`：接收任意額外的關鍵字參數（Keyword Arguments）並原封不動轉交給底層 `fit()`。"
                ),
                "shape_and_type": "`fit(X, y)` 動態設定 `scale_pos_weight_: float` ➔ `predict_proba(X)` 輸出 `ndarray (N, 2)`。",
            },
            {
                "block_title": "區塊 2：字典解包 (**kwargs) 與 imbalanced-learn Pipeline 組裝",
                "tags": ["Scikit-Learn 規範", "Python 進階語法與演算法"],
                "code": """def _get_classifier(model_name: str, imbalance: str, model_params: dict | None = None, seed: int = SEED):
    params = model_params.copy() if model_params is not None else {}
    if model_name == "M3":
        clf_params = {
            "n_estimators": 300,
            "random_state": seed,
            "n_jobs": -1,
            "class_weight": "balanced" if imbalance == "S1" else None,
        }
        clf_params.update(params)  # 若 Optuna 傳入新參數，直接覆蓋預設值！
        return ExtraTreesClassifier(**clf_params)

def build_pipeline(...) -> Pipeline:
    steps = [
        ("feature_selector", FeatureSelector(features=features)),
        ("drop_const_dup", DropConstantAndDuplicate()),
        ("winsorizer", Winsorizer(lower=0.01, upper=0.99, enabled=use_winsorizer)),
        ("large_value_indicator", LargeValueIndicator(enabled=use_large_indicator)),
        ("scaler", QuantileScaler(kind=scaler)),
        ("corr_filter", CorrelationFilter(threshold=0.95, enabled=use_corr_filter)),
    ]
    resampler = _get_resampler(imbalance, seed=seed)
    if resampler is not None:
        steps.append(("resampler", resampler))
    steps.append(("classifier", _get_classifier(model_name, imbalance, model_params, seed)))
    return Pipeline(steps)""",
                "what_it_does": (
                    "先建立分類器的預設參數字典 `clf_params`，若外部（如 Optuna 調參）有傳入 `model_params`，就用 `.update()` 覆蓋預設值，再用 `**clf_params` 解包實例化模型；"
                    "最後將所有前處理步驟、重抽樣器（若有）與分類器組裝為 `imblearn.pipeline.Pipeline`。"
                ),
                "why_written_this_way": (
                    "1. **為什麼一定要從 `imblearn.pipeline` 匯入 `Pipeline`，而不能用 `sklearn.pipeline.Pipeline`？**\n"
                    "因為 Scikit-Learn 原生的 `Pipeline` 不支援會改變樣本筆數 $N$ 的 `SMOTE`（原生轉換器只允許改變特徵欄數 $D$，不允許改變列數 $N$ 與標籤 $y$）！`imblearn.pipeline.Pipeline` 專門擴充了 `fit_resample(X, y)` 介面，且在呼叫 `predict_proba(X_val)` 時會**自動跳過 `resampler` 步驟**，確保驗證集跟測試集絕對不會被 SMOTE 污染！\n"
                    "2. `model_params.copy()`：先淺拷貝一份字典再操作，避免意外修改到呼叫端傳進來的外部字典（防止 Side Effect）。"
                ),
                "syntax_points": (
                    "• `dict.update(other_dict)`：將 `other_dict` 的鍵值對合併進來，若有相同 key 則以新值覆蓋舊值。\n"
                    "• `ExtraTreesClassifier(**clf_params)`：雙星號字典解包（Dictionary Unpacking），將 `{'n_estimators': 600, 'max_features': 0.3}` 自動展開為 `ExtraTreesClassifier(n_estimators=600, max_features=0.3)`。"
                ),
                "shape_and_type": "輸出：`imblearn.pipeline.Pipeline` 實例（包含 7～8 個具名步驟 `named_steps`）。",
            },
        ],
    },
    {
        "id": "M06",
        "code": "src/metrics.py",
        "title": "M06. 悲觀同分處理與向量化業務指標 (src/metrics.py)",
        "file_path": "bankruptcy/src/metrics.py",
        "category": "二、管線轉換與演算法核心層",
        "analogy": "公平公正且最嚴格的「裁判計分板」——遇到同分平手（Ties）時絕不給模型佔便宜，一律按最嚴格的悲觀標準計入誤警成本。",
        "purpose": (
            "實作專案核心業務指標 `alert_ratio_at_recall`（達到指定召回率需發出幾倍預警）與 `recall_at_alert_ratio`（固定預警預算下能抓出多少破產公司），"
            "並確保所有輸入陣列經過嚴格防呆檢查。"
        ),
        "calls_and_called_by": "被 `cv.py`、`tune.py`、`ensemble.py`、`thresholding.py`、`explain.py`、`final_eval.py` 呼叫。",
        "blocks": [
            {
                "block_title": "區塊 1：Fail-Fast 防禦性輸入驗證 (_validate_inputs)",
                "tags": ["防禦性設計與型別陷阱", "NumPy 向量化"],
                "code": """ArrayLike = Union[Sequence, np.ndarray, pd.Series]

def _validate_inputs(y_true: ArrayLike, score: ArrayLike) -> tuple[np.ndarray, np.ndarray, int, int]:
    if y_true is None or score is None:
        raise ValueError("y_true and score must not be None.")
    y_arr = np.asarray(y_true).ravel()
    s_arr = np.asarray(score).ravel()
    if len(y_arr) != len(s_arr):
        raise ValueError(f"Length mismatch: {len(y_arr)} vs {len(s_arr)}.")
    if np.isnan(s_arr).any():
        raise ValueError("score array contains NaN values.")
    unique_vals = np.unique(y_arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"y_true must contain only binary labels (0 or 1), got {unique_vals}.")
    p_count = int((y_arr == 1).sum())
    if p_count == 0:
        raise ValueError("y_true must contain at least one positive sample (y=1).")
    return y_arr.astype(int), s_arr.astype(float), p_count, len(y_arr)""",
                "what_it_does": (
                    "在計算任何指標前，統一把 `y_true` 與預測分數 `score` 轉為一維 NumPy 陣列，並依序檢查："
                    "是否為空、長度是否一致、分數是否有 `NaN`、標籤是否只有 `0` 和 `1`、以及是否至少有 1 家破產正樣本（避免除以 `P=0`）。"
                ),
                "why_written_this_way": (
                    "這體現了軟體工程的 **Fail-Fast（快速失敗）原則**：如果上游模型因為某種 Bug 輸出了 `NaN` 或長度不對的陣列，"
                    "與其讓它算出一個詭異的 `0.0` 混進報表裡，不如在進入指標計算的第一秒就拋出清楚說明原因的 `ValueError`，省下數小時的除錯時間！"
                ),
                "syntax_points": (
                    "• 函式名稱開頭單底線 `_validate_inputs`：Python 慣例代表「模組內部私有輔助函式（Internal Helper）」。\n"
                    "• `np.isin(unique_vals, [0, 1])`：向量化檢查陣列中的每個元素是否都屬於 `[0, 1]` 集合。"
                ),
                "shape_and_type": "輸入：任意序列 ➔ 輸出：`tuple(int64 ndarray (N,), float64 ndarray (N,), int P, int N)`。",
            },
            {
                "block_title": "區塊 2：悲觀同分預警倍數計算 (alert_ratio_at_recall)",
                "tags": ["NumPy 向量化", "Python 進階語法與演算法"],
                "code": """def alert_ratio_at_recall(y_true: ArrayLike, score: ArrayLike, target_recall: float) -> float:
    y_arr, s_arr, P, _ = _validate_inputs(y_true, score)

    # 1. 達到目標召回率至少需要抓到的正樣本整數個數: ceil(R * P)
    need = int(np.ceil(target_recall * P))
    need = max(1, min(need, P))

    # 2. 取出所有真實破產公司的分數，由高到低排序，找第 need 個人的分數當門檻
    pos_scores = np.sort(s_arr[y_arr == 1])[::-1]
    thr = pos_scores[need - 1]

    # 3. 悲觀同分原則：全體樣本中只要分數 >= thr 一律計入預警！
    flagged = int((s_arr >= thr).sum())
    return float(flagged / P)""",
                "what_it_does": (
                    "計算為了抓出至少 `target_recall`（如 95%）比例的破產公司，總共需要發出多少家預警（相對於真實破產數 `P` 的倍數）。"
                ),
                "why_written_this_way": (
                    "這 6 行程式碼濃縮了極高明的向量化思維：\n"
                    "1. 不用逐步掃描所有可能的門檻！因為要抓到 `need` 家破產公司，**最低限度的門檻 `thr` 必然就是「所有破產公司當中，分數排第 `need` 高的那一家的分數」**！\n"
                    "2. `s_arr >= thr`（包含等號 `>=`）：如果第 `need` 家破產公司的分數是 `0.50`，而剛好有 200 家正常公司也拿到同分 `0.50`（例如在 Isotonic Regression 階梯平段上），使用 `>=` 就會把這 200 家全部算進預警成本，嚴格懲罰產生同分的粗糙模型！"
                ),
                "syntax_points": (
                    "• `s_arr[y_arr == 1]`：布林遮罩索引（Boolean Indexing），只挑出真實破產樣本的分數。\n"
                    "• `np.sort(...)[::-1]`：`np.sort` 預設由小排到大，加上切片步長 `[::-1]` 瞬間反轉為**由大排到小（降序）**！\n"
                    "• `pos_scores[need - 1]`：因為 Python 索引從 `0` 開始，所以「第 `need` 高」對應的索引是 `need - 1`。"
                ),
                "shape_and_type": "`pos_scores: ndarray (P,)` ➔ `thr: float` ➔ `(s_arr >= thr): bool ndarray (N,)` ➔ 回傳 `float`（倍數 $\\ge 1.0$）。",
            },
        ],
    },
]


# ==============================================================================
# 12 Core Modules Encyclopedia (Part 2: M07 to M12)
# ==============================================================================
CODE_MODULES_PART2: list[dict[str, Any]] = [
    {
        "id": "M07",
        "code": "src/stats_utils.py",
        "title": "M07. 校正統計檢定與多重比較控制 (src/stats_utils.py)",
        "file_path": "bankruptcy/src/stats_utils.py",
        "category": "二、管線轉換與演算法核心層",
        "analogy": "科學實驗室的「防誇大審計員」——當研究人員宣稱新方法比較好時，負責扣除訓練資料重疊造成的假象（Nadeau-Bengio），並對同時測試多組模型的行為進行嚴格校正（Holm-Bonferroni）。",
        "purpose": (
            "提供重複交叉驗證專用的 `nadeau_bengio_paired_test` 與多重檢定校正 `holm_bonferroni`，確保專案遵守「無統計校正支持不得宣稱顯著」的科學鐵則。"
        ),
        "calls_and_called_by": "被 `run_imbalance.py`、`tune.py`、`ensemble.py`、`explain.py` 呼叫。",
        "blocks": [
            {
                "block_title": "區塊 1：Nadeau-Bengio 校正配對 t 檢定 (nadeau_bengio_paired_test)",
                "tags": ["NumPy 向量化", "防禦性設計與型別陷阱"],
                "code": """def nadeau_bengio_paired_test(
    diffs: Sequence[float] | np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    mean_d = float(np.mean(d))
    s2 = float(np.var(d, ddof=1))  # 樣本變異數必須除以 (k - 1)！
    c = (1.0 / k) + (1.0 / (n_splits - 1))  # 1/15 + 1/4 = 0.3167
    se_corrected = np.sqrt(max(0.0, c * s2))

    df = k - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_lower = mean_d - t_crit * se_corrected
    ci_upper = mean_d + t_crit * se_corrected

    if se_corrected < 1e-12:
        p_val = 1.0 if np.isclose(mean_d, 0.0) else 0.0
    else:
        t_stat = mean_d / se_corrected
        p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))
    return mean_d, float(ci_lower), float(ci_upper), float(p_val)""",
                "what_it_does": (
                    "接收兩個模型在 15 個外層驗證折上的指標差值陣列 `diffs`（長度 $k=15$），計算平均差異、Nadeau-Bengio 校正後的 95% 信賴區間 `[ci_lower, ci_upper]` 以及雙尾 $p$ 值。"
                ),
                "why_written_this_way": (
                    "1. **`np.var(d, ddof=1)` 的超級大坑**：NumPy 的 `np.var()` 預設 `ddof=0`（除以 $k$，母體變異數），而 Pandas 的 `.var()` 預設 `ddof=1`（除以 $k-1$，無偏樣本變異數）！在做 t 檢定時，公式要求的是無偏樣本變異數，所以在 NumPy 中**一定要手動指定 `ddof=1`**！\n"
                    "2. **`if se_corrected < 1e-12:`**：如果拿兩個一模一樣的模型去比，15 折差值全為 `0.0`，標準誤 `se_corrected` 會是 `0.0`，直接算 `mean_d / se_corrected` 會觸發 `0.0 / 0.0 = NaN`！加上 `1e-12` 門檻保護是數值計算的黃金習慣。"
                ),
                "syntax_points": (
                    "• `stats.t.ppf(0.975, df=14)`：Percent Point Function（累積分布反函數，即分位數函數），查出雙尾 $\\alpha=0.05$ 在自由度 14 下的臨界值 $t_{\\text{crit}} \\approx 2.1448$。\n"
                    "• `stats.t.cdf(|t|, df=14)`：Cumulative Distribution Function（累積機率函數），算出面積後用 `2.0 * (1.0 - cdf)` 得到雙尾 $p$ 值。"
                ),
                "shape_and_type": "輸入：`diffs: ndarray (15,)` ➔ 輸出：`tuple(mean_d, ci_lower, ci_upper, p_val)` 共 4 個 `float`。",
            },
            {
                "block_title": "區塊 2：Holm-Bonferroni 逐步校正與反向索引還原 (holm_bonferroni)",
                "tags": ["NumPy 向量化", "Python 進階語法與演算法"],
                "code": """def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    adjusted = np.empty(m, dtype=float)
    cum_max = 0.0
    for i in range(m):
        adj_p = min(1.0, (m - i) * sorted_p[i])
        cum_max = max(cum_max, adj_p)  # 強制維持單調遞增 (Step-down monotonicity)
        adjusted[i] = cum_max

    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return [float(x) for x in out]""",
                "what_it_does": (
                    "當一次進行 $m$ 組統計檢定時，將 $m$ 個原始 $p$ 值由小排到大，第 $i$ 小的 $p$ 值乘上 $(m - i)$ 倍，並取累積最大值確保校正後 $p$ 值不會倒退，最後還原回原本輸入的順序。"
                ),
                "why_written_this_way": (
                    "Holm-Bonferroni 演算法規定必須先按 $p$ 值由小到大排序計算 $(m-i)p_{(i)}$，但呼叫端傳入 `p_values` 時通常是按照 `[M1, M2, M3, M4, M5]` 的順序。"
                    "透過 `order = np.argsort(p_arr)` 搭配 `out[order] = adjusted`，不必更動呼叫端的表格順序就能精準對位回傳！"
                ),
                "syntax_points": (
                    "• `np.empty(m, dtype=float)`：當我們確定馬上會把陣列每一格都填滿時，用 `np.empty` 比 `np.zeros` 更快，因為它直接配置記憶體而不花時間先把每一格填 0。\n"
                    "• `cum_max = max(cum_max, adj_p)`：利用單一變數在迴圈中實作累積最大值（等同於 `np.maximum.accumulate`）。"
                ),
                "shape_and_type": "輸入：`Sequence[float]` 長度 $m$ ➔ 輸出：`list[float]` 長度 $m$（值域 $[0.0, 1.0]$）。",
            },
        ],
    },
    {
        "id": "M08",
        "code": "src/tune.py",
        "title": "M08. Optuna 巢狀超參數調優與 SQLite 斷點續跑 (src/tune.py)",
        "file_path": "bankruptcy/src/tune.py",
        "category": "三、實驗引擎與可解釋性層",
        "analogy": "全自動化的「AI 配方研發實驗室」——外層 15 間獨立考場評估實力，內層由 Optuna 貝氏搜尋引擎自動調配超參數，且每試完一組配方就立刻寫入 SQLite 資料庫，停電重開也能接著跑。",
        "purpose": (
            "執行 E2a（15 折外層 × 內層 CV 的無偏巢狀調參評估）與 E2b（全訓練集 10 折 CV 的最終部署超參數固化），"
            "以三點 log 預警損失作為優化目標。"
        ),
        "calls_and_called_by": "匯入 `optuna`、`cv.py`、`pipelines.py`、`metrics.py`；產出 `outputs/models/best_params/M1~M5.json`。",
        "blocks": [
            {
                "block_title": "區塊 1：Optuna 閉包目標函式 (Closure Objective) 與內層 OOF 拼接",
                "tags": ["Python 進階語法與演算法", "Scikit-Learn 規範"],
                "code": """def make_objective(model_name: str, X_outer_tr: pd.DataFrame, y_outer_tr: pd.Series, inner_splits: list[dict]):
    def objective(trial: optuna.Trial) -> float:
        imbalance, model_params = sample_hyperparameters(trial, model_name)
        oof_scores = np.zeros(len(X_outer_tr), dtype=float)

        for isp in inner_splits:
            in_tr = isp["train_idx"]
            in_val = isp["val_idx"]
            pipe = build_pipeline(
                model_name=model_name,
                imbalance=imbalance,
                scaler="quantile",
                use_winsorizer=False,
                use_corr_filter=True,
                model_params=model_params,
            )
            pipe.fit(X_outer_tr.iloc[in_tr], y_outer_tr.iloc[in_tr])
            oof_scores[in_val] = get_score(pipe, X_outer_tr.iloc[in_val])

        return compute_three_point_log_loss(y_outer_tr.values, oof_scores)
    return objective""",
                "what_it_does": (
                    "定義一個工廠函式 `make_objective`，內部包著傳給 Optuna 的 `objective(trial)`："
                    "每次 Trial 抽樣一組超參數，在內層各折訓練並預測驗證折，將全體內層預測拼成完整的 `oof_scores` 後，計算三點 log 預警倍數損失回傳給 Optuna 最小化。"
                ),
                "why_written_this_way": (
                    "1. **為什麼要用「函式裡面包函式（Closure 閉包）」？**\n"
                    "因為 `study.optimize(objective, n_trials=40)` 規定 `objective` 只能接收 `trial` 一個參數！透過外層 `make_objective(...)` 把當前的 `X_outer_tr`、`y_outer_tr` 捕獲（Capture）在閉包作用域內，內層 `objective(trial)` 就能直接存取它們，完全不需要動用骯髒的全域變數（Global Variables）！\n"
                    "2. **為什麼內層要把 OOF 拼起來算一次 Loss，而不是每折算完取平均？** 因為內層單折正樣本更少，拼成完整內層 OOF（約 141 個破產樣本）再取分位數切點，能大幅降低單折階梯離散化噪音。"
                ),
                "syntax_points": (
                    "• **Python 閉包（Closure）**：內層函式 `objective` 記住了外層函式 `make_objective` 的區域變數環境，是機器學習調參腳本最乾淨的設計模式。\n"
                    "• `oof_scores[in_val] = ...`：利用索引陣列將各驗證折的預測精準填入對應的樣本格子中。"
                ),
                "shape_and_type": "`X_outer_tr: DataFrame (4364, 95)` ➔ `oof_scores: ndarray (4364,)` ➔ 回傳 `loss: float`。",
            },
            {
                "block_title": "區塊 2：SQLite 持久化與斷點續跑 (Checkpointing & Resumability)",
                "tags": ["檔案與跨平台 I/O", "防禦性設計與型別陷阱"],
                "code": """db_path = STUDIES_DIR / f"study_{model_name}_{split_id}.db"
storage_url = f"sqlite:///{db_path.as_posix()}"
sampler = optuna.samplers.TPESampler(seed=study_seed)

study = optuna.create_study(
    study_name=f"{model_name}_{split_id}",
    direction="minimize",
    sampler=sampler,
    storage=storage_url,
    load_if_exists=True,
)
completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining = max(0, n_trials - completed_trials)
if remaining > 0:
    study.optimize(objective, n_trials=remaining)""",
                "what_it_does": (
                    "將每個模型、每一折的 Optuna 搜尋紀錄綁定到本地 SQLite 資料庫檔案 `.db`，"
                    "啟動時先檢查已經完成了幾個 Trial（`completed_trials`），只補跑剩下的次數 `remaining`！"
                ),
                "why_written_this_way": (
                    "跑完 5 個模型 × 15 個外層折 × 40 個 Trial 需要數千次模型擬合。如果不用 `storage=sqlite:///...` 與 `load_if_exists=True`，萬一跑到 95% 時電腦重開機，前面幾個小時的心血就全部歸零！"
                ),
                "syntax_points": (
                    "• `db_path.as_posix()`：SQLite 的 URL 格式要求即使是 Windows 系統也必須使用正斜線 `/`（例如 `sqlite:///C:/Users/...`），`.as_posix()` 能強制將 Windows 路徑轉為正斜線字串！"
                ),
                "shape_and_type": "`study: optuna.Study`（持久化於磁碟 SQLite 檔案）。",
            },
        ],
    },
    {
        "id": "M09",
        "code": "src/ensemble.py & thresholding.py",
        "title": "M09. ECDF 百分位集成、安全邊際與機率校準 (ensemble.py & thresholding.py)",
        "file_path": "bankruptcy/src/thresholding.py",
        "category": "三、實驗引擎與可解釋性層",
        "analogy": "銀行風控委員會的「總決策長」——把三位專家（M3, M4, M5）不同標準的評分換算成統一的百分位排名，加上安全緩衝邊際（Safety Margin）定出警戒線，並附上保序校準機率。",
        "purpose": (
            "實作 E3（模型成對檢定、ECDF 百分位集成、難抓樣本診斷）與 E4（跨折門檻轉移評估、Platt vs Isotonic 機率校準、以及 `outputs/models/final/` 全套上線元件固化）。"
        ),
        "calls_and_called_by": "讀取 `outputs/models/oof/` 與 `best_params/`；寫出最終固化模型元件至 `outputs/models/final/`。",
        "blocks": [
            {
                "block_title": "區塊 1：字典生成式建立 O(1) 原始列號至訓練集相對位置查找表",
                "tags": ["Python 進階語法與演算法", "NumPy 向量化"],
                "code": """splits_payload = load_splits()
train_orig_idx = splits_payload["train_idx"]  # 長度 5455，裝著原始 data.csv 的列號 (0~6818)

# 建立「原始 data.csv 列號 -> 訓練集內部第幾列 (0~5454)」的 O(1) 查找字典
orig_to_rel = {orig: i for i, orig in enumerate(train_orig_idx)}

for s in selected_splits:
    out_tr_idx = np.array([orig_to_rel[i] for i in s["train_idx"]])
    out_val_idx = np.array([orig_to_rel[i] for i in s["val_idx"]])""",
                "what_it_does": (
                    "將 `splits.joblib` 中記錄的「原始 `data.csv` 列號（0～6818 中抽出的 5,455 個不連續整數）」，"
                    "透過字典生成式 `orig_to_rel` 快速轉換為「訓練集矩陣 `X_train` 內部的連續列號（0～5454）」。"
                ),
                "why_written_this_way": (
                    "這是處理多層資料切分最容易出 Bug 的地方！因為 `X_train` 已經切掉了 20% 測試集，所以它的長度只有 5,455（`.iloc` 索引為 `0 ~ 5454`）；"
                    "但 `s['train_idx']` 存的是原始 `data.csv` 的列號（最大到 `6818`）。如果直接寫 `X_train.iloc[s['train_idx']]` 就會觸發 `IndexError: positional indexers are out-of-bounds`！"
                    "用 `{orig: i for i, orig in enumerate(train_orig_idx)}` 建立雜湊查找表，轉換一個索引只需 $O(1)$ 時間。"
                ),
                "syntax_points": (
                    "• `{k: v for ...}`：字典生成式（Dictionary Comprehension），一行建立鍵值反查表，比先建空字典再跑 `for` 迴圈快且清晰。"
                ),
                "shape_and_type": "`train_orig_idx: ndarray (5455,)` ➔ `orig_to_rel: dict[int, int]` (5455 keys)。",
            },
            {
                "block_title": "區塊 2：安全邊際門檻計算與 Platt / Isotonic 雙校準器擬合",
                "tags": ["Scikit-Learn 規範", "NumPy 向量化"],
                "code": """# 1. 根據目標召回率 R 與安全邊際 m 決定有效目標 R_eff = min(1.0, R + m)
eff_r = min(1.0, float(target_r + margin))
pos_scores = np.sort(ref_scores[ref_y == 1])[::-1]
P = len(pos_scores)
need = max(1, min(int(np.ceil(eff_r * P)), P))
thr = float(pos_scores[need - 1])

# 2. Platt Scaling (Sigmoid 單調校準) 與 Isotonic (階梯保序校準)
cal_platt = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
cal_platt.fit(oof_ea.reshape(-1, 1), y_train_arr)

cal_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
cal_iso.fit(oof_ea, y_train_arr)""",
                "what_it_does": (
                    "1. 將目標召回率加上安全邊際 $m$ 後，在訓練集 10 折 OOF 正樣本分數中找出對應的切點 `thr` 寫入 `thresholds.json`。\n"
                    "2. 在同一份 10 折 OOF 集成百分位分數 `oof_ea` 上，分別訓練 Platt Scaling（無正則化懲罰 `C=1e6` 之單變數羅吉斯迴歸）與 Isotonic Regression（保序迴歸並設定超出範圍自動截斷 `out_of_bounds='clip'`）。"
                ),
                "why_written_this_way": (
                    "**為什麼 `cal_platt.fit` 要寫 `oof_ea.reshape(-1, 1)`，而 `cal_iso.fit` 卻直接傳一維 `oof_ea`？**\n"
                    "這是初學者使用 Scikit-Learn 最常遇到的維度報錯（`ValueError: Expected 2D array, got 1D array instead`）！\n"
                    "因為 `LogisticRegression` 是通用多特徵模型，規定輸入 `X` 必須是二維矩陣 `(樣本數 N, 特徵數 D)`——當只有 1 個特徵時，必須用 `.reshape(-1, 1)` 把 `(5455,)` 轉成 `(5455, 1)`；而 `IsotonicRegression` 專門設計給單一變數校準，所以接受一維 `(5455,)`！"
                ),
                "syntax_points": (
                    "• `arr.reshape(-1, 1)`：`-1` 代表「請 NumPy 自動幫我算這一維是多少（即總長度 5455）」，`1` 代表第二維固定為 1 欄。這是把 1D 向量轉為 2D 直欄矩陣的標準寫法！"
                ),
                "shape_and_type": "`oof_ea: ndarray (5455,)` ➔ `.reshape(-1, 1)` 變為 `ndarray (5455, 1)`。",
            },
        ],
    },
    {
        "id": "M10",
        "code": "src/explain.py",
        "title": "M10. TreeSHAP 可解釋性與共線家族聚類 (src/explain.py)",
        "file_path": "bankruptcy/src/explain.py",
        "category": "三、實驗引擎與可解釋性層",
        "analogy": "模型的「X 光透視與解剖中心」——把黑盒子樹模型拆開，用賽局理論算出每個財務指標對每家公司破產風險的推力（SHAP），並測試拿掉特徵後模型會退步多少。",
        "purpose": (
            "執行 E5a（15 折外層置換重要性、階層式共線群組重要性與 TreeExplainer SHAP 分析）與 E5b（無洩漏巢狀 Top-$k$ 精簡特徵消融與核心特徵集篩選）。"
        ),
        "calls_and_called_by": "匯入 `shap`、`scipy.cluster.hierarchy`；產出 `e5a_*.csv`、`e5b_*.csv` 與 `core_features*.txt`。",
        "blocks": [
            {
                "block_title": "區塊 1：拆解 Pipeline 前處理與底層樹模型以計算 TreeSHAP",
                "tags": ["Scikit-Learn 規範", "Python 進階語法與演算法"],
                "code": """# 1. 逐步執行 Pipeline 中除了最後一層 classifier 與 resampler 以外的前處理步驟
X_val_trans = X_val.copy()
for step_name, transformer in fitted_pipe.steps[:-1]:
    if step_name == "resampler":
        continue
    X_val_trans = transformer.transform(X_val_trans)

# 2. 取出底層樹模型（若被 AdaptiveScaleWeightClassifier 包覆則解開 .estimator_）
clf = fitted_pipe.named_steps["classifier"]
raw_tree_model = clf.estimator_ if hasattr(clf, "estimator_") else clf

# 3. 執行 TreeExplainer 計算每個特徵對每筆樣本的 SHAP 貢獻值
explainer = shap.TreeExplainer(raw_tree_model)
shap_vals = explainer.shap_values(X_val_trans)""",
                "what_it_does": (
                    "先將驗證折原始 95 欄資料手動通過已訓練 Pipeline 的前處理層，得到實際餵給樹模型的 72 欄標準常態化 DataFrame `X_val_trans`；"
                    "再取出最底層的樹模型交給 `shap.TreeExplainer` 計算精確的 SHAP 矩陣。"
                ),
                "why_written_this_way": (
                    "`shap.TreeExplainer` 是專為決策樹設計的 C++ 高速解析演算法，它只能解析樹結構（`ExtraTrees`, `XGBoost`, `LightGBM`），看不懂外層包著的 `QuantileScaler` 或 `CorrelationFilter`！"
                    "因此我們必須用 `fitted_pipe.steps[:-1]` 切片取出所有前處理器，先把資料轉換成 72 欄，再將拆封後的 `raw_tree_model` 與 72 欄 `X_val_trans` 配對送入 `TreeExplainer`。"
                ),
                "syntax_points": (
                    "• `fitted_pipe.steps[:-1]`：列表切片，`[:-1]` 代表「從頭取到倒數第二個元素（即排除最後一個步驟 `classifier`）」。\n"
                    "• 三元條件運算式 `a if condition else b`：一行判斷是否需要解開自訂 Wrapper 取出內部的 `estimator_`。"
                ),
                "shape_and_type": "`X_val: DataFrame (N_val, 95)` ➔ `X_val_trans: DataFrame (N_val, 72)` ➔ `shap_vals: ndarray (N_val, 72)`。",
            },
        ],
    },
    {
        "id": "M11",
        "code": "src/scoring.py & score_new_data.py",
        "title": "M11. 確定性推論引擎與 CLI 命令列工具 (scoring.py & score_new_data.py)",
        "file_path": "bankruptcy/src/scoring.py",
        "category": "四、部署推論與品質保證層",
        "analogy": "正式上線的「全自動風險評分櫃檯」——載入凍結封存的模型與參考分布，對新送來的企業財報進行毫秒級確定性評分，並提供命令列介面（CLI）供外部系統串接。",
        "purpose": (
            "將訓練與實驗邏輯徹底抽離，提供純粹、無狀態、100% 可重現的推論函式庫 `scoring.py` 以及支援 `--input` / `--output` 的命令列工具 `score_new_data.py`。"
        ),
        "calls_and_called_by": "被 `final_eval.py` (E6)、`app/logic.py` (Streamlit 互動介面) 與外部使用者直接呼叫。",
        "blocks": [
            {
                "block_title": "區塊 1：強制單執行緒確定性推論與端到端評分 (score_dataset)",
                "tags": ["NumPy 向量化", "防禦性設計與型別陷阱"],
                "code": """raw_scores = {}
pct_scores = {}
for m in model_names:
    pipe = components["models"][m]
    clf = pipe.named_steps.get("classifier")
    if clf is not None and hasattr(clf, "n_jobs"):
        clf.set_params(n_jobs=1)  # 強制單執行緒，消除多核浮點數加總微差！
    raw = pipe.predict_proba(X)[:, 1]
    raw_scores[m] = raw
    pct_scores[m] = to_percentile(raw, components["references"][m])

# 計算未校準 ECDF 百分位等權平均
ea_scores = np.mean([pct_scores[m] for m in model_names], axis=0)

# 獨立套用 Platt 校準器與工作點門檻
calibrated_platt = cal_platt.predict_proba(ea_scores.reshape(-1, 1))[:, 1]""",
                "what_it_does": (
                    "依序呼叫 M3、M4、M5 三個固化 Pipeline 預測正類機率 `[:, 1]`，透過 `to_percentile` 映射為經驗百分位後取縱軸平均 `np.mean(..., axis=0)` 得到 `ea_scores`，"
                    "最後計算 Platt 校準機率與 R80～R100 四個預警布林旗標。"
                ),
                "why_written_this_way": (
                    "**為什麼推論時要特別寫 `clf.set_params(n_jobs=1)`？**\n"
                    "在電腦科學中，浮點數加法是不滿足嚴格結合律的（`(a + b) + c` 與 `a + (b + c)` 在小數點後第 16 位可能不同）。"
                    "當隨機森林或 GBDT 使用多執行緒（`n_jobs=-1`）平行加總 600 棵樹的預測值時，作業系統排程順序的微小變化可能造成 `1e-16` 的浮點數微差。"
                    "強制設為 `n_jobs=1` 保證在任何電腦、跑一萬次，每個位元（Bitwise）都 100% 完全一致！"
                ),
                "syntax_points": (
                    "• `pipe.predict_proba(X)[:, 1]`：`predict_proba` 回傳形狀 `(N, 2)` 的矩陣，第 0 欄是「正常機率 $P(y=0)$」，第 1 欄是「破產機率 $P(y=1)$」；用 `[:, 1]` 取出所有樣本的第 1 欄。\n"
                    "• `np.mean([arr1, arr2, arr3], axis=0)`：沿著第 0 軸（跨模型維度）對齊樣本取平均，輸出仍為長度 $N$ 的一維陣列。"
                ),
                "shape_and_type": "輸入：`X: DataFrame (N, 95)` ➔ `raw: ndarray (N,)` ➔ `ea_scores: ndarray (N,)` ➔ 輸出：`results_df: DataFrame (N, 14)`。",
            },
            {
                "block_title": "區塊 2：CLI 命令列參數解析與欄位空白消毒 (score_new_data.py)",
                "tags": ["CLI 與自動化測試", "檔案與跨平台 I/O"],
                "code": """df = pd.read_csv(in_path)
df.columns = df.columns.str.strip()  # 自動清除欄位名稱前後隱藏空白！

missing_features = [f for f in expected_features if f not in df.columns]
if missing_features:
    raise ValueError(f"Input data is missing {len(missing_features)} required feature(s): {missing_features}")

extra_cols = [c for c in df.columns if c not in expected_features and c != "Bankrupt?"]
...
parser = argparse.ArgumentParser(description="Score new data with frozen bankruptcy prediction models.")
parser.add_argument("--input", required=True, type=str, help="Path to input CSV.")
parser.add_argument("--output", required=True, type=str, help="Path to output CSV.")
parser.add_argument("--operating-point", default="R90", choices=["R80", "R90", "R95", "R100"])""",
                "what_it_does": (
                    "1. 讀取使用者提供的 CSV 檔，並用 `.str.strip()` 自動把所有欄位名稱頭尾的空白字元清除。\n"
                    "2. 檢查 95 個必要特徵是否齊全；若有額外欄位（如 `company_id`, `company_name`），自動保留並放在輸出 CSV 的最前面方便業務對照！\n"
                    "3. 使用標準庫 `argparse` 打造支援 `--input`, `--output`, `--operating-point` 的命令列工具。"
                ),
                "why_written_this_way": (
                    "**為什麼 `df.columns = df.columns.str.strip()` 這一行價值連城？**\n"
                    "因為原始 UCI / Kaggle 破產資料集（`data.csv`）的欄位名稱前面其實都帶有一個看不見的空格（例如 `' Debt ratio %'` 而不是 `'Debt ratio %'`）！"
                    "用 `.str.strip()` 統一消毒欄位名稱，讓使用者不論傳入有沒有帶空格的 CSV 都能正常運作。"
                ),
                "syntax_points": (
                    "• `df.columns.str.strip()`：Pandas 的向量化字串方法（String Accessor `.str`），一次對全部 95 個欄名去除首尾空白。\n"
                    "• `choices=['R80', 'R90', 'R95', 'R100']`：`argparse` 內建白名單驗證，若使用者打錯成 `--operating-point R99`，命令列會自動攔截並印出合法選項提示。"
                ),
                "shape_and_type": "CLI 輸入：`--input xxx.csv` ➔ 輸出：純淨 `output.csv` + 同名方法學說明檔 `output.README.txt`。",
            },
        ],
    },
    {
        "id": "M12",
        "code": "app/logic.py & tests/",
        "title": "M12. 互動介面純邏輯解耦與 Pytest 單元測試 (app/logic.py & tests/)",
        "file_path": "bankruptcy/app/logic.py",
        "category": "四、部署推論與品質保證層",
        "analogy": "大樓的「結構安全耐震測試中心」——把網頁畫面（UI）跟背後的數學計算（Logic）徹底分開，並寫了 64 道自動化考題（Pytest），每次改程式碼都自動驗算有沒有算錯或洩漏。",
        "purpose": (
            "1. `app/logic.py`：將 Streamlit 互動示範系統的所有數值計算（單特徵替換效應 Delta Drop / Delta Add、分位數內插、What-If 掃描曲線）抽離為不依賴 `streamlit` 的純函式。\n"
            "2. `tests/`：提供 64 項自動化測試，驗證無洩漏轉換、悲觀同分指標、凍結檔案雜湊與 UI 邏輯。"
        ),
        "calls_and_called_by": "`app/app.py` 呼叫 `app/logic.py`；`pytest` 執行 `tests/test_*.py`。",
        "blocks": [
            {
                "block_title": "區塊 1：單特徵邊際替換效應批量向量化計算 (compute_replacement_effects)",
                "tags": ["NumPy 向量化", "Python 進階語法與演算法"],
                "code": """# 為什麼不用 for 迴圈呼叫 18 次 score_rows()，而是拼成一個 (2*K + 2) 列的 DataFrame 一次算完？
rows_list = [
    user_row_df.iloc[0].to_dict(),    # Row 0: 使用者目前輸入輪廓
    median_row_df.iloc[0].to_dict(),  # Row 1: 全中位數基準輪廓
]
for feat in active_features:
    # Drop row: 把使用者的該特徵換成中位數 (看分數降多少)
    r_drop = user_row_df.iloc[0].to_dict()
    r_drop[feat] = float(medians[feat])
    rows_list.append(r_drop)

    # Add row: 從全中位數出發，只把該特徵換成使用者的值 (看分數升多少)
    r_add = median_row_df.iloc[0].to_dict()
    r_add[feat] = float(user_inputs[feat])
    rows_list.append(r_add)

batch_df = pd.DataFrame(rows_list, columns=raw_columns)
scored_batch = score_rows(batch_df, components)  # 一次性批量推論！""",
                "what_it_does": (
                    "為了分析使用者調整的 $K$ 個特徵各自對風險分數造成了多少影響（Delta Drop 與 Delta Add），"
                    "將「使用者輪廓」、「全中位數輪廓」以及 $2K$ 個替換後的虛擬輪廓拼成一張 $(2K+2)$ 列的表格 `batch_df`，只呼叫一次 `score_rows()` 就瞬間算完所有比較！"
                ),
                "why_written_this_way": (
                    "在 Scikit-Learn / XGBoost / LightGBM 中，每次呼叫 `.predict_proba()` 都有固定的 Python 函式與 C++ 轉換開銷（Overhead）。"
                    "對 18 個特徵跑迴圈呼叫 38 次單筆 `predict_proba` 需要約 1.5 秒（網頁會覺得卡頓）；"
                    "拼成一個 38 列的 `batch_df` **一次批量推論（Batch Inference）只需 0.04 秒（快 35 倍）**！這是打造流暢互動式 ML App 的關鍵秘訣。"
                ),
                "syntax_points": (
                    "• `pd.DataFrame(rows_list, columns=raw_columns)`：將多個字典組成的 `list[dict]` 一次性轉為 DataFrame，比在迴圈裡用 `pd.concat` 快非常多！"
                ),
                "shape_and_type": "`rows_list: list[dict]` 長度 $2K+2$ ➔ `batch_df: DataFrame (2K+2, 95)` ➔ 一次呼叫 `score_rows`。",
            },
            {
                "block_title": "區塊 2：用 Pytest 自動化測試證明「零資料洩漏」(test_preprocessing.py)",
                "tags": ["CLI 與自動化測試", "Scikit-Learn 規範"],
                "code": """def test_winsorizer_leakage_and_invariance():
    df_a = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    winsor = Winsorizer(lower=0.10, upper=0.90, enabled=True)
    winsor.fit(df_a)

    expected_lower = df_a["feat"].quantile(0.10)  # 1.9
    expected_upper = df_a["feat"].quantile(0.90)  # 9.1

    # 構造含有極端離群值 (-999999, +999999) 的測試集 B
    df_b = pd.DataFrame({"feat": [-999999.0, 5.0, 999999.0]})
    b_trans = winsor.transform(df_b)

    # 斷言 B 被嚴格截斷在 A 的邊界 (1.9 與 9.1)，證明 transform 絕對沒有偷看 B！
    assert b_trans["feat"].iloc[0] == pytest.approx(expected_lower)
    assert b_trans["feat"].iloc[2] == pytest.approx(expected_upper)""",
                "what_it_does": (
                    "在極小的人造資料集 `df_a`（1～10）上訓練 `Winsorizer`，然後故意餵入含有 `-999999.0` 與 `999999.0` 的測試集 `df_b` 呼叫 `.transform(df_b)`，"
                    "用 `assert` 自動檢查轉換結果是否精確停在 `df_a` 的上下界 `1.9` 與 `9.1`。"
                ),
                "why_written_this_way": (
                    "初學者常問：「怎麼知道自己寫的轉換器有沒有不小心偷看到測試集？」\n"
                    "答案就是寫一個 **「Fit on A, Transform Extreme B」單元測試**！如果 `transform(df_b)` 內部有任何一行程式碼偷算了 `df_b` 的分位數，那 `-999999` 就不會被截在 `1.9`，這道 `pytest` 測試就會立刻亮紅燈警報！"
                ),
                "syntax_points": (
                    "• `pytest.approx(expected_lower)`：在電腦中，`0.1 + 0.2 == 0.3` 其實是 `False`（因為二進位浮點數精度問題 `0.30000000000000004`）！在寫單元測試比較浮點數時，永遠要用 `pytest.approx()` 或 `np.isclose()`，而不能直接用 `==`。"
                ),
                "shape_and_type": "`pytest` 測試函式以 `test_` 開頭，無需回傳值，透過 `assert` 斷言驗證行為。",
            },
        ],
    },
]

CODE_MODULES_ENCYCLOPEDIA: list[dict[str, Any]] = CODE_MODULES_PART1 + CODE_MODULES_PART2
