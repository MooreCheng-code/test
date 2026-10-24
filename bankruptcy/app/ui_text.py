"""
app/ui_text.py

Authoritative text definitions and templates for the Streamlit application.
Strictly decoupled from logic: contains no hardcoded numeric thresholds, decimals,
percentages, or prevalence constants in text constants.
All numeric values are dynamically formatted through functions from data and metadata.
"""

from __future__ import annotations

# ==============================================================================
# Page metadata and layout
# ==============================================================================
PAGE_TITLE = "企業破產早期預警模型互動示範"
PAGE_ICON = "📊"

APP_HEADER_TITLE = "企業破產早期預警模型互動示範系統"
APP_HEADER_SUBTITLE = (
    "基於機器學習集成（ExtraTrees + XGBoost + LightGBM 之經驗累積分布百分位等權平均）"
    "之互動式風險評估與模型運作展示介面"
)

TAB_LABELS = [
    "輸入與結果",
    "為什麼",
    "模型如何運作",
    "限制與免責",
]

# ==============================================================================
# Tab 1: Input & Results
# ==============================================================================
INPUT_MODE_LABEL = "輸入數值模式選擇"
MODE_SLIDER_LABEL = "訓練分位數滑桿（建議）"
MODE_EXACT_LABEL = "精確數值輸入"

PRESET_BUTTON_NORMAL = "套用典型正常公司輪廓"
PRESET_BUTTON_BANKRUPT = "套用訓練中破產公司的典型輪廓"
PRESET_BUTTON_RESET = "重設為訓練中位數"

VIRTUAL_PROFILE_NOTICE = (
    "註：預設輪廓範例為全體訓練資料依破產標籤彙總計算之中位數虛擬輪廓，並非任何一家真實企業。"
)

EXPANDER_MAIN_TITLE = "主要財務比率輸入（依平均特徵重要性排名前列，預設展開）"
EXPANDER_ADVANCED_TITLE = "進階財務比率輸入（其餘核心輸入特徵，預設收合）"

FEATURE_NOTE_SUFFIX = "（依欄位名稱，定義待確認）"

EXPANDER_FILL_TITLE = "補值說明：未手動調整特徵之填補方式與限制"

RESULTS_SECTION_TITLE = "模型評估與風險預警判定"
RESULTS_CALCULATE_BTN = "計算破產風險分數與工作點判定"

TABLE_OPERATING_POINTS_TITLE = "各預警工作點門檻與交叉驗證歷史營運表現"

CALIBRATED_PROB_TITLE = "Platt 校準後驗機率"


def format_scale_warning(feature_name: str) -> str:
    return (
        f"特徵【{feature_name}】在訓練資料中存在跨越多個數量級的編碼或極端值，"
        "建議優先使用分位數滑桿進行操作；若切換至精確數值模式，請先核對單位與尺度。"
    )


def format_fill_explanation(n_fill_columns: int) -> str:
    return (
        f"本互動展示僅要求使用者輸入核心特徵；其餘 {n_fill_columns} 個模型實際採用之財務比率，"
        "系統預設以訓練集之中位數自動填補。\n\n"
        "【補值限制說明】：此設計代表『假設其他未輸入特徵皆處於訓練資料中位數』時的評估結果。"
        "用中位數補其他特徵可能與使用者調整的輸入在財務上不完全一致（例如相關的獲利指標）；"
        "『相似公司（k 近鄰）補值』列為第二階段選項，第一版不做。"
    )


def format_score_summary(score_sentence: str, share_ge: float) -> str:
    pct = share_ge * 100.0
    return (
        f"**{score_sentence}**\n\n"
        f"在訓練資料參考分布中，約有 **{pct:.2f}%** 的樣本風險分數大於或等於此數值。"
    )


def format_platt_note(prob: float, base_rate: float) -> str:
    prob_pct = prob * 100.0
    base_pct = base_rate * 100.0
    return (
        f"校準機率：**{prob_pct:.2f}%**\n\n"
        f"（註：此機率反映訓練資料約 {base_pct:.2f}% 的破產率，"
        "只適用於與訓練資料相近的情境，不代表真實世界的破產機率。）"
    )


def format_range_warning_message(feature: str, val: float, p01: float, p99: float) -> str:
    return (
        f"【常見範圍警示】特徵 {feature} 目前輸入值（{val:.4g}）已超出訓練資料常見範圍 "
        f"[p01: {p01:.4g}, p99: {p99:.4g}]。"
    )


def format_out_of_range_message(feature: str, val: float, f_min: float, f_max: float) -> str:
    return (
        f"【全距超出警示】特徵 {feature} 目前輸入值（{val:.4g}）已超出訓練資料全距 "
        f"[min: {f_min:.4g}, max: {f_max:.4g}]，超出訓練資料範圍，結果不可靠。"
    )


# ==============================================================================
# Tab 2: Why (Placeholders and Explanations for Phase E)
# ==============================================================================
TAB2_TITLE = "特徵影響與可解釋性分析"

TAB2_EFFECTS_TITLE = "單項特徵影響分解（Single-feature Replacement Effects）"
TAB2_EFFECTS_RADIO_LABEL = "分析視角切換"
TAB2_EFFECTS_RADIO_DROP = "Delta Drop（把該特徵換成訓練中位數後，分數下降多少）"
TAB2_EFFECTS_RADIO_ADD = "Delta Add（從中位數輪廓出發，只換上該特徵目前的值，分數上升多少）"

TAB2_EFFECTS_EXPLANATION = (
    "**指標說明：**\n"
    "- **Delta Drop**：把該特徵換成訓練中位數後，分數下降多少；正值表示目前這個值讓風險高於中位數情況。\n"
    "- **Delta Add**：從中位數輪廓出發，只換上該特徵目前的值，分數上升多少。\n"
    "- **注意事項**：兩者描述的是模型行為，不是因果；單項影響不會加總成總分；分數接近飽和時 Delta Drop 會偏小。"
)

TAB2_POSITION_TITLE = "特徵相對位置分布（Position Table）"
TAB2_POSITION_EXPLANATION = (
    "呈現各輸入特徵在訓練資料分布中的百分位落點（以訓練集分位數網格線性內插計算）。"
)

TAB2_WHATIF_TITLE = "假設分析動態掃描曲線（What-If Curve）"
TAB2_WHATIF_SELECT_FEATURE = "選擇欲進行敏感度掃描的財務特徵"
TAB2_WHATIF_GRID_LABEL = "掃描網格形式"
TAB2_WHATIF_GRID_PCT = "訓練分位數等距網格（1～99 分位，推薦）"
TAB2_WHATIF_GRID_VAL = "數值線性等距網格（p01～p99 數值等距）"

TAB2_WHATIF_EXPLANATION = (
    "固定其餘所有輸入特徵不變，沿著指定財務特徵之訓練分布掃描評估分數走勢。\n\n"
    "圖中水平虛線代表四個預警工作點門檻，紅色圓點代表您目前的輸入落點。\n"
    "**註**：樹模型的曲線呈階梯狀，可能出現微小回落。"
)


# ==============================================================================
# Tab 3: How the model works
# ==============================================================================
TAB3_TITLE = "模型架構、決策邏輯與營運成本剖析"


def format_tab3_architecture(n_raw: int, n_model: int) -> str:
    return (
        "### 1. 資料處理與前處理架構\n"
        f"- **資料特性**：包含 {n_raw} 個財務特徵欄位，破產類別具高不平衡性。\n"
        "- **交叉驗證切分**：切分固定，前處理只在訓練折內學習。\n"
        "- **前處理管線**：\n"
        "  1. `FeatureSelector`：檢核原始欄位完整性。\n"
        "  2. `DropConstantAndDuplicate`：排除常數欄位與重複欄位。\n"
        "  3. `QuantileScaler`：將特徵分位數映射至常態分布，減緩極端離群值影響。\n"
        f"  4. `CorrelationFilter`：於訓練折內過濾高度線性相關特徵，保留 {n_model} 個特徵送入基底分類器。\n\n"
        "### 2. 三模型集成與百分位平均機制\n"
        "- **基底模型**：\n"
        "  - **M3 (ExtraTrees)**：極端隨機樹分類器。\n"
        "  - **M4 (XGBoost)**：梯度提升樹分類器。\n"
        "  - **M5 (LightGBM)**：輕量梯度提升機。\n"
        "- **集成評分機制 (E_A)**：\n"
        "  三個基底模型各自預測破產機率後，不直接加總原始機率（因各模型機率尺度與分布不同），\n"
        "  而是將原始輸出分別與該模型在訓練階段固化之參考分布比對，\n"
        "  轉換為百分位排序後，再取等權平均。\n\n"
        "### 3. 工作點固化與安全邊際策略\n"
        "門檻由訓練資料的交叉驗證預測決定，再加上安全邊際；安全邊際的大小以巢狀交叉驗證評估。"
    )


TAB3_PREVALENCE_SECTION_TITLE = "市場破產盛行率對審查成本與精準率之敏感度分析"


def format_tab3_prevalence_explanation() -> str:
    return (
        "真實市場的破產率可能與訓練資料不同，需以資料來源查證。\n"
        "當市場破產率較低時，預警名單中的假陽性比例會增加，使精準率降低、每查獲一家破產公司所需審查的家數增加。\n"
        "下表展示於不同市場破產率假設下，主要工作點預期之營運負荷與精準率變化（數據取自交叉驗證敏感度分析結果）："
    )


# ==============================================================================
# Tab 4: Limitations and Disclaimers
# ==============================================================================
TAB4_TITLE = "研究限制與免責聲明"


def format_tab4_disclaimer(train_bankruptcy_rate: float) -> str:
    train_pct = train_bankruptcy_rate * 100.0
    return (
        "### 1. 示範用途說明\n"
        "本互動頁面基於一份財務比率資料集；資料來源與授權待確認。"
        "本專案僅供學習與作品集展示用途，不是商業信用評等、授信或破產判斷工具，請勿用於實際財務或投資決策。\n\n"
        "### 2. 資料集與定義限制\n"
        "資料集沒有公司識別碼與年份，可能有同一公司跨年度重複；破產的定義與特徵的會計定義待確認，特徵分組與說明皆依欄位名稱推得。\n\n"
        "### 3. 破產盛行率與母體差異\n"
        f"訓練資料的破產率約為 {train_pct:.2f}%，可能高於真實市場的破產率。"
        "真實市場的破產率可能與訓練資料不同，需以資料來源查證。"
        "頁面上的校準機率只適用於與訓練資料相近的情境，不代表真實世界機率。\n\n"
        "### 4. 固定門檻適用限制\n"
        "各工作點門檻由訓練資料求得；若新資料的分布改變，固定門檻可能不再適用。\n\n"
        "### 5. 模型盲點\n"
        "少數破產公司的財報比率看起來與正常公司相近，模型可能抓不到；若要涵蓋這類案例，需要發出較高比例的預警。"
    )
