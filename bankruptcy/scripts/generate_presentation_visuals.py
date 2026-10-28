"""
bankruptcy/scripts/generate_presentation_visuals.py

Generates dedicated, high-resolution figures for the project closure presentation:
1. e7b_seed_stability_bars.png (Slide 18)
2. system_architecture_diagram.png (Slide 19)
3. model_governance_diagram.png (Slide 21)
4. implementation_roadmap.png (Slide 22)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# Setup consistent visual styling
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def generate_seed_stability_plot():
    """Slide 18: Generates e7b_seed_stability_bars.png"""
    df = pd.read_csv(TABLES_DIR / "e7b_seed_stability.csv")
    per_seed = df[df["type"] == "per_seed"].copy()
    seeds = per_seed["seed"].astype(str).tolist()

    pr_auc_vals = per_seed["pr_auc_mean"].values
    loss_vals = per_seed["loss_mean"].values
    ar95_vals = per_seed["ar95_mean"].values

    comp_loss = df[(df["type"] == "across_seed_comparison") & (df["metric"] == "loss")].iloc[0]
    comp_ar95 = df[(df["type"] == "across_seed_comparison") & (df["metric"] == "ar95")].iloc[0]
    comp_prauc = df[(df["type"] == "across_seed_comparison") & (df["metric"] == "pr_auc")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=200)
    fig.patch.set_facecolor("#0b1120")

    metrics_meta = [
        {
            "ax": axes[0],
            "title": "PR-AUC (5 組種子 vs 折間標準誤)",
            "vals": pr_auc_vals,
            "mean": comp_prauc["mean"],
            "seed_std": comp_prauc["seed_std"],
            "fold_se": comp_prauc["fold_se"],
            "ratio": comp_prauc["ratio_seed_to_fold_se"],
            "color": "#38bdf8",
            "fmt": "{:.4f}",
            "ylabel": "PR-AUC (越大家越佳)",
            "ylim": (0.45, 0.50),
        },
        {
            "ax": axes[1],
            "title": "3 點預警損失 (5 組種子 vs 折間標準誤)",
            "vals": loss_vals,
            "mean": comp_loss["mean"],
            "seed_std": comp_loss["seed_std"],
            "fold_se": comp_loss["fold_se"],
            "ratio": comp_loss["ratio_seed_to_fold_se"],
            "color": "#10b981",
            "fmt": "{:.3f}",
            "ylabel": "預警損失 Loss (越小越佳)",
            "ylim": (1.50, 1.70),
        },
        {
            "ax": axes[2],
            "title": "AR@0.95 預警倍數 (5 組種子 vs 折間標準誤)",
            "vals": ar95_vals,
            "mean": comp_ar95["mean"],
            "seed_std": comp_ar95["seed_std"],
            "fold_se": comp_ar95["fold_se"],
            "ratio": comp_ar95["ratio_seed_to_fold_se"],
            "color": "#f59e0b",
            "fmt": "{:.2f}x",
            "ylabel": "預警倍數 Alert Ratio (越小越佳)",
            "ylim": (6.5, 9.0),
        },
    ]

    for m in metrics_meta:
        ax = m["ax"]
        ax.set_facecolor("#111827")
        bars = ax.bar(seeds, m["vals"], color=m["color"], alpha=0.85, width=0.55, edgecolor="#ffffff", linewidth=0.8)

        # Baseline mean line
        ax.axhline(m["mean"], color="#ffffff", linestyle="--", alpha=0.7, label=f"5種子均值: {m['fmt'].format(m['mean'])}")
        # Fold SE error band
        ax.axhspan(m["mean"] - m["fold_se"], m["mean"] + m["fold_se"], color="#38bdf8", alpha=0.12, label=f"折間標準誤 SE (±{m['fmt'].format(m['fold_se'])})")

        ax.set_title(m["title"], fontsize=11, fontweight="bold", color="#f1f5f9", pad=12)
        ax.set_xlabel("隨機種子 (Random Seed)", fontsize=9, color="#94a3b8")
        ax.set_ylabel(m["ylabel"], fontsize=9, color="#94a3b8")
        ax.set_ylim(m["ylim"])
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.3, color="#64748b")

        # Value labels on bars
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + (m["ylim"][1]-m["ylim"][0])*0.02,
                    m["fmt"].format(h), ha="center", va="bottom", fontsize=8, color="#ffffff", fontweight="bold")

        # Annotation badge
        ax.text(0.5, 0.08, f"種子標準差 / 折間SE = {m['ratio']*100:.1f}%\n(種子擾動極微)",
                transform=ax.transAxes, ha="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=m["color"], alpha=0.9),
                color=m["color"])
        ax.legend(loc="upper right", fontsize=7.5, facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")

    fig.suptitle("E7b 隨機種子穩定性分析：5 組跨度種子重跑 15 折管線結果對照", fontsize=13, fontweight="bold", color="#f8fafc", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "e7b_seed_stability_bars.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def generate_architecture_diagram():
    """Slide 19: Generates system_architecture_diagram.png"""
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    ax.axis("off")

    # Define 5 horizontal pipeline stages
    stages = [
        {
            "title": "1. 資料輸入層 (Input)",
            "box": (0.02, 0.15, 0.16, 0.72),
            "color": "#1e293b",
            "border": "#38bdf8",
            "items": ["企金新進財報 CSV", "歷史客戶貸後報表", "95 個原始財務比率", "無標籤推論資料集", "資料格式與型別檢核"],
        },
        {
            "title": "2. 前處理引擎 (Pipe)",
            "box": (0.22, 0.15, 0.17, 0.72),
            "color": "#1e293b",
            "border": "#a855f7",
            "items": ["DropConstant(92欄)", "QuantileScaler(常態)", "CorrFilter(|r|>0.95)", "收斂至 72 代表特徵", "推論期自動旁路 SMOTE"],
        },
        {
            "title": "3. 異質模型集成 (Ensemble)",
            "box": (0.43, 0.15, 0.18, 0.72),
            "color": "#1e293b",
            "border": "#f59e0b",
            "items": ["M3: ExtraTrees", "M4: XGBoost", "M5: LightGBM", "5,455 筆 OOF 底冊", "ECDF 百分位無量綱融合"],
        },
        {
            "title": "4. 決策與校準 (Decision)",
            "box": (0.65, 0.15, 0.16, 0.72),
            "color": "#1e293b",
            "border": "#10b981",
            "items": ["安全邊際 m=0.06 固化", "R80/R90/R95/R100 門檻", "Platt Scaling 嚴格保序", "連續排名空間切分", "規避 Isotonic Ties"],
        },
        {
            "title": "5. 業務交付層 (Output)",
            "box": (0.85, 0.15, 0.14, 0.72),
            "color": "#1e293b",
            "border": "#f43f5e",
            "items": ["紅黃綠三級預警標籤", "真實破產校準機率", "Top 3 SHAP 風險主因", "CLI 批次推論報表", "Streamlit 徵信互動工作台"],
        },
    ]

    for st in stages:
        x, y, w, h = st["box"]
        # Outer Card
        rect = plt.Rectangle((x, y), w, h, facecolor=st["color"], edgecolor=st["border"], linewidth=1.8, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        # Header Badge
        header_rect = plt.Rectangle((x, y + h - 0.12), w, 0.12, facecolor=st["border"], edgecolor="none", transform=ax.transAxes, zorder=3, alpha=0.25)
        ax.add_patch(header_rect)
        ax.text(x + w/2, y + h - 0.06, st["title"], transform=ax.transAxes, ha="center", va="center", color="#ffffff", fontsize=9.5, fontweight="bold", zorder=4)

        # Items
        for idx, it in enumerate(st["items"]):
            item_y = y + h - 0.20 - idx * 0.105
            ax.text(x + 0.015, item_y, f"> {it}", transform=ax.transAxes, ha="left", va="center", color="#cbd5e1", fontsize=8.2, zorder=4)

    # Draw Connecting Arrows between stages
    arrows = [(0.18, 0.51, 0.22, 0.51), (0.39, 0.51, 0.43, 0.51), (0.61, 0.51, 0.65, 0.51), (0.81, 0.51, 0.85, 0.51)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#38bdf8", lw=2.2, mutation_scale=15), zorder=5)

    # Title Banner
    ax.text(0.5, 0.94, "破產預測 AI 系統端到端生產級工程交付架構圖", transform=ax.transAxes, ha="center", va="center", color="#f8fafc", fontsize=13.5, fontweight="bold")
    ax.text(0.5, 0.05, "單一真相原則：CLI 工具 (score_new_data.py) 與 Web UI (app/) 100% 共享 scoring.py 評分核心",
            transform=ax.transAxes, ha="center", va="center", color="#94a3b8", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#334155", alpha=0.9))

    plt.tight_layout()
    out_path = FIGURES_DIR / "system_architecture_diagram.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def generate_model_governance_diagram():
    """Slide 21: Generates model_governance_diagram.png"""
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    ax.axis("off")

    # Three Defense Lines Boxes
    lines = [
        {
            "line": "第一道防線：輸入特徵穩定性監控 (Input Drift)",
            "cycle": "執行週期：每月定時自動排程",
            "box": (0.05, 0.64, 0.90, 0.22),
            "color": "#1e293b",
            "border": "#38bdf8",
            "bullets": [
                "監控 18 大核心特徵之群體穩定性指標 (Population Stability Index, PSI)",
                "PSI < 0.10：分佈穩定，正常放行 ｜ 0.10 <= PSI <= 0.25：輕微漂移，發出黃色提醒",
                "PSI > 0.25：觸發重大概念漂移警報，自動轉交風控長與資料科學團隊實施緊急審查",
            ],
        },
        {
            "line": "第二道防線：前線徵信覆核反饋閉環 (Feedback Loop)",
            "cycle": "執行週期：每季業務檢討與授信稽核",
            "box": (0.05, 0.36, 0.90, 0.22),
            "color": "#1e293b",
            "border": "#10b981",
            "bullets": [
                "徵信調查員實地訪查回饋閉環：建立預警名單之核貸後續追蹤檔案",
                "誤報案例（False Alarm）深度分析：多為營運流動性緊縮者，主動轉介企金財務輔導或徵提擔保",
                "漏報案例（False Negative）回溯調查：重點稽查非財務性風險、關係人交易與舞弊行為",
            ],
        },
        {
            "line": "第三道防線：演算法定期演進與門檻校正 (Model Evolution)",
            "cycle": "執行週期：每年定期重訓或觸發式重訓",
            "box": (0.05, 0.08, 0.90, 0.22),
            "color": "#1e293b",
            "border": "#a855f7",
            "bullets": [
                "定期納入最新年度實際違約企業樣本，動態更新 5,455 筆 ECDF 經驗百分位參考底冊",
                "重新校準 Platt Scaling 參數，動態對齊總體經濟利率環境與破產盛行率變化",
                "重跑 18 組 Pytest 自動化回歸測試套件，嚴格記錄版本雜湊指紋（Audit Trail）",
            ],
        },
    ]

    for ln in lines:
        x, y, w, h = ln["box"]
        rect = plt.Rectangle((x, y), w, h, facecolor=ln["color"], edgecolor=ln["border"], linewidth=1.6, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.05, ln["line"], transform=ax.transAxes, va="center", color="#ffffff", fontsize=10, fontweight="bold", zorder=3)
        ax.text(x + w - 0.02, y + h - 0.05, ln["cycle"], transform=ax.transAxes, ha="right", va="center", color=ln["border"], fontsize=8.5, fontweight="bold", zorder=3)

        for b_idx, b in enumerate(ln["bullets"]):
            by = y + h - 0.09 - b_idx * 0.042
            ax.text(x + 0.03, by, f"• {b}", transform=ax.transAxes, va="center", color="#cbd5e1", fontsize=8.2, zorder=3)

    ax.text(0.5, 0.94, "金融級負責任 AI 模型風險治理架構：三道防線閉環", transform=ax.transAxes, ha="center", va="center", color="#f8fafc", fontsize=13.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "model_governance_diagram.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def generate_implementation_roadmap():
    """Slide 22: Generates implementation_roadmap.png"""
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=200)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    ax.axis("off")

    phases = [
        {
            "phase": "Phase 1 (1～3 個月)",
            "title": "平行雙軌驗證 (Shadow Testing)",
            "box": (0.05, 0.20, 0.27, 0.65),
            "color": "#1e293b",
            "border": "#38bdf8",
            "items": [
                "行內新進案件雙軌並行比對",
                "AI 評分與傳統專家審查對齊",
                "驗證前線徵信同仁使用體驗",
                "建立第一批真實誤報與漏報回饋",
                "預警倍數落點實務校正",
            ],
            "kpi": "預期成果：驗證 1,000+ 戶，前線認同度 ≥ 85%",
        },
        {
            "phase": "Phase 2 (3～6 個月)",
            "title": "核心主機系統對接 (Core API)",
            "box": (0.36, 0.20, 0.27, 0.65),
            "color": "#1e293b",
            "border": "#10b981",
            "items": [
                "批次推論 CLI 納入每日排程",
                "授信主機 RESTful API 串接",
                "自動派工至徵信調查員工作台",
                "整合 CAMELS 既有專家評等",
                "自動產出企業 SHAP 徵信摘要卡",
            ],
            "kpi": "預期成果：全流程自動化，推論 < 50ms",
        },
        {
            "phase": "Phase 3 (6～12 個月)",
            "title": "全行政策聯動 (Bank-wide Policy)",
            "box": (0.68, 0.20, 0.27, 0.65),
            "color": "#1e293b",
            "border": "#f59e0b",
            "items": [
                "預警等級聯動動態信用額度",
                "差異化放款利率與擔保品徵提",
                "貸後高風險企業自動收傘處置",
                "納入分行放款風控 KPI 考核",
                "啟動年度模型重訓與底冊更新",
            ],
            "kpi": "預期成果：年省 7,500+ 人天，呆帳全損減半",
        },
    ]

    for p in phases:
        x, y, w, h = p["box"]
        rect = plt.Rectangle((x, y), w, h, facecolor=p["color"], edgecolor=p["border"], linewidth=1.8, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.05, p["phase"], transform=ax.transAxes, ha="center", va="center", color=p["border"], fontsize=9.2, fontweight="bold", zorder=3)
        ax.text(x + w/2, y + h - 0.11, p["title"], transform=ax.transAxes, ha="center", va="center", color="#ffffff", fontsize=10.2, fontweight="bold", zorder=3)

        for it_idx, it in enumerate(p["items"]):
            iy = y + h - 0.18 - it_idx * 0.075
            ax.text(x + 0.02, iy, f"> {it}", transform=ax.transAxes, va="center", color="#cbd5e1", fontsize=8.2, zorder=3)

        # Bottom KPI badge
        kpi_rect = plt.Rectangle((x + 0.015, y + 0.025), w - 0.03, 0.09, facecolor="#0f172a", edgecolor=p["border"], linewidth=0.8, transform=ax.transAxes, zorder=3)
        ax.add_patch(kpi_rect)
        ax.text(x + w/2, y + 0.07, p["kpi"].replace("≥", ">="), transform=ax.transAxes, ha="center", va="center", color="#f8fafc", fontsize=7.6, fontweight="bold", zorder=4)

    # Connecting Arrows
    ax.annotate("", xy=(0.36, 0.52), xytext=(0.32, 0.52), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#38bdf8", lw=2, mutation_scale=14), zorder=5)
    ax.annotate("", xy=(0.68, 0.52), xytext=(0.63, 0.52), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#38bdf8", lw=2, mutation_scale=14), zorder=5)

    ax.text(0.5, 0.94, "商業落地實施三階段路線圖 (Implementation Roadmap)", transform=ax.transAxes, ha="center", va="center", color="#f8fafc", fontsize=13.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "implementation_roadmap.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_seed_stability_plot()
    generate_architecture_diagram()
    generate_model_governance_diagram()
    generate_implementation_roadmap()
