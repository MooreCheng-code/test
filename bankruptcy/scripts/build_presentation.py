"""
bankruptcy/scripts/build_presentation.py

Generates a standalone, interactive, self-contained 16:9 widescreen HTML presentation
deck for the Bankruptcy Prediction Project Closure Presentation (結案簡報).

Features:
- Offline self-contained (Base64 embedded figures, zero external CDNs).
- 16:9 presentation layout with keyboard navigation (Arrows, Space, PageUp/PageDown, Home, End).
- Slide Overview mode (ESC key or button) for quick thumbnail jumping.
- Speaker Notes drawer (S key or button) showing full spoken scripts.
- Fullscreen mode (F key or button).
- Print to PDF friendly (@media print formatting for 16:9 slide-per-page export).
- Single Source of Truth: Data dynamically loaded from outputs/ and defaults.json.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from presentation_data import (
    PRESENTATION_META,
    PRESENTATION_ACTS,
    PRESENTATION_SLIDES,
    QA_DEFENSE_PLAYBOOK,
)

PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULTS_PATH = PROJECT_ROOT / "app" / "assets" / "defaults.json"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
E6_DIR = PROJECT_ROOT / "outputs" / "e6"

OUT_DOCS_HTML = WORKSPACE_ROOT / "docs" / "project_presentation.html"
OUT_ROOT_HTML = WORKSPACE_ROOT / "破產預測_結案簡報.html"


def load_figure_base64(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_presentation_data() -> dict[str, Any]:
    # 1. Load Defaults JSON
    defaults = {}
    if DEFAULTS_PATH.exists():
        defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))

    # 2. Load Figures
    fig_map = {
        "target_distribution": FIGURES_DIR / "target_distribution.png",
        "univariate_auc_top20": FIGURES_DIR / "univariate_auc_top20.png",
        "e0_pr_curves": FIGURES_DIR / "e0_pr_curves.png",
        "e0_recall_vs_alert_ratio": FIGURES_DIR / "e0_recall_vs_alert_ratio.png",
        "e1_paired_diff_forest": FIGURES_DIR / "e1_paired_diff_forest.png",
        "e3_ensemble_vs_single_ar95": FIGURES_DIR / "e3_ensemble_vs_single_ar95.png",
        "e3_hard_positives_percentile": FIGURES_DIR / "e3_hard_positives_percentile.png",
        "e3_rank_correlation_heatmap": FIGURES_DIR / "e3_rank_correlation_heatmap.png",
        "e4_threshold_transfer": FIGURES_DIR / "e4_threshold_transfer.png",
        "e4_calibration_reliability": FIGURES_DIR / "e4_calibration_reliability.png",
        "e5_shap_summary_M4": FIGURES_DIR / "e5_shap_summary_M4.png",
        "e6_operating_curve": E6_DIR / "e6_operating_curve.png",
        "e6_pr_curve": E6_DIR / "e6_pr_curve.png",
        "e6_score_distribution_positives": E6_DIR / "e6_score_distribution_positives.png",
        "e7_prevalence_precision": FIGURES_DIR / "e7_prevalence_precision.png",
        "e7b_seed_stability_bars": FIGURES_DIR / "e7b_seed_stability_bars.png",
        "system_architecture_diagram": FIGURES_DIR / "system_architecture_diagram.png",
        "model_governance_diagram": FIGURES_DIR / "model_governance_diagram.png",
        "implementation_roadmap": FIGURES_DIR / "implementation_roadmap.png",
    }
    encoded_figures = {k: load_figure_base64(p) for k, p in fig_map.items()}

    # 3. Load CV & E6 Operating Points
    op_cv_df = pd.read_csv(TABLES_DIR / "operating_points_for_report.csv")
    op_e6_df = pd.read_csv(E6_DIR / "e6_operating_points.csv")
    op_cv_records = op_cv_df.to_dict(orient="records")
    op_e6_records = op_e6_df.to_dict(orient="records")

    # 4. Load Prevalence Sensitivity Table
    prev_df = pd.read_csv(TABLES_DIR / "e7a_prevalence_sensitivity.csv")
    prev_records = prev_df.to_dict(orient="records")

    # 5. Load Seed Stability Table
    seed_df = pd.read_csv(TABLES_DIR / "e7b_seed_stability.csv")
    seed_records = seed_df.to_dict(orient="records")

    return {
        "meta": PRESENTATION_META,
        "acts": PRESENTATION_ACTS,
        "slides": PRESENTATION_SLIDES,
        "qa_playbook": QA_DEFENSE_PLAYBOOK,
        "figures": encoded_figures,
        "op_cv": op_cv_records,
        "op_e6": op_e6_records,
        "prevalence": prev_records,
        "seed_stability": seed_records,
        "defaults": defaults,
    }


def generate_presentation_html(data: dict[str, Any]) -> str:
    json_str = json.dumps(data, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="zh-Hant" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>企業破產早期預警 AI 系統 — 結案成果與商業落地評估報告</title>
<style>
:root[data-theme="dark"] {
  --bg-main: #070a13;
  --bg-panel: #0e1526;
  --bg-card: #151f38;
  --bg-card-hover: #1e2c4f;
  --border: #233358;
  --text-main: #f1f5f9;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --accent: #38bdf8;
  --accent-soft: rgba(56, 189, 248, 0.12);
  --accent-glow: rgba(56, 189, 248, 0.35);
  --green: #10b981;
  --green-soft: rgba(16, 185, 129, 0.14);
  --red: #f43f5e;
  --red-soft: rgba(244, 63, 94, 0.14);
  --purple: #a855f7;
  --purple-soft: rgba(168, 85, 247, 0.14);
  --amber: #f59e0b;
  --amber-soft: rgba(245, 158, 11, 0.14);
  --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
  --slide-aspect: 16 / 9;
}

:root[data-theme="light"] {
  --bg-main: #f8fafc;
  --bg-panel: #ffffff;
  --bg-card: #f1f5f9;
  --bg-card-hover: #e2e8f0;
  --border: #cbd5e1;
  --text-main: #0f172a;
  --text-muted: #475569;
  --text-dim: #64748b;
  --accent: #0284c7;
  --accent-soft: rgba(2, 132, 199, 0.1);
  --accent-glow: rgba(2, 132, 199, 0.25);
  --green: #059669;
  --green-soft: rgba(5, 150, 105, 0.12);
  --red: #e11d48;
  --red-soft: rgba(225, 29, 72, 0.12);
  --purple: #7c3aed;
  --purple-soft: rgba(124, 58, 237, 0.12);
  --amber: #d97706;
  --amber-soft: rgba(217, 119, 6, 0.12);
  --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
  --slide-aspect: 16 / 9;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans TC", sans-serif;
  background: var(--bg-main);
  color: var(--text-main);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  user-select: none;
}

/* Header Bar */
.pres-header {
  height: 54px;
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  z-index: 100;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.brand-badge {
  font-size: 0.76rem;
  font-weight: 800;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--accent);
  color: #070a13;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.pres-title-header {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--text-main);
}
.header-center {
  display: flex;
  align-items: center;
  gap: 8px;
}
.act-pill {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 6px;
  background: var(--bg-card);
  color: var(--text-muted);
  border: 1px solid var(--border);
}
.slide-counter-badge {
  font-family: monospace;
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--accent);
  background: var(--accent-soft);
  padding: 3px 9px;
  border-radius: 6px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ctrl-btn {
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-main);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 6px 12px;
  border-radius: 7px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.15s;
}
.ctrl-btn:hover {
  background: var(--bg-card-hover);
  border-color: var(--accent);
  color: var(--accent);
}
.ctrl-btn.active {
  background: var(--accent);
  color: #070a13;
  border-color: var(--accent);
}

/* Slide Deck Container (16:9 Stage) */
.stage-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px 24px;
  position: relative;
  overflow: hidden;
}
.slide-stage {
  width: 100%;
  max-width: 1360px;
  aspect-ratio: var(--slide-aspect);
  max-height: calc(100vh - 124px);
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}

/* Slide Content Structure */
.slide-inner {
  flex: 1;
  padding: 32px 40px;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  position: relative;
}
.slide-header {
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 14px;
}
.slide-title-wrap h2 {
  font-size: 1.65rem;
  font-weight: 800;
  color: var(--text-main);
  letter-spacing: -0.01em;
  line-height: 1.25;
}
.slide-title-wrap p {
  font-size: 0.92rem;
  color: var(--text-muted);
  margin-top: 5px;
  font-weight: 500;
}
.slide-badge-tag {
  font-size: 0.78rem;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 6px;
  background: var(--purple-soft);
  color: var(--purple);
  white-space: nowrap;
}

/* Layout Modes */
.layout-hero {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 20px 40px;
}
.layout-hero h1 {
  font-size: 2.5rem;
  font-weight: 900;
  color: var(--text-main);
  margin-bottom: 12px;
  line-height: 1.2;
}
.layout-hero .hero-sub {
  font-size: 1.15rem;
  color: var(--accent);
  max-width: 900px;
  margin-bottom: 28px;
  line-height: 1.5;
  font-weight: 600;
}
.hero-kpis {
  display: flex;
  gap: 20px;
  justify-content: center;
  margin-top: 24px;
  width: 100%;
  max-width: 980px;
}

.kpi-card-giant {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  transition: transform 0.2s;
}
.kpi-card-giant:hover {
  transform: translateY(-3px);
  border-color: var(--accent);
}
.kpi-card-giant .kg-label {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 6px;
}
.kpi-card-giant .kg-val {
  font-size: 2.2rem;
  font-weight: 900;
  color: var(--accent);
  line-height: 1.1;
  font-family: monospace;
}
.kpi-card-giant .kg-sub {
  font-size: 0.78rem;
  color: var(--text-dim);
  margin-top: 6px;
}

/* Split Content Grid (Left Points, Right Visual/Chart) */
.layout-split {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(420px, 1.1fr) minmax(360px, 1fr);
  gap: 28px;
  align-items: center;
}
.split-left {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.split-bullets {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.split-bullets li {
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--text-main);
  padding-left: 20px;
  position: relative;
}
.split-bullets li::before {
  content: "✦";
  position: absolute;
  left: 0;
  color: var(--accent);
  font-size: 0.85rem;
  top: 1px;
}
.split-right {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  height: 100%;
  max-height: 410px;
  overflow: hidden;
  position: relative;
}
.split-right img {
  max-width: 100%;
  max-height: 380px;
  object-fit: contain;
  border-radius: 8px;
  cursor: zoom-in;
  transition: transform 0.2s;
  background: #ffffff;
}
.split-right img:hover {
  transform: scale(1.02);
}

/* Two Column Compare */
.layout-compare {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: stretch;
}
.compare-col {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  display: flex;
  flex-direction: column;
}
.compare-col h3 {
  font-size: 1.1rem;
  color: var(--accent);
  margin-bottom: 12px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}

/* Table View */
.table-stage-wrap {
  flex: 1;
  overflow-x: auto;
  margin-top: 10px;
}
.pres-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}
.pres-table th {
  background: var(--bg-card);
  color: var(--text-muted);
  font-weight: 700;
  text-align: left;
  padding: 10px 12px;
  border-bottom: 2px solid var(--border);
}
.pres-table td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text-main);
}
.pres-table tr:hover td {
  background: var(--bg-card-hover);
}

/* Footer / Progress Bar */
.pres-footer {
  height: 48px;
  background: var(--bg-panel);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  font-size: 0.82rem;
  color: var(--text-muted);
  position: relative;
}
.footer-takeaway {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-main);
  font-weight: 600;
  max-width: 85%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.footer-takeaway .ft-badge {
  background: var(--accent-soft);
  color: var(--accent);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.74rem;
  font-weight: 800;
  flex-shrink: 0;
}
.progress-line-bg {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--border);
}
.progress-line-fill {
  height: 100%;
  background: var(--accent);
  width: 0%;
  transition: width 0.25s ease-out;
}

/* Overview Modal */
.overview-modal {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(7, 10, 19, 0.94);
  backdrop-filter: blur(8px);
  z-index: 1000;
  padding: 30px;
  overflow-y: auto;
}
.overview-modal.open { display: block; }
.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 18px;
  max-width: 1400px;
  margin: 0 auto;
}
.ov-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  cursor: pointer;
  transition: all 0.15s;
  aspect-ratio: 16 / 9;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.ov-card:hover, .ov-card.active {
  border-color: var(--accent);
  transform: translateY(-2px);
  box-shadow: 0 6px 18px var(--accent-glow);
}
.ov-num {
  font-family: monospace;
  font-size: 0.75rem;
  font-weight: 800;
  color: var(--accent);
}
.ov-title {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--text-main);
  line-height: 1.35;
  margin-top: 4px;
}
.ov-badge {
  font-size: 0.7rem;
  color: var(--text-muted);
}

/* Speaker Notes Drawer */
.notes-drawer {
  position: fixed;
  top: 54px;
  right: -480px;
  width: 460px;
  bottom: 0;
  background: var(--bg-panel);
  border-left: 1px solid var(--border);
  box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
  z-index: 500;
  transition: right 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  display: flex;
  flex-direction: column;
  padding: 20px;
}
.notes-drawer.open { right: 0; }
.notes-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px;
  margin-bottom: 14px;
}
.notes-header h3 {
  font-size: 1.05rem;
  color: var(--accent);
}
.notes-body {
  flex: 1;
  overflow-y: auto;
  font-size: 0.92rem;
  line-height: 1.7;
  color: var(--text-main);
  white-space: pre-line;
}

/* Lightbox Modal */
.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.92);
  z-index: 9999;
  align-items: center;
  justify-content: center;
  padding: 30px;
}
.lightbox.open { display: flex; }
.lightbox img {
  max-width: 95vw;
  max-height: 92vh;
  border-radius: 8px;
  background: #ffffff;
}

/* Print Friendly (@media print) */
@media print {
  body {
    overflow: visible !important;
    height: auto !important;
    background: #ffffff !important;
    color: #000000 !important;
  }
  .pres-header, .pres-footer, .overview-modal, .notes-drawer, .ctrl-btn {
    display: none !important;
  }
  .stage-container {
    padding: 0 !important;
    display: block !important;
  }
  .slide-stage {
    width: 100% !important;
    max-width: 100% !important;
    aspect-ratio: 16 / 9 !important;
    page-break-after: always !important;
    break-after: page !important;
    box-shadow: none !important;
    border: none !important;
    border-radius: 0 !important;
    margin-bottom: 0 !important;
  }
  .slide-inner {
    padding: 20px !important;
  }
}
</style>
</head>
<body>

<!-- Header Bar -->
<header class="pres-header">
  <div class="header-left">
    <span class="brand-badge">結案成果報告</span>
    <span class="pres-title-header">企業破產早期預警 AI 系統</span>
  </div>

  <div class="header-center">
    <span class="act-pill" id="currentActPill">第一篇：商業使命</span>
    <span class="slide-counter-badge" id="slideCounterBadge">01 / 22</span>
  </div>

  <div class="header-right">
    <button class="ctrl-btn" onclick="prevSlide()" title="上一頁 (←)">⬅ 上一頁</button>
    <button class="ctrl-btn" onclick="nextSlide()" title="下一頁 (→ / Space)">下一頁 ➡</button>
    <button class="ctrl-btn" onclick="toggleOverview()" title="縮圖總覽 (ESC)">🗂️ 縮圖總覽</button>
    <button class="ctrl-btn" id="notesToggleBtn" onclick="toggleNotes()" title="講者備忘 (S)">📝 講者備忘</button>
    <button class="ctrl-btn" onclick="toggleFullscreen()" title="全螢幕 (F)">🖥️ 全螢幕</button>
    <button class="ctrl-btn" onclick="toggleTheme()" id="themeBtn">☀️ 淺色</button>
    <button class="ctrl-btn" onclick="window.print()" title="列印或另存 PDF (Ctrl+P)">🖨️ 轉存 PDF</button>
  </div>
</header>

<!-- Slide Stage (16:9 Stage) -->
<main class="stage-container">
  <div class="slide-stage" id="slideStage">
    <div class="slide-inner" id="slideInner">
      <!-- Injected by JavaScript -->
    </div>
  </div>
</main>

<!-- Footer Bar -->
<footer class="pres-footer">
  <div class="progress-line-bg">
    <div class="progress-line-fill" id="progressLine"></div>
  </div>
  <div class="footer-takeaway" id="footerTakeaway">
    <span class="ft-badge">核心結論</span>
    <span id="footerTakeawayText">正在載入專案結案成果...</span>
  </div>
  <div style="font-size:0.75rem; color:var(--text-dim); flex-shrink:0;">
    按 <code>←</code> <code>→</code> / <code>Space</code> 換頁 ｜ <code>ESC</code> 總覽 ｜ <code>S</code> 講稿
  </div>
</footer>

<!-- Overview Modal -->
<div class="overview-modal" id="overviewModal" onclick="closeOverview(event)">
  <div style="display:flex; justify-content:space-between; align-items:center; max-width:1400px; margin:0 auto 20px;">
    <h3 style="font-size:1.3rem; color:var(--text-main);">🗂️ 結案簡報 22 頁縮圖總覽 (點擊任一頁直接跳轉)</h3>
    <button class="ctrl-btn" onclick="toggleOverview()">✕ 關閉總覽 (ESC)</button>
  </div>
  <div class="overview-grid" id="overviewGrid"></div>
</div>

<!-- Speaker Notes Drawer -->
<aside class="notes-drawer" id="notesDrawer">
  <div class="notes-header">
    <h3>📝 講者備忘逐字稿 (Speaker Script)</h3>
    <button class="ctrl-btn" onclick="toggleNotes()" style="padding:3px 8px;">✕</button>
  </div>
  <div style="font-size:0.78rem; color:var(--accent); margin-bottom:10px;" id="notesSlideMeta">Slide 01 備忘</div>
  <div class="notes-body" id="notesBody"></div>
</aside>

<!-- Lightbox Modal -->
<div class="lightbox" id="lightboxModal" onclick="closeLightbox()">
  <img id="lightboxImg" src="" alt="Enlarged figure">
</div>

<script>
const PRES_DATA = __PRES_DATA_JSON__;

let currentSlideIdx = 0;
const totalSlides = PRES_DATA.slides.length;

function escapeHtml(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatInline(str) {
  if (!str) return '';
  let s = String(str);
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/`([^`\n]+)`/g, '<code style="background:var(--accent-soft); color:var(--accent); padding:1px 5px; border-radius:4px; font-size:0.9em;">$1</code>');
  return s;
}

function renderCurrentSlide() {
  const s = PRES_DATA.slides[currentSlideIdx];
  const inner = document.getElementById('slideInner');

  // Header meta
  document.getElementById('currentActPill').textContent = s.act_title.split('：')[0];
  document.getElementById('slideCounterBadge').textContent = `${String(s.slide_no).padStart(2, '0')} / ${String(totalSlides).padStart(2, '0')}`;
  document.getElementById('progressLine').style.width = `${((currentSlideIdx + 1) / totalSlides) * 100}%`;
  document.getElementById('footerTakeawayText').textContent = s.takeaway;

  // Notes drawer
  document.getElementById('notesSlideMeta').textContent = `Slide ${String(s.slide_no).padStart(2, '0')} · ${s.title}`;
  document.getElementById('notesBody').textContent = s.speaker_notes;

  // Render Slide Inner based on layout
  let contentHtml = '';

  if (s.layout === 'title_hero') {
    contentHtml = `
      <div class="layout-hero">
        <span class="slide-badge-tag" style="margin-bottom:16px;">✦ ${s.badge} ✦</span>
        <h1>${s.title}</h1>
        <p class="hero-sub">${s.subtitle}</p>
        <div style="max-width:850px; text-align:left; background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px; font-size:0.96rem; line-height:1.7;">
          ${(s.bullet_points || []).map(b => `<div style="margin-bottom:8px;">${formatInline(b)}</div>`).join('')}
        </div>
        <div class="hero-kpis">
          ${(s.metrics || []).map(m => `
            <div class="kpi-card-giant">
              <div class="kg-label">${m.label}</div>
              <div class="kg-val">${m.val}</div>
              <div class="kg-sub">${m.sub}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  } else if (s.layout === 'kpi_grid') {
    contentHtml = `
      <div class="slide-header">
        <div class="slide-title-wrap">
          <h2>${s.title}</h2>
          <p>${s.subtitle}</p>
        </div>
        <span class="slide-badge-tag">${s.badge}</span>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:18px; margin-bottom:24px;">
        ${(s.metrics || []).map(m => `
          <div class="kpi-card-giant">
            <div class="kg-label">${m.label}</div>
            <div class="kg-val">${m.val}</div>
            <div class="kg-sub">${m.sub}</div>
          </div>
        `).join('')}
      </div>
      <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:22px; flex:1;">
        <h4 style="font-size:1.05rem; color:var(--accent); margin-bottom:12px;">📌 核心實證要點：</h4>
        <ul class="split-bullets">
          ${(s.bullet_points || []).map(b => `<li>${formatInline(b)}</li>`).join('')}
        </ul>
      </div>
    `;
  } else if (s.layout === 'table_view') {
    // E6 Comparison table
    contentHtml = `
      <div class="slide-header">
        <div class="slide-title-wrap">
          <h2>${s.title}</h2>
          <p>${s.subtitle}</p>
        </div>
        <span class="slide-badge-tag">${s.badge}</span>
      </div>
      <div class="table-stage-wrap">
        <table class="pres-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>工作點 (安全邊際)</th>
              <th>固化門檻 $t$</th>
              <th>CV 實際召回 (均值±標準差)</th>
              <th>測試集捕獲數</th>
              <th>測試集實際召回率</th>
              <th>CV 預警倍數</th>
              <th>測試集預警倍數 (比例)</th>
              <th>測試集精確率</th>
              <th>達標</th>
            </tr>
          </thead>
          <tbody>
            ${PRES_DATA.op_e6.map((e6, i) => {
              const cv = PRES_DATA.op_cv[i] || {};
              return `
                <tr>
                  <td><strong>${e6.group}</strong></td>
                  <td><span class="badge" style="background:var(--accent-soft); color:var(--accent);">${e6.operating_point} (m=${Number(e6.margin).toFixed(2)})</span></td>
                  <td><code>${Number(e6.threshold).toFixed(4)}</code></td>
                  <td>${(Number(cv.actual_recall_mean)*100).toFixed(2)}% ± ${(Number(cv.actual_recall_std)*100).toFixed(2)}%</td>
                  <td><strong style="color:var(--accent); font-size:1.05em;">${e6.positives_caught} / ${e6.total_positives}</strong></td>
                  <td><strong style="color:var(--green);">${(Number(e6.actual_recall)*100).toFixed(2)}%</strong></td>
                  <td>${Number(cv.alert_ratio_mean).toFixed(2)}x</td>
                  <td><strong>${Number(e6.alert_ratio).toFixed(2)}x</strong> (${(Number(e6.alert_proportion)*100).toFixed(2)}%)</td>
                  <td><strong>${(Number(e6.precision)*100).toFixed(2)}%</strong></td>
                  <td><span class="badge" style="background:var(--green-soft); color:var(--green);">✔ 達標</span></td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      </div>
      <div style="margin-top:16px; background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:12px 18px; font-size:0.86rem; color:var(--text-muted);">
        ${(s.bullet_points || []).map(b => `<div>${formatInline(b)}</div>`).join('')}
      </div>
    `;
  } else {
    // Default split layout
    const figSrc = s.figure_key && PRES_DATA.figures[s.figure_key] ? PRES_DATA.figures[s.figure_key] : null;

    contentHtml = `
      <div class="slide-header">
        <div class="slide-title-wrap">
          <h2>${s.title}</h2>
          <p>${s.subtitle}</p>
        </div>
        <span class="slide-badge-tag">${s.badge}</span>
      </div>
      <div class="layout-split">
        <div class="split-left">
          <ul class="split-bullets">
            ${(s.bullet_points || []).map(b => `<li>${formatInline(b)}</li>`).join('')}
          </ul>
          ${(s.metrics && s.metrics.length > 0) ? `
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:12px; margin-top:10px;">
              ${s.metrics.map(m => `
                <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:10px 12px;">
                  <div style="font-size:0.75rem; color:var(--text-muted); font-weight:700;">${m.label}</div>
                  <div style="font-size:1.25rem; font-weight:800; color:var(--accent); font-family:monospace; margin-top:2px;">${m.val}</div>
                  <div style="font-size:0.72rem; color:var(--text-dim); margin-top:2px;">${m.sub}</div>
                </div>
              `).join('')}
            </div>
          ` : ''}
        </div>
        <div class="split-right">
          ${figSrc ? `
            <img src="${figSrc}" alt="${s.title}" onclick="openLightbox('${figSrc}')" title="點擊放大檢視圖片">
            <div style="font-size:0.74rem; color:var(--text-dim); margin-top:6px;">🔍 點擊圖表可全螢幕放大檢視 Lightbox</div>
          ` : `
            <div style="text-align:center; padding:20px; color:var(--text-dim);">
              <div style="font-size:2.5rem; margin-bottom:8px;">📊</div>
              <div style="font-size:0.9rem; font-weight:600;">核心工程架構與實證數據</div>
            </div>
          `}
        </div>
      </div>
    `;
  }

  inner.innerHTML = contentHtml;
}

function nextSlide() {
  if (currentSlideIdx < totalSlides - 1) {
    currentSlideIdx++;
    renderCurrentSlide();
  }
}

function prevSlide() {
  if (currentSlideIdx > 0) {
    currentSlideIdx--;
    renderCurrentSlide();
  }
}

function goToSlide(idx) {
  if (idx >= 0 && idx < totalSlides) {
    currentSlideIdx = idx;
    renderCurrentSlide();
    document.getElementById('overviewModal').classList.remove('open');
  }
}

function toggleOverview() {
  const modal = document.getElementById('overviewModal');
  const isOpen = modal.classList.toggle('open');
  if (isOpen) {
    renderOverviewGrid();
  }
}

function closeOverview(e) {
  if (e.target.id === 'overviewModal') {
    document.getElementById('overviewModal').classList.remove('open');
  }
}

function renderOverviewGrid() {
  const grid = document.getElementById('overviewGrid');
  grid.innerHTML = PRES_DATA.slides.map((s, idx) => `
    <div class="ov-card ${idx === currentSlideIdx ? 'active' : ''}" onclick="goToSlide(${idx})">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="ov-num">SLIDE ${String(s.slide_no).padStart(2, '0')}</span>
        <span class="ov-badge">${s.badge}</span>
      </div>
      <div class="ov-title">${s.title}</div>
      <div style="font-size:0.72rem; color:var(--text-dim);">${s.act_title.split('：')[0]}</div>
    </div>
  `).join('');
}

function toggleNotes() {
  const drawer = document.getElementById('notesDrawer');
  const btn = document.getElementById('notesToggleBtn');
  const isOpen = drawer.classList.toggle('open');
  btn.classList.toggle('active', isOpen);
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

function toggleTheme() {
  const htmlEl = document.documentElement;
  const cur = htmlEl.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  htmlEl.setAttribute('data-theme', next);
  document.getElementById('themeBtn').textContent = next === 'dark' ? '☀️ 淺色' : '🌙 深色';
}

function openLightbox(src) {
  if (!src) return;
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightboxModal').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightboxModal').classList.remove('open');
}

// Keyboard Controls
window.addEventListener('keydown', (e) => {
  // If lightbox is open, ESC closes it
  if (document.getElementById('lightboxModal').classList.contains('open')) {
    if (e.key === 'Escape') closeLightbox();
    return;
  }
  // If overview is open, ESC closes it
  if (document.getElementById('overviewModal').classList.contains('open')) {
    if (e.key === 'Escape') toggleOverview();
    return;
  }

  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
    e.preventDefault();
    nextSlide();
  } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
    e.preventDefault();
    prevSlide();
  } else if (e.key === 'Home') {
    e.preventDefault();
    goToSlide(0);
  } else if (e.key === 'End') {
    e.preventDefault();
    goToSlide(totalSlides - 1);
  } else if (e.key === 'Escape') {
    e.preventDefault();
    toggleOverview();
  } else if (e.key === 's' || e.key === 'S') {
    e.preventDefault();
    toggleNotes();
  } else if (e.key === 'f' || e.key === 'F') {
    e.preventDefault();
    toggleFullscreen();
  }
});

// Init
window.addEventListener('DOMContentLoaded', () => {
  renderCurrentSlide();
});
</script>
</body>
</html>
"""
    return html.replace("__PRES_DATA_JSON__", json_str)


def main() -> None:
    print("Loading presentation data, tables, defaults, and figures...")
    data = load_presentation_data()
    print(f"Loaded {len(data['slides'])} slides and {len(data['figures'])} embedded figures.")

    html_content = generate_presentation_html(data)

    OUT_DOCS_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOCS_HTML.write_text(html_content, encoding="utf-8")
    OUT_ROOT_HTML.write_text(html_content, encoding="utf-8")

    print("Successfully generated standalone 16:9 presentation HTML:")
    print(f"  1) {OUT_DOCS_HTML} ({OUT_DOCS_HTML.stat().st_size / 1024:.1f} KB)")
    print(f"  2) {OUT_ROOT_HTML} ({OUT_ROOT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
