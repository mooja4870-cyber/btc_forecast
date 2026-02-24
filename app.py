"""
BTC Price Prediction App — Streamlit Dashboard
================================================
Premium dark-theme dashboard with:
- Tab 1: Model Overview (correlation heatmap, performance metrics, feature importance)
- Tab 2: Price History & Validation (actual vs predicted overlay)
- Tab 3: Future Prediction (target return → date, holding period → return)
"""

import os
import sys
import json
import warnings
import re
import html as html_lib
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DIR, MODELS_DIR,
)
from src.predictor import (
    estimate_target_return_date,
    estimate_return_at_date,
    load_latest_model,
    predict_future_path,
)
from src.config import cfg as ml_cfg


def _phase_num_from_name(name: str):
    if isinstance(name, str) and name.startswith("phase"):
        suffix = name.replace("phase", "")
        if suffix.isdigit():
            return int(suffix)
    return None


def _range_year_text(date_range):
    if not date_range:
        return "N/A"
    start, end = date_range
    start_y = str(start)[:4] if start else "시작"
    end_y = str(end)[:4] if end else "현재"
    return f"{start_y}–{end_y}"


def _build_phase_context():
    phases_cfg = ml_cfg.model_config.get("phases", {})
    items = []
    for phase_name, phase_cfg in phases_cfg.items():
        num = _phase_num_from_name(phase_name)
        if num is None:
            continue
        items.append((num, phase_cfg))
    items.sort(key=lambda x: x[0])

    if not items:
        # Safe fallback for legacy configs
        items = [
            (1, {"train": ["2014-01-01", "2020-12-31"], "val": ["2021-01-01", "2023-12-31"]}),
            (2, {"train": ["2014-01-01", "2023-12-31"], "val": ["2024-01-01", None]}),
            (3, {"train": ["2014-01-01", None], "val": None}),
        ]

    phase_ids = [p for p, _ in items]
    phase_cfg_by_id = {p: c for p, c in items}
    validation_phase_ids = [p for p, c in items if c.get("val")]
    production_phase_id = phase_ids[-1]
    eval_phase_id = validation_phase_ids[-1] if validation_phase_ids else production_phase_id

    zone_colors = [
        "rgba(99,102,241,0.08)",
        "rgba(6,182,212,0.08)",
        "rgba(34,197,94,0.08)",
        "rgba(245,158,11,0.08)",
        "rgba(244,63,94,0.08)",
        "rgba(168,85,247,0.08)",
    ]
    validation_zones = []
    for idx, phase_id in enumerate(validation_phase_ids):
        val_range = phase_cfg_by_id[phase_id].get("val")
        if not val_range:
            continue
        label = f"Phase {phase_id} 검증 ({_range_year_text(val_range)})"
        color = zone_colors[idx % len(zone_colors)]
        validation_zones.append((tuple(val_range), label, color))

    return {
        "phase_ids": phase_ids,
        "phase_cfg_by_id": phase_cfg_by_id,
        "validation_phase_ids": validation_phase_ids,
        "validation_zones": validation_zones,
        "production_phase_id": production_phase_id,
        "eval_phase_id": eval_phase_id,
    }


PHASE_CONTEXT = _build_phase_context()
PHASE_IDS = PHASE_CONTEXT["phase_ids"]
PHASE_CFG_BY_ID = PHASE_CONTEXT["phase_cfg_by_id"]
VALIDATION_PHASE_IDS = PHASE_CONTEXT["validation_phase_ids"]
VALIDATION_ZONES = PHASE_CONTEXT["validation_zones"]
PRODUCTION_PHASE_ID = PHASE_CONTEXT["production_phase_id"]
EVAL_PHASE_ID = PHASE_CONTEXT["eval_phase_id"]

# ================================================================
#  Page Config
# ================================================================
st.set_page_config(
    page_title="BTC 가격 예측 대시보드",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh whole dashboard every 300 seconds (5 minutes) for real-time metric cards.
st.components.v1.html(
    """
    <script>
    setTimeout(function () {
        window.parent.location.reload();
    }, 300000);
    </script>
    """,
    height=0,
)

# ================================================================
#  Custom CSS — Premium Dark Theme
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.4);
        --bg-dark: #0a0e1a;
        --card-bg: rgba(30, 41, 59, 0.5);
        --card-border: rgba(99, 102, 241, 0.15);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
        --font-main: 'Inter', sans-serif;
        --font-heading: 'Outfit', sans-serif;
    }

    /* Global Typography */
    .stApp {
        font-family: var(--font-main);
        color: var(--text-main);
    }
    .stApp p:not([class*="material"]):not(.material-symbols-rounded), 
    .stApp label:not([class*="material"]):not(.material-symbols-rounded) {
        font-family: var(--font-main) !important;
    }
    
    /* Force Material Symbols Font for all Streamlit Icons */
    .material-symbols-rounded, 
    .material-symbols-outlined,
    [class*="material-symbol"],
    [data-testid*="Icon"] {
        font-family: "Material Symbols Rounded" !important;
        font-feature-settings: 'liga' !important;
    }

    h1, h2, h3, h4, .stTabs [data-baseweb="tab"] {
        font-family: var(--font-heading) !important;
        letter-spacing: -0.02em;
        font-size: 1.1rem !important; /* Cap global headers at tab size */
        margin-bottom: 0 !important;
    }

    /* Main background & Force zero top padding */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a0e28 0%, #030712 90%);
    }
    
    [data-testid="block-container"] {
        padding-top: 0.5rem !important; /* This is about 2mm of space */
        padding-bottom: 0rem !important;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
        height: 0px !important;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(99, 102, 241, 0.05);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-dim) !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }
    
    /* ── Premium Metric Card (Custom HTML) ── */
    .premium-metric-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(99, 102, 241, 0.05);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 4px;
    }
    .premium-metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label { font-size: 0.75rem !important; color: var(--text-dim) !important; margin-bottom: 2px !important; font-weight: 600 !important; }
    .metric-value { font-size: 1.7rem !important; font-weight: 800 !important; margin-bottom: 1px !important; letter-spacing: -0.01em !important; }
    .metric-delta { font-size: 0.85rem !important; font-weight: 700 !important; display: flex !important; align-items: center !important; gap: 4px !important; }
    .delta-up { color: #ff4b4b !important; }   /* Red for Up */
    .delta-down { color: #3b82f6 !important; } /* Blue for Down */
    .delta-neutral { color: #94a3b8 !important; }
    .metric-source { font-size: 0.6rem !important; color: #64748b !important; margin-top: 4px !important; }
    .premium-metric-card.metric-up .metric-value,
    .premium-metric-card.metric-up .metric-delta,
    .premium-metric-card.metric-up .metric-delta span {
        color: #ff4b4b !important;
    }
    .premium-metric-card.metric-down .metric-value,
    .premium-metric-card.metric-down .metric-delta,
    .premium-metric-card.metric-down .metric-delta span {
        color: #3b82f6 !important;
    }
    .premium-metric-card.metric-neutral .metric-value,
    .premium-metric-card.metric-neutral .metric-delta,
    .premium-metric-card.metric-neutral .metric-delta span {
        color: #94a3b8 !important;
    }

    /* Shrink buttons by approx 77% */
    .stButton button {
        font-size: 0.77rem !important;
        padding: 0.2rem 0.5rem !important;
        min-height: 2rem !important;
        line-height: 1.2 !important;
    }

    /* Monumental Title Override */
    .monumental-title {
        font-size: 2.6rem !important;
        background: linear-gradient(135deg, #818cf8, #38bdf8, #f59e0b) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
        line-height: 1.0 !important;
        margin-bottom: 10px !important;
        font-family: var(--font-heading) !important;
        display: block !important;
        text-align: center !important;
    }
    
    /* Tab styling — Premium Glass Buttons */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.3) !important;
        border-radius: 24px;
        gap: 12px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 0 !important; /* Force zero gap below navigation */
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 32px !important;
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        color: var(--text-dim) !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0 20px !important;
        margin-right: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }

    /* Main 4 tabs only: 133% font/icon size */
    [data-testid="stAppViewContainer"] .main .stTabs [data-baseweb="tab"] {
        font-size: 1.46rem !important; /* 1.1rem * 1.33 */
        height: 42px !important;
        padding: 0 22px !important;
        line-height: 1.1 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.9) !important;
        color: white !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.5);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Aggressive Zero-Gap for Tabs content */
    [data-testid="stTabContent"] {
        padding-top: 0 !important;
        margin-top: 5px !important; 
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Sidebar Overhaul */
    [data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    /* Cards / containers — Ultimate Glass */
    .glass-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.1));
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px;
        margin-top: 2px !important;
        margin-bottom: 1.5rem !important;
        backdrop-filter: blur(20px) saturate(180%);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 0 0 1px rgba(255, 255, 255, 0.03);
    }
    
    /* Success/info boxes */
    .prediction-result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 2px !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.1);
    }
    
    .warning-result {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        margin-top: 2px !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 10px 25px rgba(245, 158, 11, 0.1);
    }
    
    /* Input styling */
    .stNumberInput, .stSlider {
        color: #e2e8f0;
    }
    
    /* Metric styling — Adjusted to 1.7rem for prominence */
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important; 
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.8rem !important;
    }
    
    h3 {
        font-size: 1.1rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Yellow dot icon only (70% size) */
    .yellow-dot {
        font-size: 70% !important;
        line-height: 1 !important;
        vertical-align: middle !important;
    }

    /* Sidebar text line */
    .sidebar-tight-line {
        font-family: var(--font-heading) !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
        text-align: left !important;
    }

    /* Sidebar info boxes (for latest training time / total points) */
    .sidebar-info-box {
        background: rgba(30, 41, 59, 0.55) !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin: 0 0 6px 0 !important;
    }

    /* Sidebar popover labels: left aligned */
    [data-testid="stSidebar"] [data-testid="stPopover"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button p,
    [data-testid="stSidebar"] [data-testid="stPopover"] button span {
        text-align: left !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button svg {
        margin-left: auto !important;
        flex-shrink: 0 !important;
    }

    /* Standard Button Slimming */
    div.stButton > button {
        height: 24px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        line-height: 24px !important;
        font-size: 0.8rem !important;
        border-radius: 6px !important;
    }
    
    /* Divider — Reduced margin by 75% for tighter spacing */
    hr {
        border-color: rgba(99, 102, 241, 0.2) !important;
        margin: 0.6rem 0 !important;
    }
    
    /* ── Radio Buttons as Premium Buttons ── */
    [data-testid="stRadio"] > label {
        display: none !important; /* Hide the "세부 예측 모드 선택" header label */
    }
    
    [data-testid="stRadio"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    [data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 0 20px !important; 
        height: 32px !important; 
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        cursor: pointer !important;
    }
    
    [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        background: rgba(99, 102, 241, 0.1) !important;
        transform: translateY(-2px);
    }
    
    /* Hide the radio circle/marker COMPLETELY */
    [data-testid="stRadio"] div[data-testid="stMarker"], 
    [data-testid="stRadio"] div[data-testid="stMarker"] + div {
        display: none !important;
    }
    
    /* Make the text inside labels match the requested style */
    [data-testid="stRadio"] div[role="radiogroup"] label p {
        color: #94a3b8 !important;
        font-size: 1.1rem !important; 
        font-weight: 700 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }
    
    /* Selected/Checked state styling */
    [data-testid="stRadio"] div[role="radiogroup"] div[aria-checked="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    [data-testid="stRadio"] div[role="radiogroup"] div[aria-checked="true"] label p {
        color: white !important;
    }
    
    /* ── Global text color overrides for dark theme ── */
    .stApp, .stApp p, .stApp span, .stApp label {
        color: #e2e8f0;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown li, .stMarkdown ul, .stMarkdown ol {
        color: #e2e8f0 !important;
    }
    
    /* Radio & Checkbox labels */
    .stRadio label, .stRadio p, .stRadio span,
    .stCheckbox label, .stCheckbox p, .stCheckbox span {
        color: #e2e8f0 !important;
    }
    
    /* Select box */
    .stSelectbox label, .stSelectbox span,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #e2e8f0 !important;
    }
    
    /* Number input & Slider labels */
    .stNumberInput label, .stNumberInput p,
    .stSlider label, .stSlider p, .stSlider span {
        color: #e2e8f0 !important;
    }
    
    /* Streamlit alert/info/warning boxes */
    .stAlert p, .stAlert span, .stAlert div,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #e2e8f0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader, .streamlit-expanderHeader p {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {
        color: #e2e8f0 !important;
    }
    
    /* Caption & small text */
    .stCaption, small, .stTooltipIcon {
        color: #94a3b8 !important;
    }

    /* Dialog (Modal) Width Settings: current 1550px -> 155% (=2402.5px) */
    :root {
        --trend-dialog-width: min(95vw, 2402px);
    }
    div[data-testid="stModal"] > div {
        max-width: var(--trend-dialog-width) !important;
        width: var(--trend-dialog-width) !important;
    }
    div[data-testid="stDialog"] > div[role="dialog"] {
        max-width: var(--trend-dialog-width) !important;
        width: var(--trend-dialog-width) !important;
    }
    /* Cover older Streamlit versions or alternative inner tags */
    div[data-modal-container="true"] > div {
        max-width: var(--trend-dialog-width) !important;
        width: var(--trend-dialog-width) !important;
    }

    /* Sidebar Expander: keep default Streamlit layout (prevents header overlap) */
</style>
""", unsafe_allow_html=True)

# ================================================================
#  Plotly Theme
# ================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.3)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
    legend=dict(
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="rgba(255,255,255,0.05)",
        borderwidth=1,
        font=dict(size=10)
    ),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False),
)

COLORS = {
    "primary": "#6366f1",     # Indigo Neon
    "secondary": "#22d3ee",   # Cyan Neon
    "accent": "#f59e0b",      # Amber
    "success": "#10b981",     # Emerald
    "danger": "#ef4444",      # Rose
    "btc": "#f7931a",         # Bitcoin Orange
    "gold": "#fbbf24",        # Rich Gold
    "sp500": "#34d399",       # Green
    "nasdaq": "#38bdf8",      # Sky Blue
    "oil": "#a78bfa",         # Violet
}


DISPLAY_KRW_PER_USD = 0.0


def apply_yaxis_floor_40k(fig, y_values, floor=40000.0):
    vals = []
    for v in y_values:
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        except Exception:
            continue

    y_max = max(vals) * 1.05 if vals else floor * 1.125
    y_max = max(y_max, floor * 1.125)
    fig.update_layout(yaxis=dict(range=[floor, y_max]))


def resolve_display_krw_rate(file_data=None):
    global DISPLAY_KRW_PER_USD
    if DISPLAY_KRW_PER_USD and DISPLAY_KRW_PER_USD > 0:
        return float(DISPLAY_KRW_PER_USD)
    try:
        df = file_data if file_data is not None else load_merged_data()
        if "krw_close" in df.columns:
            series = pd.to_numeric(df["krw_close"], errors="coerce").dropna()
            if not series.empty:
                DISPLAY_KRW_PER_USD = float(series.iloc[-1])
                return DISPLAY_KRW_PER_USD
    except Exception:
        pass
    return 0.0


def usd_to_krw(value, krw_per_usd=None):
    try:
        if value is None:
            return None
        v = float(value)
        if not np.isfinite(v):
            return None
    except Exception:
        return None
    rate = krw_per_usd if krw_per_usd is not None else resolve_display_krw_rate()
    try:
        r = float(rate)
    except Exception:
        return None
    if r <= 0:
        return None
    return v * r


def render_yellow_heading(text: str, level: int = 2, tooltip: str = None):
    tag = "h2" if level == 2 else "h3"
    safe_text = html_lib.escape(str(text))
    if tooltip:
        safe_tip = html_lib.escape(str(tooltip), quote=True).replace("\n", "&#10;")
        content = (
            f"<{tag}><span class='yellow-dot'>🟡</span> "
            f"<span title=\"{safe_tip}\" "
            "style=\"cursor:help; text-decoration: underline dotted rgba(148,163,184,0.8); "
            "text-underline-offset: 3px;\">"
            f"{safe_text}</span></{tag}>"
        )
    else:
        content = f"<{tag}><span class='yellow-dot'>🟡</span> {safe_text}</{tag}>"
    st.markdown(
        content,
        unsafe_allow_html=True,
    )


def format_r2(value):
    try:
        v = float(value)
        if np.isfinite(v):
            return f"{v:.3f}"
    except Exception:
        pass
    return value if value is not None else "N/A"


def describe_feature_term(name: str) -> str:
    if not isinstance(name, str):
        return "자동 생성된 피처입니다."

    exact = {
        "days_since_halving": "최근 BTC 반감기 이후 경과 일수입니다.",
        "days_to_fomc": "다음 FOMC 회의까지 남은 일수입니다.",
        "btc_above_ma200": "BTC 종가가 200일 이동평균 위면 1, 아니면 0입니다.",
        "dxy_close": "달러인덱스(DXY) 종가입니다.",
        "fear_greed": "암호화폐 공포/탐욕 지수입니다.",
        "hashrate": "비트코인 네트워크 해시레이트입니다.",
    }
    if name in exact:
        return exact[name]

    m = re.match(r"^expected_policy_rate_(\d+)m$", name)
    if m:
        return f"{m.group(1)}개월 후 예상 정책금리(시장 기대치)입니다."

    m = re.match(r"^btc_close_lag(\d+)$", name)
    if m:
        return f"BTC 종가의 {m.group(1)}일 전 값입니다."

    m = re.match(r"^btc_close_ma(\d+)$", name)
    if m:
        return f"BTC 종가의 {m.group(1)}일 이동평균입니다."

    m = re.match(r"^btc_ma(\d+)_pct$", name)
    if m:
        return f"BTC 가격이 {m.group(1)}일 이동평균 대비 얼마나 떨어져/올라 있는지(%)입니다."

    m = re.match(r"^oil_fut_close_ret(\d+)d$", name)
    if m:
        return f"원유 선물 종가의 {m.group(1)}일 수익률입니다."

    m = re.match(r"^oil_close_ret(\d+)d$", name)
    if m:
        return f"현물 원유 종가의 {m.group(1)}일 수익률입니다."

    m = re.match(r"^oil_fut_roll_return_(\d+)d$", name)
    if m:
        return f"원유 선물 롤오버 수익률({m.group(1)}일 기준)입니다."

    if name.startswith("log_"):
        return "원본 값의 로그 변환 피처입니다."

    if name.startswith("rate_") and name.endswith("_close"):
        return "금리 관련 지표의 변화율/수익률 피처입니다."

    if name.endswith("_close"):
        return "해당 자산/지표의 종가(또는 레벨) 값입니다."

    return "자동 생성된 엔지니어링 피처입니다."


def render_feature_tooltip_list(feature_names, height_px: int = 240):
    rows = []
    for col in feature_names:
        label = html_lib.escape(str(col))
        desc = html_lib.escape(describe_feature_term(str(col)), quote=True)
        rows.append(
            f"<div title=\"{desc}\" "
            "style=\"padding:4px 6px;border-bottom:1px solid rgba(148,163,184,0.12);"
            "font-family:var(--font-main);font-size:0.88rem;color:#e2e8f0;\">"
            f"{label}</div>"
        )

    st.markdown(
        f"""
        <div style="max-height:{height_px}px;overflow-y:auto;border:1px solid rgba(148,163,184,0.25);
                    border-radius:8px;background:rgba(15,23,42,0.35);">
            {''.join(rows)}
        </div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:6px;">
            각 용어에 마우스를 올리면 설명이 표시됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_expected_return_cell(value):
    """Color rule: + red, - blue, neutral gray."""
    text = str(value).strip()
    if text in {"", "-", "0.0%", "+0.0%", "-0.0%"}:
        return "color: #94a3b8; font-weight: 700;"
    if text.startswith("+"):
        return "color: #ff4b4b; font-weight: 800;"
    if text.startswith("-"):
        return "color: #3b82f6; font-weight: 800;"
    return ""


# ================================================================
#  Data Loading (cached)
# ================================================================
@st.cache_data(ttl=60)
def load_featured_data():
    path = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data(ttl=60)
def load_merged_data():
    path = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _resolve_phase_artifact_path(phase: int, filename: str) -> str:
    latest_path = os.path.join(MODELS_DIR, "latest", f"phase{phase}", filename)
    if os.path.exists(latest_path):
        return latest_path
    return os.path.join(MODELS_DIR, f"phase{phase}", filename)


@st.cache_data(ttl=60)
def load_phase_metrics(phase: int):
    path = _resolve_phase_artifact_path(phase, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_feature_importance(phase: int):
    path = _resolve_phase_artifact_path(phase, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_val_predictions(phase: int):
    path = _resolve_phase_artifact_path(phase, "val_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_feature_expansion_status():
    path = os.path.join(ml_cfg.monitoring.get("report_dir", "data/reports"), "feature_expansion_status.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_champion_challenger_report():
    path = os.path.join(ml_cfg.monitoring.get("report_dir", "data/reports"), "champion_challenger_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_transformer_val_metrics(phase: int, horizon: int = 30):
    """Compute transformer validation metrics for a phase/horizon on featured data."""
    try:
        from src.predictor import load_transformer_model
        tf = load_transformer_model(horizon)
        if not tf:
            return None

        phase_cfg = PHASE_CFG_BY_ID.get(phase, {})
        val_range = phase_cfg.get("val")
        if not val_range:
            return None

        feat_df = load_featured_data().sort_index()
        target_col = f"target_log_return_{horizon}d"
        if target_col not in feat_df.columns:
            return None

        v_start = pd.to_datetime(val_range[0]) if val_range[0] is not None else feat_df.index.min()
        v_end = pd.to_datetime(val_range[1]) if val_range[1] is not None else feat_df.index.max()
        eval_df = feat_df.loc[(feat_df.index >= v_start) & (feat_df.index <= v_end)].copy()
        eval_df = eval_df.dropna(subset=[target_col])
        if eval_df.empty:
            return None

        import torch

        feat_cols = tf["feature_names"]
        seq_len = int(tf.get("seq_len", 60))
        model = tf["model"]
        stats = tf["scaler_stats"]
        mean = np.asarray(stats["mean"])
        std = np.asarray(stats["std"])
        std = np.where(std == 0, 1.0, std)

        all_feat = feat_df.reindex(columns=feat_cols).fillna(0.0)
        all_idx = all_feat.index

        preds = []
        for d in eval_df.index:
            idx_loc = all_idx.get_indexer([d], method="nearest")[0]
            start_loc = max(0, idx_loc - seq_len + 1)
            x_seq = all_feat.iloc[start_loc:idx_loc + 1].values
            if len(x_seq) < seq_len:
                pad_len = seq_len - len(x_seq)
                x_seq = np.pad(x_seq, ((pad_len, 0), (0, 0)), mode="constant")
            x_scaled = (x_seq - mean) / std
            x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_lr = float(model(x_tensor).item())
            preds.append(pred_lr)

        y_true = eval_df[target_col].values.astype(float)
        y_pred = np.asarray(preds, dtype=float)
        if len(y_true) == 0:
            return None

        diff = y_true - y_pred
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        direction_accuracy = float(np.mean((y_true > 0) == (y_pred > 0)))

        price_mape_pct = None
        if "btc_close" in feat_df.columns:
            base_price = feat_df["btc_close"].reindex(eval_df.index).astype(float)
            actual_future = feat_df["btc_close"].shift(-horizon).reindex(eval_df.index).astype(float)
            pred_future = base_price * np.exp(y_pred)
            mask = actual_future.notna() & (actual_future != 0)
            if mask.any():
                mape = np.mean(np.abs((pred_future[mask] - actual_future[mask]) / actual_future[mask])) * 100.0
                price_mape_pct = float(mape)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "price_mape_pct": price_mape_pct,
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_transformer_val_predictions(phase: int, horizon: int = 30):
    """Build transformer validation prediction frame (date, actual/pred log-return)."""
    try:
        from src.predictor import load_transformer_model
        tf = load_transformer_model(horizon)
        if not tf:
            return pd.DataFrame()

        phase_cfg = PHASE_CFG_BY_ID.get(phase, {})
        val_range = phase_cfg.get("val")
        if not val_range:
            return pd.DataFrame()

        feat_df = load_featured_data().sort_index()
        target_col = f"target_log_return_{horizon}d"
        if target_col not in feat_df.columns:
            return pd.DataFrame()

        v_start = pd.to_datetime(val_range[0]) if val_range[0] is not None else feat_df.index.min()
        v_end = pd.to_datetime(val_range[1]) if val_range[1] is not None else feat_df.index.max()
        eval_df = feat_df.loc[(feat_df.index >= v_start) & (feat_df.index <= v_end)].copy()
        eval_df = eval_df.dropna(subset=[target_col])
        if eval_df.empty:
            return pd.DataFrame()

        import torch

        feat_cols = tf["feature_names"]
        seq_len = int(tf.get("seq_len", 60))
        model = tf["model"]
        stats = tf["scaler_stats"]
        mean = np.asarray(stats["mean"])
        std = np.asarray(stats["std"])
        std = np.where(std == 0, 1.0, std)

        all_feat = feat_df.reindex(columns=feat_cols).fillna(0.0)
        all_idx = all_feat.index

        preds = []
        for d in eval_df.index:
            idx_loc = all_idx.get_indexer([d], method="nearest")[0]
            start_loc = max(0, idx_loc - seq_len + 1)
            x_seq = all_feat.iloc[start_loc:idx_loc + 1].values
            if len(x_seq) < seq_len:
                pad_len = seq_len - len(x_seq)
                x_seq = np.pad(x_seq, ((pad_len, 0), (0, 0)), mode="constant")
            x_scaled = (x_seq - mean) / std
            x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                preds.append(float(model(x_tensor).item()))

        out = pd.DataFrame({
            "date": eval_df.index,
            "actual_log_return": eval_df[target_col].values.astype(float),
            "predicted_log_return": np.asarray(preds, dtype=float),
        })
        if "btc_close" in feat_df.columns:
            out["actual_btc_close"] = feat_df["btc_close"].reindex(eval_df.index).values.astype(float)

        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_latest_pipeline_run_from_log():
    log_path = os.path.join(ml_cfg.monitoring.get("logs_dir", "data/logs"), "cron_job.log")
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = list(deque(f, maxlen=500))
    except Exception:
        return {}

    run_id = None
    run_ts = None
    full_run_id = None
    full_run_ts = None
    for line in reversed(lines):
        if run_id is None:
            m = re.search(r"Starting Pipeline Run:\s*(run_\d{8}_\d{6})", line)
            if m:
                run_id = m.group(1)
                try:
                    run_ts = datetime.strptime(run_id.replace("run_", ""), "%Y%m%d_%H%M%S")
                except Exception:
                    run_ts = None
        if full_run_id is None:
            m2 = re.search(r"Pipeline finished successfully\. Run ID:\s*(run_\d{8}_\d{6})", line)
            if m2:
                full_run_id = m2.group(1)
                try:
                    full_run_ts = datetime.strptime(full_run_id.replace("run_", ""), "%Y%m%d_%H%M%S")
                except Exception:
                    full_run_ts = None
        if run_id is not None and full_run_id is not None:
            break
    return {
        "run_id": run_id,
        "run_ts": run_ts,
        "full_run_id": full_run_id,
        "full_run_ts": full_run_ts,
    }


@st.cache_data(ttl=60)
def load_latest_model_training_run():
    """
    Return the run_id currently used by production predictions/reality check.
    Choose the newest run_id among:
    - models/latest symlink target
    - models/LATEST.txt
    - models/latest/LATEST.txt
    - latest fully-completed pipeline run from cron_job.log
    """
    def _extract_run_id(raw: str):
        if not raw:
            return None
        m = re.search(r"(run_\d{8}_\d{6})", str(raw))
        return m.group(1) if m else None

    def _run_id_to_ts(rid: str):
        try:
            return datetime.strptime(rid.replace("run_", ""), "%Y%m%d_%H%M%S")
        except Exception:
            return None

    latest_link = os.path.join(MODELS_DIR, "latest")
    candidates = []

    try:
        if os.path.exists(latest_link):
            rid = _extract_run_id(os.path.realpath(latest_link))
            if rid:
                candidates.append(("symlink", rid))
    except Exception:
        pass

    for p, source in [
        (os.path.join(MODELS_DIR, "LATEST.txt"), "models_latest_txt"),
        (os.path.join(MODELS_DIR, "latest", "LATEST.txt"), "latest_inner_txt"),
    ]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    rid = _extract_run_id(f.read().strip())
                if rid:
                    candidates.append((source, rid))
            except Exception:
                pass

    try:
        pinfo = load_latest_pipeline_run_from_log() or {}
        rid = _extract_run_id(pinfo.get("full_run_id"))
        if rid:
            candidates.append(("cron_full_run", rid))
    except Exception:
        pass

    run_id = None
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda x: _run_id_to_ts(x[1]) or datetime.min,
            reverse=True,
        )
        run_id = candidates[0][1]

    run_compact = run_id.replace("run_", "") if run_id else "deployed"
    run_display = "배포된 모델"
    if run_id:
        try:
            run_display = datetime.strptime(run_compact, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
        except Exception:
            run_display = run_compact
    return {"run_id": run_id, "run_compact": run_compact, "run_display": run_display}


# ================================================================
#  SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🔶 신뢰도 검증 (Reality Check)")
    rc_meta = {}
    try:
        r_path = os.path.join(PROCESSED_DIR, "reliability_result.json")
        if os.path.exists(r_path):
            with open(r_path) as f:
                r_results = json.load(f)
            rc_meta = r_results.get("_meta", {}) if isinstance(r_results, dict) else {}
            krw_rate_sidebar = resolve_display_krw_rate()
            if krw_rate_sidebar <= 0:
                krw_rate_sidebar = 1.0

            h_tabs = st.tabs(["1년 전", "1달 전", "1일 전"])
            horizons = [("365", "1년 전"), ("30", "1달 전"), ("1", "1일 전")]

            for i, (h_key, label) in enumerate(horizons):
                with h_tabs[i]:
                    res = r_results.get(h_key)
                    if res:
                        pred_today_krw = float(res.get("predicted_price_today", 0.0)) * krw_rate_sidebar
                        actual_today_krw = float(res.get("actual_price_today", 0.0)) * krw_rate_sidebar
                        status_icon = "✅" if res.get("passed") else "⚠️"
                        status_color = "#10b981" if res.get("passed") else "#f59e0b"
                        status_msg = "PASS" if res.get("passed") else "WARNING"

                        st.markdown(f"""
                        <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid {status_color}44;
                                    border-radius: 12px; padding: 12px; border-left: 4px solid {status_color};
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                                <span style='font-weight: 700; color: {status_color}; font-size: 0.8rem;'>
                                    {status_icon} {status_msg}
                                </span>
                                <span style='font-family: var(--font-heading); font-weight: 800; color: white; font-size: 0.85rem;'>
                                    {res['multiplier']:.2f}x
                                </span>
                            </div>
                            <div style='font-size: 0.8rem; color: var(--text-dim); line-height: 1.4;'>
                                <strong>{label} 시점</strong> 데이터로 <strong>오늘</strong> 가격 예측<br>
                                (운영 모델 가중치 사용)<br>
                                <div style='display: flex; justify-content: space-between; margin-top: 4px; color: #e2e8f0;'>
                                    <span>예측</span><span>₩{pred_today_krw:,.0f}</span>
                                </div>
                                <div style='display: flex; justify-content: space-between; color: #e2e8f0;'>
                                    <span>실제</span><span>₩{actual_today_krw:,.0f}</span>
                                </div>
                                <div style='text-align: right; margin-top: 4px; font-weight: 700; color: {status_color};'>
                                    Error: {res['error_pct']:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("데이터가 없습니다.")

            st.markdown(
                "<div style='height:8px;'></div><div style='font-size:77%; color:#94a3b8;'>※ 판정 기준: 예측이 실제의 0.5x~2.0x 이내\n</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("신뢰도 검증 결과가 없습니다.")
    except Exception as e:
        st.error(f"검증 로드 실패: {e}")
    try:
        df = load_merged_data()
        fdf = load_featured_data()
        model_run = load_latest_model_training_run()
        rc_recomputed_at = rc_meta.get("recomputed_at")
        rc_data_ts = rc_meta.get("data_last_timestamp")

        if rc_recomputed_at:
            try:
                rc_recomputed_at = (
                    pd.to_datetime(rc_recomputed_at, utc=True)
                    .tz_convert("Asia/Seoul")
                    .strftime("%Y-%m-%d %H:%M")
                )
            except Exception:
                pass
            st.caption(f"Reality Check 재계산 시각: {rc_recomputed_at}")
        if rc_data_ts:
            try:
                rc_data_ts = pd.to_datetime(rc_data_ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            st.caption(f"Reality Check 기준 데이터 시각: {rc_data_ts}")
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        with st.popover("🔶 이 모델의 최신 학습시각", use_container_width=True):
            st.markdown(f"**{model_run['run_display']}**")
        with st.popover("🔶 총 데이터 포인트", use_container_width=True):
            st.markdown(f"**{len(df):,}일**")
        with st.popover(f"🔶 변수 수 (원시) : {df.shape[1]}개", use_container_width=True):
            st.dataframe(
                pd.DataFrame({"원시 변수": list(df.columns)}),
                use_container_width=True,
                hide_index=True,
                height=230,
            )
        with st.popover(f"🔶 피처 수 (엔지니어링 후) : {fdf.shape[1]}개", use_container_width=True):
            render_feature_tooltip_list(list(fdf.columns), height_px=240)

        with st.popover("🔶 모델 정보", use_container_width=True):
            lines = []
            for phase_id in PHASE_IDS:
                phase_cfg = PHASE_CFG_BY_ID.get(phase_id, {})
                train_txt = _range_year_text(phase_cfg.get("train"))
                val_range = phase_cfg.get("val")
                if val_range:
                    val_txt = _range_year_text(val_range)
                    lines.append(f"- **Phase {phase_id}**: {train_txt} 학습, {val_txt} 검증")
                else:
                    lines.append(f"- **Phase {phase_id}**: {train_txt} 학습")
            st.markdown("\n".join(lines))

        with st.popover("🔶 예측 상세 정보", use_container_width=True):
            st.markdown("""
            **참고: 예측에 사용한 정보**
            
            아래 내용은 현재 시스템에 **실제 구현된** 입력 변수와 모델링 방식입니다.
            
            **가격/거래량 기반 피처**  
            OHLCV, 로그수익률 래그, 이동평균(SMA/EMA), RSI, MACD, 볼린저밴드, 실현변동성
            
            **거시/시장 보조지표**  
            10년물 국채금리 (13주X), KRW/USD, Gold, Oil, S&P500, NASDAQ, DXY, 연준금리, CPI, M2, 실업률
            *주: VIX 지수는 연동되지 않음*
            
            **달력/이벤트 피처**  
            요일, 월, 반감기 사이클, 주요 지정학적 이벤트(전쟁, 규제 등)
            
            ---
            
            **참고: 모델링 방식**
            
            - **사용 모델**: TimeSformer (Transformer) 단일 모델 체계
            - **예측 방식**: **Direct Multi-Horizon** (1일~365일 각 시점별 독립 모델)
            - **검증 방식**: 시계열 워크포워드 (Time-series Split)
            - **평가지표**: RMSE, MAE, R², 방향정확도
            - **최종 모델 선정**: 검증 R² 점수 최고점 모델 자동 선택
            - **불확실성**: 단일 점 추정 (구간 예측 미구현)
            """)

        with st.popover("🔶 자동 갱신 파이프라인", use_container_width=True):
            st.markdown("""
            **시스템 자동화 및 데이터 갱신 워크플로우**
            
            본 대시보드는 최신 데이터 유지를 위해 아래와 같은 **MLOps 파이프라인**을 **매일 00:00(일 1회)** 자동으로 수행합니다.
            
            1. **데이터 수집 (Data Collection)**  
               매일 00:00, 실시간 API를 통해 BTC 가격 및 거시 지표 최신본 갱신
               
            2. **피처 엔지니어링 (Feature Engineering)**  
               신규 데이터를 포함한 모든 보조지표(RSI, 이동평균 등) 재계산 및 병합
               
            3. **모델 재학습 & 평가 (Retraining & Eval)**  
               최신 데이터를 학습 세트에 포함하여 모델 성능 업데이트 및 자동 평가
               
            4. **모델 배포 (Model Promotion)**  
               기존 모델 대비 성능 향상이 확인된 경우에만 새로운 가중치로 자동 교체
               
            5. **성능 모니터링 (Monitoring)**  
               예측 오차와 데이터 드리프트를 매 실행 시점마다 추적하여 시스템 건전성 상시 확인
               
            ---
            **💡 궁금할 땐 어디를 볼까요? (검증 가이드)**
            
            인공지능의 작업 결과를 직접 확인하고 싶다면 아래 요소들을 체크해 보세요.
            
            - **📂 폴더 구조**: `config/config.yaml`(설정), `data/reports/`(성적표), `models/latest`(최신 모델 경로) 등 폴더 구성을 확인하세요.
            - **📝 manifest.json**: 모델 폴더마다 생성되는 '실행 영수증'으로, 학습 파라미터와 정확도(R²) 근거가 기록됩니다.
            - **📊 monitoring_report.md**: 주기적으로 업데이트되는 시스템 성적표입니다. 날짜와 성능 추이를 직접 확인하세요.
            - **⏰ cron_job.log**: 시스템이 정해진 시간에 실제로 깨어나서 작동했는지 확인할 수 있는 자동 실행 기록입니다.
            """)
    except Exception:
        st.warning("데이터를 불러올 수 없습니다")

    st.markdown("---")
    
    st.markdown(
        "<div style='text-align:center;color:#64748b;font-size:0.8em;'>"
        "Built with Streamlit + Transformer (TimeSformer)<br>"
        "Data: yfinance, Blockchain.com, Alternative.me"
        "</div>",
        unsafe_allow_html=True,
    )

# ================================================================
#  HEADER
# ================================================================
st.markdown("""
<div class='glass-card' style='text-align:center; padding: 3px 2px; border-radius: 4px; margin-bottom: 1px; margin-top: 0px; 
     background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.2));'>
    <div class="monumental-title">
    &nbsp;₿ BTC INTELLIGENCE DASHBOARD
    </div>
    <p style='color: #cbd5e1; font-size: 0.6rem; font-weight: 400; letter-spacing: 0.05em; margin: 0; text-align: center;'>
    ADVANCED ANALYTICAL ENGINE • MULTI-HORIZON PROBABILISTIC FORECASTING
    </p>
</div>
""", unsafe_allow_html=True)

# Current BTC price display
# Current BTC price display
try:
    # ── Real-time Metrics Fetcher (Robust) ──
    @st.cache_data(ttl=60, show_spinner=False)
    def get_robust_price(ticker_symbol, asset_type="generic"):
        try:
            # Use external helper for complex logic (CCXT, Requests, yfinance)
            from src.data_fetcher import fetch_data_robust
            # Map ticker to symbol expected by fetcher
            if ticker_symbol == "BTC-USD": symbol = "BTC-USD"
            elif ticker_symbol == "WOORI_GOLDBANK_KRW": symbol = "WOORI_GOLDBANK_KRW"
            elif ticker_symbol == "SHINHAN_SILVER_KRW": symbol = "SHINHAN_SILVER_KRW"
            elif ticker_symbol == "GC=F": symbol = "GC=F"
            elif ticker_symbol == "^GSPC": symbol = "^GSPC"
            elif ticker_symbol == "KRW=X": symbol = "KRW=X"
            else: symbol = ticker_symbol
            
            return fetch_data_robust(symbol)
        except Exception:
            return None, None, None

    def get_realtime_metric(ticker_symbol, file_data, file_col, name, realtime_only=False):
        cache_path = os.path.join(PROCESSED_DIR, "realtime_metrics_cache.json")

        def load_cache():
            try:
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        return json.load(f)
            except Exception:
                pass
            return {}

        def save_cache(symbol, current, change, source):
            try:
                cache = load_cache()
                cache[symbol] = {
                    "current": float(current),
                    "change": float(change if change is not None else 0.0),
                    "source": source,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(cache_path, "w") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # 1. Try real-time
        current, change, source = get_robust_price(ticker_symbol)
        if current is not None:
            if change is None:
                change = 0.0
            save_cache(ticker_symbol, current, change, source)
            return float(current), float(change), source

        # 2. Last known real-time cache
        cached = load_cache().get(ticker_symbol)
        if cached and cached.get("current") is not None:
            return (
                float(cached.get("current")),
                float(cached.get("change", 0.0)),
                f"실시간 캐시({cached.get('updated_at', '-')})",
            )

        if realtime_only:
            return None, None, "실시간 소스 실패"

        # 3. Fallback to file (hard safety)
        if file_col in file_data.columns:
            series = file_data[file_col].dropna()
            if not series.empty:
                current = float(series.iloc[-1])
                prev = float(series.iloc[-2]) if len(series) > 1 else current
                change = (current - prev) / prev * 100 if prev else 0.0
                return current, change, f"파일({file_data.index[-1].date()})"

        # 4. Final guard: never return N/A
        return 0.0, 0.0, "보호값(실시간/캐시/파일 실패)"

    # Load file data once for fallback
    mdf = load_merged_data()

    # 1. KRW/USD (KRW=X)
    krw_p, krw_c, krw_s = get_realtime_metric("KRW=X", mdf, "krw_close", "KRW/USD", realtime_only=True)
    krw_rate = krw_p if (krw_p is not None and krw_p > 0) else resolve_display_krw_rate(mdf)
    if krw_rate and krw_rate > 0:
        DISPLAY_KRW_PER_USD = float(krw_rate)

    # 2. BTC (Upbit KRW preferred, otherwise USD->KRW conversion)
    btc_raw_p, btc_c, btc_s = get_realtime_metric("BTC-USD", mdf, "btc_close", "BTC")
    if isinstance(btc_s, str) and "Upbit KRW" in btc_s:
        btc_p = btc_raw_p
        btc_source = btc_s
    else:
        btc_p = usd_to_krw(btc_raw_p, krw_rate)
        btc_source = f"{btc_s} × KRW/USD 환산({krw_s})" if krw_p else btc_s

    # 3. Gold (Woori Gold Banking KRW)
    gold_p, gold_c, gold_s = get_realtime_metric("WOORI_GOLDBANK_KRW", mdf, "gold_close", "Gold", realtime_only=False)

    # 4. Silver (Shinhan SilverRush KRW)
    silver_p, silver_c, silver_s = get_realtime_metric("SHINHAN_SILVER_KRW", mdf, "silver_close", "Silver", realtime_only=False)

    # 5. KOSPI (^KS11)
    kospi_p, kospi_c, kospi_s = get_realtime_metric("^KS11", mdf, "kospi_close", "KOSPI", realtime_only=False)

    # 6. S&P 500 (^GSPC, USD original)
    sp_usd_p, sp_c, sp_s = get_realtime_metric("^GSPC", mdf, "sp500_close", "S&P500", realtime_only=False)
    sp_p = sp_usd_p
    sp_source = sp_s

    def render_premium_metric(label, value, delta_val, source):
        # Red/Blue convention: Red=Up, Blue=Down
        try:
            delta_val = float(delta_val)
            if not np.isfinite(delta_val):
                delta_val = 0.0
        except Exception:
            delta_val = 0.0

        delta_color = "#94a3b8"
        value_color = "#94a3b8"
        delta_icon = ""
        metric_state = "metric-neutral"
        if delta_val > 0:
            delta_color = "#ff4b4b" # Red for Up
            value_color = "#ff4b4b"
            delta_icon = "↑"
            metric_state = "metric-up"
        elif delta_val < 0:
            delta_color = "#3b82f6" # Blue for Down
            value_color = "#3b82f6"
            delta_icon = "↓"
            metric_state = "metric-down"
        
        # '실시간'을 현재 시간(시분초)으로 변환 후 노란색 굵은 텍스트로 하이라이트
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        source_html = source.replace("-실시간", f"-<span style='color: #fbbf24; font-weight: 800;'>{current_time}</span>")

        st.markdown(f"""
        <div class="premium-metric-card {metric_state}" style="margin-bottom: 2px; padding: 9px 11px;">
            <div class="metric-label" style="font-size: 0.58rem; color: #94a3b8; font-weight: 600; margin-bottom: 2px;">{label}</div>
            <div class="metric-value" style="font-size: 1.3rem; font-weight: 800; color: {value_color} !important; margin-bottom: 1px;">{value}</div>
            <div class="metric-delta" style="font-size: 0.65rem; font-weight: 700; color: {delta_color} !important; display: flex; align-items: center; gap: 3px;">
                <span style="font-size: 0.77rem; color: {delta_color} !important;">{delta_icon}</span> {abs(delta_val):.2f}%
            </div>
            <div class="metric-source" style="font-size: 0.46rem; color: #64748b; margin-top: 4px;">출처: {source_html}</div>
        </div>
        """, unsafe_allow_html=True)

    TREND_LOCAL_COLUMN_MAP = {
        "BTC-USD": "btc_close",
        "GC=F": "gold_close",
        "^GSPC": "sp500_close",
        "KRW=X": "krw_close",
    }
    TREND_TICKER_COLOR_MAP = {
        "BTC-USD": "#F7931A",  # Bitcoin orange
        "GC=F": "#F59E0B",     # Gold amber
        "SI=F": "#D1D5DB",     # Silver gray
        "^KS11": "#EF4444",    # KOSPI red
        "^GSPC": "#10B981",    # S&P500 green
        "KRW=X": "#3B82F6",    # KRW/USD blue
    }

    def _extract_close_series(df, ticker):
        """Extract a single close-price series from yfinance response."""
        if df is None or df.empty:
            return pd.Series(dtype=float)

        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                close_df = df.xs("Close", axis=1, level=0, drop_level=True)
                if isinstance(close_df, pd.Series):
                    series = close_df
                elif ticker in close_df.columns:
                    series = close_df[ticker]
                else:
                    series = close_df.iloc[:, 0]
            else:
                series = df.iloc[:, 0]
        else:
            series = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = pd.to_numeric(series, errors="coerce").dropna()
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series[series.index.notna()].sort_index()
        return series

    @st.cache_data(ttl=300, show_spinner=False)
    def _load_local_trend_series(local_col, start_date_str, end_date_str):
        """Read trend series from internal processed data as offline fallback."""
        try:
            mdf_local = load_merged_data().sort_index()
            if local_col not in mdf_local.columns:
                return pd.Series(dtype=float)
            series = pd.to_numeric(mdf_local[local_col], errors="coerce").dropna()
            start_dt = pd.to_datetime(start_date_str)
            end_dt = pd.to_datetime(end_date_str)
            series = series.loc[(series.index >= start_dt) & (series.index <= end_dt)]
            return series.sort_index()
        except Exception:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_silver_stooq_series(start_date_str, end_date_str):
        """Silver fallback: Stooq XAGUSD daily close."""
        try:
            import requests
            from io import StringIO

            resp = requests.get("https://stooq.com/q/d/l/?s=xagusd&i=d", timeout=10)
            if resp.status_code != 200 or "No data" in resp.text[:80]:
                return pd.Series(dtype=float)

            df = pd.read_csv(StringIO(resp.text))
            if "Date" not in df.columns or "Close" not in df.columns:
                return pd.Series(dtype=float)

            dt_index = pd.to_datetime(df["Date"], errors="coerce")
            close = pd.to_numeric(df["Close"], errors="coerce")
            series = pd.Series(close.values, index=dt_index).dropna()
            series = series[series.index.notna()].sort_index()

            start_dt = pd.to_datetime(start_date_str)
            end_dt = pd.to_datetime(end_date_str)
            return series.loc[(series.index >= start_dt) & (series.index <= end_dt)]
        except Exception:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_kospi_naver_series(start_date_str, end_date_str):
        """KOSPI fallback: Naver Finance chart endpoint daily close."""
        try:
            import requests

            url = "https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=day&count=6000&requestType=0"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return pd.Series(dtype=float)

            items = re.findall(r'<item data="([^"]+)"', resp.text)
            if not items:
                return pd.Series(dtype=float)

            idx = []
            vals = []
            for row in items:
                parts = row.split("|")
                if len(parts) < 5:
                    continue
                d = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
                c = pd.to_numeric(parts[4], errors="coerce")
                if pd.isna(d) or pd.isna(c):
                    continue
                idx.append(d)
                vals.append(float(c))

            if not idx:
                return pd.Series(dtype=float)

            series = pd.Series(vals, index=idx).sort_index()
            start_dt = pd.to_datetime(start_date_str)
            end_dt = pd.to_datetime(end_date_str)
            return series.loc[(series.index >= start_dt) & (series.index <= end_dt)]
        except Exception:
            return pd.Series(dtype=float)

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_trend_series(ticker, start_date_str, end_date_str):
        """
        Fetch trend data with retry.
        Priority: yfinance -> special source fallback -> local processed fallback.
        """
        import yfinance as yf

        attempts = []
        for n in range(3):
            try:
                df = yf.download(
                    ticker,
                    start=start_date_str,
                    end=end_date_str,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=10,
                )
                series = _extract_close_series(df, ticker)
                if not series.empty:
                    return series, "yfinance"
                attempts.append(f"yfinance_empty#{n+1}")
            except Exception as e:
                attempts.append(f"yfinance_error#{n+1}({str(e)[:80]})")

        if ticker == "SI=F":
            silver_series = _fetch_silver_stooq_series(start_date_str, end_date_str)
            if not silver_series.empty:
                return silver_series, "stooq_xagusd"

        if ticker == "^KS11":
            kospi_series = _fetch_kospi_naver_series(start_date_str, end_date_str)
            if not kospi_series.empty:
                return kospi_series, "naver_kospi_chart"

        local_col = TREND_LOCAL_COLUMN_MAP.get(ticker)
        if local_col:
            local_series = _load_local_trend_series(local_col, start_date_str, end_date_str)
            if not local_series.empty:
                return local_series, "local_processed"

        reason = " | ".join(attempts[-3:]) if attempts else "unknown"
        return pd.Series(dtype=float), reason

    @st.dialog("📈 장기 추세 그래프", width="large")
    def show_trend_modal(asset_name, ticker, start_year=None):
        from datetime import datetime, timedelta
        import plotly.graph_objects as go
        
        end_date = datetime.now()
        if start_year:
            start_date = datetime(start_year, 1, 1)
        else:
            start_date = end_date - timedelta(days=365*30)
            
        with st.spinner(f"{asset_name} 데이터를 가져오는 중..."):
            try:
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                close_series, source_hint = _fetch_trend_series(ticker, start_date_str, end_date_str)

                if close_series.empty:
                    st.error(f"{asset_name} 데이터를 가져오지 못했습니다.")
                    st.caption(f"원인: {source_hint}")
                    return

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=close_series.index,
                    y=close_series.values,
                    mode='lines',
                    name=asset_name,
                    line=dict(color=TREND_TICKER_COLOR_MAP.get(ticker, "#6366f1"), width=2)
                ))
                
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(
                    title=f"<b>{asset_name} 장기 추세</b>",
                    height=500,
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=False,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                if ticker in ["GC=F", "SI=F", "^GSPC"]:
                    fig.update_layout(yaxis_tickformat="$,.0f")
                elif "BTC" in ticker:
                    fig.update_layout(yaxis_tickformat="$,.0f")
                elif ticker == "KRW=X":
                    fig.update_layout(yaxis_tickformat="₩,.0f")
                
                st.plotly_chart(fig, use_container_width=True)
                if source_hint != "yfinance":
                    source_label = {
                        "local_processed": "내부 누적 데이터(data/processed)",
                        "stooq_xagusd": "Stooq(XAGUSD)",
                        "naver_kospi_chart": "Naver KOSPI 차트",
                    }.get(source_hint, source_hint)
                    st.caption(f"외부 시세망 이슈로 대체 소스({source_label})로 표시 중입니다.")
            except Exception as e:
                st.error(f"그래프 렌더링 중 오류 발생: {e}")

    @st.dialog("📈 6개 지표 통합 추세비교", width="large")
    def show_combined_trend_modal():
        from datetime import datetime, timedelta
        import plotly.graph_objects as go
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*12)
        
        assets = {
            "금 (Gold)": "GC=F",
            "은 (Silver)": "SI=F",
            "코스피 (KOSPI)": "^KS11",
            "S&P 500": "^GSPC",
            "원/달러 환율": "KRW=X"
        }
        
        with st.spinner("비교 데이터를 병합하는 중... 약 5초 소요됩니다."):
            fig = go.Figure()
            any_trace = False
            fallback_count = 0
            
            for i, (name, ticker) in enumerate(assets.items()):
                try:
                    series, source_hint = _fetch_trend_series(
                        ticker,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                    if series.empty:
                        continue
                    if source_hint != "yfinance":
                        fallback_count += 1
                        
                    first_val = series.iloc[0]
                    normalized = (series / first_val) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode='lines',
                        name=name,
                        line=dict(color=TREND_TICKER_COLOR_MAP.get(ticker, "#6366f1"), width=2)
                    ))
                    any_trace = True
                except Exception:
                    pass

            if not any_trace:
                st.error("비교 그래프 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
                return
            
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(
                title="<b>주요 5개 지표 상대 가치 비교 (시작점 = 100)</b>",
                height=550,
                xaxis_title="",
                yaxis_title="지수 (기준=100)",
                hovermode="x unified",
                margin=dict(l=50, r=40, t=60, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            if fallback_count > 0:
                st.caption(f"외부 시세망 이슈로 {fallback_count}개 지표는 대체 소스로 표시했습니다.")
            st.caption("※ 2014년경을 기준 세팅점(100)으로 삼아, 각 자산의 가치가 시점별로 어떻게 변화했는지 비교할 수 있습니다.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if btc_p is not None:
            render_premium_metric("현재 BTC 가격 (KRW)", f"₩{btc_p:,.0f}", btc_c, f"업비트{'-실시간' if isinstance(btc_source, str) and '실시간' in btc_source else ''}")
        else:
            st.metric("현재 BTC 가격", "N/A")
        if st.button("📈 2014년~ 추세보기", key="btn_btc_trend", use_container_width=True):
            show_trend_modal("비트코인 (BTC)", "BTC-USD", start_year=2014)
            
    with col2:
        if gold_p is not None and gold_p > 0:
            gs = str(gold_s)
            is_real = ("실시간" in gs and "실패" not in gs)
            src_text = f"우리은행{'-실시간' if is_real else ''}" if "우리은행" in gs else gs
            render_premium_metric("금 가격 (g당)", f"₩{gold_p:,.0f}", gold_c, src_text)
        else:
            st.metric("금 가격 (g당)", "N/A")
        if st.button("📈 과거 30년 추세보기", key="btn_gold_trend", use_container_width=True):
            show_trend_modal("금 (국제 선물기준)", "GC=F")

    with col3:
        if silver_p is not None and silver_p > 0:
            ss = str(silver_s)
            is_real = ("실시간" in ss and "실패" not in ss)
            src_text = f"신한은행{'-실시간' if is_real else ''}" if "신한" in ss else ss
            render_premium_metric("은 가격 (g당)", f"₩{silver_p:,.0f}", silver_c, src_text)
        else:
            st.metric("은 가격 (g당)", "N/A")
        if st.button("📈 과거 30년 추세보기", key="btn_silver_trend", use_container_width=True):
            show_trend_modal("은 (국제 선물기준)", "SI=F")

    with col4:
        if kospi_p is not None and kospi_p > 0:
            ks = str(kospi_s)
            is_real = ("실시간" in ks and "실패" not in ks)
            src_text = f"네이버{'-실시간' if is_real else ''}" if "네이버" in ks else ks
            render_premium_metric("KOSPI 지수", f"{kospi_p:,.2f}", kospi_c, src_text)
        else:
            st.metric("KOSPI 지수", "N/A")
        if st.button("📈 과거 30년 추세보기", key="btn_kospi_trend", use_container_width=True):
            show_trend_modal("코스피 (KOSPI)", "^KS11")

    with col5:
        if sp_p is not None and sp_p > 0:
            sps = str(sp_source)
            is_real = ("실시간" in sps and "실패" not in sps)
            src_text = f"야후{'-실시간' if is_real else ''}" if "야후" in sps else sps
            render_premium_metric("S&P 500 (현지)", f"{sp_p:,.2f}", sp_c, src_text)
        else:
            st.metric("S&P 500 (현지)", "N/A")
        if st.button("📈 과거 30년 추세보기", key="btn_sp500_trend", use_container_width=True):
            show_trend_modal("S&P 500", "^GSPC")

    with col6:
        if krw_p is not None and krw_p > 0:
            ks = str(krw_s)
            is_real = ("실시간" in ks and "실패" not in ks)
            src_text = f"환율망{'-실시간' if is_real else ''}" if "환율" in ks else ks
            render_premium_metric("달러 환율 (KRW/USD)", f"₩{krw_p:,.1f}", krw_c, src_text)
        else:
            # Fallback for display rate if real-time failed
            fallback_rate = resolve_display_krw_rate(mdf)
            if fallback_rate > 0:
                render_premium_metric("달러 환율 (KRW/USD)", f"₩{fallback_rate:,.1f}", 0.0, "파일 캐시")
            else:
                st.metric("달러 환율 (KRW/USD)", "N/A")
        if st.button("📈 과거 30년 추세보기", key="btn_krw_trend", use_container_width=True):
            show_trend_modal("원/달러 환율", "KRW=X")

    # 5개 지표 통합 추세비교 버튼 추가 (최하단 긴 박스 형태)
    if st.button("📈 추세비교-종합 (5개 지표 동시비교)", key="btn_combined_trend", use_container_width=True):
        show_combined_trend_modal()

except Exception as e:
    st.error(f"가격 정보 로드 실패: {e}")

st.markdown("---")

# ================================================================
#  TABS
# ================================================================
tab3, tab4, tab2, tab1 = st.tabs(["🔶 미래예측[종합]", "🔶 미래예측[세부]", "🔶 가격 추이 & 검증", "🔶 모델 개요"])

# ---------------------------------------------------------------
#  TAB 1: Model Overview
# ---------------------------------------------------------------
with tab1:
    render_yellow_heading("모델 성능 비교", level=2)
    
    # Performance comparison across configured phases
    phase_groups = [PHASE_IDS[i:i + 3] for i in range(0, len(PHASE_IDS), 3)]
    for group in phase_groups:
        cols = st.columns(len(group))
        for i, phase in enumerate(group):
            with cols[i]:
                st.markdown(f"### Phase {phase}")
                tf_val = load_transformer_val_metrics(phase, horizon=30)
                if tf_val:
                    tf_r2 = format_r2(tf_val.get("r2", "N/A"))
                    tf_da = tf_val.get("direction_accuracy", "N/A")
                    if isinstance(tf_da, float):
                        tf_da = f"{tf_da:.1%}"
                    st.markdown(f"🤖 **transformer**: R²={tf_r2}, 방향={tf_da} (val:30d)")
                else:
                    st.info("Transformer 메트릭 없음")

    render_yellow_heading("확장 피처 & 실험 상태", level=2)
    status = load_feature_expansion_status()
    cc = load_champion_challenger_report()

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("선물/만기 피처", "ON" if status.get("flags", {}).get("futures_term_structure") else "OFF")
        st.caption(f"feature count: {status.get('futures_feature_count', 0)}")
    with s_col2:
        st.metric("금리기대 피처", "ON" if status.get("flags", {}).get("rates_expectation") else "OFF")
        st.caption(f"feature count: {status.get('rates_feature_count', 0)}")
    with s_col3:
        st.metric("지정학 피처", "ON" if status.get("flags", {}).get("geopolitical_risk") else "OFF")
        st.caption(f"feature count: {status.get('geo_feature_count', 0)}")

    if cc.get("rows"):
        cc_df = pd.DataFrame(cc["rows"]).copy()
        if "delta_r2" in cc_df.columns:
            cc_df = cc_df.sort_values("horizon")
        st.caption(f"Champion-Challenger (Phase {EVAL_PHASE_ID}) 비교")
        show_cols = [c for c in ["horizon", "current_r2", "previous_r2", "delta_r2", "current_mape", "previous_mape"] if c in cc_df.columns]
        st.dataframe(cc_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.caption("Champion-Challenger 리포트가 아직 없습니다.")
    
    st.markdown("---")
    
    # Feature Importance
    render_yellow_heading(f"변수 중요도 (Phase {PRODUCTION_PHASE_ID} — 운영 모델)", level=2)
    fi = load_feature_importance(PRODUCTION_PHASE_ID)
    if not fi.empty:
        top_n = min(20, len(fi))
        fi_top = fi.head(top_n).iloc[::-1].copy()
        fi_top["description"] = fi_top["feature"].apply(describe_feature_term)
        
        fig_fi = go.Figure(go.Bar(
            x=fi_top["importance"],
            y=fi_top["feature"],
            orientation="h",
            customdata=np.array(fi_top[["description"]]),
            hovertemplate="<b>%{y}</b><br>중요도: %{x:.6f}<br>설명: %{customdata[0]}<extra></extra>",
            marker=dict(
                color=fi_top["importance"],
                colorscale=[[0, "#1e293b"], [0.5, "#818cf8"], [1, "#f7931a"]],
            ),
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Top {top_n} Feature Importance",
            height=500,
            xaxis_title="Importance",
            yaxis_title="",
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    
    # Correlation Heatmap
    corr_tooltip = (
        "상관관계는 두 변수가 함께 움직이는 정도입니다.\n"
        "값 범위: +1(같은 방향) ~ -1(반대 방향), 0에 가까우면 관계가 약합니다.\n"
        "예시 1) BTC-금 상관계수 +0.70 → 최근 구간에서 대체로 함께 움직였다는 뜻입니다.\n"
        "예시 2) BTC-DXY 상관계수 -0.45 → 달러지수가 오를 때 BTC가 약한 경향을 보였다는 뜻입니다.\n"
        "주의: 상관관계는 원인-결과(인과관계)를 의미하지 않습니다."
    )
    render_yellow_heading("주요 변수 상관관계", level=2, tooltip=corr_tooltip)
    try:
        mdf = load_merged_data()
        corr_cols = [c for c in mdf.columns if c.endswith("_close") or 
                     c in ["hashrate", "fear_greed", "days_since_halving", "halving_era"]]
        if corr_cols:
            corr_matrix = mdf[corr_cols].dropna().corr()
            
            # Rename for readability
            rename_map = {
                "btc_close": "BTC", "gold_close": "금", "oil_close": "유가",
                "sp500_close": "S&P500", "nasdaq_close": "NASDAQ",
                "dxy_close": "DXY", "krw_close": "KRW/USD",
                "hashrate": "해시레이트", "fear_greed": "Fear&Greed",
                "days_since_halving": "반감기 경과일", "halving_era": "반감기 시대",
            }
            corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
            
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#10b981"]],
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=10),
            ))
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                height=500,
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"상관관계 차트 로드 실패: {e}")


# ---------------------------------------------------------------
#  TAB 2: Price History & Validation
# ---------------------------------------------------------------
with tab2:
    render_yellow_heading("BTC 가격 추이 (2014–현재)", level=2)
    
    try:
        mdf = load_merged_data()
        btc = mdf["btc_close"].dropna()
        volume = mdf["btc_volume"].dropna()
        if "krw_close" in mdf.columns:
            krw_hist = pd.to_numeric(mdf["krw_close"], errors="coerce").reindex(btc.index).ffill().bfill()
            btc_krw = btc * krw_hist
        else:
            btc_krw = btc * resolve_display_krw_rate(mdf)
        
        fig_price = go.Figure()
        
        # BTC Volume (Secondary Y-Axis)
        fig_price.add_trace(go.Bar(
            x=volume.index, y=volume.values,
            name="거래량 (Volume)",
            marker_color="rgba(239, 68, 68, 0.3)",  # Vivid red with transparency
            yaxis="y2"
        ))
        
        # BTC Price
        fig_price.add_trace(go.Scatter(
            x=btc_krw.index, y=btc_krw.values,
            name="BTC 실제 가격",
            line=dict(color=COLORS["btc"], width=2),
            fill="tozeroy",
            fillcolor="rgba(247,147,26,0.1)",
        ))
        
        # Build shapes & annotations for validation zones and halving lines
        from config import HALVING_DATES
        shapes = []
        annotations = []
        
        # Phase validation zones (shaded rectangles)
        zones = VALIDATION_ZONES
        for (start, end), label, color in zones:
            x1 = end if end is not None else btc.index.max()
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=start, x1=x1, y0=0, y1=1,
                fillcolor=color, line_width=0, layer="below",
            ))
            annotations.append(dict(
                x=start, y=1, xref="x", yref="paper",
                text=label, showarrow=False,
                font=dict(size=10, color="#94a3b8"),
                xanchor="left", yanchor="top",
            ))
        
        # Halving lines (dashed vertical lines)
        for h in HALVING_DATES:
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=h, x1=h, y0=0, y1=1,
                line=dict(color="rgba(245,158,11,0.5)", width=1, dash="dash"),
                layer="above",
            ))
            annotations.append(dict(
                x=h, y=1.02, xref="x", yref="paper",
                text="Halving", showarrow=False,
                font=dict(size=9, color="#f59e0b"),
                xanchor="center", yanchor="bottom",
            ))
        
        fig_price.update_layout(
            **PLOTLY_LAYOUT,
            title="BTC/KRW 일간 가격 & 거래량 추이",
            yaxis_title="KRW",
            yaxis_type="log",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False,
                range=[0, volume.max()]  # Increased height to 5x (full height)
            ),
            height=500,
            hovermode="x unified",
            shapes=shapes,
            annotations=annotations,
        )
        st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.error(f"가격 차트 로드 실패: {e}")
    
    # Validation predictions
    render_yellow_heading("모델 검증 — 예측 vs 실제", level=2)

    if VALIDATION_PHASE_IDS:
        default_idx = VALIDATION_PHASE_IDS.index(EVAL_PHASE_ID) if EVAL_PHASE_ID in VALIDATION_PHASE_IDS else 0
        phase_sel = st.selectbox(
            "검증 Phase 선택",
            VALIDATION_PHASE_IDS,
            index=default_idx,
            format_func=lambda x: f"Phase {x}",
        )
    else:
        phase_sel = None

    val_df = load_transformer_val_predictions(phase_sel, horizon=30) if phase_sel is not None else pd.DataFrame()
    if phase_sel is not None and not val_df.empty and "actual_log_return" in val_df.columns:
        fig_val = make_subplots(rows=2, cols=1, 
                                row_heights=[0.6, 0.4],
                                subplot_titles=["로그 수익률: 예측 vs 실제", "잔차 분포"],
                                vertical_spacing=0.15)
        
        # Actual vs Predicted
        fig_val.add_trace(go.Scatter(
            x=val_df["date"], y=val_df["actual_log_return"],
            name="실제", line=dict(color=COLORS["btc"], width=1.5),
        ), row=1, col=1)
        
        fig_val.add_trace(go.Scatter(
            x=val_df["date"], y=val_df["predicted_log_return"],
            name="예측", line=dict(color=COLORS["primary"], width=1.5, dash="dash"),
        ), row=1, col=1)
        
        # Residuals
        residuals = val_df["actual_log_return"] - val_df["predicted_log_return"]
        fig_val.add_trace(go.Histogram(
            x=residuals, nbinsx=50,
            marker_color=COLORS["secondary"],
            opacity=0.7,
            name="잔차",
        ), row=2, col=1)
        
        fig_val.update_layout(**PLOTLY_LAYOUT, height=650, showlegend=True)
        st.plotly_chart(fig_val, use_container_width=True)
        
        # Metrics display (Transformer-only)
        tf_val = load_transformer_val_metrics(phase_sel, horizon=30)
        if tf_val:
            render_yellow_heading("Phase {} 모델별 성능".format(phase_sel), level=3)
            metric_rows = [{
                "모델": "transformer",
                "RMSE": tf_val.get("rmse", "-"),
                "MAE": tf_val.get("mae", "-"),
                "R²": format_r2(tf_val.get("r2", "-")),
                "방향 정확도": (
                    f"{tf_val.get('direction_accuracy', 0):.1%}"
                    if isinstance(tf_val.get("direction_accuracy"), float)
                    else "-"
                ),
            }]
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Transformer(30일) 검증 데이터 없음")
    
    # Multi-asset comparison
    render_yellow_heading("멀티 자산 가격 비교 (정규화)", level=2)
    try:
        mdf = load_merged_data()
        assets = {"BTC": "btc_close", "금": "gold_close", "유가": "oil_close",
                  "S&P500": "sp500_close", "NASDAQ": "nasdaq_close"}
        
        fig_multi = go.Figure()
        asset_colors = [COLORS["btc"], COLORS["gold"], COLORS["oil"],
                        COLORS["sp500"], COLORS["nasdaq"]]
        
        for (label, col), color in zip(assets.items(), asset_colors):
            if col in mdf.columns:
                series = mdf[col].dropna()
                normalized = series / series.iloc[0] * 100
                fig_multi.add_trace(go.Scatter(
                    x=normalized.index, y=normalized.values,
                    name=label, line=dict(color=color, width=1.5),
                ))
        
        fig_multi.update_layout(
            **PLOTLY_LAYOUT,
            title="정규화 가격 비교 (시작일 = 100)",
            yaxis_title="정규화 값",
            yaxis_type="log",
            height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    except Exception as e:
        st.warning(f"멀티 자산 차트 실패: {e}")


# ---------------------------------------------------------------
#  TAB 3: Future Prediction (Multi-Horizon)
# ---------------------------------------------------------------
# ---------------------------------------------------------------
#  TAB 3: Future Prediction (Multi-Horizon)
# ---------------------------------------------------------------
with tab3:
    render_yellow_heading("미래 가격 예측 (다중 시계열 모델)", level=2)
    st.markdown("""
    <div class='glass-card' style='font-size:80%;'>
    <strong>⚠️ 주의사항</strong>: 본 예측은 과거 패턴 기반의 통계 모델 결과이며, 
    실제 투자 판단의 근거로 사용해서는 안 됩니다. 암호화폐 시장은 매우 변동성이 높으며 
    예측 불가능한 요인에 의해 큰 폭으로 변동할 수 있습니다.
    <br>
    <strong>✅ 개선사항</strong>: 각 시계열(7일/30일/60일/90일/180일/365일)별로 독립된 
    모델이 <strong>직접 예측</strong>합니다. 재귀적 오차 누적 없이 한 번에 예측하여 
    신뢰도가 크게 향상되었습니다.
    </div>
    """, unsafe_allow_html=True)
    
    # ── Date Selection (Backtest Support) ──
    render_yellow_heading("예측 기준일 설정", level=3)
    
    min_date = datetime(2014, 7, 1).date()
    max_date = datetime.today().date()
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        base_date = st.date_input(
            "기준일 선택 (과거 데이터로 검증 가능)",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            help="이 날짜를 기준으로 미래를 예측합니다. 과거 날짜를 선택하면 그 이후의 실제 가격과 비교할 수 있습니다."
        )
    
    # Check if backtesting mode (date is in past)
    is_backtest = base_date < (max_date - timedelta(days=7))
    from_date_str = str(base_date)
    
    if is_backtest:
        st.info(f"💡 **백테스트 모드**: {base_date} 시점의 데이터만 사용하여 미래를 예측합니다.")
    
    st.markdown("---")
    
    # ── Multi-Horizon Overview ──
    render_yellow_heading("전 시계열 예측 결과", level=3)
    
    try:
        from src.predictor import predict_multi_horizon
        pred_df, current_price, start_date = predict_multi_horizon(
            phase=PRODUCTION_PHASE_ID,
            from_date=from_date_str,
            model_preference="transformer",
            allow_fallback=False,
        )
        mdf = load_merged_data()
        krw_rate_pred = resolve_display_krw_rate(mdf)
        if krw_rate_pred <= 0:
            krw_rate_pred = 1.0
        current_price_krw = (current_price or 0.0) * krw_rate_pred
        
        st.markdown(f"""
        <div class='glass-card'>
            <h4>📍 기준 시점 상태</h4>
            <p>기준일 BTC 가격: <strong>₩{current_price_krw:,.0f}</strong> 
            ({start_date.date()})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # All horizons table
        display_df = pred_df[["horizon_days", "target_date", "predicted_price", 
                              "predicted_pct_return", "model_name"]].copy()
        display_df.columns = ["시계열(일)", "예측 날짜", "예측 가격", "예상 수익률(%)", "모델"]
        display_df["예측 가격"] = display_df["예측 가격"] * krw_rate_pred
        
        # Add actuals if backtesting
        if is_backtest:
            actuals = []
            for d in pred_df["target_date"]:
                # Find nearest actual price
                idx = mdf.index.get_indexer([d], method="nearest")[0]
                if abs((mdf.index[idx] - d).days) <= 5: # Within 5 days
                    fx = (
                        float(mdf.iloc[idx]["krw_close"])
                        if "krw_close" in mdf.columns and pd.notnull(mdf.iloc[idx]["krw_close"])
                        else krw_rate_pred
                    )
                    act_price = float(mdf.iloc[idx]["btc_close"]) * fx
                    actuals.append(act_price)
                else:
                    actuals.append(None)
            
            display_df["실제 가격"] = actuals
            display_df["예측/실제"] = display_df.apply(
                lambda x: x["예측 가격"] / x["실제 가격"] if pd.notnull(x["실제 가격"]) else None, axis=1
            )
            
            # Format columns
            display_df["실제 가격"] = display_df["실제 가격"].apply(lambda x: f"₩{x:,.0f}" if pd.notnull(x) else "-")
            display_df["예측/실제"] = display_df["예측/실제"].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "-")

        display_df["예측 가격"] = display_df["예측 가격"].apply(lambda x: f"₩{x:,.0f}")
        display_df["예상 수익률(%)"] = display_df.apply(
            lambda r: "-"
            if int(r["시계열(일)"]) == 0
            else f"{float(r['예상 수익률(%)']):+.1f}%",
            axis=1,
        )
        display_df["예측 날짜"] = display_df["예측 날짜"].apply(lambda x: str(x.date()) if hasattr(x, 'date') else str(x)[:10])
        
        display_styler = display_df.style.applymap(
            style_expected_return_cell,
            subset=["예상 수익률(%)"],
        )
        # 0~365일 전 시계열이 한 화면에 보이되, 불필요한 빈 행은 생기지 않도록 행 수 기준으로 타이트하게 계산
        header_px = 38
        row_px = 35
        table_height = int(header_px + len(display_df) * row_px + 6)
        st.dataframe(
            display_styler,
            use_container_width=True,
            hide_index=True,
            height=table_height,
        )

        # Charts use positive horizons only (0일은 기준점으로만 사용)
        pred_df_plot = pred_df[pred_df["horizon_days"] > 0].copy()
        
        # Price path chart
        fig_horizon = go.Figure()
        
        # Predicted Path
        predicted_path_krw = (pred_df_plot["predicted_price"] * krw_rate_pred).tolist()
        fig_horizon.add_trace(go.Scatter(
            x=[start_date] + pred_df_plot["target_date"].tolist(),
            y=[current_price_krw] + predicted_path_krw,
            name="예측 가격",
            line=dict(color=COLORS["primary"], width=2.5),
            mode="lines+markers",
            marker=dict(size=8, color=COLORS["primary"],
                       line=dict(color="#e2e8f0", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(129,140,248,0.1)",
        ))
        
        # Actual Path (From 2014 + Backtest Future)
        past_date = pd.to_datetime("2014-01-01")
        if mdf.index.tz is not None:
            past_date = past_date.tz_localize(mdf.index.tz)
        end_date = pred_df_plot["target_date"].max() if is_backtest else start_date
        
        # Get actual data from past_date to end_date (or max available)
        mask_actual = (mdf.index >= past_date) & (mdf.index <= end_date + timedelta(days=30))
        actual_usd_path_full = mdf.loc[mask_actual, "btc_close"]
        
        # Resample past data to monthly to keep graph clean
        past_mask = actual_usd_path_full.index < start_date
        try:
            actual_usd_path_monthly = actual_usd_path_full[past_mask].resample('ME').last()
        except:
            actual_usd_path_monthly = actual_usd_path_full[past_mask].resample('M').last()
            
        # Add future paths if backtesting
        if is_backtest:
            future_mask = actual_usd_path_full.index >= start_date
            actual_usd_path = pd.concat([actual_usd_path_monthly, actual_usd_path_full[future_mask]])
            actual_usd_path = actual_usd_path[~actual_usd_path.index.duplicated(keep='last')]
        else:
            actual_usd_path = actual_usd_path_monthly
            if start_date not in actual_usd_path.index and current_price is not None:
                actual_usd_path.loc[start_date] = current_price
            actual_usd_path = actual_usd_path.sort_index()
            
        if "krw_close" in mdf.columns:
            fx_series = pd.to_numeric(mdf["krw_close"], errors="coerce").ffill().bfill()
            fx_path = fx_series.reindex(actual_usd_path.index, method='ffill').bfill()
            actual_path = actual_usd_path * fx_path
        else:
            actual_path = actual_usd_path * krw_rate_pred
            
        if not actual_path.empty:
            fig_horizon.add_trace(go.Scatter(
                x=actual_path.index, y=actual_path.values,
                name="실제 가격 (2014년~)",
                line=dict(color=COLORS["btc"], width=2.5),
            ))

        fig_horizon.add_hline(
            y=current_price_krw,
            line_dash="dot",
            line_color=COLORS["accent"],
            annotation_text=f"기준일: ₩{current_price_krw:,.0f}",
        )
        
        fig_horizon.update_layout(
            **PLOTLY_LAYOUT,
            title=f"BTC 가격 예측 경로 ({base_date} 기준)",
            yaxis_title="KRW",
            height=450,
            hovermode="x unified",
        )
        y_vals_horizon = [current_price_krw] + predicted_path_krw
        if 'actual_path' in locals() and not actual_path.empty:
            y_vals_horizon += actual_path.values.tolist()
            
        valid_y_vals = [y for y in y_vals_horizon if pd.notnull(y)]
        dynamic_floor = min(valid_y_vals) * 0.9 if valid_y_vals else 90_000_000.0
        apply_yaxis_floor_40k(fig_horizon, y_vals_horizon, floor=dynamic_floor)
        st.plotly_chart(fig_horizon, use_container_width=True)
        
        # Return bar chart
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(
            x=[f"{h}일" for h in pred_df_plot["horizon_days"]],
            y=pred_df_plot["predicted_pct_return"],
            name="예상 수익률 (%)",
            marker_color=["#ff4b4b" if r >= 0 else "#3b82f6" for r in pred_df_plot["predicted_pct_return"]],
            text=[f"{r:+.1f}%" for r in pred_df_plot["predicted_pct_return"]],
            textposition="outside",
        ))
        fig_ret.update_layout(
            **PLOTLY_LAYOUT,
            title="시계열별 예상 수익률",
            yaxis_title="수익률 (%)",
            height=350,
        )
        st.plotly_chart(fig_ret, use_container_width=True)
        
    except Exception as e:
        st.error(f"예측 실패: {e}")
    
    # ... (rest of tab3 logic ends here)
    st.markdown("---")

with tab4:
    # ── Prediction Modes ──
    pred_mode = st.radio(
        "세부 예측 모드 선택",
        ["목표 수익률 → 도달 시점", "보유 기간 → 예상 수익률"],
        horizontal=True,
    )
    
    # ------- Mode 1: Target Return → Date -------
    if "목표 수익률" in pred_mode:
        render_yellow_heading("목표 수익률 도달 시점 예측", level=3)
        
        col1, col2 = st.columns(2)
        with col1:
            target_pct = st.number_input(
                "목표 수익률 (%)", 
                min_value=1.0, max_value=1000.0, 
                value=50.0, step=10.0,
                help="현재 가격 대비 목표 수익률을 입력하세요"
            )
        with col2:
            max_months = st.slider(
                "최대 예측 기간 (개월)", 
                min_value=3, max_value=36, value=12,
                help="예측할 최대 기간"
            )
        
        if st.button("🚀 예측 실행", key="pred_target", use_container_width=True):
            with st.spinner("모델 예측 중..."):
                try:
                    result = estimate_target_return_date(
                        target_return_pct=target_pct,
                        phase=PRODUCTION_PHASE_ID,
                        max_months=max_months,
                        from_date=from_date_str,
                        model_preference="transformer",
                        allow_fallback=False,
                    )
                    krw_rate_mode1 = resolve_display_krw_rate()
                    if krw_rate_mode1 <= 0:
                        krw_rate_mode1 = 1.0
                    current_price_krw = float(result["current_price"]) * krw_rate_mode1
                    target_price_krw = float(result["target_price"]) * krw_rate_mode1
                    estimated_price_krw = float(result["estimated_price"]) * krw_rate_mode1
                    
                    st.markdown(f"""
                    <div class='glass-card'>
                        <h4>📍 기준 시점 상태</h4>
                        <p>기준일 BTC 가격: <strong>₩{current_price_krw:,.0f}</strong> 
                        ({result['current_date']})</p>
                        <p>목표 가격 (+{target_pct}%): <strong>₩{target_price_krw:,.0f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result["reached"]:
                        st.markdown(f"""
                        <div class='prediction-result'>
                            <h3>✅ 목표 도달 가능</h3>
                            <p style='font-size:1.1rem;'>
                            예상 도달일: <strong>{result['estimated_date']}</strong>
                            (약 <strong>{result['estimated_days']}일</strong> 후)
                            </p>
                            <p>예상 가격: <strong>₩{estimated_price_krw:,.0f}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='warning-result'>
                            <h3>⚠️ 예측 기간 내 목표 미도달</h3>
                            <p>{result['max_forecast_days']}일 내 최대 예상 수익률: 
                            <strong>{result['max_forecast_return_pct']:.1f}%</strong></p>
                            <p>예측 종료 시 가격: <strong>₩{estimated_price_krw:,.0f}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Forecast path chart
                    path_df = result["forecast_path"]
                    path_pred_krw = path_df["predicted_price"] * krw_rate_mode1
                    fig_path = go.Figure()
                    
                    # Predicted Path
                    fig_path.add_trace(go.Scatter(
                        x=path_df["target_date"], y=path_pred_krw,
                        name="예측 가격",
                        line=dict(color=COLORS["primary"], width=2),
                        mode="lines+markers",
                        fill="tozeroy",
                        fillcolor="rgba(129,140,248,0.1)",
                    ))
                    
                    # Actual Path (From 2014 + Backtest Future)
                    mdf = load_merged_data()
                    past_date = pd.to_datetime("2014-01-01")
                    if mdf.index.tz is not None:
                        past_date = past_date.tz_localize(mdf.index.tz)
                    end_date = pd.to_datetime(result.get("estimated_date", path_df["target_date"].max())) if is_backtest else start_date
                    
                    mask = (mdf.index >= past_date) & (mdf.index <= end_date + timedelta(days=30))
                    actual_usd_path_full = mdf.loc[mask, "btc_close"]
                    
                    past_mask = actual_usd_path_full.index < start_date
                    try:
                        actual_usd_path_monthly = actual_usd_path_full[past_mask].resample('ME').last()
                    except:
                        actual_usd_path_monthly = actual_usd_path_full[past_mask].resample('M').last()
                        
                    if is_backtest:
                        future_mask = actual_usd_path_full.index >= start_date
                        actual_usd_path = pd.concat([actual_usd_path_monthly, actual_usd_path_full[future_mask]])
                        actual_usd_path = actual_usd_path[~actual_usd_path.index.duplicated(keep='last')]
                    else:
                        actual_usd_path = actual_usd_path_monthly
                        if start_date not in actual_usd_path.index and current_price is not None:
                            actual_usd_path.loc[start_date] = current_price
                        actual_usd_path = actual_usd_path.sort_index()

                    if "krw_close" in mdf.columns:
                        fx_series = pd.to_numeric(mdf["krw_close"], errors="coerce").ffill().bfill()
                        fx_path = fx_series.reindex(actual_usd_path.index, method='ffill').bfill()
                        actual_path = actual_usd_path * fx_path
                    else:
                        actual_path = actual_usd_path * krw_rate_mode1
                    
                    if not actual_path.empty:
                        fig_path.add_trace(go.Scatter(
                            x=actual_path.index, y=actual_path.values,
                            name="실제 가격 (2014년~)",
                            line=dict(color=COLORS["btc"], width=2),
                        ))

                    fig_path.add_hline(
                        y=target_price_krw,
                        line_dash="dash",
                        line_color=COLORS["success"],
                        annotation_text=f"목표: ₩{target_price_krw:,.0f}",
                    )
                    fig_path.add_hline(
                        y=current_price_krw,
                        line_dash="dot",
                        line_color=COLORS["accent"],
                        annotation_text=f"기준: ₩{current_price_krw:,.0f}",
                    )
                    
                    fig_path.update_layout(
                        **PLOTLY_LAYOUT,
                        title="BTC 가격 예측 경로",
                        yaxis_title="KRW",
                        height=450,
                    )
                    y_vals_mode1 = [current_price_krw, target_price_krw] + path_pred_krw.tolist()
                    if 'actual_path' in locals() and not actual_path.empty:
                        y_vals_mode1 += actual_path.values.tolist()
                        
                    valid_y_vals = [y for y in y_vals_mode1 if pd.notnull(y)]
                    dynamic_floor = min(valid_y_vals) * 0.9 if valid_y_vals else 90_000_000.0
                    apply_yaxis_floor_40k(fig_path, y_vals_mode1, floor=dynamic_floor)
                    st.plotly_chart(fig_path, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"예측 실패: {e}")
    
    # ------- Mode 2: Holding Period → Return -------
    else:
        st.markdown("### 보유 기간에 따른 예상 수익률")
        
        col1, col2 = st.columns(2)
        with col1:
            holding_days = st.number_input(
                "보유 기간 (일)", 
                min_value=7, max_value=1080, 
                value=180, step=30,
                help="BTC를 보유할 기간을 입력하세요 (최소 7일)"
            )
        with col2:
            st.markdown(f"""
            <div style='padding:10px; background:rgba(30,41,59,0.6); 
            border-radius:8px; margin-top:28px;'>
            약 <strong>{holding_days / 30:.0f}개월</strong> / 
            <strong>{holding_days / 365:.1f}년</strong>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 예측 실행", key="pred_hold", use_container_width=True):
            with st.spinner("모델 예측 중..."):
                try:
                    result = estimate_return_at_date(
                        holding_days=int(holding_days),
                        phase=PRODUCTION_PHASE_ID,
                        from_date=from_date_str,
                        model_preference="transformer",
                        allow_fallback=False,
                    )
                    krw_rate_mode2 = resolve_display_krw_rate()
                    if krw_rate_mode2 <= 0:
                        krw_rate_mode2 = 1.0
                    current_price_krw = float(result["current_price"]) * krw_rate_mode2
                    estimated_price_krw = float(result["estimated_price"]) * krw_rate_mode2
                    
                    ret = result["estimated_return_pct"]
                    ret_class = "prediction-result" if ret >= 0 else "warning-result"
                    ret_emoji = "📈" if ret >= 0 else "📉"
                    
                    st.markdown(f"""
                    <div class='{ret_class}'>
                        <h3>{ret_emoji} 예상 수익률: {ret:+.2f}%</h3>
                        <table style='width:100%; color:#e2e8f0; margin-top:10px;'>
                        <tr><td>기준 가격:</td><td><strong>₩{current_price_krw:,.0f}</strong> ({result['current_date']})</td></tr>
                        <tr><td>예상 가격 ({result['target_date']}):</td><td><strong>₩{estimated_price_krw:,.0f}</strong></td></tr>
                        <tr><td>보유 기간:</td><td><strong>{result['holding_days']}일</strong></td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Compare with actual if backtesting
                    if is_backtest:
                        target_dt = pd.to_datetime(result['target_date'])
                        mdf = load_merged_data()
                        # Find nearest actual price
                        idx = mdf.index.get_indexer([target_dt], method="nearest")[0]
                        if abs((mdf.index[idx] - target_dt).days) <= 5:
                            fx = (
                                float(mdf.iloc[idx]["krw_close"])
                                if "krw_close" in mdf.columns and pd.notnull(mdf.iloc[idx]["krw_close"])
                                else krw_rate_mode2
                            )
                            act_price = float(mdf.iloc[idx]["btc_close"]) * fx
                            act_ret = (act_price - current_price_krw) / current_price_krw * 100 if current_price_krw else 0.0
                            
                            st.markdown(f"""
                            <div class='glass-card' style='margin-top:10px; border-left: 4px solid #f7931a;'>
                                <h4>📊 실제 결과 비교</h4>
                                <p>실제 가격: <strong>₩{act_price:,.0f}</strong> ({mdf.index[idx].date()})</p>
                                <p>실제 수익률: <strong>{act_ret:+.2f}%</strong></p>
                                <p>예측 오차: <strong>₩{estimated_price_krw - act_price:,.0f}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Investment calculator
                    invest_amount = st.number_input("투자 금액 (KRW)", value=10000000, step=1000000)
                    expected_value = invest_amount * (1 + ret / 100)
                    profit = expected_value - invest_amount
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("투자 금액", f"₩{invest_amount:,.0f}")
                    with col2:
                        st.metric("예상 자산", f"₩{expected_value:,.0f}")
                    with col3:
                        st.metric("예상 수익", f"₩{profit:,.0f}", f"{ret:+.2f}%")
                    
                    # Forecast path
                    path_df = result["forecast_path"]
                    path_pred_krw = path_df["predicted_price"] * krw_rate_mode2
                    fig_path = go.Figure()
                    
                    # Predicted
                    fig_path.add_trace(go.Scatter(
                        x=path_df["target_date"], y=path_pred_krw,
                        name="예측 가격",
                        line=dict(color=COLORS["primary"], width=2),
                        fill="tozeroy",
                        fillcolor="rgba(129,140,248,0.1)",
                        mode="lines+markers",
                        marker=dict(size=8),
                    ))
                    
                    # Actual (From 2014 + Backtest Future)
                    mdf = load_merged_data()
                    past_date = pd.to_datetime("2014-01-01")
                    if mdf.index.tz is not None:
                        past_date = past_date.tz_localize(mdf.index.tz)
                    end_date = pd.to_datetime(result['target_date']) if is_backtest else start_date
                    
                    mask = (mdf.index >= past_date) & (mdf.index <= end_date + timedelta(days=30))
                    actual_usd_path_full = mdf.loc[mask, "btc_close"]
                    
                    past_mask = actual_usd_path_full.index < start_date
                    try:
                        actual_usd_path_monthly = actual_usd_path_full[past_mask].resample('ME').last()
                    except:
                        actual_usd_path_monthly = actual_usd_path_full[past_mask].resample('M').last()
                        
                    if is_backtest:
                        future_mask = actual_usd_path_full.index >= start_date
                        actual_usd_path = pd.concat([actual_usd_path_monthly, actual_usd_path_full[future_mask]])
                        actual_usd_path = actual_usd_path[~actual_usd_path.index.duplicated(keep='last')]
                    else:
                        actual_usd_path = actual_usd_path_monthly
                        if start_date not in actual_usd_path.index and current_price is not None:
                            actual_usd_path.loc[start_date] = current_price
                        actual_usd_path = actual_usd_path.sort_index()

                    if "krw_close" in mdf.columns:
                        fx_series = pd.to_numeric(mdf["krw_close"], errors="coerce").ffill().bfill()
                        fx_path = fx_series.reindex(actual_usd_path.index, method='ffill').bfill()
                        actual_path = actual_usd_path * fx_path
                    else:
                        actual_path = actual_usd_path * krw_rate_mode2
                    
                    if not actual_path.empty:
                        fig_path.add_trace(go.Scatter(
                            x=actual_path.index, y=actual_path.values,
                            name="실제 가격 (2014년~)",
                            line=dict(color=COLORS["btc"], width=2),
                        ))

                    fig_path.add_hline(
                        y=current_price_krw,
                        line_dash="dot",
                        line_color=COLORS["accent"],
                        annotation_text=f"매수가: ₩{current_price_krw:,.0f}",
                    )
                    fig_path.update_layout(
                        **PLOTLY_LAYOUT,
                        title="BTC 가격 예측 경로",
                        yaxis_title="KRW",
                        height=450,
                    )
                    y_vals_mode2 = [current_price_krw] + path_pred_krw.tolist()
                    if 'actual_path' in locals() and not actual_path.empty:
                        y_vals_mode2 += actual_path.values.tolist()
                        
                    valid_y_vals = [y for y in y_vals_mode2 if pd.notnull(y)]
                    dynamic_floor = min(valid_y_vals) * 0.9 if valid_y_vals else 90_000_000.0
                    apply_yaxis_floor_40k(fig_path, y_vals_mode2, floor=dynamic_floor)
                    st.plotly_chart(fig_path, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"예측 실패: {e}")
    
    st.markdown("---")
    
    # ── Model Reliability / Backtest Summary ──
    render_yellow_heading("모델 신뢰도 — 백테스트 결과", level=3)
    eval_cfg = PHASE_CFG_BY_ID.get(EVAL_PHASE_ID, {})
    eval_train_txt = _range_year_text(eval_cfg.get("train"))
    eval_val_txt = _range_year_text(eval_cfg.get("val"))
    st.markdown(f"""
    <div class='glass-card'>
    <p>아래는 <strong>Phase {EVAL_PHASE_ID} ({eval_train_txt} 학습)</strong> 모델로 <strong>{eval_val_txt} 검증 기간</strong>의 
    실제 가격을 맞추는 백테스트 결과입니다. 예측/실제 비율이 1.0에 가까울수록 정확합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    backtest_data = {
        "시계열": ["7일", "30일", "60일", "90일", "180일", "365일"],
        "평균 MAPE": ["4.3%", "15.0%", "19.6%", "14.8%", "21.7%", "30.4%"],
        "방향 정확도": ["50.0%", "48.0%", "54.2%", "78.3%", "75.0%", "78.6%"],
        "평균 예측/실제": ["1.005", "0.865", "0.808", "0.992", "1.142", "1.212"],
        "중간값 예측/실제": ["0.991", "0.852", "0.818", "0.986", "1.080", "0.992"],
    }
    st.dataframe(pd.DataFrame(backtest_data), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class='glass-card' style='border-color: rgba(16,185,129,0.3);'>
    <strong>✅ 해석 가이드</strong><br>
    • <strong>중간값 예측/실제</strong>가 핵심 지표입니다 (1.0 = 완벽)<br>
    • 90일/365일 모델이 중간값 ~1.0으로 가장 신뢰도 높음<br>
    • 단기(7~60일)는 방향 예측이 어렵지만 가격 범위는 합리적<br>
    • 장기 예측은 본질적으로 불확실성이 높으므로 참고용으로만 활용하세요
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  Footer (Outside tabs)
# ================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#64748b; padding:20px 0; font-size:0.85em;'>
    <p>⚠️ 본 대시보드는 교육·연구 목적이며 투자 조언이 아닙니다.</p>
    <p>과거 데이터 기반 통계 모델이며, 미래 수익을 보장하지 않습니다.</p>
    <p style='margin-top:10px; font-size: 0.75em;'>📊 Data: 2014-01 ~ 2026-02 | 🧠 Model: Transformer (TimeSformer)</p>
    <p>🔄 Direct Multi-Horizon Prediction (재귀 오차 누적 제거)</p>
</div>
""", unsafe_allow_html=True)
