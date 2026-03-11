# UI Overview for BTC Price Prediction Dashboard

## Overview
This markdown file documents the **frontend UI** implementation of the Streamlit application located in `app.py`. The UI is built with a premium dark theme, custom CSS, and interactive Streamlit components.

---

## 1. Page Configuration
```python
st.set_page_config(
    page_title="BTC 가격 예측 대시보드",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)
```
Sets the page title, icon, layout, and expands the sidebar by default.

---

## 2. Auto‑Refresh (Every 5 minutes)
```python
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
```
Injects a tiny HTML snippet that reloads the whole page after 300 seconds.

---

## 3. Premium Dark Theme (Custom CSS)
```python
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
    /* Global typography */
    .stApp {font-family: var(--font-main); color: var(--text-main);}
    .stApp p, .stApp label {font-family: var(--font-main) !important;}
    /* Material icons */
    .material-symbols-rounded, .material-symbols-outlined, [class*="material-symbol"], [data-testid*="Icon"] {
        font-family: "Material Symbols Rounded" !important;
        font-feature-settings: 'liga' !important;
    }
    /* Header fonts */
    h1, h2, h3, h4, .stTabs [data-baseweb="tab"] {font-family: var(--font-heading) !important;}
    /* Background */
    .stApp {background: radial-gradient(circle at 10% 20%, #0a0e28 0%, #030712 90%);}
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 0 20px rgba(99,102,241,0.05);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {transform: translateY(-4px); border-color: rgba(99,102,241,0.4);}
    div[data-testid="stMetric"] label {color: var(--text-dim) !important; font-weight: 500 !important;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#fff; font-size:1.1rem; font-weight:800; text-shadow:0 0 10px rgba(255,255,255,0.1);}
    /* Premium metric card (custom HTML) */
    .premium-metric-card {background: var(--card-bg); border:1px solid var(--card-border); border-radius:12px; padding:15px; backdrop-filter:blur(16px);}
    .premium-metric-card:hover {transform:translateY(-4px); border-color:rgba(99,102,241,0.4);}
    .metric-label {font-size:0.75rem; color:var(--text-dim); margin-bottom:2px; font-weight:600;}
    .metric-value {font-size:1.7rem; font-weight:800; margin-bottom:1px; letter-spacing:-0.01em;}
    .metric-delta {font-size:0.85rem; font-weight:700; display:flex; align-items:center; gap:4px;}
    .delta-up {color:#ff4b4b;}
    .delta-down {color:#3b82f6;}
    .delta-neutral {color:#94a3b8;}
    /* Buttons */
    .stButton button {font-size:0.77rem; padding:0.2rem 0.5rem; min-height:2rem; line-height:1.2;}
    /* Monumental title */
    .monumental-title {font-size:2.6rem; background:linear-gradient(135deg,#818cf8,#38bdf8,#f59e0b); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:900; line-height:1; margin-bottom:10px; font-family:var(--font-heading); text-align:center;}
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {background:rgba(15,23,42,0.3); border-radius:24px; gap:12px; padding:6px; border:1px solid rgba(255,255,255,0.05);}
    .stTabs [data-baseweb="tab"] {height:32px; background:rgba(15,23,42,0.4); border:1px solid rgba(255,255,255,0.05); border-radius:8px; color:var(--text-dim); font-weight:700; font-size:1.1rem; padding:0 20px; margin-right:8px; display:flex; align-items:center; justify-content:center; transition:all 0.4s cubic-bezier(0.16,1,0.3,1); box-shadow:0 4px 15px rgba(0,0,0,0.2);}
    .stTabs [data-baseweb="tab"]:hover {background:rgba(255,255,255,0.05); color:white;}
    .stTabs [aria-selected="true"] {background:rgba(99,102,241,0.9); color:white; box-shadow:0 0 15px rgba(99,102,241,0.5);}
</style>
""", unsafe_allow_html=True)
```
A large CSS block that defines colors, fonts, glass‑morphism cards, animated tabs, and more.

---

## 4. Sidebar (Reality Check Section)
```python
with st.sidebar:
    st.markdown("### 🔶 신뢰도 검증 (Reality Check)")
    # Load reliability_result.json and display three horizon tabs
    r_path = os.path.join(PROCESSED_DIR, "reliability_result.json")
    if os.path.exists(r_path):
        with open(r_path) as f:
            r_results = json.load(f)
        h_tabs = st.tabs(["1년 전", "1달 전", "1일 전"])
        horizons = [("365", "1년 전"), ("30", "1달 전"), ("1", "1일 전")]
        for i, (h_key, label) in enumerate(horizons):
            with h_tabs[i]:
                res = r_results.get(h_key)
                if res:
                    # Show predicted vs actual price, status icon, etc.
                    ...
```
Creates a sidebar with a title, three tabs for different prediction horizons, and displays reliability metrics.

---

## 5. Main Content – Popovers & Metric Cards
The main page uses a series of **popovers** to expose model metadata, next run time, data statistics, feature counts, etc.
```python
with st.popover("🔶 이 모델의 최신 학습시각", use_container_width=True):
    st.markdown(f"**{model_run['run_display']}**")

with st.popover("🔶 차기 자동학습 예정시각", use_container_width=True):
    st.markdown(f"**{model_run.get('next_run_str', '매일 00:00 (KST)')}**")
    st.caption("※ GitHub Actions 스케줄러 상황에 따라 약간의 지연이 발생할 수 있습니다.")

with st.popover("🔶 총 데이터 포인트", use_container_width=True):
    st.markdown(f"**{len(df):,}일**")
```
These popovers give the user quick access to model‑related information without cluttering the main view.

---

## 6. Tabs for Different Views (Not fully shown here)
The application defines several top‑level tabs (e.g., Model Overview, Price History & Validation, Future Prediction). Each tab contains metric cards, charts (Plotly), and data tables. The exact tab definitions are later in `app.py` and follow the pattern:
```python
tab1, tab2, tab3, tab4 = st.tabs(["Model Overview", "Price History & Validation", "Future Prediction", "..."])
with tab1:
    # Metric cards, charts, etc.
```

---

## 7. Helper Functions for Styling Cells
```python
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
```
Used with `st.dataframe(...).style.applymap(style_expected_return_cell)` to colour‑code return values.

---

## 8. Summary
The UI is entirely built with **Streamlit** components, heavily customised via CSS for a premium dark‑theme experience. Key visual elements include:
- Full‑width layout with auto‑refresh.
- Sidebar containing reliability checks.
- Popovers for model metadata.
- Metric cards with glass‑morphism styling.
- Animated glass‑style tabs.
- Consistent typography using **Inter** and **Outfit** Google fonts.

Feel free to edit `app.py` to adjust any of these sections, or copy the snippets above into a new Streamlit project.

---

*Generated by Antigravity – your AI coding assistant.*
