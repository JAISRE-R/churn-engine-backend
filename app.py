"""
churn-engine/app.py
====================
Streamlit Application Entry Point.

Live Customer Churn Risk Assessment Engine
— Real-Time Recommendation Overrides for E-Commerce —

This module bootstraps the application:
  1. Loads the project configuration.
  2. Initialises the logging system.
  3. Ensures all required directories exist.
  4. Renders a rich, styled multi-page dashboard.

Run with:
    streamlit run app.py

Phase 1 Status
--------------
All pipeline modules are scaffolded. This entry point renders
placeholder pages for every section. Real logic will be wired
in subsequent phases.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on the Python path so `src` is importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Assessment Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    None,
        "Report a bug": None,
        "About": (
            "**Live Customer Churn Risk Assessment Engine**\n\n"
            "Real-Time Recommendation Overrides for E-Commerce\n\n"
            "Phase 1 — Project Foundation"
        ),
    },
)

# ---------------------------------------------------------------------------
# Bootstrap: config + logging + directories
# ---------------------------------------------------------------------------
from src.utils import load_config, setup_logging, ensure_project_dirs

config = load_config()
logger = setup_logging(config)
ensure_project_dirs(config)
logger.info("Churn Engine dashboard starting — Phase 1")

# ---------------------------------------------------------------------------
# Dashboard page modules
# ---------------------------------------------------------------------------
from dashboard.pages import (
    pg_home,
    pg_dataset,
    pg_event_stream,
    pg_churn_score,
    pg_decision_gate,
    pg_recommendations,
    pg_override,
    pg_evaluation,
    pg_about,
    pg_settings,
    pg_impact_evaluation,
    pg_sim_settings,
    pg_audit_log,
)

# ---------------------------------------------------------------------------
# Custom CSS — Clean White Professional SaaS Design
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables: Clean White SaaS ── */
:root {
    --bg-primary:    #ffffff;
    --bg-secondary:  #f6f8fa;
    --bg-card:       #ffffff;
    --border-color:  #d0d7de;
    --border-light:  #eaeef2;
    --accent-blue:   #0969da;
    --accent-blue-subtle: #ddf4ff;
    --accent-green:  #1a7f37;
    --accent-green-subtle: #dafbe1;
    --accent-amber:  #9a6700;
    --accent-amber-subtle: #fff8c5;
    --accent-red:    #cf222e;
    --accent-red-subtle: #ffebe9;
    --text-primary:  #1f2328;
    --text-secondary:#424a53;
    --text-muted:    #656d76;
    --font-main:     'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-mono:     'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --radius:        8px;
    --shadow-sm:     0 1px 2px 0 rgba(31, 35, 40, 0.04);
    --shadow-md:     0 3px 6px -1px rgba(31, 35, 40, 0.08);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main Content Area ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
    background-color: var(--bg-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}

[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-primary) !important;
}

/* ── Page Header ── */
.page-header {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
}

.page-title {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0.25rem 0 0.25rem 0 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em;
}

.page-subtitle {
    font-size: 0.98rem !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

/* ── Phase Badge ── */
.phase-badge {
    display: inline-block;
    background-color: var(--accent-blue-subtle);
    border: 1px solid #54aeff;
    color: var(--accent-blue);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    margin-bottom: 0.4rem;
}

/* ── Phase Cards (Home page) ── */
.phase-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.15s ease;
}

.phase-card.active {
    border-color: var(--accent-green);
    background: #f6fbf7;
}

.phase-tag {
    font-size: 0.75rem;
    font-weight: 700;
    font-family: var(--font-mono);
    color: var(--accent-blue);
    min-width: 70px;
}

.phase-desc {
    flex: 1;
    color: var(--text-primary);
    font-size: 0.9rem;
}

.phase-status {
    font-size: 0.8rem;
    color: var(--text-muted);
    white-space: nowrap;
}

.phase-card.active .phase-status {
    color: var(--accent-green);
    font-weight: 600;
}

/* ── Sidebar Navigation ── */
.nav-section-header {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    padding: 1.1rem 0 0.35rem 0;
}

/* ── Streamlit Widgets Overrides ── */
.stSelectbox > div > div,
.stSlider > div,
.stMetric {
    background: #ffffff !important;
    border-radius: var(--radius);
}

.stMetric {
    border: 1px solid var(--border-color);
    padding: 0.85rem 1.1rem;
    box-shadow: var(--shadow-sm);
}

[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border-color) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Alerts ── */
.stAlert {
    border-radius: var(--radius) !important;
    border-left-width: 4px !important;
}

/* ── Code blocks ── */
.stCodeBlock {
    border-radius: var(--radius);
    border: 1px solid var(--border-color);
    font-family: var(--font-mono) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border-color) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── JSON viewer ── */
.element-container .stJson {
    background: #ffffff !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

/* ── Sidebar radio buttons ── */
[data-testid="stSidebar"] .stRadio {
    gap: 0.2rem !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 0.4rem 0.75rem !important;
    border-radius: 6px !important;
    transition: background 0.12s ease !important;
    cursor: pointer !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(9, 105, 218, 0.08) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
PAGES: dict[str, dict] = {
    # MAIN PRODUCT
    "📊 Dashboard":            {"module": pg_home},
    "⚡ Live Activity":         {"module": pg_event_stream},
    "🎯 Customer Risk":        {"module": pg_churn_score},
    "💡 Recommendations":      {"module": pg_recommendations},
    "🔄 Interventions":        {"module": pg_override},
    "📋 Audit Log":            {"module": pg_audit_log},
    "📈 Impact Evaluation":    {"module": pg_impact_evaluation},

    # SECONDARY / TECHNICAL
    "🗄️ Dataset & EDA":        {"module": pg_dataset},
    "🧪 Model Evaluation":     {"module": pg_evaluation},
    "🚦 Data Quality / System Diagnostics": {"module": pg_decision_gate},
    "🎛️ Simulation Settings":   {"module": pg_sim_settings},
    "⚙️ Settings":             {"module": pg_settings},
    "ℹ️ About":                {"module": pg_about},
}


def on_main_change():
    if st.session_state.main_nav_radio:
        st.session_state.selected_page = st.session_state.main_nav_radio

def on_tech_change():
    if st.session_state.tech_nav_radio:
        st.session_state.selected_page = st.session_state.tech_nav_radio


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page name."""

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "📊 Dashboard"

    with st.sidebar:
        # Logo / branding
        st.markdown("""
        <div style="
            padding: 1rem 0.5rem;
            border-bottom: 1px solid #d0d7de;
            margin-bottom: 0.75rem;
        ">
            <div style="
                font-size: 1.35rem;
                font-weight: 800;
                color: #0969da;
                letter-spacing: -0.02em;
                line-height: 1.1;
            ">
                🎯 Churn Engine
            </div>
            <div style="
                font-size: 0.68rem;
                color: #656d76;
                margin-top: 0.2rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            ">
                Live Risk &amp; Retention Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main product section
        main_options = [
            "📊 Dashboard",
            "⚡ Live Activity",
            "🎯 Customer Risk",
            "💡 Recommendations",
            "🔄 Interventions",
            "📋 Audit Log",
            "📈 Impact Evaluation"
        ]
        st.markdown('<div class="nav-section-header">MAIN PRODUCT</div>', unsafe_allow_html=True)
        main_idx = None
        if st.session_state.selected_page in main_options:
            main_idx = main_options.index(st.session_state.selected_page)
            
        st.radio(
            label="Main Product",
            options=main_options,
            index=main_idx,
            key="main_nav_radio",
            on_change=on_main_change,
            label_visibility="collapsed"
        )

        # Event source mode selector
        st.markdown('<div class="nav-section-header">EVENT SOURCE</div>', unsafe_allow_html=True)
        if "event_source_mode" not in st.session_state:
            st.session_state.event_source_mode = "simulation"

        current_mode = st.session_state.event_source_mode
        mode_options = ["⚡ Simulation Mode", "🌐 Live Store Mode"]
        mode_idx = 0 if current_mode == "simulation" else 1

        selected_mode_label = st.radio(
            label="Event Source Mode",
            options=mode_options,
            index=mode_idx,
            key="event_source_mode_radio",
            label_visibility="collapsed"
        )
        new_mode = "simulation" if "Simulation" in selected_mode_label else "live"
        if new_mode != current_mode:
            st.session_state.event_source_mode = new_mode
            if "stream_manager" in st.session_state:
                sm = st.session_state.stream_manager
                if new_mode == "simulation":
                    if not sm._running:
                        sm.start()
                else:
                    if sm._running:
                        sm.stop()
            st.rerun()

        # Dynamic System status values
        model_label = "❌ Not Loaded"
        stream_label = "⏸ Idle"
        thresh_label = "0.50"
        gate_label = "🟢 Active"

        if st.session_state.get("event_source_mode") == "live":
            stream_label = "🌐 LIVE STORE"
            thresh_label = "0.50"
            model_label = "✅ Live Bridge"
        elif "stream_manager" in st.session_state:
            sm = st.session_state.stream_manager
            stats = sm.get_stats()
            if stats["status"] == "running":
                stream_label = "🟢 SIMULATING"
            else:
                stream_label = "⏸ Stopped"
            
            if sm.predictor and sm.predictor.model is not None:
                raw_model = sm.predictor.model.model
                model_label = f"✅ {raw_model.__class__.__name__}"
                thresh_label = f"{sm.predictor._churn_threshold:.2f}"
                gate_label = "🟢 Active"

        # System status panel
        st.markdown("---")
        st.markdown('<div class="nav-section-header">System Status</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.78rem;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#656d76;">Model</span>
                <span style="color:#0969da; font-weight:600;">{model_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#656d76;">Stream</span>
                <span style="color:#1a7f37; font-weight:600;">{stream_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#656d76;">Gate</span>
                <span style="color:#1a7f37; font-weight:600;">{gate_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#656d76;">Threshold</span>
                <span style="color:#0969da; font-weight:600;">{thresh_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "v1.0.0-production · Python · Streamlit\n\n"
            "Academic Project · E-Commerce Retention"
        )

    return st.session_state.selected_page


def main() -> None:
    """Application entry point."""

    # ---------------------------------------------------------------------------
    # Global Bootstrap / Initialization & Auto-Start Simulator
    # ---------------------------------------------------------------------------
    from src.event_stream import StreamManager, CustomerStateManager
    
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        sm = StreamManager(st.session_state.config, st.session_state.state_manager)
        st.session_state.stream_manager = sm
        
        # Auto-start simulated event stream at launch
        if not sm._running:
            sm.start()
            logger.info("Automatically started StreamManager background thread at launch.")

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Render the selected page module
    page_info = PAGES[selected_page]
    page_info["module"].render()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#656d76; font-size:0.75rem; padding:0.5rem;'>"
        "Live Customer Churn Risk Assessment Engine &nbsp;·&nbsp; "
        "Business Analytics Interface &nbsp;·&nbsp; "
        "Built with Streamlit &amp; Python"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
