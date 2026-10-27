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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.event_stream import CustomerStateManager

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
    pg_override,
    pg_evaluation,
    pg_about,
    pg_settings,
    pg_impact_evaluation,
    pg_sim_settings,
    pg_audit_log,
)

# ---------------------------------------------------------------------------
# Custom CSS — Enterprise Light Dashboard
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #F4F6F9;
    --bg-secondary:  #EEF1F5;
    --bg-card:       #ffffff;
    --border-color:  #E1E5EB;
    --border-light:  #EBEEF2;
    --accent-blue:   #4263A5;
    --accent-blue-subtle: #EAF0FA;
    --accent-green:  #10B981;
    --accent-green-subtle: #E7F7F1;
    --accent-amber:  #F59E0B;
    --accent-amber-subtle: #FFF5DF;
    --accent-red:    #F45B69;
    --accent-red-subtle: #FFF0F1;
    --text-primary:  #1F2937;
    --text-secondary:#4B5563;
    --text-muted:    #6B7280;
    --font-main:     'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-mono:     'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --radius:        12px;
    --shadow-sm:     0 2px 8px 0 rgba(31, 41, 55, 0.04);
    --shadow-md:     0 5px 16px -4px rgba(31, 41, 55, 0.08);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main Content Area ── */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
    background-color: var(--bg-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
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
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: var(--shadow-sm);
}

.page-title {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0.15rem 0 0.25rem 0 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em;
}

.page-subtitle {
    font-size: 0.9rem !important;
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
    padding: 0.7rem 0.85rem;
    min-height: 100px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}

.stMetric:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
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

.platform-header {
    background: #fff;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin: 0 0 1rem;
    box-shadow: var(--shadow-sm);
}

.platform-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    min-height: 3.4rem;
}

.platform-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 11px;
    background: #EAF0FA;
    color: #4263A5;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.platform-title {
    color: var(--text-primary);
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.035em;
    line-height: 1.4;
}

.platform-context {
    color: var(--text-muted);
    font-size: 0.72rem;
    margin-top: 0.15rem;
}

.profile-placeholder {
    border-left: 1px solid var(--border-light);
    padding-left: 0.8rem;
    min-height: 2.6rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    white-space: nowrap;
}

.profile-placeholder strong {
    color: var(--text-primary);
    font-size: 0.78rem;
}

.profile-placeholder span {
    color: var(--text-muted);
    font-size: 0.68rem;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
}

.stButton > button {
    border-radius: 9px;
    border-color: var(--border-color);
    color: var(--text-secondary);
    font-weight: 600;
}

.stButton > button:focus-visible,
.stSelectbox [role="combobox"]:focus-visible {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
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
    background: rgba(66, 99, 165, 0.08) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
def render_model_performance_impact() -> None:
    """Render the existing offline evaluation and impact simulation pages together."""
    pg_evaluation.render()
    st.markdown("---")
    pg_impact_evaluation.render()


MAIN_PAGES = {
    "Overview": pg_home.render,
    "Live Monitoring": pg_event_stream.render,
    "Customer Risk": pg_churn_score.render,
    "Retention Interventions": pg_override.render,
    "Model Performance & Impact": render_model_performance_impact,
}

ADMIN_PAGES = {
    "Audit Logs": pg_audit_log.render,
    "Dataset Inspection": pg_dataset.render,
    "Decision Gate Diagnostics": pg_decision_gate.render,
    "Simulation Settings": pg_sim_settings.render,
    "Application Settings": pg_settings.render,
    "About": pg_about.render,
}

PAGES = {**MAIN_PAGES, **ADMIN_PAGES}


def on_main_change() -> None:
    selected_page = st.session_state.main_nav_radio
    if selected_page in MAIN_PAGES:
        st.session_state.selected_page = selected_page
        st.session_state.admin_nav_radio = None


def on_admin_change() -> None:
    selected_page = st.session_state.admin_nav_radio
    if selected_page in ADMIN_PAGES:
        st.session_state.selected_page = selected_page
        st.session_state.main_nav_radio = None


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page name."""

    if st.session_state.get("selected_page") not in PAGES:
        st.session_state.selected_page = "Overview"

    main_options = list(MAIN_PAGES)
    admin_options = list(ADMIN_PAGES)

    if st.session_state.get("main_nav_radio") not in main_options:
        st.session_state.main_nav_radio = (
            st.session_state.selected_page
            if st.session_state.selected_page in main_options
            else None
        )
    if st.session_state.get("admin_nav_radio") not in admin_options:
        st.session_state.admin_nav_radio = (
            st.session_state.selected_page
            if st.session_state.selected_page in admin_options
            else None
        )

    with st.sidebar:
        # Logo / branding
        st.markdown("""
        <div style="
            padding: 1rem 0.5rem;
            border-bottom: 1px solid #d0d7de;
            margin-bottom: 0.75rem;
        ">
            <div style="
                font-size: 1.15rem;
                font-weight: 700;
                color: #1F2937;
                letter-spacing: -0.02em;
                line-height: 1.1;
            ">
                Churn Engine
            </div>
            <div style="
                font-size: 0.68rem;
                color: #656d76;
                margin-top: 0.2rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            ">
                Customer risk &amp; retention platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-section-header">MAIN</div>', unsafe_allow_html=True)
        st.radio(
            label="Main navigation",
            options=main_options,
            index=(
                main_options.index(st.session_state.selected_page)
                if st.session_state.selected_page in main_options
                else None
            ),
            key="main_nav_radio",
            on_change=on_main_change,
            label_visibility="collapsed"
        )

        with st.expander(
            "Administration & Diagnostics",
            expanded=st.session_state.selected_page in admin_options,
        ):
            st.radio(
                label="Administration and diagnostics navigation",
                options=admin_options,
                index=(
                    admin_options.index(st.session_state.selected_page)
                    if st.session_state.selected_page in admin_options
                    else None
                ),
                key="admin_nav_radio",
                on_change=on_admin_change,
                label_visibility="collapsed",
            )

        # Event source mode selector
        st.markdown('<div class="nav-section-header">EVENT SOURCE</div>', unsafe_allow_html=True)
        if "event_source_mode" not in st.session_state:
            st.session_state.event_source_mode = "simulation"

        current_mode = st.session_state.event_source_mode
        mode_options = ["Simulation Mode", "Live Store Mode"]
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

        # Show runtime facts only when they are available from the active source.
        model_label = "Unavailable"
        stream_label = "Idle"
        thresh_label = "Unavailable"
        gate_label = "Unavailable"

        if st.session_state.get("event_source_mode") == "live":
            stream_label = "Live Store Mode"
            gate_label = "Backend event data"
        elif "stream_manager" in st.session_state:
            sm = st.session_state.stream_manager
            stats = sm.get_stats()
            if stats["status"] == "running":
                stream_label = "Running"
            else:
                stream_label = "Stopped"
            
            if sm.predictor and sm.predictor.model is not None:
                raw_model = sm.predictor.model.model
                model_label = raw_model.__class__.__name__
                thresh_label = f"{sm.predictor._churn_threshold:.2f}"
                gate_label = "Configured"

        # System status panel
        st.markdown("---")
        st.markdown('<div class="nav-section-header">System Status</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.78rem;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#6B7280;">Model</span>
                <span style="color:#4263A5; font-weight:600;">{model_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#6B7280;">Stream</span>
                <span style="color:#10B981; font-weight:600;">{stream_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#6B7280;">Gate</span>
                <span style="color:#10B981; font-weight:600;">{gate_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#6B7280;">Threshold</span>
                <span style="color:#4263A5; font-weight:600;">{thresh_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "v1.0.0-production · Python · Streamlit\n\n"
            "Academic Project · E-Commerce Retention"
        )

    return st.session_state.selected_page


def render_platform_header(state: CustomerStateManager) -> None:
    """Render the product header and truthful Overview filter controls."""
    countries = sorted({
        str(event.get("country")).strip()
        for history in state.customer_history.values()
        for event in history
        if event.get("country") and str(event.get("country")).strip()
    })
    region_options = ["All regions", *countries]

    st.markdown(
        """
        <div class="platform-header platform-brand">
            <div class="platform-logo" aria-label="Customer retention engine">CR</div>
            <div>
                <div class="platform-title">GLOBAL E-COMMERCE PARTNER RETENTION HUB</div>
                <div class="platform-context">Customer churn assessment · Local dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    region_col, timeframe_col, notification_col, profile_col = st.columns(
        [1.2, 1.5, 1.5, 1.6], vertical_alignment="center"
    )
    with region_col:
        if st.session_state.get("dashboard_region_filter") not in region_options:
            st.session_state.dashboard_region_filter = "All regions"
        st.selectbox(
            "Region",
            options=region_options,
            key="dashboard_region_filter",
            help="Filters the Overview cohort and returned event records by recorded country; it is not a deployment-region setting.",
        )
    with timeframe_col:
        st.selectbox(
            "Timeframe",
            options=["All available history", "Last 30 days", "Last 7 days"],
            key="dashboard_timeframe_filter",
            help="Filters cohort membership by recorded event time; customer features still use full available history.",
        )
    with notification_col:
        st.button(
            "Notifications · —",
            disabled=True,
            use_container_width=True,
            help="No notification feed is configured.",
        )
    with profile_col:
        st.markdown(
            """
            <div class="profile-placeholder">
                <strong>Dashboard operator</strong>
                <span>Local profile placeholder · not authenticated</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Filters apply to the Overview cohort and returned event panels. "
        "Customer features use full available history; backend total-event counts remain unfiltered."
    )


def main() -> None:
    """Application entry point."""

    # ---------------------------------------------------------------------------
    # Global Bootstrap / Initialization & Auto-Start Simulator
    # ---------------------------------------------------------------------------
    from src.event_stream import CustomerStateManager, StreamManager
    
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

    render_platform_header(st.session_state.state_manager)

    # Render the selected page module
    PAGES[selected_page]()

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
