"""
churn-engine/dashboard/pages/pg_event_stream.py
=================================================
Live Activity Page — Real-time event stream monitor.
Supports both Simulation Mode (synthetic simulator thread)
and Live Store Mode (connected to FastAPI live event buffer).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.event_stream import CustomerStateManager, StreamManager
from src.utils import load_config

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def _get_api_base_url() -> str | None:
    """Read the backend URL from Streamlit secrets without production localhost fallback."""
    try:
        api_url = st.secrets["API_BASE_URL"]
    except (KeyError, StreamlitSecretNotFoundError):
        is_production = os.getenv("CHURN_ENVIRONMENT", "").casefold() == "production"
        api_url = None if is_production else DEFAULT_API_BASE_URL

    if not isinstance(api_url, str) or not api_url.strip():
        if os.getenv("CHURN_ENVIRONMENT", "").casefold() == "production":
            return None
        return DEFAULT_API_BASE_URL
    api_url = api_url.strip().rstrip("/")
    is_production = os.getenv("CHURN_ENVIRONMENT", "").casefold() == "production"
    if is_production and not api_url.casefold().startswith("https://"):
        return None
    return api_url


def fetch_live_events(
    api_url: str = DEFAULT_API_BASE_URL,
    limit: int = 50,
) -> tuple[bool, list[dict], int | None]:
    """Fetch recent live events from FastAPI admin endpoint."""
    url = f"{api_url.strip().rstrip('/')}/api/admin/events/recent?limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StreamlitDashboard/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as res:
            if res.status == 200:
                data = json.loads(res.read().decode("utf-8"))
                if not isinstance(data, dict):
                    st.warning("FastAPI returned an invalid response while fetching recent events.")
                    return False, [], None
                events = data.get("events", [])
                if not isinstance(events, list) or not all(
                    isinstance(event, dict) for event in events
                ):
                    st.warning("FastAPI returned an invalid event list.")
                    return False, [], None
                total_events = data.get("total_events")
                if not isinstance(total_events, int) or total_events < 0:
                    total_events = None
                return True, events, total_events
            st.warning(f"FastAPI returned HTTP {res.status} while fetching recent events.")
    except urllib.error.HTTPError as exc:
        st.warning(f"FastAPI returned HTTP {exc.code} while fetching recent events.")
    except TimeoutError:
        st.warning("The request to FastAPI timed out. Check the backend and try again.")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            st.warning("The request to FastAPI timed out. Check the backend and try again.")
        else:
            st.warning("Could not connect to FastAPI. Check the backend and API_BASE_URL setting.")
    except (json.JSONDecodeError, UnicodeDecodeError):
        st.warning("FastAPI returned an invalid response while fetching recent events.")
    except ValueError:
        st.warning("API_BASE_URL is not a valid URL.")
    return False, [], None


def render() -> None:
    """Render the Live Activity page."""
    # Ensure configuration in session state
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    mode = st.session_state.get("event_source_mode", "simulation")

    if mode == "live":
        render_live_store_mode()
    else:
        render_simulation_mode()


def _format_timestamp(value: object) -> str:
    timestamp = str(value or "")
    if "T" in timestamp:
        return timestamp.replace("T", " ").split(".")[0]
    return timestamp or "—"


def _format_probability(value: object) -> str:
    try:
        return f"{float(value):.2%}" if value is not None else "No data"
    except (TypeError, ValueError):
        return "No data"


def render_live_store_mode() -> None:
    """Render recent events fetched from the live FastAPI event buffer."""
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Live Monitoring</h1>
        <p class="page-subtitle">
            Live Store Mode · events fetched from the FastAPI storefront event buffer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    api_url = _get_api_base_url()
    if api_url is None:
        st.error(
            "Configure the Streamlit API_BASE_URL secret with the deployed HTTPS backend URL."
        )
        connected, live_events, total_events = False, [], None
    else:
        connected, live_events, total_events = fetch_live_events(
            api_url=api_url, limit=50
        )

    status_columns = st.columns(3)
    status_columns[0].metric(
        "FastAPI connection", "Connected" if connected else "Unavailable"
    )
    status_columns[1].metric(
        "Total backend events",
        f"{total_events:,}" if connected and total_events is not None else "Unavailable",
    )
    status_columns[2].metric(
        "Recent events returned", f"{len(live_events):,}" if connected else "Unavailable"
    )
    st.markdown("### Recent live events")
    if live_events:
        table_rows = []
        for event in live_events[:25]:
            table_rows.append({
                "Timestamp": _format_timestamp(event.get("timestamp")),
                "Customer ID": event.get("customer_id", "—"),
                "Event": str(event.get("event_type", "—")).replace("_", " ").title(),
                "Product ID": event.get("product_id") or "—",
                "Churn risk": _format_probability(event.get("churn_probability")),
                "Risk level": event.get("risk_level", "No data"),
                "Decision gate": event.get("decision", "No data"),
                "Strategy": event.get("strategy", "No data"),
            })
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No recent events are available."
            if connected
            else "Live events are unavailable until FastAPI reconnects."
        )

    refresh_rate = float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5))
    time.sleep(refresh_rate)
    st.rerun()


def render_simulation_mode() -> None:
    """Render recent events produced by the synthetic StreamManager."""
    sm: StreamManager = st.session_state.stream_manager
    stats = sm.get_stats()
    is_running = stats["status"] == "running"

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Live Monitoring</h1>
        <p class="page-subtitle">
            Simulation Mode · synthetic events scored by the local StreamManager.
        </p>
    </div>
    """, unsafe_allow_html=True)

    status_columns = st.columns(3)
    status_columns[0].metric("Simulation status", "Running" if is_running else "Stopped")
    status_columns[1].metric("Events this session", f"{stats['total_events']:,}")
    status_columns[2].metric(
        "Processing rate", f"{stats['events_per_second']:.2f} events/sec"
    )
    st.markdown("### Recent simulated events")
    df_feed = sm.get_recent_events(20)
    if len(df_feed) > 0:
        rows = []
        for _, event in df_feed.iterrows():
            prediction = sm.latest_predictions.get(event["customer_id"], {})
            rows.append({
                "Timestamp": _format_timestamp(event.get("event_timestamp")),
                "Customer ID": event.get("customer_id", "—"),
                "Event": str(event.get("event_type", "—")).replace("_", " ").title(),
                "Product ID": event.get("product_id") or "—",
                "Churn risk": _format_probability(prediction.get("churn_probability")),
                "Risk level": prediction.get("risk_level", "No data"),
                "Decision gate": prediction.get("intervention_status", "No data"),
            })
        st.dataframe(
            pd.DataFrame(rows[::-1]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No simulated events are available yet.")

    if is_running:
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
