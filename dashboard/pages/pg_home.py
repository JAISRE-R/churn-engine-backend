"""
churn-engine/dashboard/pages/pg_home.py
==========================================
Overview Page — customer risk and intervention activity.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager
from dashboard.pages.pg_event_stream import _get_api_base_url, fetch_live_events


def _filtered_customer_ids(state: CustomerStateManager) -> list[str]:
    """Return customers with recorded events matching the Overview filters."""
    selected_region = st.session_state.get("dashboard_region_filter", "All regions")
    selected_timeframe = st.session_state.get(
        "dashboard_timeframe_filter", "All available history"
    )
    cutoff = None
    if selected_timeframe == "Last 7 days":
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    elif selected_timeframe == "Last 30 days":
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)

    customer_ids = []
    for customer_id, history in state.customer_history.items():
        matching_events = []
        for event in history:
            if selected_region != "All regions" and str(event.get("country", "")) != selected_region:
                continue
            timestamp = pd.to_datetime(
                event.get("event_timestamp"), errors="coerce", utc=True
            )
            if pd.isna(timestamp) or (cutoff is not None and timestamp < cutoff):
                continue
            matching_events.append(event)
        if matching_events:
            customer_ids.append(customer_id)
    return customer_ids


def _filtered_live_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the Overview region and timeframe filters to returned live events."""
    selected_region = st.session_state.get("dashboard_region_filter", "All regions")
    selected_timeframe = st.session_state.get(
        "dashboard_timeframe_filter", "All available history"
    )
    cutoff = None
    if selected_timeframe == "Last 7 days":
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    elif selected_timeframe == "Last 30 days":
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)

    filtered = []
    for event in events:
        region = event.get("country", event.get("region"))
        if selected_region != "All regions" and str(region or "") != selected_region:
            continue
        timestamp = pd.to_datetime(
            event.get("timestamp", event.get("event_timestamp")),
            errors="coerce",
            utc=True,
        )
        if cutoff is not None and (pd.isna(timestamp) or timestamp < cutoff):
            continue
        filtered.append(event)
    return filtered


def _runtime_feature_importance(sm: StreamManager) -> pd.DataFrame:
    """Read model feature importance only when the loaded runtime artifact exposes it."""
    predictor = getattr(sm, "predictor", None)
    pipeline = getattr(predictor, "model", None)
    feature_names = getattr(pipeline, "feature_names", None)
    model = getattr(pipeline, "model", None)
    if feature_names is None or model is None:
        return pd.DataFrame()

    if hasattr(model, "feature_importances_"):
        raw_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_values = model.coef_
        if getattr(raw_values, "ndim", 1) > 1:
            raw_values = raw_values[0]
        raw_values = [abs(value) for value in raw_values]
    else:
        return pd.DataFrame()

    try:
        names = list(feature_names)
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError):
        return pd.DataFrame()
    if len(names) != len(values) or not values:
        return pd.DataFrame()

    importance = pd.DataFrame({"Feature": names, "Importance": values})
    importance = importance.replace([math.inf, -math.inf], pd.NA).dropna()
    return importance.sort_values("Importance", ascending=False).head(8)


def _feature_value(features: pd.DataFrame, customer_id: str, name: str) -> float | None:
    if name not in features.columns or customer_id not in features.index:
        return None
    value = pd.to_numeric(pd.Series([features.loc[customer_id, name]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def _risk_color(value: Any) -> str:
    normalized = str(value).strip().upper()
    if normalized in {"CRITICAL", "HIGH"}:
        return "#F45B69"
    if normalized in {"MEDIUM", "ELEVATED"}:
        return "#F59E0B"
    return "#10B981"


def _render_intervention_activity(sm: StreamManager, mode: str, live_events: list[dict[str, Any]]) -> None:
    st.markdown("### Recent intervention activity")
    st.caption(
        "Intervention decisions and offer records are not evidence that a customer was retained."
    )
    rows = []
    if mode == "simulation":
        records = sm.override_engine.get_audit_log()
        for record in reversed(records[-8:]):
            products = record.get("recommended_products")
            rows.append({
                "Timestamp": record.get("timestamp", "Unavailable"),
                "Customer ID": record.get("customer_id", "Unavailable"),
                "Decision": "Intervention triggered",
                "Risk": record.get("risk_level", "Unavailable"),
                "Product IDs": ", ".join(map(str, products)) if products else "No product recorded",
                "Outcome": "Not recorded",
                "Source": "Local simulation/runtime audit",
            })
    else:
        for event in live_events:
            decision = event.get("decision") or event.get("intervention_status")
            if not decision or str(decision).upper() in {"NORMAL", "NONE", "NO_INTERVENTION"}:
                continue
            products = event.get("recommended_products")
            rows.append({
                "Timestamp": event.get("timestamp", "Unavailable"),
                "Customer ID": event.get("customer_id", "Unavailable"),
                "Decision": str(decision).replace("_", " ").title(),
                "Risk": event.get("risk_level", "Unavailable"),
                "Product IDs": ", ".join(map(str, products)) if isinstance(products, list) else "Unavailable",
                "Outcome": "Not recorded",
                "Source": "Live backend event",
            })
        rows = rows[:8]

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No intervention decisions are available for the selected source.")


def render() -> None:
    """Render the Overview dashboard."""
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)
    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm: StreamManager = st.session_state.stream_manager
    state: CustomerStateManager = st.session_state.state_manager
    stats = sm.get_stats()
    mode = st.session_state.get("event_source_mode", "simulation")
    model_loaded = getattr(getattr(sm, "predictor", None), "model", None) is not None

    if model_loaded and not sm.latest_predictions and mode == "simulation":
        from collections import deque

        for customer_id in list(state.customer_history.keys())[:30]:
            features = state.get_features(customer_id)
            prediction = sm.predictor.predict(customer_id, features)
            sm.latest_predictions[customer_id] = prediction
            history = sm.prediction_history.setdefault(
                customer_id, deque(maxlen=sm.risk_history_size)
            )
            history.append({
                "timestamp": prediction["timestamp"],
                "churn_probability": prediction["churn_probability"],
                "threshold": prediction["threshold"],
                "risk_level": prediction["risk_level"],
                "intervention_status": prediction["intervention_status"],
            })

    live_connected: bool | None = None
    live_events: list[dict[str, Any]] = []
    live_total_events = None
    if mode == "live":
        live_connected, live_events, live_total_events = fetch_live_events(
            api_url=_get_api_base_url(),
            limit=50,
        )
        if live_connected:
            live_events = _filtered_live_events(live_events)

    st.markdown(
        """
        <div class="page-header">
            <h1 class="page-title">Overview</h1>
            <p class="page-subtitle">A current view of customer risk, data availability, and recorded intervention decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_label = (
        "Live Store Mode · FastAPI connected"
        if mode == "live" and live_connected
        else "Live Store Mode · FastAPI unavailable"
        if mode == "live"
        else "Simulation Mode · local StreamManager"
    )
    stream_status = (
        "Connected" if live_connected else "Unavailable"
    ) if mode == "live" else ("Running" if stats["status"] == "running" else "Stopped")
    st.caption(f"**System status:** {stream_status} · **Data source:** {source_label}")

    filtered_ids = _filtered_customer_ids(state)
    scored = {}
    if mode == "simulation" and model_loaded:
        scored = {
            customer_id: sm.latest_predictions[customer_id]
            for customer_id in filtered_ids
            if customer_id in sm.latest_predictions
            and pd.notna(sm.latest_predictions[customer_id].get("churn_probability"))
        }

    at_risk = {
        customer_id: prediction
        for customer_id, prediction in scored.items()
        if float(prediction["churn_probability"]) >= float(prediction.get("threshold", 1.0))
    }
    probabilities = [
        float(prediction["churn_probability"])
        for prediction in scored.values()
        if math.isfinite(float(prediction["churn_probability"]))
    ]
    average_probability = sum(probabilities) / len(probabilities) if probabilities else None

    local_customer_count = len(filtered_ids) if mode == "simulation" else None
    metric_data = [
        (
            "At-risk customers",
            f"{len(at_risk):,}" if scored else "Unavailable",
            "Scored customers at or above their recorded decision threshold."
            if scored else "No current scored cohort is available for this data source.",
        ),
        (
            "At-risk GMV",
            "Unavailable",
            "The available data has no GMV field. Lifetime purchase value is a different proxy.",
        ),
        (
            "Projected churn rate",
            f"{average_probability:.1%}" if average_probability is not None else "Unavailable",
            f"Mean predicted churn probability across {len(scored):,} currently scored customers; "
            "this is a cohort score average, not an observed population churn rate."
            if scored else "A complete scored cohort is not available.",
        ),
        (
            "High-value partners at risk",
            "Unavailable",
            "No partner classification or configured high-value cutoff is available.",
        ),
        (
            "Customers in local event history",
            f"{local_customer_count:,}" if local_customer_count is not None else "Unavailable",
            "Distinct customers with recorded local events matching the Overview filters."
            if local_customer_count is not None else "Live Store Mode does not expose a complete local customer cohort.",
        ),
        (
            "Win-back success rate",
            "Unavailable",
            "No observed retention outcome records are available.",
        ),
    ]
    for row_start in (0, 3):
        columns = st.columns(3)
        for column, (label, value, explanation) in zip(
            columns, metric_data[row_start:row_start + 3]
        ):
            column.metric(label, value, help=explanation)

    if mode == "live":
        total_events = live_total_events if live_connected else None
    else:
        loaded_event_count = getattr(state, "loaded_event_count", None)
        if loaded_event_count is None:
            loaded_event_count = max(
                0,
                sum(len(history) for history in state.customer_history.values())
                - stats["total_events"],
            )
        total_events = loaded_event_count + stats["total_events"]
    event_count_label = f"{total_events:,}" if total_events is not None else "Unavailable"
    if mode == "simulation":
        st.caption(
            f"Events processed: {event_count_label} · "
            f"Scored cohort: {len(scored):,} customers"
        )
    else:
        st.caption(
            f"Backend total events: {event_count_label} · "
            "The live event endpoint does not expose a complete customer scoring cohort."
        )

    st.markdown("### Churn risk segmentation")
    st.caption(
        "Customer-level churn probability against lifetime purchase value proxy (purchase totals in recorded history; not GMV). "
        "The selected cohort is based on matching event activity; feature values use full available customer history. "
        "Value quadrants use the displayed cohort median, not a configured eligibility or business threshold."
    )
    segmentation_rows = []
    if mode == "simulation" and model_loaded:
        for customer_id, prediction in scored.items():
            features = state.get_features(customer_id)
            value = _feature_value(features, customer_id, "monetary_total")
            probability = pd.to_numeric(
                pd.Series([prediction.get("churn_probability")]), errors="coerce"
            ).iloc[0]
            if value is not None and pd.notna(probability):
                segmentation_rows.append({
                    "Customer ID": customer_id,
                    "Churn probability": float(probability),
                    "Lifetime purchase value proxy": value,
                    "Risk": str(prediction.get("risk_level", "Unavailable")).upper(),
                })

    if segmentation_rows:
        segmentation = pd.DataFrame(segmentation_rows)
        value_midpoint = segmentation["Lifetime purchase value proxy"].median()
        fig = px.scatter(
            segmentation,
            x="Churn probability",
            y="Lifetime purchase value proxy",
            color="Risk",
            color_discrete_map={
                "CRITICAL": "#F45B69",
                "HIGH": "#F45B69",
                "MEDIUM": "#F59E0B",
                "LOW": "#10B981",
            },
            hover_data={
                "Customer ID": True,
                "Churn probability": ":.1%",
                "Lifetime purchase value proxy": ":,.2f",
                "Risk": True,
            },
            template="plotly_white",
        )
        threshold_values = [
            float(prediction["threshold"])
            for prediction in scored.values()
            if prediction.get("threshold") is not None
        ]
        configured_threshold = threshold_values[0] if threshold_values else None
        if configured_threshold is not None:
            fig.add_vline(
                x=configured_threshold,
                line_dash="dash",
                line_color="#F45B69",
                annotation_text="Decision threshold",
            )
        fig.add_hline(
            y=value_midpoint,
            line_dash="dot",
            line_color="#9CA3AF",
            annotation_text="Displayed cohort median",
        )
        fig.update_layout(
            height=330,
            margin={"l": 10, "r": 10, "t": 25, "b": 10},
            legend_title_text="Model risk tier",
            xaxis={"tickformat": ".0%", "title": "Churn probability"},
            yaxis_title="Lifetime purchase value proxy",
        )
        fig.update_traces(marker={"size": 9, "opacity": 0.78})
        st.plotly_chart(fig, use_container_width=True)
    elif mode == "live":
        st.info("Customer-level risk segmentation is unavailable: the live event feed does not provide a complete scored customer cohort.")
    elif not model_loaded:
        st.info("Customer-level risk segmentation is unavailable because the local model artifact is not loaded.")
    else:
        st.info("No scored customers with recorded purchase-value features match the selected Overview filters.")

    panel_left, panel_right = st.columns([1.05, 1.35], gap="large")
    with panel_left:
        st.markdown("### Early warning indicators")
        st.caption(
            "Share of the displayed customer cohort meeting existing behavioral indicators. "
            "These are descriptive signals, not causal findings."
        )
        indicator_rows = []
        if mode == "simulation" and scored:
            indicator_rules = [
                ("Days since event > 30", "recency_days", lambda value: value > 30, "High"),
                ("Days since purchase > 45", "days_since_last_purchase", lambda value: value > 45, "High"),
                ("Cart abandonment ≥ 50%", "cart_abandonment_rate", lambda value: value >= 0.50, "Medium"),
                ("Support tickets ≥ 2", "support_ticket_count", lambda value: value >= 2, "Medium"),
                ("No events in past 7 days", "events_7d", lambda value: value == 0, "Medium"),
            ]
            for label, feature, rule, severity in indicator_rules:
                values = []
                for customer_id in scored:
                    value = _feature_value(state.get_features(customer_id), customer_id, feature)
                    if value is not None:
                        values.append(value)
                if values:
                    indicator_rows.append({
                        "Indicator": label,
                        "Share of cohort": sum(rule(value) for value in values) / len(values),
                        "Severity": severity,
                        "Customers with feature": len(values),
                    })
        if indicator_rows:
            indicator_df = pd.DataFrame(indicator_rows)
            fig_indicators = go.Figure()
            for severity, color in [
                ("High", "#F45B69"),
                ("Medium", "#F59E0B"),
            ]:
                subset = indicator_df[indicator_df["Severity"] == severity]
                if not subset.empty:
                    fig_indicators.add_trace(go.Bar(
                        x=subset["Share of cohort"],
                        y=subset["Indicator"],
                        orientation="h",
                        name=severity,
                        marker_color=color,
                        customdata=subset[["Customers with feature"]].to_numpy(),
                        hovertemplate=(
                            "%{y}<br>Share: %{x:.1%}<br>"
                            "Customers with feature: %{customdata[0]}<extra></extra>"
                        ),
                    ))
            fig_indicators.update_layout(
                template="plotly_white",
                barmode="overlay",
                height=300,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis={"tickformat": ".0%", "title": "Share of customers"},
                yaxis_title=None,
                legend_title_text="Existing signal severity",
            )
            st.plotly_chart(fig_indicators, use_container_width=True)
        else:
            st.info("No customer-level behavior is available for this indicator summary.")

        importance = _runtime_feature_importance(sm) if model_loaded else pd.DataFrame()
        if not importance.empty:
            with st.expander("Runtime model feature importance", expanded=False):
                st.caption("Importance from the loaded local model artifact; not causal attribution.")
                fig_importance = px.bar(
                    importance.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    template="plotly_white",
                )
                fig_importance.update_traces(marker_color="#4263A5")
                fig_importance.update_layout(
                    height=280,
                    margin={"l": 10, "r": 10, "t": 10, "b": 10},
                    xaxis_title="Model importance",
                    yaxis_title=None,
                    showlegend=False,
                )
                st.plotly_chart(fig_importance, use_container_width=True)

    with panel_right:
        st.markdown("### At-risk customer drilldown")
        st.caption("Risk score does not establish eligibility for a product-specific retention offer.")
        search = st.text_input(
            "Search at-risk customers",
            placeholder="Search customer ID or risk tier",
            key="overview_customer_search",
        )
        table_rows = []
        feature_cache: dict[str, pd.DataFrame] = {}
        for customer_id, prediction in at_risk.items():
            features = state.get_features(customer_id)
            feature_cache[customer_id] = features
            row = {
                "Customer ID": customer_id,
                "Customer segment": "Unavailable",
                "Churn probability": float(prediction["churn_probability"]),
                "Risk": prediction.get("risk_level", "Unavailable"),
                "Behavioral indicators": "Unavailable",
                "Lifetime purchase value proxy": _feature_value(
                    features, customer_id, "monetary_total"
                ),
            }
            abandonment = _feature_value(features, customer_id, "cart_abandonment_rate")
            recency = _feature_value(features, customer_id, "recency_days")
            purchase_count = _feature_value(features, customer_id, "purchase_count")
            indicators = []
            if abandonment is not None:
                indicators.append(f"Cart abandonment {abandonment:.0%}")
            if recency is not None:
                indicators.append(f"Last event {recency:.0f}d ago")
            if purchase_count is not None:
                indicators.append(f"{purchase_count:.0f} recorded purchases")
            if indicators:
                row["Behavioral indicators"] = " · ".join(indicators)
            table_rows.append(row)

        customer_table = pd.DataFrame(table_rows)
        if not customer_table.empty and search.strip():
            matches = customer_table.astype(str).apply(
                lambda column: column.str.contains(search.strip(), case=False, na=False)
            ).any(axis=1)
            customer_table = customer_table[matches]
        if not customer_table.empty:
            customer_table["Churn probability"] = customer_table["Churn probability"].map(
                lambda value: f"{value:.1%}"
            )
            customer_table["Lifetime purchase value proxy"] = customer_table[
                "Lifetime purchase value proxy"
            ].map(lambda value: f"{value:,.2f}" if pd.notna(value) else "Unavailable")
            styled_table = customer_table.style.map(
                lambda value: (
                    "color: #F45B69; font-weight: 700"
                    if str(value).upper() in {"HIGH", "CRITICAL"}
                    else "color: #F59E0B; font-weight: 700"
                    if str(value).upper() == "MEDIUM"
                    else "color: #10B981; font-weight: 700"
                    if str(value).upper() == "LOW"
                    else ""
                ),
                subset=["Risk"],
            )
            st.dataframe(styled_table, use_container_width=True, hide_index=True)
            selectable_customers = customer_table["Customer ID"].tolist()
            selected_customer = st.selectbox(
                "Customer to inspect",
                options=selectable_customers,
                key="overview_inspect_customer",
            )
            if st.button("Open customer risk details", type="secondary"):
                st.session_state.dashboard_inspect_customer = selected_customer
                st.session_state.dashboard_customer_risk_selector = selected_customer
                st.session_state.selected_page = "Customer Risk"
                st.session_state.main_nav_radio = "Customer Risk"
                st.session_state.admin_nav_radio = None
                st.rerun()
        elif at_risk:
            st.info("No at-risk customer matches the current search.")
        elif scored:
            st.info("No scored customers currently meet their configured decision threshold.")
        elif mode == "live":
            st.info("Customer risk details are unavailable from the recent-events-only live feed.")
        else:
            st.info("No customer risk scores are available for the selected Overview filters.")

    _render_intervention_activity(sm, mode, live_events)

    if stats["status"] == "running" or mode == "live":
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
