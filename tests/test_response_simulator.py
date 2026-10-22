"""
churn-engine/tests/test_response_simulator.py
=============================================
Phase 7 Unit Tests — Intervention Response Simulator and Financial Models.

Validates:
  1. Simulator initialization and config/policy binding.
  2. Deterministic persona resolution.
  3. Paired simulation determinism and reproducibility.
  4. Control vs. Treatment cohort calculations (conversion uplift & relative uplift).
  5. Retention and churn rate formulas.
  6. Financial impact models (revenue, discount costs, net margin, and ROI).
  7. Threshold sensitivity what-if analysis monotonicity and trade-offs.
  8. Single-event response simulation contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.response_simulator import (
    InterventionSimulator,
    SimulationSummary,
    get_customer_persona,
)
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def simulator(config):
    return InterventionSimulator(config=config, gross_margin=0.30, random_seed=42)


# ===========================================================================
# 1. Initialization and Configuration
# ===========================================================================

def test_simulator_initialization(simulator):
    """Test simulator attributes and default configurations."""
    assert simulator.gross_margin == 0.30
    assert simulator.random_seed == 42
    assert simulator.default_threshold in [0.40, 0.50]
    assert simulator.discounts["MEDIUM"] == 0.05
    assert simulator.discounts["HIGH"] == 0.15
    assert simulator.discounts["CRITICAL"] == 0.25


def test_get_customer_persona_deterministic():
    """Test deterministic persona resolution from Customer ID."""
    p1 = get_customer_persona("C-1000", seed=42)
    p2 = get_customer_persona("C-1000", seed=42)
    assert p1 == p2
    assert p1 in ["loyal", "high_value", "window_shopper", "discount_hunter", "at_risk", "returning"]


# ===========================================================================
# 2. Simulation Determinism and Reproducibility
# ===========================================================================

def test_simulation_reproducibility(simulator):
    """Running simulation twice with the exact same seed produces identical outcomes."""
    res1 = simulator.run_simulation(threshold=0.50, seed=42)
    res2 = simulator.run_simulation(threshold=0.50, seed=42)

    assert res1.total_customers == res2.total_customers
    assert res1.flagged_customers == res2.flagged_customers
    assert res1.control_conversions == res2.control_conversions
    assert res1.treatment_conversions == res2.treatment_conversions
    assert res1.control_conversion_rate == res2.control_conversion_rate
    assert res1.treatment_conversion_rate == res2.treatment_conversion_rate
    assert res1.retention_uplift == res2.retention_uplift
    assert res1.net_margin_impact == res2.net_margin_impact


# ===========================================================================
# 3. Control vs. Treatment Uplift Calculations
# ===========================================================================

def test_control_vs_treatment_calculation(simulator):
    """Verify that intervention overrides generate positive conversion uplift."""
    res = simulator.run_simulation(threshold=0.50, seed=42)

    assert isinstance(res, SimulationSummary)
    assert res.total_customers == 500
    assert res.flagged_customers > 0

    # Conversions
    assert res.treatment_conversions >= res.control_conversions
    assert 0.0 <= res.control_conversion_rate <= 1.0
    assert 0.0 <= res.treatment_conversion_rate <= 1.0

    # Absolute and relative conversion uplift
    expected_abs_uplift = round(res.treatment_conversion_rate - res.control_conversion_rate, 4)
    assert abs(res.absolute_conversion_uplift - expected_abs_uplift) < 1e-4

    if res.control_conversion_rate > 0:
        expected_rel_uplift = round(res.absolute_conversion_uplift / res.control_conversion_rate, 4)
        assert abs(res.relative_conversion_uplift - expected_rel_uplift) < 1e-4


# ===========================================================================
# 4. Retention and Churn Rate Formulas
# ===========================================================================

def test_retention_and_churn_rates(simulator):
    """Verify retention and churn arithmetic identities."""
    res = simulator.run_simulation(threshold=0.50, seed=42)

    # Churn rate is complement of retention rate
    assert abs((res.control_retention_rate + res.control_churn_rate) - 1.0) < 1e-4
    assert abs((res.treatment_retention_rate + res.treatment_churn_rate) - 1.0) < 1e-4

    # Retention uplift
    expected_ret_uplift = round(res.treatment_retention_rate - res.control_retention_rate, 4)
    assert abs(res.retention_uplift - expected_ret_uplift) < 1e-4

    # Treatment churn rate should be lower or equal to control churn rate
    assert res.treatment_churn_rate <= res.control_churn_rate


# ===========================================================================
# 5. Financial Impact Calculations
# ===========================================================================

def test_financial_impact_calculations(simulator):
    """Verify revenue, discount cost, net margin, and ROI calculations."""
    res = simulator.run_simulation(threshold=0.50, seed=42)

    assert res.baseline_revenue > 0.0
    assert res.intervention_revenue > 0.0
    assert res.total_discount_cost >= 0.0

    # Net revenue impact identity
    expected_net_rev = round(res.intervention_revenue - res.baseline_revenue, 2)
    assert abs(res.net_revenue_impact - expected_net_rev) < 1.0

    # Baseline margin identity
    expected_ctl_margin = round(res.baseline_revenue * simulator.gross_margin, 2)
    assert abs(res.baseline_margin - expected_ctl_margin) < 1.0

    # Net margin impact identity
    expected_net_margin = round(res.intervention_margin - res.baseline_margin, 2)
    assert abs(res.net_margin_impact - expected_net_margin) < 1.0

    # ROI formula
    if res.total_discount_cost > 0:
        expected_roi = round(res.net_margin_impact / res.total_discount_cost, 4)
        assert abs(res.return_on_discount_spend - expected_roi) < 1e-4


# ===========================================================================
# 6. Threshold What-If Sensitivity Analysis
# ===========================================================================

def test_threshold_what_if_sensitivity(simulator):
    """Test threshold sensitivity monotonic properties across cutoffs."""
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    df_sens = simulator.evaluate_threshold_sensitivity(thresholds=thresholds)

    assert len(df_sens) == 5
    assert "threshold" in df_sens.columns
    assert "customers_flagged" in df_sens.columns
    assert "net_margin_impact" in df_sens.columns
    assert "unnecessary_discount_count" in df_sens.columns

    # As threshold increases, fewer customers should be flagged (monotonic non-increasing)
    flagged = df_sens["customers_flagged"].tolist()
    assert all(flagged[i] >= flagged[i + 1] for i in range(len(flagged) - 1))

    # Unnecessary discounts (false positives) should also decrease as threshold tightens
    fp_counts = df_sens["unnecessary_discount_count"].tolist()
    assert all(fp_counts[i] >= fp_counts[i + 1] for i in range(len(fp_counts) - 1))


# ===========================================================================
# 7. Single-Event Response Simulation
# ===========================================================================

def test_single_event_response_simulation(simulator):
    """Test individual event response simulation for live feeds and audit logs."""
    # Test unflagged (0% discount)
    resp_unflagged = simulator.simulate_single_event_response(
        customer_id="C-1001",
        churn_probability=0.20,
        risk_level="LOW",
        discount_offered=0.0,
        aov=100.0,
    )
    assert resp_unflagged["discount_cost"] == 0.0
    assert "Standard baseline recommendation" in resp_unflagged["action_desc"]

    # Test flagged (15% discount)
    resp_flagged = simulator.simulate_single_event_response(
        customer_id="C-1002",
        churn_probability=0.65,
        risk_level="HIGH",
        discount_offered=0.15,
        aov=100.0,
    )
    assert "Applied 15% retention coupon" in resp_flagged["action_desc"]
    assert resp_flagged["outcome_status"] in ["CONVERTED_WITH_DISCOUNT", "NO_CONVERSION"]
    if resp_flagged["simulated_converted"]:
        assert resp_flagged["discount_cost"] == 15.0  # 15% of 100
        assert resp_flagged["simulated_revenue"] == 85.0  # 100 - 15
        # Gross margin = 85 - COGS(70) = 15 = 100 * (0.30 - 0.15)
        assert resp_flagged["margin_contribution"] == 15.0
