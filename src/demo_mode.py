"""Server-owned customer assignments for the explicitly enabled local demo."""

from __future__ import annotations

import os
from typing import Any


DEMO_AT_RISK_CUSTOMERS = frozenset({
    "C-1000",
    "C-1002",
    "C-1108",
    "C-1327",
    "C-1420",
})

DEMO_NOT_AT_RISK_CUSTOMERS = frozenset({
    "C-1001",
    "C-1042",
    "C-1211",
    "C-1388",
    "C-1499",
})

_PRODUCTION_MARKERS = (
    "RENDER",
    "FLY_APP_NAME",
    "DYNO",
    "K_SERVICE",
    "AWS_LAMBDA_FUNCTION_NAME",
    "WEBSITE_INSTANCE_ID",
)


def local_demo_enabled(config: dict[str, Any]) -> bool:
    """Require an explicit local opt-in and fail closed in production environments."""
    project_environment = str(config.get("project", {}).get("environment", "")).casefold()
    runtime_environment = os.getenv("CHURN_ENVIRONMENT", "").casefold()
    production_marked = any(os.getenv(name) for name in _PRODUCTION_MARKERS)

    return (
        os.getenv("CHURN_LOCAL_DEMO_MODE", "").casefold() == "true"
        and runtime_environment == "local"
        and project_environment not in {"production", "prod"}
        and not production_marked
    )


def demo_classification(customer_id: str) -> str | None:
    if customer_id in DEMO_AT_RISK_CUSTOMERS:
        return "DEMO_AT_RISK"
    if customer_id in DEMO_NOT_AT_RISK_CUSTOMERS:
        return "DEMO_NOT_AT_RISK"
    return None
