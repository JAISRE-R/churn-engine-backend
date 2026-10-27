"""
churn-engine/api.py
===================
FastAPI Thin Bridge Layer.

Connects the external customer-facing frontend to the existing Python
churn engine (CustomerStateManager, ChurnPredictor, DecisionGate,
BaselineRecommender, and OverrideEngine).

Pipeline Flow:
--------------
HTTP Event
    ↓
CustomerEvent translation
    ↓
Event normalization
    ↓
CustomerStateManager.update_state()
    ↓
CustomerStateManager.get_features()
    ↓
ChurnPredictor.predict()
    ↓
DecisionGate.decide()
    ↓
BaselineRecommender.recommend()
    ↓
OverrideEngine.process()
    ↓
Product ID Adaptation & Region Mapping
    ↓
CustomerExperience JSON response
"""

from __future__ import annotations

import os
import sys
import uuid
import logging
import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, ensure_project_dirs
from src.event_stream import CustomerStateManager, CustomerEvent
from src.predictor import ChurnPredictor
from src.decision_gate import DecisionGate
from src.recommender import BaselineRecommender
from src.override_engine import OverrideEngine
from src.checkout_ledger import CheckoutLedger
from src.demo_mode import (
    DEMO_AT_RISK_CUSTOMERS,
    DEMO_NOT_AT_RISK_CUSTOMERS,
    demo_classification,
    local_demo_enabled,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Domain Mapping & Constants
# ===========================================================================

EVENT_TYPE_MAP: dict[str, str] = {
    "product_view": "product_click",
    "support_request": "support_contact",
    "wishlist_add": "wishlist_click",
    "wishlist_remove": "wishlist_click",
    "cart_view": "page_view",
    "cart_update": "add_to_cart",
}

COUNTRY_TO_REGION: dict[str, str] = {
    "US": "North America",
    "CA": "North America",
    "MX": "Latin America",
    "GB": "Europe",
    "DE": "Europe",
    "FR": "Europe",
    "IT": "Europe",
    "ES": "Europe",
    "NL": "Europe",
    "AU": "Asia-Pacific",
    "JP": "Asia-Pacific",
    "SG": "Asia-Pacific",
    "IN": "Asia-Pacific",
    "NZ": "Asia-Pacific",
    "BR": "Latin America",
    "AR": "Latin America",
    "AE": "Middle East",
    "SA": "Middle East",
}

REGIONS: list[str] = [
    "Asia-Pacific",
    "North America",
    "Europe",
    "Latin America",
    "Middle East",
]

# Frontend catalog: 36 products (P-1001 to P-1036)
VALID_CATALOG_IDS: list[str] = [f"P-{i}" for i in range(1001, 1037)]
VALID_CATALOG_SET: set[str] = set(VALID_CATALOG_IDS)

CATEGORY_CATALOG_MAP: dict[str, list[str]] = {
    "Laptops": ["P-1001", "P-1002", "P-1003", "P-1004"],
    "Smartphones": ["P-1005", "P-1006", "P-1007", "P-1008"],
    "Headphones": ["P-1009", "P-1010", "P-1011", "P-1012"],
    "Accessories": ["P-1013", "P-1014", "P-1015", "P-1016"],
    "Electronics": ["P-1017", "P-1018", "P-1019", "P-1020"],
    "Smart Home": ["P-1021", "P-1022", "P-1023", "P-1024"],
    "Wearables": ["P-1025", "P-1026", "P-1027", "P-1028"],
    "Gaming": ["P-1029", "P-1030", "P-1031", "P-1032"],
    "Office Equipment": ["P-1033", "P-1034", "P-1035", "P-1036"],
    "Fashion & Apparel": ["P-1013", "P-1025", "P-1026", "P-1027"],
    "Home & Living": ["P-1021", "P-1022", "P-1023", "P-1024"],
    "Beauty & Health": ["P-1027", "P-1028", "P-1016"],
    "Sports & Outdoors": ["P-1028", "P-1029", "P-1030"],
}


def resolve_region(customer_id: str, country: str | None = None) -> str:
    """Deterministically map customer identity or country to a geographical region."""
    if country and country in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[country]
    # Stable deterministic hashing by customer_id matching frontend demo region
    h = 0
    for c in customer_id:
        h = (h * 31 + ord(c)) % 100000
    return REGIONS[h % len(REGIONS)]


def adapt_recommendations(
    raw_recommendations: list[dict[str, Any]],
    offers_authorized: bool = False,
) -> list[dict[str, Any]]:
    """Adapt backend recommendation product IDs to valid frontend catalog products.
    
    Guarantees:
    - If rec product_id is in frontend catalog (P-1001..P-1036), keep it intact.
    - If non-catalog ID (e.g. P-101, COUPON-CRIT-25), resolve deterministically
      to a valid catalog product from the matching category.
    - Preserves backend reason, authorized product-level discounts, and ranking.
    - Avoids duplicate product IDs in the returned list.
    """
    adapted: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for rec in raw_recommendations:
        pid_val = rec.get("product_id")
        pid = str(pid_val).strip() if pid_val is not None and not (isinstance(pid_val, float) and pd.isna(pid_val)) else ""
        category_val = rec.get("category")
        category = str(category_val).strip() if category_val is not None and not (isinstance(category_val, float) and pd.isna(category_val)) else "General"
        reason_val = rec.get("reason")
        reason = str(reason_val).strip() if reason_val is not None and not (isinstance(reason_val, float) and pd.isna(reason_val)) else "Recommended for you"
        item_discount = rec.get("discount", 0)
        if (
            not offers_authorized
            or pid not in VALID_CATALOG_SET
            or not isinstance(item_discount, (int, float))
            or not 0 < item_discount <= 100
        ):
            item_discount = 0

        # 1. Direct hit in catalog
        if pid in VALID_CATALOG_SET and pid not in used_ids:
            target_pid = pid
        else:
            # 2. Deterministic category or hash-based fallback
            candidates = CATEGORY_CATALOG_MAP.get(category, VALID_CATALOG_IDS)
            hash_idx = sum(ord(c) for c in pid) if pid else 0
            
            # Find first unused candidate
            target_pid = None
            for i in range(len(candidates)):
                cand = candidates[(hash_idx + i) % len(candidates)]
                if cand not in used_ids:
                    target_pid = cand
                    break
            
            # If all category candidates used, pick from full catalog
            if target_pid is None:
                for cand in VALID_CATALOG_IDS:
                    if cand not in used_ids:
                        target_pid = cand
                        break
            
            if target_pid is None:
                target_pid = candidates[0]

        used_ids.add(target_pid)
        adapted.append({
            "product_id": target_pid,
            "reason": reason,
            "discount": int(item_discount),
        })

    return adapted


# ===========================================================================
# Pydantic Request & Response Schemas
# ===========================================================================

class EventRequest(BaseModel):
    customer_id: str = Field(..., description="Customer identifier, e.g. C-1327")
    session_id: str = Field(..., description="Session identifier")
    event_type: str = Field(..., description="Action name, e.g. page_view, add_to_cart")
    product_id: Optional[str] = Field(None, description="Interacted product ID")
    category: Optional[str] = Field(None, description="Product category")
    quantity: Optional[int] = Field(None, description="Item quantity")
    order_value: Optional[float] = Field(None, description="Order total in USD")
    query: Optional[str] = Field(None, description="Search query string")
    timestamp: Optional[str] = Field(None, description="ISO timestamp string")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)


class CheckoutLineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    quantity: int = Field(ge=1, le=99)


class CheckoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1)
    lines: list[CheckoutLineRequest] = Field(..., min_length=1, max_length=36)


class CheckoutFinalizeRequest(CheckoutRequest):
    order_id: str = Field(..., min_length=1, max_length=128)


class RecommendationItem(BaseModel):
    product_id: str
    reason: Optional[str] = None
    discount: Optional[int] = 0


class CustomerExperienceResponse(BaseModel):
    customer_id: str
    region: Optional[str] = None
    churn_probability: Optional[float] = None
    risk_level: Optional[str] = None
    decision: Optional[str] = None
    strategy: Optional[str] = None
    discount: Optional[int] = 0
    headline: Optional[str] = None
    message: Optional[str] = None
    recommendation_type: Optional[str] = None
    recommendations: list[RecommendationItem] = []
    demo_mode: bool = False
    demo_classification: Optional[Literal["DEMO_AT_RISK", "DEMO_NOT_AT_RISK"]] = None


class DemoCustomerOption(BaseModel):
    customer_id: str
    label: str


class DemoCustomersResponse(BaseModel):
    available: bool
    demo_mode: bool
    customers: list[DemoCustomerOption]


class LiveEventRecord(BaseModel):
    timestamp: str
    customer_id: str
    session_id: str
    event_type: str
    product_id: Optional[str] = None
    category: Optional[str] = None
    order_value: Optional[float] = None
    churn_probability: float
    risk_level: str
    decision: str
    strategy: str
    discount: int = 0
    recommendations: list[dict[str, Any]] = []
    region: str


class AdminRecentEventsResponse(BaseModel):
    status: str
    total_events: int
    count: int
    events: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    model_type: str
    threshold: float
    customers_loaded: int


class CheckoutLineResponse(BaseModel):
    product_id: str
    quantity: int
    unit_price: float
    discount_percent: int
    discount_value: float
    line_total: float


class CheckoutQuoteResponse(BaseModel):
    customer_id: str
    lines: list[CheckoutLineResponse]
    subtotal: float
    discount_value: float
    shipping: float
    tax: float
    total: float


class CheckoutFinalizationResponse(CheckoutQuoteResponse):
    order_id: str
    retention_offer_status: Literal["REDEEMED", "NOT_REDEEMED"]
    demo_mode: bool
    experience: CustomerExperienceResponse


CATALOG_PRICES: dict[str, Decimal] = {
    "P-1001": Decimal("1449"), "P-1002": Decimal("2199"),
    "P-1003": Decimal("749"), "P-1004": Decimal("1099"),
    "P-1005": Decimal("549"), "P-1006": Decimal("1699"),
    "P-1007": Decimal("329"), "P-1008": Decimal("219"),
    "P-1009": Decimal("179"), "P-1010": Decimal("119"),
    "P-1011": Decimal("399"), "P-1012": Decimal("129"),
    "P-1013": Decimal("749"), "P-1014": Decimal("199"),
    "P-1015": Decimal("69"), "P-1016": Decimal("129"),
    "P-1017": Decimal("89"), "P-1018": Decimal("149"),
    "P-1019": Decimal("79"), "P-1020": Decimal("45"),
    "P-1021": Decimal("59"), "P-1022": Decimal("89"),
    "P-1023": Decimal("549"), "P-1024": Decimal("899"),
    "P-1025": Decimal("279"), "P-1026": Decimal("89"),
    "P-1027": Decimal("159"), "P-1028": Decimal("139"),
    "P-1029": Decimal("649"), "P-1030": Decimal("289"),
    "P-1031": Decimal("219"), "P-1032": Decimal("249"),
    "P-1033": Decimal("99"), "P-1034": Decimal("179"),
    "P-1035": Decimal("149"), "P-1036": Decimal("349"),
}


def round_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


from collections import deque
import threading


class LiveEventBuffer:
    """Thread-safe in-memory rolling buffer for recent live events received by FastAPI."""
    def __init__(self, maxlen: int = 1000):
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total_events = 0

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(record)
            self._total_events += 1

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)
            return items[::-1][:limit]

    def get_total_count(self) -> int:
        with self._lock:
            return self._total_events

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._total_events = 0


# ===========================================================================
# Application Factory
# ===========================================================================

def create_app(
    config: Optional[dict[str, Any]] = None,
    state_manager: Optional[CustomerStateManager] = None,
    predictor: Optional[ChurnPredictor] = None,
    decision_gate: Optional[DecisionGate] = None,
    recommender: Optional[BaselineRecommender] = None,
    override_engine: Optional[OverrideEngine] = None,
    live_event_buffer: Optional[LiveEventBuffer] = None,
    checkout_ledger: Optional[CheckoutLedger] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()
    
    setup_logging(config)
    ensure_project_dirs(config)

    app = FastAPI(
        title="Churn Engine API",
        description="FastAPI thin bridge connecting e-commerce storefront to churn risk engine",
        version="1.0.0",
    )

    # CORS configuration
    production = os.getenv("CHURN_ENVIRONMENT", "").casefold() == "production"
    cors_origins_env = os.getenv("CORS_ORIGINS")
    if production:
        if os.getenv("CHURN_LOCAL_DEMO_MODE", "").casefold() != "false":
            raise RuntimeError(
                "CHURN_LOCAL_DEMO_MODE=false must be explicitly configured in production"
            )
        if not cors_origins_env:
            raise RuntimeError("CORS_ORIGINS must be configured in production")
    else:
        cors_origins_env = cors_origins_env or (
            "http://localhost:8080,http://localhost:5173,http://localhost:3000,"
            "http://127.0.0.1:8080,http://127.0.0.1:5173"
        )
    allowed_origins = [
        origin.strip() for origin in cors_origins_env.split(",") if origin.strip()
    ]
    if production and (
        not allowed_origins
        or any(
            not origin.casefold().startswith("https://")
            or "localhost" in origin.casefold()
            or "127.0.0.1" in origin
            for origin in allowed_origins
        )
    ):
        raise RuntimeError(
            "Production CORS_ORIGINS must contain public HTTPS frontend origins only"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Service instances
    app.state.config = config
    app.state.state_manager = state_manager or CustomerStateManager(config)
    app.state.predictor = predictor or ChurnPredictor(config)
    app.state.decision_gate = decision_gate or DecisionGate(config)
    app.state.recommender = recommender or BaselineRecommender(config)
    app.state.override_engine = override_engine or OverrideEngine(config)
    app.state.live_event_buffer = live_event_buffer or LiveEventBuffer(maxlen=1000)
    app.state.local_demo_enabled = local_demo_enabled(config)
    if checkout_ledger is None:
        configured_ledger_path = os.getenv("CHECKOUT_LEDGER_PATH")
        if production:
            if not configured_ledger_path:
                raise RuntimeError(
                    "CHECKOUT_LEDGER_PATH must be explicitly configured in production"
                )
            ledger_path = Path(configured_ledger_path).expanduser()
            if not ledger_path.is_absolute() or not ledger_path.parent.is_dir():
                raise RuntimeError(
                    "CHECKOUT_LEDGER_PATH must be an absolute path with an existing parent directory"
                )
        else:
            ledger_path = (
                Path(configured_ledger_path).expanduser()
                if configured_ledger_path
                else PROJECT_ROOT / "data" / "checkout_ledger.sqlite3"
            )
        checkout_ledger = CheckoutLedger(ledger_path)
    app.state.checkout_ledger = checkout_ledger

    def execute_churn_pipeline(
        customer_id: str,
        reference_date: Optional[datetime] = None,
        redeemed_product_ids: Optional[set[str]] = None,
    ) -> CustomerExperienceResponse:
        """Run the exact sequence: Features -> Predictor -> Gate -> Recommender -> Override."""
        sm: CustomerStateManager = app.state.state_manager
        pred: ChurnPredictor = app.state.predictor
        gate: DecisionGate = app.state.decision_gate
        recom: BaselineRecommender = app.state.recommender
        override: OverrideEngine = app.state.override_engine

        # 1. Fetch features
        features = sm.get_features(customer_id, reference_date=reference_date)

        # 2. Churn Predictor
        pred_res = pred.predict(customer_id, features)
        prob = pred_res["churn_probability"]
        risk_lvl = pred_res["risk_level"]

        # 3. Decision Gate (lookup previous probability for transition detection)
        prev_prob = None
        for item in reversed(gate.get_history()):
            if item["customer_id"] == customer_id:
                prev_prob = item["churn_probability"]
                break

        decision_res = gate.decide(
            customer_id=customer_id,
            churn_probability=prob,
            previous_probability=prev_prob,
            risk_level=risk_lvl
        )

        # 4. Baseline Recommender
        cust_history = sm.customer_history.get(customer_id, [])
        base_recs = recom.recommend(customer_id, cust_history)

        # 5. Override Engine
        demo_active = app.state.local_demo_enabled
        demo_class = demo_classification(customer_id) if demo_active else None
        offers_allowed = not demo_active or customer_id in DEMO_AT_RISK_CUSTOMERS
        if redeemed_product_ids is None:
            redeemed_product_ids = app.state.checkout_ledger.get_redeemed_product_ids(
                customer_id
            )
        final_recs = override.process(
            baseline_recs=base_recs,
            decision_res=decision_res,
            customer_events=cust_history,
            offers_allowed=offers_allowed,
            redeemed_product_ids=redeemed_product_ids,
        )

        # 6. Build response
        is_override = (
            final_recs.get("strategy") == "RETENTION_OVERRIDE"
            and decision_res.get("decision") == "INTERVENTION_REQUIRED"
            and offers_allowed
        )
        tier_discount_pct = int(final_recs.get("metadata", {}).get("discount_pct", 0))

        # Look up country for customer
        country = cust_history[0].get("country", "") if cust_history else ""
        region = resolve_region(customer_id, country)

        if is_override:
            headline = "Special offer for you"
            rec_type = "retention_offer"
        else:
            headline = "Recommended for you"
            rec_type = "personalized"

        adapted_items = adapt_recommendations(
            final_recs.get("recommendations", []),
            offers_authorized=is_override,
        )
        has_product_offer = any(item["discount"] > 0 for item in adapted_items)
        message = (
            f"Save {tier_discount_pct}% on selected products."
            if is_override and has_product_offer
            else (
                "We have recommendations based on your activity."
                if is_override
                else None
            )
        )

        return CustomerExperienceResponse(
            customer_id=customer_id,
            region=region,
            churn_probability=prob,
            risk_level=risk_lvl,
            decision=decision_res.get("decision", "NORMAL"),
            strategy="RETENTION_OVERRIDE" if is_override else "BASELINE",
            discount=0,
            headline=headline,
            message=message,
            recommendation_type=rec_type,
            recommendations=[RecommendationItem(**item) for item in adapted_items],
            demo_mode=demo_active,
            demo_classification=demo_class,
        )

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/api/demo/customers", response_model=DemoCustomersResponse)
    def get_demo_customers():
        customers = [
            DemoCustomerOption(
                customer_id=customer_id,
                label=f"Demo customer {customer_id}",
            )
            for customer_id in sorted(
                DEMO_AT_RISK_CUSTOMERS | DEMO_NOT_AT_RISK_CUSTOMERS
            )
        ]
        return DemoCustomersResponse(
            available=bool(customers),
            demo_mode=app.state.local_demo_enabled,
            customers=customers,
        )

    @app.get("/health", response_model=HealthResponse)
    def health_check():
        pred: ChurnPredictor = app.state.predictor
        gate: DecisionGate = app.state.decision_gate
        sm: CustomerStateManager = app.state.state_manager

        model_loaded = pred.model is not None
        model_name = "None"
        if model_loaded:
            inner_model = getattr(pred.model, "model", pred.model)
            model_name = inner_model.__class__.__name__

        return HealthResponse(
            status="ok",
            service="churn-engine",
            model_loaded=model_loaded,
            model_type=model_name,
            threshold=gate.threshold,
            customers_loaded=len(sm.customer_history),
        )

    @app.post("/api/events", response_model=CustomerExperienceResponse)
    def post_event(req: EventRequest):
        sm: CustomerStateManager = app.state.state_manager

        # Parse timestamp safely
        if req.timestamp:
            try:
                event_time = pd.to_datetime(req.timestamp).replace(tzinfo=None)
            except Exception:
                event_time = datetime.utcnow()
        else:
            event_time = datetime.utcnow()

        # Event normalization
        normalized_type = EVENT_TYPE_MAP.get(req.event_type, req.event_type)

        # Contextual customer profile attributes
        history = sm.customer_history.get(req.customer_id, [])
        country = history[0].get("country", "US") if history else "US"
        age_days = history[0].get("customer_age_days", 180) if history else 180
        device = history[0].get("device_type", "desktop") if history else "desktop"

        # Build dataclass
        cust_event = CustomerEvent(
            event_id=f"evt-web-{uuid.uuid4().hex[:8]}",
            customer_id=req.customer_id,
            event_timestamp=event_time,
            event_type=normalized_type,
            session_id=req.session_id,
            product_id=req.product_id,
            order_value=req.order_value,
            coupon_used=bool(req.metadata.get("coupon_used", False) if req.metadata else False),
            page_url=f"/shop/{req.event_type}",
            device_type=device,
            country=country,
            customer_age_days=age_days,
            churn=0,
        )

        # 1. Update State
        sm.update_state(cust_event)

        # 2. Run Pipeline
        pipeline_res = execute_churn_pipeline(req.customer_id, reference_date=event_time)

        # 3. Record to live event buffer for dashboard observation
        try:
            recs_list = []
            for r in pipeline_res.recommendations:
                if hasattr(r, "model_dump"):
                    recs_list.append(r.model_dump())
                elif hasattr(r, "dict"):
                    recs_list.append(r.dict())
                elif isinstance(r, dict):
                    recs_list.append(r)
                else:
                    recs_list.append({
                        "product_id": getattr(r, "product_id", ""),
                        "reason": getattr(r, "reason", None),
                        "discount": getattr(r, "discount", 0),
                    })

            live_record = {
                "timestamp": event_time.isoformat() if hasattr(event_time, "isoformat") else str(event_time),
                "customer_id": req.customer_id,
                "session_id": req.session_id,
                "event_type": req.event_type,
                "product_id": req.product_id,
                "category": req.category,
                "order_value": req.order_value,
                "churn_probability": pipeline_res.churn_probability,
                "risk_level": pipeline_res.risk_level,
                "decision": pipeline_res.decision,
                "strategy": pipeline_res.strategy,
                "discount": pipeline_res.discount,
                "recommendations": recs_list,
                "region": pipeline_res.region,
            }
            app.state.live_event_buffer.append(live_record)
        except Exception as exc:
            logger.warning("Failed to append event to live_event_buffer: %s", exc)

        return pipeline_res

    @app.get("/api/experience/{customer_id}", response_model=CustomerExperienceResponse)
    def get_experience(customer_id: str):
        if not customer_id or not customer_id.strip():
            raise HTTPException(status_code=400, detail="customer_id must not be empty")
        
        # Query experience without mutating history
        return execute_churn_pipeline(customer_id.strip())

    def build_checkout_quote(
        customer_id: str,
        lines: list[CheckoutLineRequest],
        redeemed_product_ids: Optional[set[str]] = None,
    ) -> CheckoutQuoteResponse:
        for line in lines:
            if line.product_id not in CATALOG_PRICES:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown product_id: {line.product_id}",
                )

        experience = execute_churn_pipeline(
            customer_id, redeemed_product_ids=redeemed_product_ids
        )
        authorized_offers: dict[str, int] = {}
        if (
            experience.decision == "INTERVENTION_REQUIRED"
            and experience.strategy == "RETENTION_OVERRIDE"
        ):
            authorized_offers = {
                recommendation.product_id: recommendation.discount
                for recommendation in experience.recommendations
                if recommendation.product_id in CATALOG_PRICES
                and recommendation.discount is not None
                and 0 < recommendation.discount <= 100
            }

        quoted_lines: list[CheckoutLineResponse] = []
        subtotal = Decimal("0")
        discount_total = Decimal("0")
        for line in lines:
            unit_price = CATALOG_PRICES[line.product_id]
            discount_percent = authorized_offers.get(line.product_id, 0)
            original_line_total = unit_price * line.quantity
            line_discount = round_money(
                original_line_total * Decimal(discount_percent) / Decimal("100")
            )
            final_line_total = original_line_total - line_discount
            subtotal += original_line_total
            discount_total += line_discount
            quoted_lines.append(
                CheckoutLineResponse(
                    product_id=line.product_id,
                    quantity=line.quantity,
                    unit_price=float(unit_price),
                    discount_percent=discount_percent,
                    discount_value=float(line_discount),
                    line_total=float(final_line_total),
                )
            )

        discounted_subtotal = subtotal - discount_total
        shipping = (
            Decimal("0")
            if discounted_subtotal == 0 or discounted_subtotal >= Decimal("75")
            else Decimal("9.95")
        )
        tax = round_money(discounted_subtotal * Decimal("0.08"))
        total = round_money(discounted_subtotal + shipping + tax)

        return CheckoutQuoteResponse(
            customer_id=customer_id,
            lines=quoted_lines,
            subtotal=float(subtotal),
            discount_value=float(discount_total),
            shipping=float(shipping),
            tax=float(tax),
            total=float(total),
        )

    @app.post("/api/checkout/quote", response_model=CheckoutQuoteResponse)
    def checkout_quote(req: CheckoutRequest):
        customer_id = req.customer_id.strip()
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id must not be empty")
        return build_checkout_quote(customer_id, req.lines)

    @app.post(
        "/api/checkout/finalize",
        response_model=CheckoutFinalizationResponse,
    )
    def finalize_checkout(req: CheckoutFinalizeRequest):
        customer_id = req.customer_id.strip()
        order_id = req.order_id.strip()
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id must not be empty")
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id must not be empty")

        signature = json.dumps(
            {
                "customer_id": customer_id,
                "lines": sorted(
                    (line.product_id, line.quantity) for line in req.lines
                ),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        ledger: CheckoutLedger = app.state.checkout_ledger
        with ledger.transaction() as connection:
            previous = ledger.get_finalization(connection, order_id)
            if previous:
                if (
                    previous["customer_id"] != customer_id
                    or previous["request_signature"] != signature
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="order_id was already used for a different checkout",
                    )
                return CheckoutFinalizationResponse.model_validate_json(
                    previous["response_json"]
                )

            redeemed_product_ids = ledger.get_redeemed_product_ids(
                customer_id, connection
            )
            quote = build_checkout_quote(
                customer_id,
                req.lines,
                redeemed_product_ids=redeemed_product_ids,
            )
            newly_redeemed_product_ids = {
                line.product_id for line in quote.lines if line.discount_percent > 0
            }

            refreshed_experience = execute_churn_pipeline(
                customer_id,
                redeemed_product_ids=(
                    redeemed_product_ids | newly_redeemed_product_ids
                ),
            )
            finalized = CheckoutFinalizationResponse(
                **quote.model_dump(),
                order_id=order_id,
                retention_offer_status=(
                    "REDEEMED" if quote.discount_value > 0 else "NOT_REDEEMED"
                ),
                demo_mode=app.state.local_demo_enabled,
                experience=refreshed_experience,
            )
            ledger.save_finalization(
                connection,
                order_id=order_id,
                customer_id=customer_id,
                request_signature=signature,
                response_json=finalized.model_dump_json(),
                redeemed_product_ids=newly_redeemed_product_ids,
            )
            return finalized

    @app.get("/api/admin/events/recent", response_model=AdminRecentEventsResponse)
    def get_admin_recent_events(limit: int = 50):
        clamped_limit = max(1, min(1000, limit))
        buf: LiveEventBuffer = app.state.live_event_buffer
        events = buf.get_recent(limit=clamped_limit)
        return {
            "status": "ok",
            "total_events": buf.get_total_count(),
            "count": len(events),
            "events": events,
        }

    return app


# Default ASGI application instance
app = create_app()
