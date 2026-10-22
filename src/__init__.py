"""
churn-engine/src/__init__.py
============================
Package initialiser for the `src` module.

Exposes version metadata and provides a convenience import of
the project-level configuration so downstream modules can do:

    from src import config

instead of reconstructing the path each time.
"""

# TODO Phase 2: expose key pipeline functions after they are implemented.

__version__: str = "1.0.0-phase1"
__author__: str = "Churn Engine Team"
__description__: str = (
    "Live Customer Churn Risk Assessment Engine — "
    "Real-Time Recommendation Override System"
)

# Lazy import guard — config may not be available in bare test environments.
try:
    from src.utils import load_config as _load_config  # noqa: F401
except ImportError:
    pass  # Silently ignore during initial scaffolding
