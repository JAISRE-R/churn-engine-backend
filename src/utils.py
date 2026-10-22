"""
churn-engine/src/utils.py
==========================
Shared utility functions used across all pipeline modules.

Responsibilities
----------------
- Load and validate the YAML configuration file.
- Bootstrap the logging system from config settings.
- Provide filesystem helpers (directory creation, path resolution).
- Provide lightweight timer / profiling decorators.
- Expose a thin wrapper around joblib for model serialisation.

Phase 1 Status
--------------
All classes and functions are scaffolded with docstrings and TODO
markers. No business logic is implemented yet.
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Module-level logger (bootstrapped before full logging setup)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file's location
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


# ===========================================================================
# Configuration
# ===========================================================================

def load_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the project YAML configuration as a dictionary.

    Parameters
    ----------
    config_path : Path | str
        Absolute or relative path to the YAML config file.
        Defaults to ``configs/config.yaml`` relative to the project root.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the given path.
    yaml.YAMLError
        If the YAML content is malformed.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg["project"]["name"]
    'Live Churn Risk Assessment Engine'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Ensure 'configs/config.yaml' exists in the project root."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    logger.debug("Configuration loaded from: %s", config_path)
    return config


# ===========================================================================
# Logging Setup
# ===========================================================================

def setup_logging(config: dict[str, Any] | None = None) -> logging.Logger:
    """Initialise the project-wide logging configuration.

    Sets up a rotating file handler (if enabled) and a stream handler
    using settings from the ``logging`` section of the config dictionary.

    Parameters
    ----------
    config : dict[str, Any] | None
        Full project configuration dict (output of :func:`load_config`).
        If ``None``, a minimal default configuration is applied.

    Returns
    -------
    logging.Logger
        The root logger configured according to project settings.

    Notes
    -----
    Call this function **once** at application startup (e.g., in ``app.py``)
    before any module-level code produces log output.

    TODO Phase 2
    ------------
    - Add structured JSON logging handler for production deployments.
    - Integrate with external log aggregation (e.g., ELK, CloudWatch).
    """
    log_cfg = (config or {}).get("logging", {})

    level_name: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_name, logging.INFO)

    log_format: str = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    date_fmt: str = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    formatter = logging.Formatter(fmt=log_format, datefmt=date_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove pre-existing handlers to avoid duplicate messages
    root_logger.handlers.clear()

    # --- Stream Handler (always present) ---
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # --- Rotating File Handler (optional) ---
    if log_cfg.get("to_file", True):
        log_dir: Path = PROJECT_ROOT / log_cfg.get("logs", "logs")
        ensure_dir(log_dir)
        log_filename: str = log_cfg.get("log_filename", "churn_engine.log")
        log_filepath: Path = log_dir / log_filename

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_filepath,
            maxBytes=log_cfg.get("max_bytes", 10_485_760),
            backupCount=log_cfg.get("backup_count", 5),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.debug("File logging enabled at: %s", log_filepath)

    logger.info(
        "Logging initialised | level=%s | to_file=%s",
        level_name,
        log_cfg.get("to_file", True),
    )
    return root_logger


# ===========================================================================
# Filesystem Helpers
# ===========================================================================

def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and all parents) if it does not already exist.

    Parameters
    ----------
    path : Path | str
        Directory path to create.

    Returns
    -------
    Path
        The resolved, guaranteed-to-exist directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs(config: dict[str, Any]) -> None:
    """Create all required project directories defined in config ``paths``.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dict.

    Notes
    -----
    This function is called at startup to guarantee that all path-dependent
    modules have a valid directory structure to write into.
    """
    paths_cfg: dict[str, str] = config.get("paths", {})
    for key, rel_path in paths_cfg.items():
        if rel_path:
            full_path = PROJECT_ROOT / rel_path
            ensure_dir(full_path)
            logger.debug("Directory ready: %s (%s)", key, full_path)


def resolve_path(relative: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve a relative path key from config into an absolute Path.

    Parameters
    ----------
    relative : str
        Either a raw relative path string or a key from ``config.paths``.
    config : dict[str, Any] | None
        Full project config. If provided, ``relative`` is looked up as a
        key first, then treated as a literal path.

    Returns
    -------
    Path
        Absolute path resolved against the project root.

    TODO Phase 2
    ------------
    - Support environment-variable overrides for cloud deployments.
    """
    if config:
        paths_cfg = config.get("paths", {})
        relative = paths_cfg.get(relative, relative)
    return PROJECT_ROOT / relative


# ===========================================================================
# Timing / Profiling
# ===========================================================================

def timer(func):
    """Decorator that logs the wall-clock execution time of a function.

    Usage
    -----
    >>> @timer
    ... def heavy_computation():
    ...     time.sleep(1)

    TODO Phase 3
    ------------
    - Integrate with a metrics registry (e.g., Prometheus) for production.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(
            "⏱  %s.%s completed in %.4f seconds",
            func.__module__,
            func.__qualname__,
            elapsed,
        )
        return result
    return wrapper


# ===========================================================================
# Model Serialisation Helpers
# ===========================================================================

def save_model(model: Any, filepath: Path | str) -> None:
    """Serialise a fitted model to disk using joblib.

    Parameters
    ----------
    model : Any
        A fitted scikit-learn-compatible estimator.
    filepath : Path | str
        Destination file path (typically ``models/churn_model.joblib``).

    Raises
    ------
    ImportError
        If joblib is not installed in the current environment.

    TODO Phase 3
    ------------
    - Add versioning metadata (timestamp, training data hash, metrics).
    - Implement model registry integration.
    """
    import joblib  # Lazy import so Phase 1 works without fitting models.

    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    joblib.dump(model, filepath)
    logger.info("Model saved → %s", filepath)


def load_model(filepath: Path | str) -> Any:
    """Load a serialised model from disk using joblib.

    Parameters
    ----------
    filepath : Path | str
        Path to the serialised model file.

    Returns
    -------
    Any
        The deserialised estimator object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.

    TODO Phase 4
    ------------
    - Validate model schema / version before loading.
    - Support remote model stores (S3, GCS, Azure Blob).
    """
    import joblib

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info("Model loaded ← %s", filepath)
    return model


# ===========================================================================
# Miscellaneous Helpers
# ===========================================================================

def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns
    -------
    Path
        Absolute project root path.
    """
    return PROJECT_ROOT


def banner(text: str, width: int = 70) -> str:
    """Return a formatted banner string for CLI / log output.

    Parameters
    ----------
    text : str
        The message to display inside the banner.
    width : int
        Total width of the banner.

    Returns
    -------
    str
        Multi-line banner string.

    Examples
    --------
    >>> print(banner("Phase 1 Complete"))
    ══════════════════════════════════════════════════════════════════════
      Phase 1 Complete
    ══════════════════════════════════════════════════════════════════════
    """
    line = "═" * width
    return f"\n{line}\n  {text}\n{line}\n"
