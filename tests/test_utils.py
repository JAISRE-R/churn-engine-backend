"""
churn-engine/tests/test_utils.py
==================================
Unit tests for src/utils.py.

Phase 1: Tests verifying that the configuration loads correctly
and that directory creation helpers work as expected.

Phase 2+: Additional tests will be added as modules are implemented.
"""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for src.utils.load_config()."""

    def test_config_loads_without_error(self) -> None:
        """The default config.yaml should load without raising an exception."""
        from src.utils import load_config
        cfg = load_config()
        assert isinstance(cfg, dict), "Config should be a dict"

    def test_config_has_required_top_level_keys(self) -> None:
        """Config must contain the essential top-level keys."""
        from src.utils import load_config
        cfg = load_config()
        required_keys = ["project", "paths", "data", "model", "decision_gate", "logging"]
        for key in required_keys:
            assert key in cfg, f"Missing top-level config key: '{key}'"

    def test_churn_threshold_in_valid_range(self) -> None:
        """Churn threshold must be a probability in (0, 1)."""
        from src.utils import load_config
        cfg = load_config()
        threshold = cfg["decision_gate"]["churn_threshold"]
        assert 0.0 < threshold < 1.0, f"Invalid churn threshold: {threshold}"

    def test_config_raises_on_missing_file(self) -> None:
        """load_config() should raise FileNotFoundError for a nonexistent path."""
        from src.utils import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Filesystem Helper Tests
# ---------------------------------------------------------------------------

class TestEnsureDir:
    """Tests for src.utils.ensure_dir()."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """ensure_dir() should create a directory that does not yet exist."""
        from src.utils import ensure_dir
        target = tmp_path / "new_subdir" / "nested"
        result = ensure_dir(target)
        assert result.exists(), "Directory should have been created"
        assert result.is_dir(), "Path should be a directory"

    def test_idempotent_on_existing_dir(self, tmp_path: Path) -> None:
        """ensure_dir() should not raise when the directory already exists."""
        from src.utils import ensure_dir
        ensure_dir(tmp_path)  # Already exists
        ensure_dir(tmp_path)  # Should be a no-op


# ---------------------------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------------------------

class TestBanner:
    """Tests for src.utils.banner()."""

    def test_banner_contains_text(self) -> None:
        """Banner output should contain the input text."""
        from src.utils import banner
        result = banner("Phase 1 Complete")
        assert "Phase 1 Complete" in result

    def test_banner_is_string(self) -> None:
        """Banner should return a string."""
        from src.utils import banner
        assert isinstance(banner("test"), str)


# ---------------------------------------------------------------------------
# TODO Markers for Future Phases
# ---------------------------------------------------------------------------

# TODO Phase 2: Add tests for DataLoader.load_csv()
# TODO Phase 2: Add tests for DataCleaner.drop_duplicates()
# TODO Phase 2: Add tests for DataCleaner.handle_missing_values()
# TODO Phase 3: Add tests for ModelTrainer.train() (with a tiny synthetic dataset)
# TODO Phase 3: Add tests for compute_classification_metrics()
# TODO Phase 4: Add tests for EventSimulator.generate_event()
# TODO Phase 4: Add tests for EventBuffer.push() and drain()
# TODO Phase 5: Add tests for DecisionGate.evaluate()
# TODO Phase 5: Add tests for OverrideEngine.override()
