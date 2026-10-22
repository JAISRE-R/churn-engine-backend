"""
churn-engine/src/preprocessing.py
===================================
Data Cleaning and Preprocessing Pipeline.

Responsibilities
----------------
- Load raw customer / event data from CSV.
- Handle missing values using configurable strategies.
- Remove duplicate records.
- Normalise and standardise column names.
- Cast columns to clean dtypes (timestamps, categories, integers).
- Save clean event data to `data/processed/cleaned_events.parquet`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_project_root

logger = logging.getLogger(__name__)


# ===========================================================================
# DataLoader
# ===========================================================================

class DataLoader:
    """Loads raw customer behavioural data from CSV files.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.data_cfg: dict[str, Any] = config.get("data", {})
        self.root: Path = get_project_root()
        self.raw_path: Path = self.root / config["paths"]["data_raw"]
        logger.debug("DataLoader initialised | raw_path=%s", self.raw_path)

    def load_csv(self, filename: str | None = None) -> pd.DataFrame:
        """Load raw data from a CSV file.

        Parameters
        ----------
        filename : str | None
            CSV filename within the raw data directory.
            If None, reads ``data.raw_filename`` from config.

        Returns
        -------
        pd.DataFrame
            Raw, unprocessed DataFrame.
        """
        filename = filename or self.data_cfg.get("raw_filename", "raw_events.csv")
        filepath = self.raw_path / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Raw dataset file not found: {filepath}\n"
                "Run `python src/generate_data.py` first to generate synthetic data."
            )

        logger.info("Loading raw dataset from %s ...", filepath)
        df = pd.read_csv(filepath, low_memory=False)
        logger.info("Loaded raw dataset: %d rows × %d columns", *df.shape)
        return df


# ===========================================================================
# DataCleaner
# ===========================================================================

class DataCleaner:
    """Cleans a raw DataFrame by handling missing values, duplicates, and dtypes.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        logger.debug("DataCleaner initialised.")

    def drop_duplicates(self, df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
        """Remove exact duplicate rows from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        subset : list[str] | None

        Returns
        -------
        pd.DataFrame
        """
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=subset).copy()
        dropped = initial_len - len(df_clean)
        logger.info("Deduplication complete: %d duplicate rows dropped", dropped)
        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using domain rules.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df_clean = df.copy()

        # Fill order_value nulls (non-purchases) with 0.0
        if "order_value" in df_clean.columns:
            df_clean["order_value"] = df_clean["order_value"].fillna(0.0)

        # Fill product_id nulls with 'NONE'
        if "product_id" in df_clean.columns:
            df_clean["product_id"] = df_clean["product_id"].fillna("NONE")

        # Fill coupon_used nulls with False
        if "coupon_used" in df_clean.columns:
            df_clean["coupon_used"] = df_clean["coupon_used"].fillna(False).astype(bool)

        logger.info("Handled missing values across %d columns", len(df_clean.columns))
        return df_clean

    def normalise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names to lowercase snake_case.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df_clean = df.copy()
        df_clean.columns = (
            df_clean.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return df_clean

    def cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast DataFrame columns to clean pandas dtypes.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df_clean = df.copy()

        if "event_timestamp" in df_clean.columns:
            df_clean["event_timestamp"] = pd.to_datetime(df_clean["event_timestamp"], errors="coerce")

        if "churn" in df_clean.columns:
            df_clean["churn"] = df_clean["churn"].astype(int)

        categorical_cols = ["event_type", "device_type", "country"]
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype("category")

        return df_clean


# ===========================================================================
# Preprocessing Pipeline
# ===========================================================================

class PreprocessingPipeline:
    """Orchestrates the full data loading and cleaning workflow.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root: Path = get_project_root()
        self.loader = DataLoader(config)
        self.cleaner = DataCleaner(config)
        self.processed_dir: Path = self.root / config["paths"]["data_processed"]
        ensure_dir(self.processed_dir)
        logger.info("PreprocessingPipeline ready.")

    def run(self, filename: str | None = None) -> pd.DataFrame:
        """Execute the complete preprocessing pipeline.

        Parameters
        ----------
        filename : str | None
            Source CSV filename. Falls back to config if None.

        Returns
        -------
        pd.DataFrame
            Fully cleaned DataFrame saved to `data/processed/cleaned_events.parquet`.
        """
        logger.info("Executing PreprocessingPipeline ...")
        df_raw = self.loader.load_csv(filename)
        df_norm = self.cleaner.normalise_column_names(df_raw)
        df_dedup = self.cleaner.drop_duplicates(df_norm, subset=["event_id"])
        df_imputed = self.cleaner.handle_missing_values(df_dedup)
        df_clean = self.cleaner.cast_dtypes(df_imputed)

        out_path = self.processed_dir / "cleaned_events.parquet"
        df_clean.to_parquet(out_path, index=False)
        logger.info("Saved clean event dataset → %s | Shape: %d rows × %d cols", out_path, *df_clean.shape)

        return df_clean
