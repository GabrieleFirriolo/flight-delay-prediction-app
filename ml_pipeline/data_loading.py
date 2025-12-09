"""
Data loading and preprocessing module.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml_pipeline.config import (
    DATA_DIR,
    FLIGHTS_CSV,
    AIRPORTS_CSV,
    AIRLINES_CSV,
    TRAIN_MONTHS,
    TEST_MONTHS,
    LEAKAGE_COLS,
)


def load_raw_data(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load raw CSV files from the data directory.

    Returns:
        Dictionary with keys 'flights', 'airports', 'airlines' mapping to DataFrames.
    """
    data_dir = Path(data_dir)

    flights_path = data_dir / FLIGHTS_CSV
    airports_path = data_dir / AIRPORTS_CSV
    airlines_path = data_dir / AIRLINES_CSV

    for path in [flights_path, airports_path, airlines_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    flights = pd.read_csv(flights_path, low_memory=False)
    airports = pd.read_csv(airports_path)
    airlines = pd.read_csv(airlines_path)

    return {
        "flights": flights,
        "airports": airports,
        "airlines": airlines,
    }


def train_test_split_by_month(
    flights: pd.DataFrame,
    train_months: list = TRAIN_MONTHS,
    test_months: list = TEST_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = flights[flights["MONTH"].isin(train_months)].copy()
    df_test = flights[flights["MONTH"].isin(test_months)].copy()

    return df_train, df_test


def clean_flights(
    flights_raw: pd.DataFrame,
    leakage_cols: List[str] = LEAKAGE_COLS,
) -> pd.DataFrame:
    flights = flights_raw.copy()
    flights = flights[(flights["CANCELLED"] == 0) & (flights["DIVERTED"] == 0)]
    flights = flights[
        flights["ARRIVAL_DELAY"].notna()
        & flights["SCHEDULED_DEPARTURE"].notna()
        & flights["TAIL_NUMBER"].notna()
    ]
    cols_to_drop = [c for c in leakage_cols if c in flights.columns]
    flights = flights.drop(columns=cols_to_drop)
    return flights


def add_time_features(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()

    df["DEP_HOUR"] = (df["SCHEDULED_DEPARTURE"] // 100) % 24
    df["ARR_HOUR"] = (df["SCHEDULED_ARRIVAL"] // 100) % 24

    # Vectorized time-of-day bucketing: night 0-5, morning 6-11,
    # afternoon 12-17, evening 18-23 (hours are already in [0, 23]).
    slot_bins = [-1, 5, 11, 17, 23]
    slot_labels = ["night", "morning", "afternoon", "evening"]
    df["DEP_TIME_SLOT"] = pd.cut(df["DEP_HOUR"], bins=slot_bins, labels=slot_labels).astype("object")
    df["ARR_TIME_SLOT"] = pd.cut(df["ARR_HOUR"], bins=slot_bins, labels=slot_labels).astype("object")

    peak_hours = [7, 8, 9, 17, 18, 19, 20]
    df["DEP_IS_PEAK_HOUR"] = df["DEP_HOUR"].isin(peak_hours).astype(int)
    df["ARR_IS_PEAK_HOUR"] = df["ARR_HOUR"].isin(peak_hours).astype(int)

    red_eye_hours = [0, 1, 2, 3, 4, 5, 21, 22, 23]
    df["DEP_IS_RED_EYE"] = df["DEP_HOUR"].isin(red_eye_hours).astype(int)
    df["ARR_IS_RED_EYE"] = df["ARR_HOUR"].isin(red_eye_hours).astype(int)

    return df


def add_calendar_flags(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    df["IS_SUMMER"] = df["MONTH"].isin([6, 7, 8]).astype(int)
    df["IS_HOLIDAY_SEASON"] = df["MONTH"].isin([7, 8, 12]).astype(int)
    return df


def add_cyclic_features(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()

    cyclic_configs = [
        ("MONTH", 12, "MONTH"),
        ("DAY_OF_WEEK", 7, "DAY"),
        ("DEP_HOUR", 24, "DEP_HOUR"),
        ("ARR_HOUR", 24, "ARR_HOUR"),
    ]

    for col, period, prefix in cyclic_configs:
        df[f"{prefix}_SIN"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{prefix}_COS"] = np.cos(2 * np.pi * df[col] / period)

    return df


def preprocess_flights(flights_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_flights(flights_raw)
    df = add_time_features(df)
    df = add_calendar_flags(df)
    df = add_cyclic_features(df)
    return df
