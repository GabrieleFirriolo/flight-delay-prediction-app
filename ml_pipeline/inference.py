"""
Inference module for online feature building.

Transforms raw flight data into model-ready features using
pre-computed statistics and encoding maps from training.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from ml_pipeline.data_loading import (
    add_time_features,
    add_calendar_flags,
    add_cyclic_features,
)
from ml_pipeline.features import (
    add_route_features,
    add_hub_features,
    add_historical_delay_features,
)


def _parse_series_string(series_str: str) -> Dict[str, float]:
    """
    Parse a pandas Series string representation into a dictionary.
    
    Handles format like:
    "AIRLINE\nAA     3.705432\nAS    -0.798636\n..."
    """
    if not isinstance(series_str, str):
        return series_str if isinstance(series_str, dict) else {}
    
    result = {}
    lines = series_str.strip().split("\n")
    
    for line in lines[1:]:  # Skip header line
        line = line.strip()
        if not line or "..." in line or line.startswith("Name:") or line.startswith("dtype:"):
            continue
        
        # Split on whitespace, last part is the value
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                value = float(parts[-1])
                result[key] = value
            except ValueError:
                continue
    
    return result


def _apply_one_hot_encoding(
    df: pd.DataFrame,
    column: str,
    categories: List[str],
) -> pd.DataFrame:
    """
    Apply one-hot encoding for a single column with known categories.
    
    Args:
        df: Input DataFrame.
        column: Column to encode.
        categories: List of category values (excluding baseline).
    
    Returns:
        DataFrame with one-hot columns added and original column removed.
    """
    result = df.copy()
    
    if column not in result.columns:
        return result
    
    value = result[column].iloc[0]
    
    for cat in categories:
        col_name = f"{column}_{cat}"
        result[col_name] = 1 if value == cat else 0
    
    result = result.drop(columns=[column])
    return result


def _apply_target_encoding(
    df: pd.DataFrame,
    column: str,
    encoding_map: Union[Dict[str, float], str],
    global_mean: float,
) -> pd.DataFrame:
    """
    Apply target encoding for a single column using pre-computed map.
    
    Args:
        df: Input DataFrame.
        column: Column to encode.
        encoding_map: Dictionary or Series string mapping category -> encoded value.
        global_mean: Fallback value for unseen categories.
    
    Returns:
        DataFrame with target-encoded column (original removed).
    """
    result = df.copy()
    
    if column not in result.columns:
        return result
    
    # Parse encoding map if it's a string
    if isinstance(encoding_map, str):
        encoding_map = _parse_series_string(encoding_map)
    
    value = result[column].iloc[0]
    encoded_value = encoding_map.get(value, global_mean)
    
    new_col = f"{column}_TE"
    result[new_col] = encoded_value
    result = result.drop(columns=[column])
    
    return result


def _ensure_columns(
    df: pd.DataFrame,
    required_columns: List[str],
) -> pd.DataFrame:
    """
    Ensure DataFrame has all required columns in correct order.
    
    Missing columns are filled with 0.
    """
    result = df.copy()
    
    for col in required_columns:
        if col not in result.columns:
            result[col] = 0
    
    return result[required_columns]


def build_features_from_raw(
    raw_flight: pd.DataFrame,
    feature_stats: Dict,
    encoding_info_cls: Dict,
    encoding_info_reg: Dict,
    feature_columns_cls: List[str],
    feature_columns_reg: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform raw flight data into model-ready feature matrices.
    
    Args:
        raw_flight: Single-row DataFrame with columns:
            MONTH, DAY_OF_WEEK, SCHEDULED_DEPARTURE, SCHEDULED_ARRIVAL,
            ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIRLINE, DISTANCE
        feature_stats: From metadata["feature_stats"], contains:
            - hub_airports: list of hub airport codes
            - delay_stats: dict with route/airline/hour delay statistics
        encoding_info_cls: From metadata["encoding_info"]["classification"]
        encoding_info_reg: From metadata["encoding_info"]["regression"]
        feature_columns_cls: List of feature column names for classifier
        feature_columns_reg: List of feature column names for regressor
    
    Returns:
        Tuple of (X_cls, X_reg) DataFrames ready for model.predict()
    """
    df = raw_flight.copy()
    
    # Step 1: Time features (DEP_HOUR, ARR_HOUR, time slots, peak/red-eye flags)
    df = add_time_features(df)
    
    # Step 2: Calendar flags (IS_WEEKEND, IS_SUMMER, IS_HOLIDAY_SEASON)
    df = add_calendar_flags(df)
    
    # Step 3: Cyclic encoding (MONTH_SIN/COS, DAY_SIN/COS, DEP_HOUR_SIN/COS, ARR_HOUR_SIN/COS)
    df = add_cyclic_features(df)
    
    # Step 4: Route features (ROUTE, DISTANCE_CAT)
    df = add_route_features(df)
    
    # Step 5: Hub features
    hub_airports = set(feature_stats.get("hub_airports", []))
    df = add_hub_features(df, hub_airports)
    
    # Step 6: Historical delay features
    delay_stats = feature_stats.get("delay_stats", {})
    df = add_historical_delay_features(df, delay_stats)
    
    # Step 7: Build classification features
    df_cls = df.copy()
    
    # One-hot encode for classification: DEP_TIME_SLOT, DISTANCE_CAT, ARR_TIME_SLOT
    df_cls = _apply_one_hot_encoding(df_cls, "DEP_TIME_SLOT", ["evening", "morning", "night"])
    df_cls = _apply_one_hot_encoding(df_cls, "DISTANCE_CAT", ["medium", "short"])
    df_cls = _apply_one_hot_encoding(df_cls, "ARR_TIME_SLOT", ["evening", "morning", "night"])
    
    # Target encode high-cardinality columns for classification
    te_maps_cls = encoding_info_cls.get("target_encoding_maps", {})
    global_mean_cls = encoding_info_cls.get("global_mean", 0.0)
    
    for col in ["ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"]:
        te_map = te_maps_cls.get(col, {})
        df_cls = _apply_target_encoding(df_cls, col, te_map, global_mean_cls)
    
    # Ensure all required columns exist and are in correct order
    X_cls = _ensure_columns(df_cls, feature_columns_cls)
    
    # Step 8: Build regression features
    df_reg = df.copy()
    
    # Target encode high-cardinality columns for regression
    te_maps_reg = encoding_info_reg.get("target_encoding_maps", {})
    global_mean_reg = encoding_info_reg.get("global_mean", 0.0)
    
    for col in ["ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"]:
        te_map = te_maps_reg.get(col, {})
        df_reg = _apply_target_encoding(df_reg, col, te_map, global_mean_reg)
    
    # Ensure all required columns exist and are in correct order
    X_reg = _ensure_columns(df_reg, feature_columns_reg)

    return X_cls, X_reg


@lru_cache(maxsize=2)
def load_model_bundle(model_dir: str = "models") -> Dict[str, Any]:
    """
    Load the trained model + inference metadata once and cache it.

    Returns a dict with the model, its raw metadata, feature stats and the
    per-stage encoding info, so callers don't have to re-read them from disk
    (loading joblib + parsing the metadata JSON on every prediction is slow).
    Cached per ``model_dir`` via ``lru_cache``.
    """
    from ml_pipeline.serialization import load_model

    model, metadata = load_model(Path(model_dir))
    encoding_info = metadata.get("encoding_info", {})
    return {
        "model": model,
        "metadata": metadata,
        "feature_stats": metadata.get("feature_stats", {}),
        "encoding_info_cls": encoding_info.get("classification", {}),
        "encoding_info_reg": encoding_info.get("regression", {}),
    }


def predict_flight(
    *,
    month: int,
    day_of_week: int,
    scheduled_departure_hour: int,
    scheduled_arrival_hour: int,
    origin_airport_code: str,
    destination_airport_code: str,
    airline_code: str,
    distance: float,
    model_dir: str = "models",
) -> Dict[str, Any]:
    """
    Run the two-stage model on a single flight described by pre-departure fields.

    This is the single inference entrypoint shared by the REST ``/predict``
    endpoint and the agent's prediction tool, so both paths produce identical
    results from the same code.
    """
    bundle = load_model_bundle(model_dir)
    model = bundle["model"]

    raw_flight = pd.DataFrame([{
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "SCHEDULED_DEPARTURE": scheduled_departure_hour * 100,
        "SCHEDULED_ARRIVAL": scheduled_arrival_hour * 100,
        "ORIGIN_AIRPORT": origin_airport_code,
        "DESTINATION_AIRPORT": destination_airport_code,
        "AIRLINE": airline_code,
        "DISTANCE": distance,
    }])

    X_cls, X_reg = build_features_from_raw(
        raw_flight=raw_flight,
        feature_stats=bundle["feature_stats"],
        encoding_info_cls=bundle["encoding_info_cls"],
        encoding_info_reg=bundle["encoding_info_reg"],
        feature_columns_cls=model.feature_columns_cls,
        feature_columns_reg=model.feature_columns_reg,
    )

    outputs = model.predict(X_cls, X_reg)

    return {
        "delay_probability": float(outputs["delay_probability"][0]),
        "delayed": bool(outputs["delay_class"][0]),
        "delay_minutes": float(outputs["delay_minutes"][0]),
    }
