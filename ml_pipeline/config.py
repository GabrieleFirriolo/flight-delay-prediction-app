"""
Configuration module for the ML pipeline.

"""

from pathlib import Path
from typing import List

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
FLIGHTS_CSV = "flights.csv"
AIRPORTS_CSV = "airports.csv"
AIRLINES_CSV = "airlines.csv"

# =============================================================================
# BUSINESS CONSTANTS
# =============================================================================
DELAY_THRESHOLD = 15  # minutes - threshold for "delayed" classification
MAX_DELAY = 180  # minutes - cap for regression target (outlier handling)
HUB_QUANTILE = 0.90  # top 10% airports by traffic are considered hubs

# =============================================================================
# TEMPORAL SPLIT
# =============================================================================
TRAIN_MONTHS = list(range(1, 11))  # January to October
TEST_MONTHS = list(range(11, 13))  # November and December

# =============================================================================
# Features that are only known AFTER departure/arrival
# =============================================================================
LEAKAGE_COLS: List[str] = [
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "ELAPSED_TIME",
    "AIR_TIME",
    "WHEELS_ON",
    "TAXI_IN",
    "ARRIVAL_TIME",
    "DIVERTED",
    "CANCELLED",
    "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
]

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Features for CLASSIFICATION stage
FEATURES_CLS: List[str] = [
    "MONTH_SIN", "MONTH_COS",
    "DAY_SIN", "DAY_COS",
    "DEP_HOUR_SIN", "DEP_HOUR_COS",
    "ARR_HOUR_SIN", "ARR_HOUR_COS",
    "IS_WEEKEND", "IS_SUMMER", "IS_HOLIDAY_SEASON",
    "DEP_TIME_SLOT", "DEP_IS_PEAK_HOUR", "DEP_IS_RED_EYE",
    "ARR_TIME_SLOT", "ARR_IS_PEAK_HOUR", "ARR_IS_RED_EYE",
    "DISTANCE_CAT",
    "ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE",
    "IS_ORIGIN_HUB", "IS_DEST_HUB",
    "ROUTE_DELAY_RATE_15", "AIRLINE_DELAY_RATE_15", "HOUR_DELAY_RATE_15",
]

# Features for REGRESSION stage
FEATURES_REG: List[str] = [
    "MONTH_SIN", "MONTH_COS",
    "DAY_SIN", "DAY_COS",
    "DEP_HOUR_SIN", "DEP_HOUR_COS",
    "ARR_HOUR_SIN", "ARR_HOUR_COS",
    "IS_WEEKEND", "IS_SUMMER", "IS_HOLIDAY_SEASON",
    "DISTANCE",
    "ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE",
    "IS_ORIGIN_HUB", "IS_DEST_HUB",
    "ROUTE_DELAY_MEAN", "AIRLINE_DELAY_MEAN", "HOUR_DELAY_MEAN",
]

# Low-cardinality categorical features (one-hot encoded)
CAT_LOW_CLS: List[str] = ["DEP_TIME_SLOT", "DISTANCE_CAT", "ARR_TIME_SLOT"]
CAT_LOW_REG: List[str] = [] 

# High-cardinality categorical features (target encoded)
CAT_HIGH: List[str] = ["ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"]

# =============================================================================
# BEST MODEL HYPERPARAMETERS (from analysis.ipynb GridSearch)
# =============================================================================
BEST_CLASSIFIER = "LightGBM"
BEST_CLASSIFIER_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 10,
    "n_estimators": 200,
}

BEST_REGRESSOR = "LightGBM"
BEST_REGRESSOR_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
}