"""
Domain feature engineering module.
"""

from typing import Set, Tuple

import pandas as pd

from ml_pipeline.config import DELAY_THRESHOLD, HUB_QUANTILE


def add_route_features(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()

    df["ROUTE"] = df["ORIGIN_AIRPORT"] + "-" + df["DESTINATION_AIRPORT"]

    # Vectorized distance bucketing: short <500, medium [500, 1500), long >=1500.
    df["DISTANCE_CAT"] = pd.cut(
        df["DISTANCE"],
        bins=[0, 500, 1500, float("inf")],
        labels=["short", "medium", "long"],
        right=False,
    ).astype("object")

    return df


def compute_hub_airports(flights: pd.DataFrame, quantile: float = HUB_QUANTILE) -> Set[str]:
    departures = flights["ORIGIN_AIRPORT"].value_counts()
    arrivals = flights["DESTINATION_AIRPORT"].value_counts()
    traffic = departures.add(arrivals, fill_value=0)
    threshold = traffic.quantile(quantile)
    hub_airports = set(traffic[traffic >= threshold].index)

    return hub_airports


def add_hub_features(
    flights: pd.DataFrame,
    hub_airports: Set[str],
) -> pd.DataFrame:
    df = flights.copy()
    df["IS_ORIGIN_HUB"] = df["ORIGIN_AIRPORT"].isin(hub_airports).astype(int)
    df["IS_DEST_HUB"] = df["DESTINATION_AIRPORT"].isin(hub_airports).astype(int)

    return df


def compute_historical_delay_stats(
    df_train: pd.DataFrame,
    delay_threshold: int = DELAY_THRESHOLD,
) -> dict:
    global_delay_mean = df_train["ARRIVAL_DELAY"].mean()
    global_delay_rate = (df_train["ARRIVAL_DELAY"] > delay_threshold).mean()

    route_delay_mean = df_train.groupby("ROUTE")["ARRIVAL_DELAY"].mean()
    route_delay_rate = df_train.groupby("ROUTE")["ARRIVAL_DELAY"].apply(
        lambda x: (x > delay_threshold).mean()
    )

    airline_delay_mean = df_train.groupby("AIRLINE")["ARRIVAL_DELAY"].mean()
    airline_delay_rate = df_train.groupby("AIRLINE")["ARRIVAL_DELAY"].apply(
        lambda x: (x > delay_threshold).mean()
    )

    hour_delay_mean = df_train.groupby("DEP_HOUR")["ARRIVAL_DELAY"].mean()
    hour_delay_rate = df_train.groupby("DEP_HOUR")["ARRIVAL_DELAY"].apply(
        lambda x: (x > delay_threshold).mean()
    )

    return {
        "global_delay_mean": global_delay_mean,
        "global_delay_rate": global_delay_rate,
        "route_delay_mean": route_delay_mean,
        "route_delay_rate": route_delay_rate,
        "airline_delay_mean": airline_delay_mean,
        "airline_delay_rate": airline_delay_rate,
        "hour_delay_mean": hour_delay_mean,
        "hour_delay_rate": hour_delay_rate,
    }


def add_historical_delay_features(
    df: pd.DataFrame,
    stats: dict,
) -> pd.DataFrame:
    result = df.copy()

    global_mean = stats["global_delay_mean"]
    global_rate = stats["global_delay_rate"]

    result["ROUTE_DELAY_MEAN"] = result["ROUTE"].map(stats["route_delay_mean"]).fillna(global_mean)
    result["ROUTE_DELAY_RATE_15"] = result["ROUTE"].map(stats["route_delay_rate"]).fillna(global_rate)
    result["AIRLINE_DELAY_MEAN"] = result["AIRLINE"].map(stats["airline_delay_mean"]).fillna(global_mean)
    result["AIRLINE_DELAY_RATE_15"] = result["AIRLINE"].map(stats["airline_delay_rate"]).fillna(global_rate)
    result["HOUR_DELAY_MEAN"] = result["DEP_HOUR"].map(stats["hour_delay_mean"]).fillna(global_mean)
    result["HOUR_DELAY_RATE_15"] = result["DEP_HOUR"].map(stats["hour_delay_rate"]).fillna(global_rate)

    return result


def engineer_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full feature engineering pipeline for train and test sets.

    Computes statistics on training data only, then applies to both sets.

    Args:
        df_train: Training DataFrame (preprocessed).
        df_test: Test DataFrame (preprocessed).

    Returns:
        Tuple of (df_train_featured, df_test_featured, feature_stats).
    """
    df_train = add_route_features(df_train)
    df_test = add_route_features(df_test)

    hub_airports = compute_hub_airports(df_train)  # computed on train only
    df_train = add_hub_features(df_train, hub_airports)
    df_test = add_hub_features(df_test, hub_airports)

    delay_stats = compute_historical_delay_stats(df_train)  # computed on train only
    df_train = add_historical_delay_features(df_train, delay_stats)
    df_test = add_historical_delay_features(df_test, delay_stats)

    feature_stats = {
        "hub_airports": hub_airports,
        "delay_stats": delay_stats,
    }

    return df_train, df_test, feature_stats
