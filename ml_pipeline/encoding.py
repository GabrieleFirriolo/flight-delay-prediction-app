"""
Feature encoding module.
"""

from typing import Dict, List, Tuple

import pandas as pd

from ml_pipeline.config import (
    FEATURES_CLS,
    FEATURES_REG,
    CAT_LOW_CLS,
    CAT_LOW_REG,
    CAT_HIGH,
)


def one_hot_encode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply one-hot encoding to specified columns.

    Ensures train and test have aligned columns.

    Args:
        df_train: Training DataFrame.
        df_test: Test DataFrame.
        columns: List of column names to one-hot encode.

    Returns:
        Tuple of (encoded_train, encoded_test) with aligned columns.
    """
    train_result = df_train.copy()
    test_result = df_test.copy()

    for col in columns:
        if col not in train_result.columns:
            continue

        dummies_train = pd.get_dummies(train_result[col], prefix=col, drop_first=True)
        dummies_test = pd.get_dummies(test_result[col], prefix=col, drop_first=True)

        # Align columns - test may have missing categories
        for c in dummies_train.columns:
            if c not in dummies_test.columns:
                dummies_test[c] = 0
        dummies_test = dummies_test[dummies_train.columns]

        train_result = pd.concat(
            [train_result.drop(columns=[col]), dummies_train], axis=1
        )
        test_result = pd.concat(
            [test_result.drop(columns=[col]), dummies_test], axis=1
        )

    return train_result, test_result


def target_encode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    target_source: pd.DataFrame,
    target_col: str,
    global_mean: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    """
    Apply target encoding to high-cardinality categorical columns.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        columns: Columns to target encode.
        target_source: DataFrame containing target column (for computing means).
        target_col: Name of the target column.
        global_mean: Fallback for unseen categories.

    Returns:
        Tuple of (encoded_train, encoded_test, encoding_maps).
    """
    train_result = X_train.copy()
    test_result = X_test.copy()
    encoding_maps = {}

    for col in columns:
        if col not in train_result.columns:
            continue

        te_map = target_source.groupby(col)[target_col].mean()
        encoding_maps[col] = te_map

        new_col = f"{col}_TE"
        train_result[new_col] = train_result[col].map(te_map).fillna(global_mean)
        test_result[new_col] = test_result[col].map(te_map).fillna(global_mean)

        train_result = train_result.drop(columns=[col])
        test_result = test_result.drop(columns=[col])

    return train_result, test_result, encoding_maps


def build_classification_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str] = FEATURES_CLS,
    cat_low: List[str] = CAT_LOW_CLS,
    cat_high: List[str] = CAT_HIGH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Build feature matrices for classification stage.

    Args:
        df_train: Training DataFrame with all features.
        df_test: Test DataFrame with all features.
        features: List of feature column names to use.
        cat_low: Low-cardinality columns for one-hot encoding.
        cat_high: High-cardinality columns for target encoding.

    Returns:
        Tuple of:
            - X_train_cls: Training feature matrix
            - X_test_cls: Test feature matrix
            - y_train: Training target (ARRIVAL_DELAY)
            - y_test: Test target (ARRIVAL_DELAY)
            - months_train: Training months for CV
            - encoding_info: Dictionary with encoding maps
    """
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    y_train = df_train["ARRIVAL_DELAY"].reset_index(drop=True)
    y_test = df_test["ARRIVAL_DELAY"].reset_index(drop=True)
    months_train = df_train["MONTH"].reset_index(drop=True)
    global_mean = df_train["ARRIVAL_DELAY"].mean()

    X_train, X_test = one_hot_encode(X_train, X_test, cat_low)
    X_train, X_test, te_maps = target_encode(
        X_train, X_test, cat_high, df_train, "ARRIVAL_DELAY", global_mean
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    encoding_info = {
        "target_encoding_maps": te_maps,
        "global_mean": global_mean,
        "feature_columns": list(X_train.columns),
    }

    return X_train, X_test, y_train, y_test, months_train, encoding_info


def build_regression_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str] = FEATURES_REG,
    cat_low: List[str] = CAT_LOW_REG,
    cat_high: List[str] = CAT_HIGH,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build feature matrices for regression stage.

    Args:
        df_train: Training DataFrame with all features.
        df_test: Test DataFrame with all features.
        features: List of feature column names to use.
        cat_low: Low-cardinality columns for one-hot encoding.
        cat_high: High-cardinality columns for target encoding.

    Returns:
        Tuple of:
            - X_train_reg: Training feature matrix
            - X_test_reg: Test feature matrix
            - encoding_info: Dictionary with encoding maps
    """
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    global_mean = df_train["ARRIVAL_DELAY"].mean()

    if cat_low:
        X_train, X_test = one_hot_encode(X_train, X_test, cat_low)

    X_train, X_test, te_maps = target_encode(
        X_train, X_test, cat_high, df_train, "ARRIVAL_DELAY", global_mean
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    encoding_info = {
        "target_encoding_maps": te_maps,
        "global_mean": global_mean,
        "feature_columns": list(X_train.columns),
    }

    return X_train, X_test, encoding_info
