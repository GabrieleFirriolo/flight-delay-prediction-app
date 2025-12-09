"""
Model training module.
"""

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from lightgbm import LGBMClassifier, LGBMRegressor

from ml_pipeline.config import (
    BEST_CLASSIFIER,
    BEST_CLASSIFIER_PARAMS,
    BEST_REGRESSOR,
    BEST_REGRESSOR_PARAMS,
    DELAY_THRESHOLD,
    MAX_DELAY,
)
from ml_pipeline.models import TwoStageDelayModel

logger = logging.getLogger(__name__)


def _build_classifier(imbalance_ratio: float) -> LGBMClassifier:
    """Build LightGBM classifier with best params from notebook experiments."""
    return LGBMClassifier(
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        class_weight={0: 1, 1: imbalance_ratio},
        **BEST_CLASSIFIER_PARAMS,
    )


def _build_regressor() -> LGBMRegressor:
    """Build LightGBM regressor with best params from notebook experiments."""
    return LGBMRegressor(
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        **BEST_REGRESSOR_PARAMS,
    )


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    delay_threshold: int = DELAY_THRESHOLD,
) -> Tuple[Any, str]:
    """
    Train the classification model using pre-tuned hyperparameters.

    Returns:
        Tuple of (trained_model, model_name).
    """
    logger.info("Training classifier...")

    y_train_cls = (y_train > delay_threshold).astype(int)
    class_counts = y_train_cls.value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]

    start_time = time.time()
    classifier = _build_classifier(imbalance_ratio)
    classifier.fit(X_train, y_train_cls)
    elapsed = time.time() - start_time

    logger.info(f"Classifier trained in {elapsed:.1f}s")

    return classifier, BEST_CLASSIFIER


def find_optimal_threshold(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Find optimal classification threshold that maximizes F1 score."""
    y_proba = model.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_proba)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return thresholds[np.argmax(f1_scores)]


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    delay_threshold: int = DELAY_THRESHOLD,
    max_delay: int = MAX_DELAY,
) -> Tuple[Any, str]:
    """
    Train the regression model using pre-tuned hyperparameters.
    Only trains on samples with delay > threshold.

    Returns:
        Tuple of (trained_model, model_name).
    """
    logger.info("Training regressor...")

    late_mask = y_train > delay_threshold
    X_train_late = X_train[late_mask].reset_index(drop=True)
    y_train_late = y_train[late_mask].reset_index(drop=True)
    y_train_late_clip = y_train_late.clip(lower=delay_threshold, upper=max_delay)

    start_time = time.time()
    regressor = _build_regressor()
    regressor.fit(X_train_late, y_train_late_clip)
    elapsed = time.time() - start_time

    logger.info(f"Regressor trained in {elapsed:.1f}s")

    return regressor, BEST_REGRESSOR


def train_two_stage_model(
    X_train_cls: pd.DataFrame,
    X_train_reg: pd.DataFrame,
    y_train: pd.Series,
    delay_threshold: int = DELAY_THRESHOLD,
    max_delay: int = MAX_DELAY,
) -> Tuple[TwoStageDelayModel, Dict[str, Any]]:
    """
    Train the complete two-stage delay prediction model.

    Uses the pre-tuned hyperparameters found via GridSearch in analysis.ipynb
    (no GridSearch at training time - faster, reproducible training).
    """
    classifier, cls_name = train_classifier(X_train_cls, y_train, delay_threshold)

    y_train_cls_binary = (y_train > delay_threshold).astype(int)
    optimal_threshold = find_optimal_threshold(classifier, X_train_cls, y_train_cls_binary)

    regressor, reg_name = train_regressor(X_train_reg, y_train, delay_threshold, max_delay)

    model = TwoStageDelayModel(
        classifier=classifier,
        regressor=regressor,
        classification_threshold=optimal_threshold,
        delay_threshold=delay_threshold,
        max_delay=max_delay,
        classifier_name=cls_name,
        regressor_name=reg_name,
        feature_columns_cls=list(X_train_cls.columns),
        feature_columns_reg=list(X_train_reg.columns),
    )

    return model, {"optimal_threshold": optimal_threshold}
