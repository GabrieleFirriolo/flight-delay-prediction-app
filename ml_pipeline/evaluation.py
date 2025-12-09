"""
Model evaluation module.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml_pipeline.config import DELAY_THRESHOLD
from ml_pipeline.models import TwoStageDelayModel

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate classification performance.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        threshold: Classification threshold.

    Returns:
        Dictionary with classification metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=["On-time", "Delayed"],
        output_dict=True,
    )
    metrics["classification_report"] = report

    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_value: float = None,
) -> Dict[str, float]:
    """
    Evaluate regression performance.

    Args:
        y_true: True delay values in minutes.
        y_pred: Predicted delay values in minutes.
        baseline_value: Value for baseline comparison (e.g., mean).

    Returns:
        Dictionary with regression metrics.
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_error": np.mean(y_pred - y_true),
        "std_error": np.std(y_pred - y_true),
    }

    if baseline_value is not None:
        baseline_pred = np.full_like(y_true, baseline_value, dtype=float)
        metrics["baseline_mae"] = mean_absolute_error(y_true, baseline_pred)
        metrics["mae_improvement"] = metrics["baseline_mae"] - metrics["mae"]

    return metrics


def evaluate_two_stage_model(
    model: TwoStageDelayModel,
    X_test_cls: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_test: pd.Series,
    delay_threshold: int = DELAY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the two-stage model.

    Args:
        model: Trained TwoStageDelayModel.
        X_test_cls: Test features for classification.
        X_test_reg: Test features for regression.
        y_test: True delay values in minutes.
        delay_threshold: Threshold for delay classification.

    Returns:
        Dictionary with all evaluation metrics.
    """
    y_test_cls = (y_test > delay_threshold).astype(int)

    y_proba = model.predict_proba_delay(X_test_cls)
    cls_metrics = evaluate_classifier(
        y_test_cls.values,
        y_proba,
        model.classification_threshold,
    )

    late_mask = y_test > delay_threshold
    X_test_late_reg = X_test_reg[late_mask]
    y_test_late = y_test[late_mask]

    if len(y_test_late) > 0:
        y_pred_reg = model.regressor.predict(X_test_late_reg)
        baseline_mean = y_test_late.mean()
        reg_metrics = evaluate_regressor(
            y_test_late.values,
            y_pred_reg,
            baseline_value=baseline_mean,
        )
    else:
        reg_metrics = {}

    return {
        "classification": cls_metrics,
        "regression": reg_metrics,
        "test_set_size": len(y_test),
        "delayed_flights_count": int(late_mask.sum()),
        "delay_threshold": delay_threshold,
    }


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    cls = metrics.get("classification", {})
    reg = metrics.get("regression", {})

    print(f"""
TEST SET OVERVIEW
-----------------
- Total flights: {metrics.get('test_set_size', 'N/A'):,}
- Delayed flights (>{metrics.get('delay_threshold', 15)} min): {metrics.get('delayed_flights_count', 'N/A'):,}

STAGE 1 - CLASSIFICATION
------------------------
- Accuracy:  {cls.get('accuracy', 0):.4f}
- Precision: {cls.get('precision', 0):.4f}
- Recall:    {cls.get('recall', 0):.4f}
- F1-score:  {cls.get('f1_score', 0):.4f}
- ROC-AUC:   {cls.get('roc_auc', 0):.4f}

STAGE 2 - REGRESSION
--------------------
- MAE:  {reg.get('mae', 0):.2f} min
- RMSE: {reg.get('rmse', 0):.2f} min
- R²:   {reg.get('r2', 0):.4f}
- Baseline MAE: {reg.get('baseline_mae', 0):.2f} min
""")


def get_metrics_for_api(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for API response.

    Converts numpy types to Python native types for JSON serialization.

    Args:
        metrics: Raw metrics dictionary.

    Returns:
        JSON-serializable metrics dictionary.
    """
    def convert_value(v):
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        return v

    return convert_value(metrics)
