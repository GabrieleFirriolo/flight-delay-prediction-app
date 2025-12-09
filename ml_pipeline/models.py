"""
Model definitions module.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ml_pipeline.config import DELAY_THRESHOLD, MAX_DELAY

logger = logging.getLogger(__name__)


@dataclass
class TwoStageDelayModel:
    """
    Two-stage model for flight delay prediction.

    Stage 1 (Classification): Predicts if a flight will be delayed (>15 min).
    Stage 2 (Regression): Predicts delay magnitude for flights classified as delayed.

    Attributes:
        classifier: Trained classification model (LightGBM).
        regressor: Trained regression model (LightGBM).
        classification_threshold: Probability threshold for delay classification.
        delay_threshold: Minutes threshold for delay definition (default: 15).
        max_delay: Maximum delay cap for regression (default: 180).
        classifier_name: Name of the classifier algorithm.
        regressor_name: Name of the regressor algorithm.
        feature_columns_cls: Feature column names for classification.
        feature_columns_reg: Feature column names for regression.
        metadata: Additional metadata (training date, metrics, etc.).
    """

    classifier: Any = None
    regressor: Any = None
    classification_threshold: float = 0.5
    delay_threshold: int = DELAY_THRESHOLD
    max_delay: int = MAX_DELAY
    classifier_name: str = ""
    regressor_name: str = ""
    feature_columns_cls: list = field(default_factory=list)
    feature_columns_reg: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict_proba_delay(self, X_cls: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of delay > threshold.

        Args:
            X_cls: Feature matrix for classification.

        Returns:
            Array of delay probabilities.
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")

        return self.classifier.predict_proba(X_cls)[:, 1]


    def predict(
        self,
        X_cls: pd.DataFrame,
        X_reg: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Full prediction with all outputs.

        Args:
            X_cls: Feature matrix for classification.
            X_reg: Feature matrix for regression.

        Returns:
            Dictionary with:
                - delay_probability: Probability of delay
                - delay_class: Binary delay prediction
                - delay_minutes: Predicted delay in minutes
        """
        proba = self.predict_proba_delay(X_cls)
        delay_class = (proba >= self.classification_threshold).astype(int)

        # Regression for predicted delayed flights
        delay_minutes = np.zeros(len(X_cls))
        delay_mask = delay_class.astype(bool)
        if delay_mask.sum() > 0:
            delay_minutes[delay_mask] = self.regressor.predict(
                X_reg.iloc[delay_mask]
            )

        return {
            "delay_probability": proba,
            "delay_class": delay_class,
            "delay_minutes": delay_minutes,
        }

    def get_feature_importance(self) -> Dict[str, pd.Series]:
        result = {}

        if self.classifier is not None and hasattr(self.classifier, "feature_importances_"):
            result["classifier"] = pd.Series(
                self.classifier.feature_importances_,
                index=self.feature_columns_cls,
            ).sort_values(ascending=False)

        if self.regressor is not None and hasattr(self.regressor, "feature_importances_"):
            result["regressor"] = pd.Series(
                self.regressor.feature_importances_,
                index=self.feature_columns_reg,
            ).sort_values(ascending=False)

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification_threshold": self.classification_threshold,
            "delay_threshold": self.delay_threshold,
            "max_delay": self.max_delay,
            "classifier_name": self.classifier_name,
            "regressor_name": self.regressor_name,
            "feature_columns_cls": self.feature_columns_cls,
            "feature_columns_reg": self.feature_columns_reg,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        classifier: Any = None,
        regressor: Any = None,
    ) -> "TwoStageDelayModel":
        return cls(
            classifier=classifier,
            regressor=regressor,
            classification_threshold=data.get("classification_threshold", 0.5),
            delay_threshold=data.get("delay_threshold", DELAY_THRESHOLD),
            max_delay=data.get("max_delay", MAX_DELAY),
            classifier_name=data.get("classifier_name", ""),
            regressor_name=data.get("regressor_name", ""),
            feature_columns_cls=data.get("feature_columns_cls", []),
            feature_columns_reg=data.get("feature_columns_reg", []),
            metadata=data.get("metadata", {}),
        )
