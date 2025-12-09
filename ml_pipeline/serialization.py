"""
Model serialization module.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from ml_pipeline.config import MODELS_DIR
from ml_pipeline.models import TwoStageDelayModel

logger = logging.getLogger(__name__)

# Default model filename
DEFAULT_MODEL_FILENAME = "two_stage_delay_model.joblib"
DEFAULT_METADATA_FILENAME = "model_metadata.json"


def _to_serializable(value: Any) -> Any:
    """
    Recursively convert a value into a JSON-serializable structure.

    Pandas Series/DataFrame (e.g. target-encoding maps, which can hold thousands
    of entries) are converted with ``to_dict()`` so they survive the round-trip
    intact. Without this, ``json.dump(default=str)`` would stringify a Series
    into a *truncated* ``repr`` (with "..."), silently dropping most entries.
    """
    if hasattr(value, "to_dict"):  # pandas Series / DataFrame
        return _to_serializable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (set, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def save_model(
    model: TwoStageDelayModel,
    path: Path = None,
    feature_stats: Dict[str, Any] = None,
    evaluation_metrics: Dict[str, Any] = None,
    encoding_info: Dict[str, Any] = None,
) -> Path:
    """
    Save trained model and metadata to disk.

    Creates a directory with:
    - model.joblib: The trained model object
    - metadata.json: Model configuration and metrics

    Args:
        model: Trained TwoStageDelayModel.
        path: Directory path to save model. Defaults to MODELS_DIR.
        feature_stats: Feature engineering statistics (hub airports, delay stats).
        evaluation_metrics: Model evaluation metrics.
        encoding_info: Feature encoding information.

    Returns:
        Path to the saved model directory.
    """
    if path is None:
        path = MODELS_DIR

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / DEFAULT_MODEL_FILENAME
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    metadata = {
        "model_info": model.to_dict(),
        "saved_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }

    if feature_stats is not None:
        metadata["feature_stats"] = _to_serializable(feature_stats)

    if evaluation_metrics is not None:
        metadata["evaluation_metrics"] = _to_serializable(evaluation_metrics)

    if encoding_info is not None:
        metadata["encoding_info"] = _to_serializable(encoding_info)

    metadata_path = path / DEFAULT_METADATA_FILENAME
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")

    return path


def load_model(
    path: Path = None,
) -> Tuple[TwoStageDelayModel, Dict[str, Any]]:
    """
    Load trained model and metadata from disk.

    Args:
        path: Directory path containing saved model. Defaults to MODELS_DIR.

    Returns:
        Tuple of (model, metadata).

    Raises:
        FileNotFoundError: If model files are not found.
    """
    if path is None:
        path = MODELS_DIR

    path = Path(path)

    model_path = path / DEFAULT_MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    metadata_path = path / DEFAULT_METADATA_FILENAME
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded from {metadata_path}")
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")

    return model, metadata


def model_exists(path: Path = None) -> bool:
    """
    Check if a saved model exists at the given path.

    Args:
        path: Directory path to check. Defaults to MODELS_DIR.

    Returns:
        True if model file exists, False otherwise.
    """
    if path is None:
        path = MODELS_DIR

    path = Path(path)
    model_path = path / DEFAULT_MODEL_FILENAME

    return model_path.exists()


def get_model_info(path: Path = None) -> Optional[Dict[str, Any]]:
    """
    Get model metadata without loading the full model.

    Args:
        path: Directory path containing saved model. Defaults to MODELS_DIR.

    Returns:
        Metadata dictionary or None if not found.
    """
    if path is None:
        path = MODELS_DIR

    path = Path(path)
    metadata_path = path / DEFAULT_METADATA_FILENAME

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        return json.load(f)
