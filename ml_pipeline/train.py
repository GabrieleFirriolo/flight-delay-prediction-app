"""
Main training script for the flight delay prediction model.

This script orchestrates the full training pipeline:
1. Load raw data
2. Preprocess and clean
3. Engineer features
4. Encode for models
5. Train two-stage model
6. Evaluate on test set
7. Save model and metrics

Usage:
    python -m ml_pipeline.train
    python -m ml_pipeline.train --data-dir ./data --output-dir ./models
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from ml_pipeline.config import DATA_DIR, MODELS_DIR
from ml_pipeline.data_loading import load_raw_data, train_test_split_by_month, preprocess_flights
from ml_pipeline.features import engineer_features
from ml_pipeline.encoding import build_classification_matrices, build_regression_matrices
from ml_pipeline.training import train_two_stage_model
from ml_pipeline.evaluation import evaluate_two_stage_model, print_evaluation_summary, get_metrics_for_api
from ml_pipeline.serialization import save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train flight delay prediction model"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing CSV data files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Directory to save trained model (default: {MODELS_DIR})",
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("FLIGHT DELAY PREDICTION - MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Started at: {start_time.isoformat()}")

    try:
        logger.info("[1/7] Loading raw data...")
        data = load_raw_data(args.data_dir)
        flights_raw = data["flights"]

        logger.info("[2/7] Preprocessing data...")
        flights = preprocess_flights(flights_raw)

        logger.info("[3/7] Splitting data temporally...")
        df_train, df_test = train_test_split_by_month(flights)

        logger.info("[4/7] Engineering features...")
        df_train, df_test, feature_stats = engineer_features(df_train, df_test)

        logger.info("[5/7] Building model matrices...")
        X_train_cls, X_test_cls, y_train, y_test, months_train, cls_encoding = \
            build_classification_matrices(df_train, df_test)
        X_train_reg, X_test_reg, reg_encoding = \
            build_regression_matrices(df_train, df_test)

        logger.info("[6/7] Training two-stage model...")
        model, training_results = train_two_stage_model(
            X_train_cls=X_train_cls,
            X_train_reg=X_train_reg,
            y_train=y_train,
        )

        logger.info("[7/7] Evaluating model...")
        evaluation_metrics = evaluate_two_stage_model(
            model=model,
            X_test_cls=X_test_cls,
            X_test_reg=X_test_reg,
            y_test=y_test,
        )
        print_evaluation_summary(evaluation_metrics)

        logger.info("Saving model...")
        model.metadata = {
            "trained_at": start_time.isoformat(),
            "data_dir": str(args.data_dir),
            "train_samples": len(df_train),
            "test_samples": len(df_test),
        }

        save_path = save_model(
            model=model,
            path=args.output_dir,
            feature_stats=feature_stats,
            evaluation_metrics=get_metrics_for_api(evaluation_metrics),
            encoding_info={
                "classification": cls_encoding,
                "regression": reg_encoding,
            },
        )

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Model saved to: {save_path}")
        logger.info(f"Total duration: {duration}")
        logger.info("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
