"""
FastAPI application for flight delay prediction.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml_pipeline.inference import load_model_bundle, predict_flight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")


from app.schemas import (
    FlightRequest,
    PredictionResponse,
    HealthResponse,
    ClassificationMetrics,
    RegressionMetrics,
    PerformanceResponse,
)

def _extract_core_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only core metrics from full evaluation metrics."""
    cls_raw = metrics.get("classification", {})
    reg_raw = metrics.get("regression", {})
    
    return {
        "classification": {
            "accuracy": cls_raw.get("accuracy", 0.0),
            "precision": cls_raw.get("precision", 0.0),
            "recall": cls_raw.get("recall", 0.0),
            "f1_score": cls_raw.get("f1_score", 0.0),
            "roc_auc": cls_raw.get("roc_auc", 0.0),
        },
        "regression": {
            "mae": reg_raw.get("mae", 0.0),
            "rmse": reg_raw.get("rmse", 0.0),
            "r2": reg_raw.get("r2", 0.0),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the shared (cached) model bundle on startup."""
    try:
        bundle = load_model_bundle(str(MODEL_DIR))
        app.state.metadata = bundle["metadata"]
        app.state.model_ready = True
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        app.state.metadata = {}
        app.state.model_ready = False
    yield


app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount agent router
from assistant.router import router as agent_router
app.include_router(agent_router, prefix="/agent", tags=["Agent"])


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    model_loaded = getattr(app.state, "model_ready", False)
    version = app.state.metadata.get("version", "unknown")
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=version,
    )


@app.get("/model/performance", response_model=PerformanceResponse)
def model_performance():
    """Return core model evaluation metrics."""
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics = app.state.metadata.get("evaluation_metrics", {})
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics available")

    core = _extract_core_metrics(metrics)
    return PerformanceResponse(
        classification=ClassificationMetrics(**core["classification"]),
        regression=RegressionMetrics(**core["regression"]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightRequest):
    """Predict delay for a single flight."""
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predict_flight(**flight.model_dump(), model_dir=str(MODEL_DIR))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictionResponse(**result)
