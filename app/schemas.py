from pydantic import BaseModel, Field


class FlightRequest(BaseModel):
    """
    Input for flight delay prediction.

    Provide flight information as known before departure.
    """
    month: int = Field(ge=1, le=12)                       # 1-12
    day_of_week: int = Field(ge=1, le=7)                  # 1=Monday, 7=Sunday
    scheduled_departure_hour: int = Field(ge=0, le=23)    # 0-23
    scheduled_arrival_hour: int = Field(ge=0, le=23)      # 0-23
    origin_airport_code: str = Field(min_length=1)        # e.g. "JFK"
    destination_airport_code: str = Field(min_length=1)   # e.g. "LAX"
    airline_code: str = Field(min_length=1)               # e.g. "AA"
    distance: float = Field(ge=0)                         # miles


class PredictionResponse(BaseModel):
    delay_probability: float
    delayed: bool
    delay_minutes: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class RegressionMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float


class PerformanceResponse(BaseModel):
    classification: ClassificationMetrics
    regression: RegressionMetrics
