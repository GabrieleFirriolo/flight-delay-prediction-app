"""
Pydantic schemas for the Agent.
"""

from typing import Optional
from pydantic import BaseModel


class FlightSlots(BaseModel):
    """Slot state for flight information collection."""
    month: Optional[int] = None
    day_of_week: Optional[int] = None
    scheduled_departure_hour: Optional[int] = None
    scheduled_arrival_hour: Optional[int] = None
    origin_airport_code: Optional[str] = None
    destination_airport_code: Optional[str] = None
    airline_code: Optional[str] = None
    distance: Optional[float] = None
    
    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        return all([
            self.month is not None,
            self.day_of_week is not None,
            self.scheduled_departure_hour is not None,
            self.scheduled_arrival_hour is not None,
            self.origin_airport_code is not None,
            self.destination_airport_code is not None,
            self.airline_code is not None,
            self.distance is not None,
        ])
    
    def missing_fields(self) -> list:
        """Return list of missing field names."""
        missing = []
        if self.month is None:
            missing.append("month")
        if self.day_of_week is None:
            missing.append("day_of_week")
        if self.scheduled_departure_hour is None:
            missing.append("scheduled_departure_hour")
        if self.scheduled_arrival_hour is None:
            missing.append("scheduled_arrival_hour")
        if self.origin_airport_code is None:
            missing.append("origin_airport_code")
        if self.destination_airport_code is None:
            missing.append("destination_airport_code")
        if self.airline_code is None:
            missing.append("airline_code")
        if self.distance is None:
            missing.append("distance")
        return missing


class PredictionResult(BaseModel):
    """Prediction result from the model."""
    delay_probability: float
    delayed: bool
    delay_minutes: float


class ChatRequest(BaseModel):
    """Request for agent chat endpoint."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from agent chat endpoint."""
    reply: str
    slots: FlightSlots
    prediction: Optional[PredictionResult] = None
