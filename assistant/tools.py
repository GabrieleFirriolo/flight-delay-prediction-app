"""
LangChain tools for the Flight Delay Prediction Agent.
"""

import json
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

from ml_pipeline.inference import predict_flight
from assistant.rag import (
    search_airports as rag_search_airports,
    search_airlines as rag_search_airlines,
    estimate_distance as rag_estimate_distance,
)


@tool
def search_airports(query: str) -> str:
    """
    Search for airports by name, city, or code.
    
    Use this tool when you need to find the IATA code for an airport
    mentioned by the user in natural language.
    
    Args:
        query: Natural language description of the airport (e.g. "New York JFK", "Milano Linate")
    
    Returns:
        JSON list of matching airports with code, name, city, country.
    """
    results = rag_search_airports(query, k=5)
    return json.dumps(results, ensure_ascii=False)


@tool
def search_airlines(query: str) -> str:
    """
    Search for airlines by name or code.
    
    Use this tool when you need to find the IATA code for an airline
    mentioned by the user in natural language.
    
    Args:
        query: Natural language description of the airline (e.g. "American Airlines", "Delta")
    
    Returns:
        JSON list of matching airlines with code and name.
    """
    results = rag_search_airlines(query, k=5)
    return json.dumps(results, ensure_ascii=False)


@tool
def get_current_time() -> str:
    """
    Get the current time.
    """
    now = datetime.now()
    return json.dumps({
        "datetime": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "day_of_week": now.isoweekday(),  # 1=Monday, 7=Sunday
        "hour": now.hour,
        "minute": now.minute,
        "weekday_name": now.strftime("%A"),
    })


@tool
def calculate_distance(origin_airport_code: str, destination_airport_code: str) -> str:
    """
    Calculate the distance in miles between two airports.
    
    Use this tool to get the flight distance when you have both airport codes.
    
    Args:
        origin_airport_code: IATA code of the origin airport (e.g. "JFK")
        destination_airport_code: IATA code of the destination airport (e.g. "LAX")
    
    Returns:
        JSON with success status and distance in miles, or error message.
    """
    distance = rag_estimate_distance(origin_airport_code, destination_airport_code)
    
    if distance is None:
        return json.dumps({
            "success": False,
            "error": f"Could not calculate distance. One or both airports not found: {origin_airport_code}, {destination_airport_code}"
        })
    
    return json.dumps({
        "success": True,
        "distance_miles": round(distance, 1),
        "origin": origin_airport_code,
        "destination": destination_airport_code,
    })


@tool
def update_flight_slots(
    month: Optional[int] = None,
    day_of_week: Optional[int] = None,
    scheduled_departure_hour: Optional[int] = None,
    scheduled_arrival_hour: Optional[int] = None,
    origin_airport_code: Optional[str] = None,
    destination_airport_code: Optional[str] = None,
    airline_code: Optional[str] = None,
    distance: Optional[float] = None,
) -> str:
    """
    Update the known flight slots with fields you have extracted or confirmed.
    
    IMPORTANT: Call this tool EVERY TIME you extract new information from the user message.
    Only include fields you are confident about. Partial updates are fine.
    
    Args:
        month: Month of the flight (1-12)
        day_of_week: Day of week (1=Monday, 7=Sunday)
        scheduled_departure_hour: Scheduled departure hour (0-23)
        scheduled_arrival_hour: Scheduled arrival hour (0-23)
        origin_airport_code: IATA code of origin airport
        destination_airport_code: IATA code of destination airport
        airline_code: IATA code of the airline
        distance: Flight distance in miles
    
    Returns:
        JSON confirming which slots were updated.
    """
    data = {
        "month": month,
        "day_of_week": day_of_week,
        "scheduled_departure_hour": scheduled_departure_hour,
        "scheduled_arrival_hour": scheduled_arrival_hour,
        "origin_airport_code": origin_airport_code,
        "destination_airport_code": destination_airport_code,
        "airline_code": airline_code,
        "distance": distance,
    }
    updated = {k: v for k, v in data.items() if v is not None}
    return json.dumps({"updated_slots": updated})


@tool
def call_predict_api(
    month: int,
    day_of_week: int,
    scheduled_departure_hour: int,
    scheduled_arrival_hour: int,
    origin_airport_code: str,
    destination_airport_code: str,
    airline_code: str,
    distance: float,
) -> str:
    """
    Call the flight delay prediction with complete flight information.
    
    ONLY call this tool when ALL required fields are known and validated.
    Do not call with missing or guessed values.
    
    Args:
        month: Month of the flight (1-12)
        day_of_week: Day of week (1=Monday, 7=Sunday)
        scheduled_departure_hour: Scheduled departure hour (0-23)
        scheduled_arrival_hour: Scheduled arrival hour (0-23)
        origin_airport_code: IATA code of origin airport
        destination_airport_code: IATA code of destination airport
        airline_code: IATA code of the airline
        distance: Flight distance in miles
    
    Returns:
        JSON with prediction results: delay_probability, delayed, delay_minutes.

    Runs inference in-process (via the shared, cached model) rather than making an
    HTTP call back to /predict, since the agent and the API live in the same
    process.
    """
    try:
        result = predict_flight(
            month=month,
            day_of_week=day_of_week,
            scheduled_departure_hour=scheduled_departure_hour,
            scheduled_arrival_hour=scheduled_arrival_hour,
            origin_airport_code=origin_airport_code,
            destination_airport_code=destination_airport_code,
            airline_code=airline_code,
            distance=distance,
        )
        return json.dumps({
            "success": True,
            "delay_probability": result["delay_probability"],
            "delayed": result["delayed"],
            "delay_minutes": result["delay_minutes"],
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Prediction failed: {str(e)}",
        })


# List of all tools for the agent
AGENT_TOOLS = [
    search_airports,
    search_airlines,
    get_current_time,
    calculate_distance,
    update_flight_slots,
    call_predict_api,
]
