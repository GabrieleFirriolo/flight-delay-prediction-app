"""
RAG module for airports and airlines data.

Provides FAISS-based retrieval and distance calculation.
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from assistant.config import AIRPORTS_CSV, AIRLINES_CSV


@lru_cache(maxsize=1)
def load_airports() -> pd.DataFrame:
    """Load airports CSV into DataFrame."""
    return pd.read_csv(AIRPORTS_CSV)


@lru_cache(maxsize=1)
def load_airlines() -> pd.DataFrame:
    """Load airlines CSV into DataFrame."""
    return pd.read_csv(AIRLINES_CSV)


def get_airport_by_code(code: str) -> Optional[Dict]:
    """Get airport info by IATA code."""
    airports = load_airports()
    match = airports[airports["IATA_CODE"] == code.upper()]
    if len(match) == 0:
        return None
    row = match.iloc[0]
    return {
        "code": row["IATA_CODE"],
        "name": row["AIRPORT"],
        "city": row["CITY"],
        "state": row.get("STATE", ""),
        "country": row["COUNTRY"],
        "latitude": row["LATITUDE"],
        "longitude": row["LONGITUDE"],
    }


def get_airline_by_code(code: str) -> Optional[Dict]:
    """Get airline info by IATA code."""
    airlines = load_airlines()
    match = airlines[airlines["IATA_CODE"] == code.upper()]
    if len(match) == 0:
        return None
    row = match.iloc[0]
    return {
        "code": row["IATA_CODE"],
        "name": row["AIRLINE"],
    }


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    R = 3958.8  # Earth radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def estimate_distance(origin_code: str, dest_code: str) -> Optional[float]:
    """
    Estimate flight distance in miles between two airports.
    
    Returns None if either airport is not found.
    """
    origin = get_airport_by_code(origin_code)
    dest = get_airport_by_code(dest_code)
    
    if origin is None or dest is None:
        return None
    
    return _haversine_miles(
        origin["latitude"], origin["longitude"],
        dest["latitude"], dest["longitude"]
    )


def _create_airport_documents() -> List[Document]:
    """Create documents for airport FAISS index."""
    airports = load_airports()
    docs = []
    
    for _, row in airports.iterrows():
        text = f"Airport {row['IATA_CODE']}: {row['AIRPORT']}, {row['CITY']}, {row['COUNTRY']}"
        metadata = {
            "type": "airport",
            "code": row["IATA_CODE"],
            "name": row["AIRPORT"],
            "city": row["CITY"],
            "country": row["COUNTRY"],
        }
        docs.append(Document(page_content=text, metadata=metadata))
    
    return docs


def _create_airline_documents() -> List[Document]:
    """Create documents for airline FAISS index."""
    airlines = load_airlines()
    docs = []
    
    for _, row in airlines.iterrows():
        text = f"Airline {row['IATA_CODE']}: {row['AIRLINE']}"
        metadata = {
            "type": "airline",
            "code": row["IATA_CODE"],
            "name": row["AIRLINE"],
        }
        docs.append(Document(page_content=text, metadata=metadata))
    
    return docs


_airports_index: Optional[FAISS] = None
_airlines_index: Optional[FAISS] = None


@lru_cache(maxsize=1)
def _get_embeddings():
    """Get embeddings model (loaded once and shared across indices)."""
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )


def get_airports_index() -> FAISS:
    """Get or create FAISS index for airports."""
    global _airports_index
    if _airports_index is None:
        docs = _create_airport_documents()
        embeddings = _get_embeddings()
        _airports_index = FAISS.from_documents(docs, embeddings)
    return _airports_index


def get_airlines_index() -> FAISS:
    """Get or create FAISS index for airlines."""
    global _airlines_index
    if _airlines_index is None:
        docs = _create_airline_documents()
        embeddings = _get_embeddings()
        _airlines_index = FAISS.from_documents(docs, embeddings)
    return _airlines_index


def search_airports(query: str, k: int = 5) -> List[Dict]:
    """
    Search airports by natural language query.
    
    Returns list of matching airports with code, name, city, country.
    """
    index = get_airports_index()
    results = index.similarity_search(query, k=k)
    
    return [
        {
            "code": doc.metadata["code"],
            "name": doc.metadata["name"],
            "city": doc.metadata["city"],
            "country": doc.metadata["country"],
        }
        for doc in results
    ]


def search_airlines(query: str, k: int = 5) -> List[Dict]:
    """
    Search airlines by natural language query.
    
    Returns list of matching airlines with code and name.
    """
    index = get_airlines_index()
    results = index.similarity_search(query, k=k)
    
    return [
        {
            "code": doc.metadata["code"],
            "name": doc.metadata["name"],
        }
        for doc in results
    ]
