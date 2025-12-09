"""
Configuration for the Flight Delay Prediction Agent.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Data paths
DATA_DIR = Path("data")
AIRPORTS_CSV = DATA_DIR / "airports.csv"
AIRLINES_CSV = DATA_DIR / "airlines.csv"

# API configuration
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")

# LLM configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
