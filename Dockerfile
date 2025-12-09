FROM python:3.11-slim-bookworm

# Install system dependencies required by LightGBM and native Python wheels
# - libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY ml_pipeline/ ./ml_pipeline/
COPY app/ ./app/
COPY models/ ./models/
COPY assistant/ ./assistant/


COPY data/airlines.csv ./data/airlines.csv
COPY data/airports.csv ./data/airports.csv

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
