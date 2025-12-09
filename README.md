# Flight Delay Prediction — ML model behind a conversational ReAct agent

<p align="center">
  <img src="assets/demo.gif" alt="The chat agent collecting flight details, disambiguating the departure airport, and returning a delay prediction" width="820">
</p>

An end-to-end system that takes a machine-learning model all the way to a
product: a trained model, a service that exposes it, a client that consumes it,
and — the part this project leans into — an **LLM agent** that turns free-form
natural language ("*I'm flying from New York to L.A. with Delta tomorrow
evening*") into a structured prediction request.

This repository is **not** about chasing state-of-the-art accuracy on flight
delays (the pre-departure signal is genuinely weak — more on that below). It is
about a realistic, well-factored path from *notebook* to *deployed,
conversational ML feature*, with an emphasis on the **serving architecture** and
the **agentic / RAG layer**.

---

## Table of contents

- [What it does](#what-it-does)
- [System architecture](#system-architecture)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [1. The ML model (in brief)](#1-the-ml-model-in-brief)
- [2. Serving layer — FastAPI](#2-serving-layer--fastapi)
- [3. Client app — Next.js chat](#3-client-app--nextjs-chat)
- [4. The agentic layer (in depth)](#4-the-agentic-layer-in-depth)
  - [Why an agent at all](#why-an-agent-at-all)
  - [The ReAct loop](#the-react-loop)
  - [Slot-filling as the agent's job](#slot-filling-as-the-agents-job)
  - [The toolset](#the-toolset)
  - [RAG: retrieval as an entity-resolution tool](#rag-retrieval-as-an-entity-resolution-tool)
  - [Guardrails](#guardrails)
- [Running it](#running-it)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Honest limitations & next steps](#honest-limitations--next-steps)

---

## What it does

You describe a flight in plain language. The assistant asks for anything it's
missing, resolves ambiguous references (city names → airport codes, airline
names → IATA codes, "tomorrow evening" → an actual date/time), computes the route
distance, and only then runs the model — returning a delay probability and, if a
delay is expected, an estimated number of minutes, phrased as a natural sentence.

> **User:** *"I'm flying from New York to Los Angeles with Delta at 8pm tomorrow."*
> **Agent:** *"New York has multiple airports (JFK / LGA / EWR) — which one are you leaving from? And what's the scheduled arrival hour?"*
> *(...once all fields are known...)*
> **Agent:** *"Your Delta flight from JFK to LAX departing around 8pm has roughly a 34% chance of arriving late, likely by about 25 minutes if it is. This is a statistical estimate based on historical data."*

---

## System architecture

```
┌──────────────────────┐        POST /agent/chat          ┌───────────────────────────────────────┐
│   Next.js chat UI     │  ───────────────────────────▶    │            FastAPI service            │
│  (Flight Deck Console)│  ◀───────────────────────────    │             (app/main.py)             │
│  session_id + message │        { reply, slots,           │                                       │
└──────────────────────┘          prediction }            │   /health   /model/performance        │
                                                           │   /predict  ── direct model inference │
                                                           │   /agent/chat ── mounts the agent ────┼──┐
                                                           └───────────────────────────────────────┘  │
                                                                                                       ▼
                                                              ┌────────────────────────────────────────────────┐
                                                              │              Agent (assistant/)                  │
                                                              │                                                  │
                                                              │   ReAct loop  ⇄  Gemini 2.5 Flash (LangChain)    │
                                                              │        │                                         │
                                                              │        ▼  tool calls                             │
                                                              │   ┌───────────────────────────────────────┐     │
                                                              │   │ search_airports / search_airlines  ────┼──▶ FAISS (RAG)
                                                              │   │ get_current_time                      │     │  in-memory index over
                                                              │   │ calculate_distance  (Haversine)       │     │  airports.csv / airlines.csv
                                                              │   │ update_flight_slots (slot memory)     │     │
                                                              │   │ call_predict_api  ─────────────────────┼──▶ TwoStageDelayModel
                                                              │   └───────────────────────────────────────┘     │  (ml_pipeline, LightGBM)
                                                              └────────────────────────────────────────────────┘
```

Everything runs in a single Python process: the agent calls the model
**in-process** (no HTTP self-request), while `/predict` remains available as a
standalone REST endpoint for non-conversational consumers.

---

## Tech stack

| Layer | Choice |
|---|---|
| Model | **LightGBM** (two-stage: classifier + regressor), scikit-learn utilities |
| Serving | **FastAPI** + Uvicorn, Pydantic schemas |
| Agent | **LangChain** tool-calling over **Google Gemini 2.5 Flash** |
| Retrieval | **FAISS** (in-memory) + `sentence-transformers` multilingual embeddings |
| Client | **Next.js 16 / React 19 / Tailwind 4** |
| Packaging | Docker + docker-compose |

---

## Repository layout

```
ml_pipeline/     Training + inference package (data → features → model → artifacts)
app/             FastAPI service: /predict, /health, /model/performance, mounts the agent
assistant/       The agent: ReAct loop, tools, FAISS RAG, prompt, chat router
frontend/        Next.js chat UI (the "Flight Deck Console")
models/          Serialized artifacts (two_stage_delay_model.joblib + model_metadata.json)
data/            airports.csv, airlines.csv (flights.csv is downloaded separately)
analysis.ipynb   EDA + experimentation notebook
Dockerfile       API + agent image
docker-compose.yml
```

---

## 1. The ML model (in brief)

The predictive core is intentionally small; the interesting part is that the
**exact same feature engineering runs offline (training) and online
(inference)** so there is no train/serve skew.

**Dataset.** [US DOT Flight Delays (Kaggle)](https://www.kaggle.com/datasets/usdot/flight-delays) — ~5.8M US domestic flights (2015).

**The framing — two stages.** Delay is both a *classification* and a *regression*
problem, so the model (`ml_pipeline/models.py`, `TwoStageDelayModel`) is split:

1. **Stage 1 — classifier** (`LGBMClassifier`): `P(arrival delay > 15 min)`.
2. **Stage 2 — regressor** (`LGBMRegressor`): *how many* minutes, trained **only
   on genuinely delayed flights** and applied only to rows Stage 1 flags as late.
   The regression target is clipped to `[15, 180]` minutes to tame outliers.

This avoids a single regressor being dragged toward zero by the on-time majority.

**Leakage discipline.** Anything only knowable *after* departure/arrival
(`DEPARTURE_DELAY`, `TAXI_OUT`, `AIR_TIME`, the post-hoc delay-reason columns, …)
is dropped up front (`config.LEAKAGE_COLS`). Cancelled and diverted flights are
removed.

**Feature engineering** (`ml_pipeline/features.py`, `data_loading.py`) — all
reproduced at inference time:
- *Time*: departure/arrival hour, time-of-day slots, peak-hour and red-eye flags.
- *Calendar*: weekend / summer / holiday-season flags.
- *Cyclic*: sin/cos encodings for month, day-of-week and hour (so "23:00" and
  "00:00" are neighbours).
- *Route*: `ORIGIN-DESTINATION` pair and a distance bucket.
- *Hubs*: airports above the 90th traffic percentile (**fit on train only**).
- *Historical delay stats*: mean delay and delay-rate per route / airline /
  hour (**fit on train only**, mapped onto both splits; unseen keys fall back to
  global averages).

**No temporal leakage.** The split is **temporal** — Jan–Oct train, Nov–Dec test
(`train_test_split_by_month`) — and every fitted statistic (hubs, historical
aggregates, target-encoding maps, class imbalance ratio) is computed **strictly
on the training months** and then applied to the test months.

**Encoding** (`ml_pipeline/encoding.py`): one-hot for low-cardinality categories,
**target encoding** for high-cardinality ones (`ROUTE`, `ORIGIN_AIRPORT`,
`DESTINATION_AIRPORT`, `AIRLINE`). The encoding maps are serialized so inference
reuses the exact same mapping.

**Imbalance** is handled with `class_weight` (the train imbalance ratio), and the
classification threshold is tuned to maximise F1 on a precision/recall curve
rather than left at the naïve 0.5.

**Reported metrics** (`models/model_metadata.json`, Nov–Dec hold-out):

| Task | Metric | Value |
|---|---|---|
| Classification | ROC-AUC | ~0.61 |
| Classification | Recall (delayed) | ~0.58 |
| Classification | Precision (delayed) | ~0.22 |
| Regression (delayed only) | MAE | ~39.8 min (baseline 43.0) |

**Read this honestly:** with only *pre-departure schedule/route metadata*, delay
is weakly predictable — the high-signal drivers (real-time weather, upstream
aircraft, airport congestion) simply aren't in the dataset. The model beats the
mean baseline but is far from an oracle, and the API surfaces that uncertainty as
a *probability*, which the agent communicates as such. The engineering value is
the leak-free, train/serve-consistent pipeline, not the ceiling of the score.

**Artifacts & inference reconstruction.** `serialization.py` writes the model as
joblib plus a JSON metadata sidecar holding `feature_stats`, per-stage
`encoding_info`, and the exact feature-column order. At request time,
`inference.build_features_from_raw` takes a single raw flight row, replays every
feature function, applies the saved encoding maps, then pads any missing column
to 0 and reorders to the training layout — guaranteeing the matrix the model sees
online is identical in shape to the one it trained on.

---

## 2. Serving layer — FastAPI

`app/main.py` loads the model once on startup (via a lifespan handler) and
exposes four endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Direct inference from 8 structured fields |
| `POST` | `/agent/chat` | Conversational entrypoint (mounts the agent router) |
| `GET`  | `/health` | Liveness + whether the model loaded |
| `GET`  | `/model/performance` | Core classification/regression metrics from metadata |

Inputs/outputs are validated by Pydantic schemas (`app/schemas.py`), errors map
to proper HTTP status codes, and CORS is opened for the local frontend origin.
The `/predict` endpoint and the agent share the **same** `build_features_from_raw`
→ `model.predict` path, so a REST caller and the chatbot get identical results.

---

## 3. Client app — Next.js chat

`frontend/` is a single-screen chat UI ("Flight Deck Console", Next.js 16 /
React 19 / Tailwind 4). It:

- generates and persists a `session_id` in `localStorage` so the backend can keep
  per-conversation slot state across turns;
- POSTs `{ message, session_id }` to `/agent/chat` and renders the `reply`;
- adds a lightweight **client-side typewriter effect** (variable per-character
  delay, longer pauses on punctuation) to simulate streaming — the backend
  currently replies in a single shot, so this is presentation-only.

It is deliberately thin: all intelligence lives server-side in the agent.

---

## 4. The agentic layer (in depth)

This is the heart of the project. Code lives in `assistant/`:
`agent.py` (loop + state), `tools.py` (tools), `rag.py` (FAISS retrieval),
`prompts/system_default.md` (system prompt), `router.py` (FastAPI route).

### Why an agent at all

The model needs **8 precise structured inputs**: month, day-of-week, departure
hour, arrival hour, origin IATA, destination IATA, airline IATA, and distance in
miles. Real users don't speak in IATA codes and day-of-week numbers — they say
"*next Friday evening from Milan to JFK with American*". Something has to bridge
messy language and the strict schema. That bridge is the agent, and the design
principle throughout is: **the LLM reasons and orchestrates, but every fact is
produced by a deterministic tool — the model is never allowed to guess a value.**

### The ReAct loop

The agent implements the **ReAct** pattern (*Reason → Act → Observe*, repeat) as
a hand-rolled tool-calling loop in `run_agent_turn` (`assistant/agent.py`), rather
than delegating to a framework's opaque agent executor. Each user turn runs up to
**8 iterations** of:

1. **Reason** — the LLM (Gemini 2.5 Flash via LangChain, `temperature=0.3`) is
   invoked with the system prompt, the current slot state, and the conversation
   history. It decides what to do next.
2. **Act** — if it emits tool calls, each is dispatched to the matching Python
   function. (It may call several tools in one step, e.g. resolve two airports
   and an airline together.)
3. **Observe** — each tool's JSON result is appended to the message list as a
   `ToolMessage`, becoming context for the next reasoning step.
4. The loop ends the moment the LLM returns a plain-text answer instead of a tool
   call (or the 8-iteration cap trips the fallback).

Why hand-rolled? **Control.** The loop is where the hard guardrails live — the
prediction tool is gated on complete state, slots are extracted deterministically
from tool traffic, and a rich "prediction context" is injected right before the
final answer so the model narrates *only* the numbers the tools produced. That
level of interception is awkward to bolt onto a black-box agent runner.

### Slot-filling as the agent's job

Conversation state is a `SessionState` per `session_id` (in `agent.py`), holding
the message history, a `FlightSlots` object, and the last prediction. The 8
fields are the "slots"; the whole turn is a **slot-filling** exercise.

Rather than trusting the LLM to track state in prose, slots are updated
**deterministically** by inspecting tool traffic (`_update_slots_from_tool_calls`):
values that flow through `update_flight_slots`, `calculate_distance` or
`call_predict_api`, and results returned inside `ToolMessage`s, are written
straight into `FlightSlots`. The current slot JSON is re-injected into the prompt
every turn, so the model always sees ground truth and never re-asks for something
already known. `FlightSlots.is_complete()` / `missing_fields()` drive both the
prediction gate and the clarification questions.

### The toolset

Six tools are bound to the model (`assistant/tools.py`):

| Tool | What it does |
|---|---|
| `search_airports(query)` | FAISS semantic search over airports → candidate IATA codes |
| `search_airlines(query)` | FAISS semantic search over airlines → IATA code |
| `get_current_time()` | Grounds relative dates ("tomorrow", "next Monday") in a real timestamp |
| `calculate_distance(origin, dest)` | Great-circle miles between two airports (Haversine) |
| `update_flight_slots(...)` | Writes confirmed/extracted values into slot memory |
| `call_predict_api(...)` | Runs the model on the completed 8-field payload |

`call_predict_api` deliberately does **not** make an HTTP call back to `/predict`
(that would deadlock a single-worker server calling itself). Instead
`_predict_directly` loads the model and runs the shared inference path in-process.
The `/predict` REST endpoint still exists for external, non-conversational
callers — same code underneath.

### RAG: retrieval as an entity-resolution tool

The retrieval layer (`assistant/rag.py`) is **RAG applied to entity resolution**,
not document Q&A. The problem it solves: an LLM asked for "the IATA code for
Linate" will happily *hallucinate* one. Grounding the answer in an authoritative
lookup is the fix.

At startup the app builds two **in-memory FAISS** indexes — one over
`airports.csv`, one over `airlines.csv` — embedding each row (e.g.
`"Airport JFK: John F Kennedy International, New York, USA"`) with the
multilingual model `paraphrase-multilingual-MiniLM-L12-v2`. When the agent calls
`search_airports("aeroporto di Milano")`, the query is embedded and matched by
cosine similarity (`k=5`), returning real codes and metadata.

The principles this illustrates:

- **Retrieval as a tool, not a preamble.** Classic RAG prepends retrieved chunks
  to every prompt. Here retrieval is *agent-invoked, on demand* — the model
  fetches only when it actually needs to resolve an entity. This is the
  "RAG-as-tool" / tool-augmented pattern and it keeps prompts lean.
- **Semantic + multilingual matching** beats string matching for fuzzy, misspelt
  or non-English inputs ("Milano", "NYC", "American" → AA).
- **Grounding kills hallucination.** Codes come from the dataset, never from the
  model's parametric memory; ambiguity ("New York → JFK/LGA/EWR") is surfaced back
  to the user instead of silently guessed.
- **Same source of truth for RAG and features.** The airports the agent resolves
  and the coordinates the distance tool uses come from the same CSVs the model was
  trained against.

### Guardrails

- **Hard prediction gate:** `call_predict_api` is blocked until all 8 slots are
  filled — enforced in the loop, independent of the prompt, so even a
  mis-behaving model can't predict on partial data.
- **No guessing:** the system prompt forbids inventing field values and requires
  asking the user to disambiguate (e.g. multi-airport cities).
- **Bounded work:** the 8-iteration cap prevents runaway tool loops.
- **Graceful fallbacks:** if the loop ends without a clean answer, the agent
  returns either the missing-fields prompt or a minimal prediction summary rather
  than an error.
- **Scope + style:** answers stay on-topic (flight delays), in the user's
  language, plain-text, and always close with a note that the number is a
  statistical estimate.

---

## Running it

### Prerequisites

- Python **3.11**
- Node.js (for the frontend)
- A **Google Gemini API key** ([get one here](https://aistudio.google.com/app/apikey)) for the chat agent
- To *retrain*: the Kaggle `flights.csv` placed in `data/` (a trained model is already committed, so this is optional)

### Environment

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY
```

The `/predict`, `/health` and `/model/performance` endpoints work without a key;
only `/agent/chat` requires `GOOGLE_API_KEY`.

### Backend — Docker (recommended)

```bash
docker compose up --build
# API at http://localhost:8000  (try http://localhost:8000/health)
```

### Backend — local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-api.txt                 # minimal runtime deps (API + agent)
python -m uvicorn app.main:app --reload --port 8000
```

Use `requirements.txt` (the full set) if you also want to run the notebook or
retrain the model.

### (Optional) retrain

```bash
# place data/flights.csv first, then:
pip install -r requirements.txt
python -m ml_pipeline.train --data-dir ./data --output-dir ./models
```

### Frontend

```bash
cd frontend
npm install
# optional: echo 'NEXT_PUBLIC_CHAT_ENDPOINT=http://localhost:8000/agent/chat' > .env.local
npm run dev
# UI at http://localhost:3000
```

---

## API reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "month": 12, "day_of_week": 5,
    "scheduled_departure_hour": 20, "scheduled_arrival_hour": 23,
    "origin_airport_code": "JFK", "destination_airport_code": "MHT",
    "airline_code": "AA", "distance": 199.2
  }'
```

```json
{ "delay_probability": 0.34, "delayed": false, "delay_minutes": 0.0 }
```

### `POST /agent/chat`

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Flight from JFK to MHT next Friday with American, 8pm to 11pm", "session_id": "demo"}'
```

Returns `{ reply, slots, prediction }`. Reuse the same `session_id` across turns
to keep the slot-filling conversation going.

---

## Configuration

| Variable | Where | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | `.env` | Gemini key for the agent (required for `/agent/chat`) |
| `PREDICT_API_URL` | `.env` | Kept for completeness; agent predicts in-process by default |
| `NEXT_PUBLIC_CHAT_ENDPOINT` | `frontend/.env.local` | Agent endpoint the UI calls (default `http://127.0.0.1:8000/agent/chat`) |

Model behaviour (delay threshold, max delay, hub quantile, train/test months,
hyperparameters) lives in `ml_pipeline/config.py`; LLM model/temperature in
`assistant/config.py`.

---

## Honest limitations & next steps

- **Prediction ceiling.** Pre-departure metadata is inherently weak signal;
  meaningful gains need weather, upstream-aircraft and congestion feeds.
- **Shipped model is a demo artifact.** The committed `models/` files exist so
  the API runs out of the box; for full results, retrain from the Kaggle
  `flights.csv` with `python -m ml_pipeline.train`, which fits and serializes the
  complete feature stats and encoding maps used at inference.
- **Session state is in-memory.** Sessions live in a process-local dict — fine
  for a demo, but a real deployment would use Redis (and horizontal scaling would
  otherwise split a conversation across workers).
- **FAISS is rebuilt at startup** and covers only airports/airlines; a persisted
  index would cut cold-start time.
- **Non-streaming responses.** The agent replies in one shot (the UI fakes
  streaming); streaming tool-call and token events would improve perceived
  latency.
- **Observability.** Structured logging and conversation tracing would help with
  prompt iteration and debugging.
- **Explainability.** Surfacing per-prediction feature importance in the API and
  letting the agent explain the top delay drivers would make answers more useful.

---

Created and finalized on 9 December 2025. Licensed under [MIT](LICENSE).
