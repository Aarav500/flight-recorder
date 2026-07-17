# syntax=docker/dockerfile:1
# Single container: FastAPI serves the built React app from /app/web/dist on :8000.

# ---- Stage 1: build the React app ----
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: python server ----
FROM python:3.12-slim AS app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FLIGHTRECORDER_RUNS=/opt/flight-recorder/runs \
    FLIGHTRECORDER_STATIC=/app/web/dist

COPY pyproject.toml README.md ./
COPY flightrecorder/ ./flightrecorder/
RUN pip install --no-cache-dir ".[server]"

COPY --from=web /web/dist ./web/dist
RUN mkdir -p /opt/flight-recorder/runs
# Bake the real persisted benign run so the honest demo (the false-positive replay) ships in the
# image. Source: HF dataset Aarav500/fr-gentle-artifact, gentle_seed0.jsonl (onset 94, S~37.5).
COPY runs/gentle_seed0.jsonl /opt/flight-recorder/runs/gentle_seed0.jsonl

EXPOSE 8000
# --factory: call create_app(), which reads FLIGHTRECORDER_RUNS / FLIGHTRECORDER_STATIC.
CMD ["uvicorn", "flightrecorder.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
