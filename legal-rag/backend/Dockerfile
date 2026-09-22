# Legal RAG API image.
# By default this runs fully offline (mock LLM + mock embedder). To use real
# models, uncomment the optional deps in requirements.txt, rebuild, and set the
# USE_REAL_* env vars at runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code and the seed corpora.
COPY backend/ /app/backend/
COPY data/ /app/data/

# The app resolves DATA_DIR relative to the repo root by default; make it
# explicit for the container layout.
ENV DATA_DIR=/app/data

WORKDIR /app/backend
EXPOSE 8000

# Uvicorn serves the FastAPI app; startup builds the in-memory indexes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
