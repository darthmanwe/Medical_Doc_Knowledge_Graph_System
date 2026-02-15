FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model at build time (optional, can be done at runtime)
RUN python -m spacy download en_core_web_sm 2>/dev/null || true

COPY app/ ./app/
COPY Task_Files/ ./Task_Files/

# Default: run uvicorn (overridden by docker-compose for tests)
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
