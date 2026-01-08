FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY services/vectorizer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Accept model name as build argument
ARG MODEL_NAME=paraphrase-MiniLM-L3-v2

# Pre-download the model to avoid downloading at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Copy application code
COPY services/vectorizer/server.py .

# Run the server
CMD ["python", "server.py"]
