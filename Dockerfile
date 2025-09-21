# Dockerfile for OraculAI staging
# Build with: docker build -t oraculai:staging .
# Run with: docker run -e OPENAI_API_KEY -e PINECONE_API_KEY -e PINECONE_INDEX -p 5001:5001 oraculai:staging

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

EXPOSE 5001

# Prefer gunicorn if available, otherwise fall back to flask run for convenience
# Use Render's $PORT if provided, otherwise default to 5001 for local/dev runs.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} app:app --timeout 120 --workers 1 || python -m flask run --host=0.0.0.0 --port=${PORT:-5001}"]
