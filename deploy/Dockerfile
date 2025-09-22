# Stage 1: Builder
# This stage installs all dependencies and builds the vector index.
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Build the index. Secrets are passed securely via buildkit secrets.
# This script checks for the existence of each secret before exporting it
# to prevent the build from failing if a secret is not provided.
RUN --mount=type=secret,id=PINECONE_API_KEY \
    --mount=type=secret,id=PINECONE_ENVIRONMENT \
    --mount=type=secret,id=OPENAI_API_KEY \
    /bin/sh -c ' \
        if [ -f /run/secrets/PINECONE_API_KEY ]; then export PINECONE_API_KEY=$(cat /run/secrets/PINECONE_API_KEY); fi; \
        if [ -f /run/secrets/PINECONE_ENVIRONMENT ]; then export PINECONE_ENVIRONMENT=$(cat /run/secrets/PINECONE_ENVIRONMENT); fi; \
        if [ -f /run/secrets/OPENAI_API_KEY ]; then export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY); fi; \
        python manage.py build \
    '

# Stage 2: Final image
# This stage takes the built app and index from the builder stage
# and puts it into a clean, smaller image.
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user
RUN useradd --create-home appuser
USER appuser

# Copy installed python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and the generated index from the builder stage
COPY --from=builder /app /app

EXPOSE 5001

# Run the application
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} app:app --timeout 120 --workers 1"]
