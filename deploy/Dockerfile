# syntax = docker/dockerfile:1.7

# FORCE REBUILD: 2025-09-24T03:10:00Z - Apply memory optimizations
# Stage 1: Builder
# This stage installs all dependencies and builds the vector index.
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only the necessary files for dependency installation and build script
COPY deploy/requirements.txt requirements.txt
COPY manage.py app.py ./

# Install dependencies with pip optimizations
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy the rest of the application source code (e.g., ./sources)
COPY . .

# Run the build process with secrets
RUN --mount=type=secret,id=PINECONE_API_KEY,required=false \
        --mount=type=secret,id=PINECONE_ENVIRONMENT,required=false \
        --mount=type=secret,id=OPENAI_API_KEY,required=false \
        sh -lc "set -e; \
          if [ -s /run/secrets/PINECONE_API_KEY ] && \
             [ -s /run/secrets/PINECONE_ENVIRONMENT ] && \
             [ -s /run/secrets/OPENAI_API_KEY ]; then \
            export PINECONE_API_KEY=\"$(cat /run/secrets/PINECONE_API_KEY)\"; \
            export PINECONE_ENVIRONMENT=\"$(cat /run/secrets/PINECONE_ENVIRONMENT)\"; \
            export OPENAI_API_KEY=\"$(cat /run/secrets/OPENAI_API_KEY)\"; \
            python manage.py build; \
          else \
            echo '[Index] Pinecone/OpenAI secrets missing; skipping baked index.'; \
          fi"

# Stage 2: Final image
# This stage takes the built app and index from the builder stage
# and puts it into a clean, smaller image.
FROM python:3.11-slim

# Install only essential packages to reduce warnings
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Create a non-root user
RUN useradd --create-home --uid 1000 appuser

# Copy only necessary binaries and python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/python* /usr/local/bin/

# Copy application code and the generated index from the builder stage
COPY --from=builder /app /app

# Ensure app directory is owned by non-root user, then drop privileges
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

# Run the application
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} deploy.app:app --workers 1 --timeout 120 --log-level info --access-logfile - --error-logfile -"]
