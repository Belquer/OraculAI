# syntax = docker/dockerfile:1.7

# Stage 1: Builder
# This stage installs all dependencies and builds the vector index.
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only the necessary files for dependency installation and build script
COPY requirements.txt manage.py app.py ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code (e.g., ./sources)
COPY . .

# Run the build process with secrets
RUN --mount=type=secret,id=PINECONE_API_KEY,required=false \
        --mount=type=secret,id=PINECONE_ENVIRONMENT,required=false \
        --mount=type=secret,id=OPENAI_API_KEY,required=false \
        bash -lc "set -e; \
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

WORKDIR /app

# Create a non-root user
RUN useradd --create-home --uid 1000 appuser

# Copy installed python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and the generated index from the builder stage
COPY --from=builder /app /app

# Ensure app directory is owned by non-root user, then drop privileges
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

# Run the application
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} app:app --timeout 120 --workers 1"]
