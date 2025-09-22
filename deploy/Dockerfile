FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps for build stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python deps in builder
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copy full repo into builder and run index build (CI will provide secrets)
COPY . /app/

ARG PINECONE_API_KEY
ARG PINECONE_ENVIRONMENT
ARG OPENAI_API_KEY
ENV PINECONE_API_KEY=${PINECONE_API_KEY}
ENV PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN python manage.py build || true

# Final image
FROM python:3.11-slim
WORKDIR /app

# Install runtime deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copy app files and built index from builder
COPY --from=builder /app /app

EXPOSE 5001

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} app:app --timeout 120 --workers 1"]
