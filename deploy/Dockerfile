FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copy runtime files only
COPY app.py /app/
COPY templates/ /app/templates/
COPY manage.py /app/

EXPOSE 5001

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5001} app:app --timeout 120 --workers 1 || python -m flask run --host=0.0.0.0 --port=${PORT:-5001}"]
