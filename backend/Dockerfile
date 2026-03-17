FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    inotify-tools \
    && rm -rf /var/lib/apt/lists/*

# FIX #15: Set PYTHONPATH in the image itself.
# docker-compose sets it via `environment:`, but if the image is run directly
# with `docker run`, the env block is absent and module imports break.
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
