# Use lightweight official Python runtime
FROM python:3.14-slim

WORKDIR /app

# Install system utilities and compiler components
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy frozen dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application source code, pre-trained model, and sensor cache
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY gunicorn.conf.py .

# Define container runtime environment variables
ENV PYTHONUTF8=1
ENV PORT=5000

# Expose production port
EXPOSE 5000

# Run the Flask app with the Gunicorn production server
CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.api.app:app"]
