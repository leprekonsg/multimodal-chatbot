FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY *.py ./
COPY static/ ./static/

# Create image storage directory
RUN mkdir -p /app/images

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "server.py"]
