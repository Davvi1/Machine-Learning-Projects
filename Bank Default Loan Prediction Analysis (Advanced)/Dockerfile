# Use an official lightweight Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# --- ✅ Add system dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first (helps with caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
