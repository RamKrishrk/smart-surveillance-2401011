# Use lightweight Python base image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

    # Set working directory inside container
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

    # Copy application source code
COPY . .


# Expose Flask application port
EXPOSE 5000


# Start Flask application
CMD ["python", "app.py"]
