FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY app/ app/
COPY .env.example .env

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create storage directories
RUN mkdir -p storage/videos storage/frames storage/chroma

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/videos')"

# Run both services using a startup script
CMD ["bash", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &\
     sleep 5 && \
     streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0"]
