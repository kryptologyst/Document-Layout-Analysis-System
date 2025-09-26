# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/sample_data

# Generate sample data
RUN python generate_sample_data.py

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///data/document_analysis.db
ENV DEFAULT_MODEL=layoutparser
ENV USE_GPU=false

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Document Layout Analysis System..."\n\
echo "API Server: http://localhost:8000"\n\
echo "Web Interface: http://localhost:8501"\n\
echo ""\n\
# Start API server in background\n\
python api.py &\n\
API_PID=$!\n\
\n\
# Start Streamlit app\n\
streamlit run app.py --server.port 8501 --server.address 0.0.0.0\n\
\n\
# Wait for API server\n\
wait $API_PID' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/start.sh"]
