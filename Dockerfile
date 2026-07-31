# SNIM — Smart Network Intrusion Monitor
# Dockerfile for deployment on Render, Railway, or any OCI-compatible host.
# Also compatible with Hugging Face Spaces (sdk: docker).

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the data directory exists (traffic_log.db will be created at runtime)
RUN mkdir -p data

# Expose the port Streamlit will listen on
EXPOSE 7860

# Health-check so orchestrators know when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
#CMD ["streamlit", "run", "app.py", \
#     "--server.port=7860", \
#     "--server.address=0.0.0.0", \
#     "--server.headless=true", \
#     "--browser.gatherUsageStats=false"]

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860} --server.headless=true --browser.gatherUsageStats=false