# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml .
COPY uv.lock .

# Install uv and dependencies
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY .env .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the FastAPI app
CMD ["uv", "run", "python", "src/05_fastapi/chat_api.py"]