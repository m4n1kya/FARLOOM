FROM python:3.12-slim

WORKDIR /app

# Install build essentials if needed for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy repository
COPY . /app

# Install package and dependencies
RUN pip install --no-cache-dir -e .

# Hugging Face Spaces defaults to port 7860, Render uses $PORT
ENV PORT=7860
EXPOSE 7860

# Start FARLOOM knowledge engine
CMD ["sh", "-c", "python -m uvicorn graphrag_knowledge_engine.api:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
