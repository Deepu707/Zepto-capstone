FROM python:3.11-slim

# System deps (build tools for chromadb / tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY docs ./docs
COPY ingest.py graph.py prompt_template.py ./
COPY app ./app

# Pre-build the ChromaDB collection inside the image,
# so the container starts ready to serve.
RUN python ingest.py

# Default to MOCK_LLM=1 (graded baseline)
ENV MOCK_LLM=1
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]