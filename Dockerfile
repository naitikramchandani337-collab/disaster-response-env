FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app/       ./app/
COPY server/    ./server/
COPY openenv.yaml .
COPY inference.py .
COPY baseline_results.json .
COPY README.md .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV ENV_BASE_URL="http://localhost:7860"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
