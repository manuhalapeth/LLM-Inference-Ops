FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi==0.115.6 uvicorn==0.34.0 httpx==0.28.1 && apt-get update -qq && apt-get install -y -qq procps >/dev/null && rm -rf /var/lib/apt/lists/*
COPY server.py client.py run_server.sh /app/
CMD ["sh", "/app/run_server.sh"]
