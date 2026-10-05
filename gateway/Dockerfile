FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model's tokenizer into the image, so the gateway can count prompt
# tokens exactly (for the token budget harness) without any network at startup.
ARG TOKENIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
RUN python -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('${TOKENIZER_MODEL}').save('/srv/tokenizer.json')"

COPY app ./app

EXPOSE 8080
# GATEWAY_WORKERS > 1 runs several processes; their Prometheus metrics are merged
# through files in PROMETHEUS_MULTIPROC_DIR, cleared at every start.
CMD if [ "${GATEWAY_WORKERS:-1}" -gt 1 ]; then \
      export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus && rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR; \
    fi; \
    exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --no-access-log --workers "${GATEWAY_WORKERS:-1}"
