#!/usr/bin/env bash
# Phase 3 measurements, run on the GPU box after setup_gpu_box.sh says "Ready".
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

OUT=results/03_harnesses_and_traces
mkdir -p "$OUT"
export VLLM_CONFIG=baseline_tracing
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 OTEL_SERVICE_NAME=agent

echo "==> Stack with tracing on (vLLM sends spans to Jaeger)"
docker compose up -d --build
python3 scripts/smoke_test.py --wait 1800 > /dev/null
docker compose logs --no-color --no-log-prefix vllm 2>/dev/null \
  | grep -E "Available KV cache memory|GPU KV cache size|otlp|[Tt]racing|Application startup complete" | head -20 > "$OUT/vllm_startup.log" || true

echo "==> Installing the CrewAI agent"
make -s agent-env

echo "==> Evals with harnesses on"
python3 evals/run_evals.py --label harnesses_on

echo "==> Failure drills"
python3 scripts/failure_drills.py

echo "==> One traced agent run"
.venv/bin/python agent/app.py --phase 03_harnesses_and_traces

echo "==> Evals with harnesses off (same requests, nothing stops them)"
HARNESSES_ENABLED=false docker compose up -d gateway
sleep 5
python3 scripts/smoke_test.py --wait 120 > /dev/null
python3 evals/run_evals.py --label harnesses_off
docker compose up -d gateway
sleep 5

echo "==> Saving traces and gateway logs"
sleep 5  # let the last spans reach Jaeger
python3 scripts/export_traces.py
docker compose logs --no-color --no-log-prefix gateway 2>/dev/null | grep -E '"event": "(rejected|upstream_error)"|"outcome": "(timeout|client_disconnected)"' > "$OUT/gateway_failures.log" || true

echo "==> Done. Results are in $OUT/"
ls "$OUT"
