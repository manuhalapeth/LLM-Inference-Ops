#!/usr/bin/env bash
# Phase 1 measurements, run on the GPU box after setup_gpu_box.sh says "Ready".
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

AGENT_RUNS="${AGENT_RUNS:-3}"

echo "==> Making sure the stack has the Phase 1 services"
docker compose up -d --build
python3 scripts/smoke_test.py --wait 1800 > /dev/null

echo "==> Installing the CrewAI agent"
make -s agent-env

echo "==> Tracing single requests through every layer"
python3 scripts/trace_request.py --repeats 5

echo "==> Running the agent ${AGENT_RUNS} times"
for i in $(seq "$AGENT_RUNS"); do
  .venv/bin/python agent/app.py
  sleep 5
done

echo "==> Done. Results are in results/01_end_to_end/"
ls results/01_end_to_end/
