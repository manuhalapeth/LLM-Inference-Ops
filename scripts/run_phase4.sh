#!/usr/bin/env bash
# Phase 4: break one GPU. Run on the GPU box after setup_gpu_box.sh says "Ready".
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

OUT=results/04_breaking_one_gpu
mkdir -p "$OUT/baseline"

# Measure the GPU, not the per user limits: rate limits and load shedding are
# raised out of the way. Every other harness (token budgets, injection) stays on.
export VLLM_CONFIG=baseline RUN_LABEL=baseline
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=4096

echo "==> Stack: baseline engine, gateway with load test limits"
docker compose up -d --build
python3 scripts/smoke_test.py --wait 1800 > /dev/null
docker compose logs --no-color --no-log-prefix vllm 2>/dev/null \
  | grep -E "Available KV cache memory|GPU KV cache size|max_num_seqs|Application startup complete" | head -10 > "$OUT/vllm_startup.log" || true

echo "==> Sampling container CPU and memory every 10 s"
( while true; do
    docker stats --no-stream --format '{"t": '"$(date +%s)"', "name": "{{.Name}}", "cpu": "{{.CPUPerc}}", "mem": "{{.MemUsage}}"}'
    sleep 10
  done ) > "$OUT/baseline/container_stats.jsonl" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

echo "==> Locust: 1 to 512 users, 60 s per step (~13 min). Live UI on :8089"
docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust

echo "==> Per step analysis"
python3 loadtest/analyze_load.py --label baseline

echo "==> Concurrent CrewAI crews"
make -s agent-env
.venv/bin/python scripts/agent_load.py

kill $SAMPLER 2>/dev/null || true
echo "==> Done. Results are in $OUT/"
ls "$OUT" "$OUT/baseline"
