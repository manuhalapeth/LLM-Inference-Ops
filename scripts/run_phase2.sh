#!/usr/bin/env bash
# Phase 2 measurements, run on the GPU box after setup_gpu_box.sh says "Ready".
# Runs everything twice on the same machine: once with the baseline engine,
# once with Mooncake added as a KV cache tier in CPU memory.
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

OUT=results/02_kv_cache_mooncake
mkdir -p "$OUT"
BASE=(docker compose)
MOONCAKE=(docker compose -f docker-compose.yml -f docker-compose.mooncake.yml)

wait_ready() {
  python3 scripts/smoke_test.py --wait 1800 > /dev/null
}

save_vllm_log() {
  docker compose logs --no-color --no-log-prefix vllm 2>/dev/null \
    | grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Application startup complete|[Mm]ooncake|kv_connector|KVConnector" \
    | grep -v "operation_time" | head -60 > "$OUT/vllm_startup_$1.log" || true
}

echo "==> [1/2] Baseline: vLLM as in Phase 1"
export VLLM_CONFIG=baseline
"${BASE[@]}" up -d --build
wait_ready
save_vllm_log baseline
python3 scripts/trace_request.py --phase 02_kv_cache_mooncake --label baseline --repeats 5
python3 scripts/kv_cache_experiment.py --label baseline

echo "==> [2/2] Mooncake: building vLLM with Mooncake and restarting it"
export VLLM_CONFIG=mooncake_store
"${MOONCAKE[@]}" up -d --build
# NGINX resolves vLLM's address once at startup; the recreated container may have a new one.
"${MOONCAKE[@]}" restart nginx
wait_ready
save_vllm_log mooncake
"${MOONCAKE[@]}" logs --no-color --no-log-prefix mooncake-master 2>/dev/null | tail -40 > "$OUT/mooncake_master.log" || true
python3 scripts/trace_request.py --phase 02_kv_cache_mooncake --label mooncake --repeats 5
python3 scripts/kv_cache_experiment.py --label mooncake

echo "==> Done. Results are in $OUT/"
ls "$OUT"
