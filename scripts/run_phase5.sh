#!/usr/bin/env bash
# Phase 5: profile and tune one GPU. Run on the GPU box after setup_gpu_box.sh says "Ready".
# Every config is the baseline plus ONE change (vllm/configs/tune_*.yaml), run
# through the same Locust load as Phase 4, on the same machine.
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase5.sh
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

PHASE=05_profiling_and_tuning
OUT=results/$PHASE
mkdir -p "$OUT"
export LOAD_PHASE=$PHASE
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=4096
TUNING_STEPS="${TUNING_STEPS:-64,128,192,256,384,512,768}"
LATENCY_STEPS="${LATENCY_STEPS:-1,4,16,64}"
CONFIGS="${CONFIGS:-baseline tune_max_num_seqs_512 tune_max_num_seqs_1024 tune_batched_tokens_8192 tune_no_prefix_caching tune_kv_cache_fp8 tune_fp8_weights}"

start_vllm() {  # $1 = config name; returns non-zero if vLLM doesn't come up
  export VLLM_CONFIG=$1
  docker compose up -d gateway vllm >/dev/null 2>&1
  if ! python3 scripts/smoke_test.py --wait 1200 > /dev/null 2>&1; then
    echo "!! $1: vLLM did not start, see $OUT/failed_$1.log"
    docker compose logs --no-color --no-log-prefix --tail 120 vllm > "$OUT/failed_$1.log" 2>&1
    return 1
  fi
  docker compose logs --no-color --no-log-prefix vllm 2>/dev/null \
    | grep -E "non-default args|Available KV cache memory|GPU KV cache size|Model loading took|[Qq]uantiz|kv_cache_dtype|max_num_seqs|enable_prefix_caching|[Ss]peculative|Application startup complete" \
    | head -20 > "$OUT/vllm_startup_$1.log" || true
}

run_load() {  # $1 = label, $2 = steps, $3 = seconds per step
  mkdir -p "$OUT/$1"
  RUN_LABEL=$1 LOAD_STEPS=$2 STEP_SECONDS=$3 docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust \
    > "$OUT/$1/locust_console.log" 2>&1
  python3 loadtest/analyze_load.py --phase $PHASE --label "$1" --warmup-s 10 | tail -12
}

echo "==> Throughput sweeps: one config at a time ($TUNING_STEPS users, 45 s per step)"
for cfg in $CONFIGS; do
  echo "==> [$cfg]"
  start_vllm "$cfg" && run_load "$cfg" "$TUNING_STEPS" 45
done

echo "==> Latency sweeps at low load: baseline vs speculative decoding ($LATENCY_STEPS users)"
for cfg in baseline tune_ngram_speculative; do
  echo "==> [latency_$cfg]"
  start_vllm "$cfg" && run_load "latency_$cfg" "$LATENCY_STEPS" 45
done

echo "==> Profiling the baseline at 128 users"
if start_vllm profile_baseline; then
  rm -rf results/profiles/* 2>/dev/null || true
  mkdir -p "$OUT/profile_load"
  RUN_LABEL=profile_load LOAD_STEPS=128 STEP_SECONDS=75 docker compose --profile loadtest run --rm locust \
    > "$OUT/profile_load/locust_console.log" 2>&1 &
  LOCUST=$!
  sleep 40
  curl -s -X POST localhost:8000/start_profile > /dev/null
  sleep 15
  curl -s -X POST localhost:8000/stop_profile > /dev/null
  wait $LOCUST
  sleep 10  # let the profiler finish writing
  ls -la results/profiles/
  python3 loadtest/analyze_profile.py results/profiles --label baseline_128_users
fi

echo "==> Comparing every config with the baseline"
python3 loadtest/compare_tuning.py

echo "==> Back to the baseline"
start_vllm baseline || true
echo "==> Done. Results are in $OUT/"
ls "$OUT"
