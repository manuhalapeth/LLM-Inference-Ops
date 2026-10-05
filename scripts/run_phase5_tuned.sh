#!/usr/bin/env bash
# Phase 5, part 2: the combined tuned config, plus an eval check that FP8 keeps
# answers correct. Run on the GPU box after run_phase5.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

PHASE=05_profiling_and_tuning
OUT=results/$PHASE
export LOAD_PHASE=$PHASE
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=4096
TUNING_STEPS="${TUNING_STEPS:-64,128,192,256,384,512,768}"

start_vllm() {
  export VLLM_CONFIG=$1
  docker compose up -d gateway vllm >/dev/null 2>&1
  if ! python3 scripts/smoke_test.py --wait 1200 > /dev/null 2>&1; then
    echo "!! $1: vLLM did not start, see $OUT/failed_$1.log"
    docker compose logs --no-color --no-log-prefix --tail 120 vllm > "$OUT/failed_$1.log" 2>&1
    return 1
  fi
  docker compose logs --no-color --no-log-prefix vllm 2>/dev/null \
    | grep -E "non-default args|Available KV cache memory|GPU KV cache size|Model loading took|[Qq]uantiz|kv_cache_dtype|max_num_seqs|Application startup complete" \
    | head -20 > "$OUT/vllm_startup_$1.log" || true
}

echo "==> Evals on the baseline (BF16), for comparison"
start_vllm baseline && python3 evals/run_evals.py --phase $PHASE --label quality_baseline

echo "==> [tuned] FP8 weights + FP8 KV cache + max-num-seqs 512"
if start_vllm tuned; then
  python3 evals/run_evals.py --phase $PHASE --label quality_tuned
  mkdir -p "$OUT/tuned"
  RUN_LABEL=tuned LOAD_STEPS=$TUNING_STEPS STEP_SECONDS=45 docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust \
    > "$OUT/tuned/locust_console.log" 2>&1
  python3 loadtest/analyze_load.py --phase $PHASE --label tuned --warmup-s 10 | tail -12
fi

echo "==> Comparing every config with the baseline"
python3 loadtest/compare_tuning.py

echo "==> Back to the baseline"
start_vllm baseline || true
echo "==> Done."
