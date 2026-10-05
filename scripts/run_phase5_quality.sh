#!/usr/bin/env bash
# Phase 5, part 3: which FP8 change breaks answer quality? Evals on each one alone.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PHASE=05_profiling_and_tuning
OUT=results/$PHASE
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=4096
for cfg in tune_fp8_weights tune_kv_cache_fp8; do
  echo "==> [quality_$cfg]"
  export VLLM_CONFIG=$cfg
  docker compose up -d gateway vllm >/dev/null 2>&1
  if python3 scripts/smoke_test.py --wait 1200 > /dev/null 2>&1; then
    python3 evals/run_evals.py --phase $PHASE --label "quality_$cfg"
  else
    echo "!! $cfg did not start"
  fi
done
echo "==> Back to the baseline"
export VLLM_CONFIG=baseline
docker compose up -d gateway vllm >/dev/null 2>&1
python3 scripts/smoke_test.py --wait 1200 > /dev/null 2>&1
echo "==> Done."
