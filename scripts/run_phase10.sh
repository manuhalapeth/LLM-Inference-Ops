#!/usr/bin/env bash
# Phase 10: four GPUs, and does routing by cache beat routing by load? Needs 4 GPUs.
# Run on the GPU box after setup_gpu_box.sh says "Ready":
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase10.sh
#
# [1] Scaling on vLLM v0.34.0: 1, 2, 4 copies, least connections, the Phase 4 mix
# [2] Routing at 4 copies, on conversations (each user's prefix grows and repeats):
#     NGINX round robin, least connections, hash per user, then vLLM's router
#     with round robin, power of two, consistent hash, cache aware and KV aware
# [3] cache aware with a tighter and a looser balance threshold
# vLLM servers stay up between the routing runs (only the router and NGINX
# change), so every routing run sees the same four servers.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

PHASE="${PHASE10_DIR:-10_routing_four_gpus}"
OUT=results/$PHASE
mkdir -p "$OUT"
export GATEWAY_WORKERS="${GATEWAY_WORKERS:-16}" LOCUST_PROCESSES="${LOCUST_PROCESSES:-16}"
TOPO_EXTRA=(--config tune_fp8_weights_kv_events)
. scripts/lib_compose.sh
use_vllm v0.34.0
N=$(( GPUS >= 4 ? 4 : GPUS ))
echo "==> $GPUS GPUs, routing runs on $N copies"

BASE_STEPS="${BASE_STEPS:-64 128 192 256 384 512}"   # users per server, as in Phases 5 and 6
SESSION_STEPS="${SESSION_STEPS:-$((64 * N)),$((128 * N)),$((192 * N)),$((256 * N))}"
SCALES="${SCALES:-1 2 4}"
NGINX_LBS="${NGINX_LBS:-round_robin least_conn user_hash}"
ROUTER_POLICIES="${ROUTER_POLICIES:-round_robin power_of_two consistent_hash cache_aware kv_aware}"
RUN_SCALING="${RUN_SCALING:-1}" RUN_THRESHOLDS="${RUN_THRESHOLDS:-1}"
scaled_steps() { local out=""; for u in $BASE_STEPS; do out="$out,$((u * $1))"; done; echo "${out#,}"; }

start_sampler
nvidia-smi topo -m > "$OUT/nvidia_smi_topo.txt" 2>&1

if [ "$RUN_SCALING" = 1 ]; then
  for n in $SCALES; do
    [ "$n" -le "$GPUS" ] || continue
    echo "==> [1] $n copies, least connections"
    apply --replicas "$n" --lb least_conn && run_load "scale_${n}x" "$(scaled_steps "$n")" mix
  done
  python3 loadtest/compare_tuning.py --phase "$PHASE" --baseline scale_1x --labels scale_1x scale_2x scale_4x \
    --price-per-hour "${GPU_HOURLY_PRICE_USD:-1.0}" | tail -8
fi

router_snapshot() {  # what the router decided, from its own metrics
  C exec -T vllm-router python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:29000/metrics').read().decode())" \
    > "$OUT/$1/router_metrics.txt" 2>/dev/null || true
}

for lb in $NGINX_LBS; do
  echo "==> [2] NGINX $lb, $N copies, conversations"
  apply --replicas "$N" --lb "$lb" && run_load "nginx_${lb}" "$SESSION_STEPS" sessions
done
for policy in $ROUTER_POLICIES; do
  echo "==> [2] vLLM router $policy, $N copies, conversations"
  if apply --replicas "$N" --router "$policy"; then
    run_load "router_${policy}" "$SESSION_STEPS" sessions
    router_snapshot "router_${policy}"
    C logs --no-color --no-log-prefix vllm-router 2>&1 | tail -200 > "$OUT/router_${policy}/router.log"
  fi
done

if [ "$RUN_THRESHOLDS" = 1 ]; then
  for t in 16 256; do
    echo "==> [3] cache aware, balance threshold $t"
    apply --replicas "$N" --router cache_aware --balance-abs-threshold "$t" \
      && run_load "router_cache_aware_bal${t}" "$SESSION_STEPS" sessions && router_snapshot "router_cache_aware_bal${t}"
  done
fi

echo "==> Comparing routing"
python3 loadtest/compare_tuning.py --phase "$PHASE" --baseline nginx_least_conn \
  --labels $(for lb in $NGINX_LBS; do printf "nginx_%s " "$lb"; done) $(for p in $ROUTER_POLICIES; do printf "router_%s " "$p"; done) \
  router_cache_aware_bal16 router_cache_aware_bal256 --price-per-hour "${GPU_HOURLY_PRICE_USD:-1.0}" | tail -16

kill $SAMPLER 2>/dev/null || true
echo "==> Done. Results are in $OUT/"
ls "$OUT"
