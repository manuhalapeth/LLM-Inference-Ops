#!/usr/bin/env bash
# Phase 6: scale out on one machine with several GPUs. Run on the GPU box after
# setup_gpu_box.sh says "Ready". Safe experiments first, Mooncake last.
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase6.sh
#
# Every vLLM server runs the Phase 5 recommended config (FP8 weights).
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

PHASE=06_scaling_out
OUT=results/$PHASE
mkdir -p "$OUT"
export LOAD_PHASE=$PHASE
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=100000
GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPUS=${GPUS_OVERRIDE:-$GPUS}
TOPO_EXTRA=(--gpus "$GPUS" --gateway-workers "${GATEWAY_WORKERS_P6:-8}")
[ -n "${FAKE_DIR:-}" ] && TOPO_EXTRA+=(--fake-dir "$FAKE_DIR")
STEP_SECONDS_P6="${STEP_SECONDS_P6:-45}"
RUN_TP4="${RUN_TP4:-1}"
RUN_MOONCAKE="${RUN_MOONCAKE:-1}"
RUN_CORE="${RUN_CORE:-1}"   # experiments 1 to 4 (set to 0 to rerun only the Mooncake ones)
BASE_STEPS="${BASE_STEPS:-64 128 192 256 384 512}"   # users per vLLM server at each step
SESSION_BASE="${SESSION_BASE:-128 256}"
FAILOVER_USERS_PER_SERVER="${FAILOVER_USERS_PER_SERVER:-128}"
FAILOVER_SECONDS="${FAILOVER_SECONDS:-300}" FAILOVER_STOP_AT="${FAILOVER_STOP_AT:-60}" FAILOVER_DOWN_FOR="${FAILOVER_DOWN_FOR:-90}"
echo "==> $GPUS GPUs"

C() { docker compose -f docker-compose.yml -f generated/compose.json "$@"; }

apply() {  # topology.py arguments; starts that setup and waits until it answers
  python3 scripts/topology.py "${TOPO_EXTRA[@]}" "$@" || return 1
  set -a; . generated/env; set +a
  docker compose rm -sf vllm >/dev/null 2>&1   # the single-server vLLM, if it's still up
  C up -d --build --remove-orphans >/dev/null 2>&1
  C exec -T nginx nginx -s reload >/dev/null 2>&1
  # Wait for EVERY vLLM server to be healthy, not just the first one to answer:
  # otherwise requests reach servers that are still loading and fail.
  local deadline=$(( $(date +%s) + 1500 ))
  until [ -z "$(C ps --format '{{.Service}} {{.Health}}' | grep -E '^vllm-' | grep -v ' healthy$')" ]; do
    if [ "$(date +%s)" -gt "$deadline" ]; then break; fi
    sleep 5
  done
  if ! python3 scripts/smoke_test.py --wait 300 > /dev/null 2>&1; then
    echo "!! $TOPOLOGY did not come up"
    C logs --no-color --tail 80 > "$OUT/failed_$(echo "$TOPOLOGY" | tr ' =,+' '____').log" 2>&1
    return 1
  fi
  sleep 15  # let every server finish warming up, not just the first to answer
  for s in $(C config --services | grep -E '^vllm-'); do
    C logs --no-color --no-log-prefix "$s" 2>/dev/null \
      | grep -E "non-default args|Available KV cache memory|GPU KV cache size|Model loading took|tensor_parallel|[Mm]ooncake|Application startup complete" \
      | head -12 > "$OUT/vllm_startup_${s}_$(echo "$TOPOLOGY" | tr ' =,+' '____').log" || true
  done
  echo "==> up: $TOPOLOGY"
}

run_load() {  # label, steps, mode
  mkdir -p "$OUT/$1"
  RUN_LABEL=$1 LOAD_STEPS=$2 STEP_SECONDS=$STEP_SECONDS_P6 LOAD_MODE=$3 \
    docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust > "$OUT/$1/locust_console.log" 2>&1
  LOAD_MODE=$3 python3 loadtest/analyze_load.py --phase $PHASE --label "$1" --warmup-s 10 | tail -10
}

scaled_steps() {  # the Phase 5 steps times the number of servers
  local n=$1 out=""
  for u in $BASE_STEPS; do out="$out,$((u * n))"; done
  echo "${out#,}"
}

( while true; do
    docker stats --no-stream --format '{"t": '"$(date +%s)"', "name": "{{.Name}}", "cpu": "{{.CPUPerc}}", "mem": "{{.MemUsage}}"}'
    sleep 10
  done ) > "$OUT/container_stats.jsonl" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

MAXN=$(( GPUS >= 4 ? 4 : GPUS ))
SESSION_STEPS=$(for u in $SESSION_BASE; do printf "%s," $((u * MAXN)); done); SESSION_STEPS=${SESSION_STEPS%,}

if [ "$RUN_CORE" = 1 ]; then
echo "==> [1] Scaling: ${SCALES:-1 2 4} copies, one per GPU, round robin"
for n in ${SCALES:-1 2 4}; do
  [ "$n" -le "$GPUS" ] || continue
  apply --replicas "$n" --lb round_robin && run_load "scale_${n}x" "$(scaled_steps "$n")" mix
done

echo "==> [2] Load balancing with conversations ($SESSION_STEPS users, $MAXN copies)"
for lb in round_robin least_conn user_hash; do
  apply --replicas "$MAXN" --lb "$lb" && run_load "lb_${lb}" "$SESSION_STEPS" sessions
done

echo "==> [3] Failover: stop one copy under load, start it again"
if apply --replicas "$MAXN" --lb least_conn; then
  mkdir -p "$OUT/failover"
  RUN_LABEL=failover LOAD_STEPS=$((FAILOVER_USERS_PER_SERVER * MAXN)) STEP_SECONDS=$FAILOVER_SECONDS LOAD_MODE=mix \
    docker compose --profile loadtest run --rm locust > "$OUT/failover/locust_console.log" 2>&1 &
  LOCUST=$!
  sleep "$FAILOVER_STOP_AT"; echo "{\"event\": \"stop vllm-1\", \"t\": $(date +%s.%N)}" > "$OUT/failover/events.tmp"
  C stop -t 0 vllm-1 >/dev/null 2>&1
  sleep "$FAILOVER_DOWN_FOR"; echo "{\"event\": \"start vllm-1\", \"t\": $(date +%s.%N)}" >> "$OUT/failover/events.tmp"
  C start vllm-1 >/dev/null 2>&1
  wait $LOCUST
  python3 -c "import json,sys; print(json.dumps([json.loads(l) for l in open('$OUT/failover/events.tmp')]))" > "$OUT/failover/events.json"
  python3 loadtest/analyze_timeline.py --phase $PHASE --label failover | tail -30
fi

echo "==> [4] Tensor parallel: one model split across 2 GPUs (vs 2 copies in [1])"
if [ "$GPUS" -ge 2 ] && apply --replicas 1 --tp 2; then
  python3 evals/run_evals.py --phase $PHASE --label quality_tp2 | tail -2
  run_load tp2 "$(scaled_steps 2)" mix
fi
if [ "$RUN_TP4" = 1 ] && [ "$GPUS" -ge 4 ] && apply --replicas 1 --tp 4; then
  run_load tp4 "$(scaled_steps 4)" mix
fi

echo "==> Comparing scaling and tensor parallel"
python3 loadtest/compare_tuning.py --phase $PHASE --baseline scale_1x --labels scale_1x scale_2x scale_4x tp2 tp4
fi

if [ "$RUN_MOONCAKE" = 1 ]; then
  echo "==> [5] Mooncake store shared by every copy, conversations, round robin"
  apply --replicas "$MAXN" --lb round_robin --mooncake-store && run_load "lb_round_robin_mooncake" "$SESSION_STEPS" sessions

  echo "==> [6] Disaggregated serving: 1 prefill + 1 decode server (vs 2 copies in [1])"
  if [ "${RUN_PD:-1}" = 1 ] && [ "$GPUS" -ge 2 ] && apply --pd 1,1; then
    python3 evals/run_evals.py --phase $PHASE --label quality_pd_1p1d | tail -2
    run_load pd_1p1d "$(scaled_steps 2)" mix
  fi
fi

kill $SAMPLER 2>/dev/null || true
echo "==> Done. Results are in $OUT/"
ls "$OUT"
