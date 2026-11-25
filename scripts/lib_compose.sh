# Shared by run_phase9.sh, run_phase10.sh and run_phase11.sh: bring up a setup
# written by topology.py, wait until every vLLM server answers, run a Locust
# sweep and analyze it. The same steps as run_phase6.sh, in one place.
#
# Callers set PHASE and OUT, and TOPO_EXTRA (array) if they need more topology
# arguments, then: . scripts/lib_compose.sh

GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPUS=${GPUS_OVERRIDE:-$GPUS}
[ -n "${TOPO_EXTRA+x}" ] || TOPO_EXTRA=()
TOPO_EXTRA+=(--gpus "$GPUS" --gateway-workers "${GATEWAY_WORKERS:-8}")
[ -n "${FAKE_DIR:-}" ] && TOPO_EXTRA+=(--fake-dir "$FAKE_DIR")
STEP_SECONDS="${STEP_SECONDS:-45}"
export LOAD_PHASE=$PHASE
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=100000

C() { docker compose -f docker-compose.yml -f generated/compose.json "$@"; }

use_vllm() {  # tag, e.g. v0.34.0: every vLLM server after this runs that version
  export VLLM_TAG=$1 VLLM_IMAGE=vllm/vllm-openai:$1
  echo "==> vLLM $1"
}

slug() { echo "$1" | tr ' =,+/' '_____'; }

apply() {  # topology.py arguments; starts that setup and waits until it answers
  python3 scripts/topology.py "${TOPO_EXTRA[@]}" "$@" || return 1
  set -a; . generated/env; set +a
  docker compose rm -sf vllm >/dev/null 2>&1   # the single-server vLLM, if it's still up
  C up -d --build --remove-orphans >/dev/null 2>&1
  C exec -T nginx nginx -s reload >/dev/null 2>&1
  # Wait for EVERY vLLM server to be healthy, not just the first one to answer.
  local deadline=$(( $(date +%s) + 1500 ))
  until [ -z "$(C ps --format '{{.Service}} {{.Health}}' | grep -E '^vllm-' | grep -v ' healthy$')" ]; do
    if [ "$(date +%s)" -gt "$deadline" ]; then break; fi
    sleep 5
  done
  if ! python3 scripts/smoke_test.py --wait 300 > /dev/null 2>&1; then
    echo "!! $TOPOLOGY did not come up"
    C logs --no-color --tail 200 > "$OUT/failed_$(slug "$TOPOLOGY")_${VLLM_TAG:-v0.30.0}.log" 2>&1
    return 1
  fi
  sleep 15  # let every server finish warming up
  for s in $(C config --services | grep -E '^vllm-'); do
    C logs --no-color --no-log-prefix "$s" 2>/dev/null \
      | grep -E "non-default args|Available KV cache memory|GPU KV cache size|Model loading took|tensor_parallel|[Mm]ooncake|[Nn]ixl|NIXL|UCX|kv_cache_dtype|k_scale|Application startup complete" \
      | head -20 > "$OUT/vllm_startup_${s}_$(slug "$TOPOLOGY")_${VLLM_TAG:-v0.30.0}.log" || true
  done
  echo "==> up: $TOPOLOGY (${VLLM_TAG:-v0.30.0})"
}

run_load() {  # label, steps, mode
  mkdir -p "$OUT/$1"
  RUN_LABEL=$1 LOAD_STEPS=$2 STEP_SECONDS=$STEP_SECONDS LOAD_MODE=$3 \
    docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust > "$OUT/$1/locust_console.log" 2>&1
  LOAD_MODE=$3 python3 loadtest/analyze_load.py --phase "$PHASE" --label "$1" --warmup-s 10 | tail -10
}

save_logs() {  # name: every container's last lines, for anything that went wrong
  C logs --no-color --tail 400 > "$OUT/$1.log" 2>&1
}

start_sampler() {  # container CPU and memory every 10 s, for the whole session
  ( while true; do
      docker stats --no-stream --format '{"t": '"$(date +%s)"', "name": "{{.Name}}", "cpu": "{{.CPUPerc}}", "mem": "{{.MemUsage}}"}'
      sleep 10
    done ) >> "$OUT/container_stats.jsonl" 2>/dev/null &
  SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null || true' EXIT
}
