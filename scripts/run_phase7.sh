#!/usr/bin/env bash
# Phase 7: GPU sharing with HAMi, and autoscaling with KEDA, on k3s.
# Run on the GPU box after setup_k3s.sh says "Ready".
#
#   cd ~/LLM-Inference-Ops && ./scripts/run_phase7.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}

PHASE=07_gpu_slicing_hami
OUT=results/$PHASE
mkdir -p "$OUT" generated
STEPS="${STEPS:-32,64,128,192,256}"
STEP_SECONDS="${STEP_SECONDS:-60}"
PROBE_ALONE_S="${PROBE_ALONE_S:-60}"
SCENARIOS="${SCENARIOS:-isolated shared shared_cores}"
RUN_AUTOSCALE="${RUN_AUTOSCALE:-1}"
AUTOSCALE_STEPS="${AUTOSCALE_STEPS:-64,64,384,384,384,384,384,384,64,64,64,64,64}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
SMALL_URL="${SMALL_URL:-http://localhost:8001}"
FAKE_ARGS=()
[ -n "${FAKE_DIR:-}" ] && FAKE_ARGS=(--fake --fake-dir "$FAKE_DIR")
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | paste -sd, -)
UUIDS=${UUIDS:-GPU-fake0,GPU-fake1}
K=(kubectl -n llmops)

record() {  # snapshot of where everything is running
  { date -u; "${K[@]}" get pods -o wide; echo; nvidia-smi 2>/dev/null; } > "$OUT/$1/placement.txt" 2>&1
}

deploy() {  # scenario name
  "${K[@]}" delete deploy vllm-large vllm-small --ignore-not-found --wait >/dev/null 2>&1
  "${K[@]}" delete scaledobject vllm-large-scaler --ignore-not-found >/dev/null 2>&1
  "${K[@]}" wait --for=delete pod -l component=vllm --timeout 300s >/dev/null 2>&1
  python3 scripts/k8s_manifests.py "$1" --gpu-uuids "$UUIDS" "${FAKE_ARGS[@]}" > "generated/k8s_$1.json"
  "${K[@]}" apply -f "generated/k8s_$1.json" >/dev/null
  if ! "${K[@]}" wait --for=condition=Ready pod -l component=vllm --timeout 1800s; then
    echo "!! $1: vLLM pods did not become ready"
    mkdir -p "$OUT/$1"
    { "${K[@]}" get pods -o wide; "${K[@]}" describe pods -l component=vllm; "${K[@]}" logs -l component=vllm --tail 80; } > "$OUT/$1/failed.log" 2>&1
    return 1
  fi
  python3 scripts/smoke_test.py --url "$GATEWAY_URL" --wait 300 > /dev/null || return 1
  sleep 10
}

locust() {  # label, steps, seconds per step
  mkdir -p "$OUT/$1"
  docker run --rm --network host -v "$PWD/loadtest:/mnt/loadtest:ro" -v "$PWD/$OUT/$1:/results" --user 0 \
    -e GATEWAY_URL="$GATEWAY_URL" -e LOAD_STEPS="$2" -e STEP_SECONDS="$3" -e LOAD_MODE=mix -e RESULTS_DIR=/results \
    locustio/locust:2.46.6 -f /mnt/loadtest/locustfile.py --autostart --autoquit 5 --processes 8 \
    --html /results/locust_report.html --csv /results/locust > "$OUT/$1/locust_console.log" 2>&1
}

for s in $SCENARIOS; do
  echo "==> [$s]"
  deploy "$s" || continue
  mkdir -p "$OUT/$s"; record "$s"
  steps_n=$(echo "$STEPS" | tr ',' '\n' | wc -l)
  probe_s=$(( PROBE_ALONE_S + steps_n * STEP_SECONDS + 20 ))
  echo "{\"start\": $(date +%s.%N), \"concurrency\": 4, \"url\": \"$SMALL_URL\"}" > "$OUT/$s/probe_meta.json"
  python3 loadtest/probe.py --url "$SMALL_URL" --model small --concurrency 4 --duration "$probe_s" --out "$OUT/$s/probe.jsonl" &
  PROBE=$!
  sleep "$PROBE_ALONE_S"            # tenant B alone first
  locust "$s" "$STEPS" "$STEP_SECONDS"
  wait $PROBE
  python3 loadtest/analyze_load.py --phase $PHASE --label "$s" --warmup-s 10 --vllm-selector 'model_name="llm"' | tail -8
  python3 loadtest/analyze_tenants.py --phase $PHASE --label "$s" | tail -9
done

if [ "$RUN_AUTOSCALE" = 1 ]; then
  echo "==> [autoscale] 1 to 2 pods with KEDA"
  if deploy autoscale; then
    mkdir -p "$OUT/autoscale"; record autoscale
    ( while true; do
        echo "{\"t\": $(date +%s.%N), \"replicas\": $("${K[@]}" get deploy vllm-large -o jsonpath='{.status.replicas}' 2>/dev/null || echo 0), \"ready\": $("${K[@]}" get deploy vllm-large -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)}"
        sleep 5
      done ) > "$OUT/autoscale/replicas.jsonl" 2>/dev/null &
    WATCH=$!
    locust autoscale "$AUTOSCALE_STEPS" "$STEP_SECONDS"
    sleep 120; kill $WATCH 2>/dev/null
    record autoscale
    python3 loadtest/analyze_timeline.py --phase $PHASE --label autoscale --bucket-s 15 | tail -6
    "${K[@]}" get events --sort-by=.lastTimestamp 2>/dev/null | grep -iE "scaled|keda|SuccessfulRescale" > "$OUT/autoscale/scale_events.txt" || true
  fi
fi

echo "==> Done. Results are in $OUT/"
ls "$OUT"
