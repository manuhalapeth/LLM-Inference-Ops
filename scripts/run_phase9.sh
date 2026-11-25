#!/usr/bin/env bash
# Phase 9: disaggregated prefill and decode, second attempt. Needs 2 GPUs.
# Run on the GPU box after setup_gpu_box.sh says "Ready":
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase9.sh
#
# [0] Why Phase 6 crashed: the client hang up test on v0.30.0, then v0.34.0
# [1] The baseline on v0.34.0: 2 copies, least connections
# [2] 1 prefill + 1 decode, KV moved by Mooncake (TCP)
# [3] 1 prefill + 1 decode, KV moved by NIXL (UCX)
# Each of [1] to [3] runs the Phase 6 mix and the interference load, and the
# 22 case eval set. Every server runs FP8 weights, as in Phase 6.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

PHASE="${PHASE9_DIR:-09_disaggregated_serving}"
OUT=results/$PHASE
mkdir -p "$OUT"
. scripts/lib_compose.sh
[ "$GPUS" -ge 2 ] || { echo "!! needs 2 GPUs, found $GPUS"; exit 1; }

MIX_STEPS="${MIX_STEPS:-128,256,384,512,768,1024}"          # Phase 6's 2 copy steps
INTERFERENCE_STEPS="${INTERFERENCE_STEPS:-16,32,64,128,256}"
RUN_REPRO="${RUN_REPRO:-1}"
SETUPS="${SETUPS:-copies pd_mooncake pd_nixl}"
start_sampler
nvidia-smi topo -m > "$OUT/nvidia_smi_topo.txt" 2>&1

if [ "$RUN_REPRO" = 1 ]; then
  for tag in v0.30.0 v0.34.0; do
    echo "==> [0] Client hang ups during the KV handover, vLLM $tag"
    use_vllm "$tag"
    # No smoke test request here: on v0.30.0 that single request is enough to crash it.
    if SMOKE=0 apply --pd 1,1 --pd-connector mooncake; then
      python3 scripts/pd_abort_repro.py --phase "$PHASE" --label "abort_$tag" | tail -5
      C logs --no-color --no-log-prefix vllm-decode-0 > "$OUT/abort_${tag}_decode_full.log" 2>&1 || true
      grep -E -B2 -A28 "Traceback|AssertionError|EngineDeadError" "$OUT/abort_${tag}_decode_full.log" \
        | head -120 > "$OUT/abort_${tag}_decode.log" || true
    fi
  done
fi

use_vllm v0.34.0
measure() {  # label prefix, KV stats on (1) or off (0)
  local p=$1 kv=$2
  python3 evals/run_evals.py --phase "$PHASE" --label "quality_$p" | tail -2
  for mode in mix interference; do
    steps=$MIX_STEPS; [ "$mode" = interference ] && steps=$INTERFERENCE_STEPS
    [ "$kv" = 1 ] && python3 scripts/kv_transfer_stats.py snapshot "$OUT/kv_before.json"
    run_load "${p}_${mode}" "$steps" "$mode"
    if [ "$kv" = 1 ]; then
      python3 scripts/kv_transfer_stats.py snapshot "$OUT/kv_after.json"
      python3 scripts/kv_transfer_stats.py diff "$OUT/kv_before.json" "$OUT/kv_after.json" > "$OUT/${p}_${mode}/kv_transfer.json"
      cat "$OUT/${p}_${mode}/kv_transfer.json"
    fi
    # A crash shows up as errors; keep the logs that explain it.
    C ps --format '{{.Service}} {{.State}}' | grep -E '^vllm-' | grep -qv running && save_logs "${p}_${mode}_crash"
    C logs --no-color --no-log-prefix 2>&1 | grep -qE "AssertionError|EngineDeadError" && save_logs "${p}_${mode}_engine_error"
  done
  rm -f "$OUT/kv_before.json" "$OUT/kv_after.json"
}

for setup in $SETUPS; do
  case $setup in
    copies)      echo "==> [1] 2 copies, least connections"; apply --replicas 2 --lb least_conn && measure copies 0 ;;
    pd_mooncake) echo "==> [2] 1 prefill + 1 decode, Mooncake"; apply --pd 1,1 --pd-connector mooncake && measure pd_mooncake 1 ;;
    pd_nixl)     echo "==> [3] 1 prefill + 1 decode, NIXL"; apply --pd 1,1 --pd-connector nixl && measure pd_nixl 1 ;;
    pd_nixl_hostpid)
      echo "==> [5] 1 prefill + 1 decode, NIXL with the servers in the host PID namespace (UCX shared memory)"
      apply --pd 1,1 --pd-connector nixl_hostpid && measure pd_nixl_hostpid 1 ;;
    pd_mooncake_tuned)
      echo "==> [4] 1 prefill + 1 decode, Mooncake with a deeper TCP queue, 60 s abort timeout, recompute on failure"
      apply --pd 1,1 --pd-connector mooncake_tuned && measure pd_mooncake_tuned 1 ;;
  esac
done

echo "==> Comparing"
for mode in mix interference; do
  python3 loadtest/compare_tuning.py --phase "$PHASE" --baseline "copies_$mode" \
    --labels "copies_$mode" "pd_mooncake_$mode" "pd_nixl_$mode" "pd_mooncake_tuned_$mode" "pd_nixl_hostpid_$mode" --price-per-hour "${GPU_HOURLY_PRICE_USD:-1.0}" | tail -12
done

kill $SAMPLER 2>/dev/null || true
echo "==> Done. Results are in $OUT/"
ls "$OUT"
