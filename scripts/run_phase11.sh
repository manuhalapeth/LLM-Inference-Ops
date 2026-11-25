#!/usr/bin/env bash
# Phase 11: can a calibrated FP8 KV cache keep answer quality? One GPU, vLLM v0.34.0.
# Run on the GPU box after setup_gpu_box.sh says "Ready":
#
#   cd ~/LLM-Inference-Ops && git pull && ./scripts/run_phase11.sh
#
# [0] Calibrate two checkpoints with llm-compressor (FP8_DYNAMIC weights, KV
#     scales per layer and per KV head) and report each layer's scales
# [1] Quality, every config: the 22 case eval set and 500 GSM8K problems
# [2] Throughput, every config that loads: the Phase 5 sweep
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

PHASE="${PHASE11_DIR:-11_fp8_kv_calibration}"
OUT=results/$PHASE
mkdir -p "$OUT"
export LOAD_PHASE=$PHASE VLLM_TAG=v0.34.0 VLLM_IMAGE=vllm/vllm-openai:v0.34.0
export RATE_LIMIT_RPM=1000000 RATE_LIMIT_TPM=1000000000 MAX_IN_FLIGHT=4096
HF_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CKPT_TENSOR=llmops/qwen2.5-7b-fp8dyn-kv-tensor
CKPT_HEAD=llmops/qwen2.5-7b-fp8dyn-kv-head
STEPS="${STEPS:-64,128,192,256,384,512,768}"
GSM8K_N="${GSM8K_N:-500}"
RUN_CALIBRATION="${RUN_CALIBRATION:-1}" RUN_QUALITY="${RUN_QUALITY:-1}" RUN_SWEEPS="${RUN_SWEEPS:-1}"

# label | vLLM config | model (blank = the base model)
CONFIGS="
baseline_bf16|baseline|
bf16w_kv_fp8_scale1|tune_kv_cache_fp8|
fp8w_kv_bf16|tune_fp8_weights|
fp8w_kv_fp8_scale1|p11_fp8w_kv_fp8|
fp8w_kv_fp8_per_token_head|p11_fp8w_kv_fp8_per_token_head|
ckpt_kv_bf16|p11_ckpt_kv_auto|$CKPT_TENSOR
ckpt_kv_fp8_tensor|p11_ckpt_kv_fp8|$CKPT_TENSOR
ckpt_kv_fp8_head|p11_ckpt_kv_fp8|$CKPT_HEAD
"
SWEEP_LABELS="${SWEEP_LABELS:-fp8w_kv_bf16 fp8w_kv_fp8_scale1 fp8w_kv_fp8_per_token_head ckpt_kv_bf16 ckpt_kv_fp8_tensor ckpt_kv_fp8_head}"

if [ "$RUN_CALIBRATION" = 1 ]; then
  docker compose down >/dev/null 2>&1   # calibration needs the whole GPU
  for strategy in tensor attn_head; do
    ckpt=$CKPT_TENSOR; [ "$strategy" = attn_head ] && ckpt=$CKPT_HEAD
    if [ -f "$HF_DIR/$ckpt/config.json" ]; then echo "==> [0] $ckpt already calibrated"; continue; fi
    echo "==> [0] Calibrating $strategy -> $ckpt"
    docker run --rm --gpus all --ipc host -v "$HF_DIR:/hf" -v "$PWD/scripts:/scripts:ro" -e HF_HOME=/hf \
      -e HF_TOKEN="${HF_TOKEN:-}" pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime \
      bash -c "pip install -q llmcompressor==0.15.0 && python3 /scripts/calibrate_kv_fp8.py --model $BASE_MODEL --strategy $strategy --out /hf/$ckpt" \
      > "$OUT/calibrate_${strategy}.log" 2>&1 || echo "!! calibration $strategy failed, see $OUT/calibrate_${strategy}.log"
    tail -3 "$OUT/calibrate_${strategy}.log"
    [ -f "$HF_DIR/$ckpt/config.json" ] && python3 scripts/kv_scales_report.py "$HF_DIR/$ckpt" --out "$OUT/kv_scales_${strategy}.json" | tail -14
  done
fi

start_vllm() {  # label config model; non-zero if vLLM doesn't come up
  export VLLM_CONFIG=$2 MODEL_NAME=${3:+/root/.cache/huggingface/$3}
  MODEL_NAME=${MODEL_NAME:-$BASE_MODEL}
  docker compose up -d gateway vllm >/dev/null 2>&1
  if ! python3 scripts/smoke_test.py --wait 1200 > /dev/null 2>&1; then
    echo "!! $1: vLLM did not start, see $OUT/failed_$1.log"
    docker compose logs --no-color --no-log-prefix --tail 150 vllm > "$OUT/failed_$1.log" 2>&1
    return 1
  fi
  docker compose logs --no-color --no-log-prefix vllm 2>/dev/null \
    | grep -E "non-default args|Available KV cache memory|GPU KV cache size|Model loading took|[Qq]uantiz|kv_cache_dtype|k_scale|v_scale|scaling factor|[Aa]ttention backend|Using .* backend|Application startup complete" \
    | head -30 > "$OUT/vllm_startup_$1.log" || true
}

each_config() {  # callback: runs it for every config line
  # Read the list first: docker compose run reads stdin and would eat the lines.
  local lines line label cfg model
  mapfile -t lines <<< "$CONFIGS"
  for line in "${lines[@]}"; do
    IFS='|' read -r label cfg model <<< "$line"
    [ -n "$label" ] && "$1" "$label" "$cfg" "$model" </dev/null
  done
}

quality() {
  echo "==> [1] $1"
  start_vllm "$1" "$2" "$3" || return 0
  python3 evals/run_evals.py --phase "$PHASE" --label "quality_$1" | tail -2
  python3 evals/gsm8k.py --phase "$PHASE" --label "$1" --n "$GSM8K_N" | tail -2
}

sweep() {
  case " $SWEEP_LABELS " in *" $1 "*) ;; *) return 0 ;; esac
  echo "==> [2] $1"
  start_vllm "$1" "$2" "$3" || return 0
  mkdir -p "$OUT/$1"
  RUN_LABEL=$1 LOAD_STEPS=$STEPS STEP_SECONDS=45 docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust \
    > "$OUT/$1/locust_console.log" 2>&1
  python3 loadtest/analyze_load.py --phase "$PHASE" --label "$1" --warmup-s 10 | tail -10
}

[ "$RUN_QUALITY" = 1 ] && each_config quality
[ "$RUN_SWEEPS" = 1 ] && each_config sweep

echo "==> Comparing"
python3 loadtest/compare_tuning.py --phase "$PHASE" --baseline fp8w_kv_bf16 --labels $SWEEP_LABELS \
  --price-per-hour "${GPU_HOURLY_PRICE_USD:-1.0}" | tail -10
echo "==> Done. Results are in $OUT/"
ls "$OUT"
