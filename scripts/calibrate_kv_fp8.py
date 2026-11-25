"""Calibrate FP8 scales for the KV cache (and FP8 weights) with llm-compressor.

Phase 5's FP8 KV cache broke answers (17 x 23 = "253") because vLLM stores keys
and values in FP8 with a scale of 1.0 unless the checkpoint provides one. FP8
(e4m3) holds about +-448 with 3 bits of mantissa, so a scale of 1.0 assumes
every layer's keys and values fit that range and use it well. This script runs
calibration text through the model, records how large each layer's keys and
values really get, and saves a checkpoint with those scales (k_scale, v_scale).

Weights use FP8_DYNAMIC (per channel FP8 weights, per token FP8 activations),
the offline version of Phase 5's recommended FP8 weights.

    strategy tensor    one scale per layer for K and one for V
    strategy attn_head one scale per KV head

Runs inside a PyTorch container on the GPU box (run_phase11.sh does this):

    python3 calibrate_kv_fp8.py --strategy tensor --out /hf/llmops/qwen2.5-7b-fp8dyn-kv-tensor
"""

import argparse
import json
import time
from pathlib import Path

from compressed_tensors.quantization import QuantizationArgs
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--strategy", choices=["tensor", "attn_head"], default="tensor")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dataset", default="perfectblend", help="an llm-compressor dataset name")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"],
        kv_cache_scheme=QuantizationArgs(num_bits=8, type="float", strategy=args.strategy,
                                         dynamic=False, symmetric=True),
    )
    oneshot(model=model, dataset=args.dataset, splits=f"train[:{args.samples}]", recipe=recipe,
            max_seq_length=args.max_seq_len, num_calibration_samples=args.samples)

    out = Path(args.out)
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)
    (out / "calibration.json").write_text(json.dumps({**vars(args), "seconds": round(time.time() - start)}, indent=2))
    print(f"saved {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
