"""How much precision do keys and values lose in FP8, with scale 1.0 vs calibrated?

Runs a few real prompts through the model (on CPU, BF16), captures every
layer's keys (after RoPE, as they go into the cache) and values, and rounds
them to FP8 e4m3 and back three ways:

    scale1      x / 1.0                    vLLM's default without calibration
    calibrated  x / k_scale (per layer)    the scale llm-compressor stored
    per_token   x / (max|x| over the head / 448), per token and head  (fp8_per_token_head)

For each it reports the relative error (|x - fp8(x)| / |x|, averaged over
elements, weighted by size so tiny elements don't dominate) and the share
of elements below FP8's smallest normal number (2**-6 after scaling), where
the format runs out of mantissa bits.

    python3 kv_fp8_error.py --model Qwen/Qwen2.5-7B-Instruct --scales /hf/llmops/<ckpt>/.. --out kv_fp8_error.json

Needs torch and transformers (run_phase11.sh runs it in the PyTorch image).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FP8_MAX = 448.0
MIN_NORMAL = 2.0 ** -6

PROMPTS = [
    "What is 17 multiplied by 23? Show your working.",
    "Convert 100 degrees Celsius to Fahrenheit and explain the formula.",
    "Return a JSON object with the keys name, age and city for a fictional person.",
    "Summarize this incident: at 09:12 the payments API started returning 500 errors after a config deploy. "
    "32 minutes of impact, 4,100 customers affected, rolled back at 09:44. Follow ups: add a canary stage.",
]


def roundtrip(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.to(torch.float32) * scale


def stats(x: torch.Tensor, scale: torch.Tensor) -> dict:
    x = x.float()
    err = (x - roundtrip(x, scale)).abs()
    scaled = (x / scale).abs()
    return {"rel_error": float(err.sum() / x.abs().sum().clamp_min(1e-30)),
            "below_min_normal": float((scaled < MIN_NORMAL).float().mean()),
            # A per token scale puts the largest element at exactly 448; float rounding can
            # push it a hair above, which isn't clipping.
            "clipped": float((scaled > FP8_MAX * (1 + 1e-5)).float().mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scales", required=True, help="kv_scales_tensor.json from kv_scales_report.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    calib = {r["layer"]: r for r in json.loads(Path(args.scales).read_text())["layers"]}
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    cfg = model.config
    n_kv, head_dim = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads

    acc: dict[int, dict[str, list]] = {}
    for prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids, use_cache=True)
        cache = out.past_key_values
        for i in range(cfg.num_hidden_layers):
            layer = cache.layers[i] if hasattr(cache, "layers") else None
            k = layer.keys if layer is not None else cache[i][0]
            v = layer.values if layer is not None else cache[i][1]
            k, v = k[0].float(), v[0].float()  # [kv_heads, tokens, head_dim]
            a = acc.setdefault(i, {"k": [], "v": []})
            a["k"].append(k)
            a["v"].append(v)

    rows = []
    for i in sorted(acc):
        k, v = torch.cat(acc[i]["k"], dim=1), torch.cat(acc[i]["v"], dim=1)
        row = {"layer": i, "k_abs_median": float(k.abs().median()), "v_abs_median": float(v.abs().median())}
        for kind, x in (("k", k), ("v", v)):
            one = torch.tensor(1.0)
            cal = torch.tensor(calib[i][f"{kind}_scale_max"])
            per_tok = (x.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
            for name, scale in (("scale1", one), ("calibrated", cal), ("per_token", per_tok)):
                for metric, value in stats(x, scale).items():
                    row[f"{kind}_{name}_{metric}"] = value
        rows.append(row)
        print(f"layer {i:>2}  K err {row['k_scale1_rel_error']:.3%} / {row['k_calibrated_rel_error']:.3%} / {row['k_per_token_rel_error']:.3%}"
              f"   V err {row['v_scale1_rel_error']:.3%} / {row['v_calibrated_rel_error']:.3%} / {row['v_per_token_rel_error']:.3%}"
              f"   V below normal at scale 1: {row['v_scale1_below_min_normal']:.1%}")

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    summary = {f"{kind}_{name}_{m}": mean(f"{kind}_{name}_{m}") for kind in "kv"
               for name in ("scale1", "calibrated", "per_token") for m in ("rel_error", "below_min_normal", "clipped")}
    print(json.dumps(summary, indent=2))
    tokens = sum(t.shape[1] for t in acc[0]["k"])
    Path(args.out).write_text(json.dumps({"prompts": PROMPTS, "tokens": tokens, "summary": summary, "layers": rows}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
