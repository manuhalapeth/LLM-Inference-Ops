"""Which part of an FP8 KV cache hurts Qwen2.5, and why? A simulation in HF transformers.

vLLM's FP8 KV cache broke GSM8K (92% -> 6%) on every attention kernel and with
every scale (Phase 11). This script reproduces the cache in plain PyTorch:
keys and values are rounded to FP8 e4m3 and back as they enter the cache
(after RoPE, as vLLM stores them), and the model reads them from there. On a
long real text it compares the next token predictions with the BF16 run:

    top1_agree   share of positions where the most likely next token is the same
    kl           mean KL divergence of the next token distribution (nats)

Variants, each changing one thing:

    k_and_v            what vLLM does (scale 1.0)
    k_only / v_only    which tensor is the problem
    k_and_v_calibrated one scale per layer from the calibrated checkpoint
    k_per_channel      keys get one scale per channel (dimension), values per token
    k_centered         keys minus each channel's mean before rounding, added back after:
                       tests whether a large constant offset per channel is what
                       eats the precision

Also reports, per layer, how large each key channel's mean is compared with
how much it varies from token to token.

    python3 kv_fp8_attention_sim.py --scales kv_scales_tensor.json --out kv_fp8_attention_sim.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicLayer

FP8_MAX = 448.0
TEXT = (Path(__file__).with_name("kv_sim_text.txt"))


def fp8(x: torch.Tensor, scale) -> torch.Tensor:
    return ((x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


def make_quantizer(variant: str, layer_scales: dict):
    def k_fn(k, layer):
        if variant in ("v_only",):
            return k
        if variant == "k_and_v_calibrated":
            return fp8(k, layer_scales[layer]["k"])
        if variant == "k_per_channel":
            s = (k.float().abs().amax(dim=(0, 2), keepdim=True) / FP8_MAX).clamp_min(1e-12)  # [1, heads, 1, dim]
            return fp8(k, s)
        if variant == "k_centered":
            mean = k.float().mean(dim=2, keepdim=True)
            return (fp8(k.float() - mean, 1.0).float() + mean).to(k.dtype)
        return fp8(k, 1.0)

    def v_fn(v, layer):
        if variant in ("k_only", "k_centered", "k_per_channel"):
            return v if variant == "k_only" else fp8(v, 1.0)
        if variant == "k_and_v_calibrated":
            return fp8(v, layer_scales[layer]["v"])
        return fp8(v, 1.0)

    return k_fn, v_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scales", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    scales = {r["layer"]: {"k": r["k_scale_max"], "v": r["v_scale_max"]}
              for r in json.loads(Path(args.scales).read_text())["layers"]}
    text = TEXT.read_text()
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens].to(dev)

    # Hook the cache: every key and value passes through here after RoPE.
    original_update = DynamicLayer.update
    state = {"variant": None, "layer": 0, "keys": {}}

    def update(self, key_states, value_states, *a, **kw):
        layer = state["layer"] % model.config.num_hidden_layers
        state["layer"] += 1
        if state["variant"] == "capture":
            state["keys"][layer] = key_states.detach().float().cpu()
        elif state["variant"]:
            k_fn, v_fn = make_quantizer(state["variant"], scales)
            key_states, value_states = k_fn(key_states, layer), v_fn(value_states, layer)
        return original_update(self, key_states, value_states, *a, **kw)

    DynamicLayer.update = update

    def run(variant):
        state.update(variant=variant, layer=0)
        with torch.no_grad():
            return model(ids, use_cache=True).logits[0].float()

    ref = run("capture")
    ref_logp = torch.log_softmax(ref, -1)
    results = {}
    for variant in ["k_and_v", "k_only", "v_only", "k_and_v_calibrated", "k_per_channel", "k_centered"]:
        logits = run(variant)
        logp = torch.log_softmax(logits, -1)
        kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)
        results[variant] = {"top1_agree": float((logits.argmax(-1) == ref.argmax(-1)).float().mean()),
                            "kl_mean": float(kl.mean()), "kl_p95": float(kl.quantile(0.95))}
        print(f"{variant:<20} top1 agree {results[variant]['top1_agree']:.1%}   KL {results[variant]['kl_mean']:.4f}")

    # Per layer: each key channel's mean against its spread across tokens.
    layers = []
    for layer, k in sorted(state["keys"].items()):
        k = k[0]  # [heads, tokens, dim]
        mean, std = k.mean(dim=1), k.std(dim=1)  # [heads, dim]
        ratio = mean.abs() / std.clamp_min(1e-6)
        err = (k - fp8(k, 1.0).float()).abs().mean(dim=1)  # rounding error per channel
        layers.append({"layer": layer, "max_abs_mean": float(mean.abs().max()), "max_mean_over_std": float(ratio.max()),
                       "channels_mean_over_std_gt_10": int((ratio > 10).sum()),
                       "max_error_over_std": float((err / std.clamp_min(1e-6)).max())})
    DynamicLayer.update = original_update
    Path(args.out).write_text(json.dumps({"tokens": int(ids.shape[1]), "variants": results, "key_channels": layers}, indent=2))
    print(json.dumps(layers[:3] + layers[-2:], indent=1))


if __name__ == "__main__":
    main()
