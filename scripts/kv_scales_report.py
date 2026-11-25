"""What the calibrated KV scales say about why scale 1.0 failed.

A calibrated FP8 checkpoint stores k_scale and v_scale per layer (or per KV
head): scale = largest value seen / 448, the largest FP8 e4m3 number. So each
scale gives back the largest key or value that layer produced on the
calibration text. With vLLM's default scale of 1.0:

    largest value > 448     clipped: the biggest keys are cut to 448
    largest value < ~1      most values sit near FP8's small end, where steps are coarse

    python3 scripts/kv_scales_report.py /root/.cache/huggingface/llmops/qwen2.5-7b-fp8dyn-kv-tensor

Standard library only (reads the safetensors files directly).
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

FP8_MAX = 448.0
_LAYER = re.compile(r"layers\.(\d+)\.")


def read_scales(path: Path) -> dict[str, list[float]]:
    """Every tensor named *k_scale or *v_scale in one .safetensors file."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__" or not name.endswith(("k_scale", "v_scale")):
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dtype = meta["dtype"]
            if dtype == "F32":
                vals = list(struct.unpack(f"<{len(raw) // 4}f", raw))
            elif dtype == "BF16":
                vals = [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]
            elif dtype == "F16":
                vals = list(struct.unpack(f"<{len(raw) // 2}e", raw))
            else:
                continue
            out[name] = vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", help="write the report as JSON here")
    args = parser.parse_args()

    scales = {}
    for f in sorted(Path(args.checkpoint).glob("*.safetensors")):
        scales.update(read_scales(f))
    if not scales:
        sys.exit("no k_scale / v_scale tensors found")

    layers: dict[int, dict] = {}
    for name, vals in scales.items():
        m = _LAYER.search(name)
        if not m:
            continue
        kind = "k" if name.endswith("k_scale") else "v"
        layers.setdefault(int(m.group(1)), {})[kind] = vals
    rows = []
    for i in sorted(layers):
        row = {"layer": i}
        for kind in ("k", "v"):
            vals = layers[i].get(kind, [])
            if vals:
                row[f"{kind}_scale_max"] = max(vals)
                row[f"{kind}_absmax"] = max(vals) * FP8_MAX   # the largest key / value seen
                row[f"{kind}_heads"] = len(vals)
        rows.append(row)

    k = [r["k_absmax"] for r in rows if "k_absmax" in r]
    v = [r["v_absmax"] for r in rows if "v_absmax" in r]
    summary = {
        "layers": len(rows),
        "k_absmax_max": max(k), "k_absmax_min": min(k),
        "v_absmax_max": max(v), "v_absmax_min": min(v),
        "layers_k_clipped_at_scale_1": sum(x > FP8_MAX for x in k),
        "layers_v_clipped_at_scale_1": sum(x > FP8_MAX for x in v),
        "layers_k_below_1": sum(x < 1 for x in k),
        "layers_v_below_1": sum(x < 1 for x in v),
    }
    print(f"{'layer':>5} {'K largest':>10} {'V largest':>10}")
    for r in rows:
        flag = "  <- K clipped at scale 1.0" if r.get("k_absmax", 0) > FP8_MAX else ""
        print(f"{r['layer']:>5} {r.get('k_absmax', 0):>10.1f} {r.get('v_absmax', 0):>10.2f}{flag}")
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "summary": summary, "layers": rows}, indent=2))


if __name__ == "__main__":
    main()
