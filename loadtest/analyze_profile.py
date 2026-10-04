"""Where GPU time goes: summarize a PyTorch profiler trace from vLLM.

Reads the Chrome trace files vLLM's profiler writes (*.json.gz), groups every
GPU kernel by what kind of work it does, and measures how busy the GPU was
(time with at least one kernel running, out of the whole window).

    python3 loadtest/analyze_profile.py results/profiles --label baseline_128_users

Standard library only.
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import save_run  # noqa: E402

# Order matters: the first pattern that matches a kernel name wins.
CATEGORIES = [
    ("attention", r"flash|attn|attention|fmha|paged|mla|decode_kernel|prefill_kernel"),
    ("matmul (GEMM)", r"gemm|cutlass|cublas|nvjet|matmul|wgmma|xmma|_mma|sm\d+_|ampere|hopper|blackwell|splitk"),
    ("normalization", r"norm|rms"),
    ("activation", r"silu|gelu|act_and_mul|swiglu"),
    ("rotary embedding", r"rotary|rope"),
    ("sampling", r"sampl|top_?k|top_?p|softmax|argmax|gumbel|sort|penalt|logit"),
    ("memory and copies", r"memcpy|memset|copy|cat_|_cat|index|gather|scatter|fill|reshape|elementwise|vectorized|unrolled"),
]
_COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in CATEGORIES]


def categorize(kernel: str) -> str:
    return next((name for name, rx in _COMPILED if rx.search(kernel)), "other")


def load_kernels(trace_dir: Path) -> list[dict]:
    kernels = []
    for path in sorted(trace_dir.rglob("*.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            data = json.load(f)
        events = data["traceEvents"] if isinstance(data, dict) else data
        kernels += [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel" and e.get("dur")]
    return kernels


def busy_time(kernels: list[dict]) -> float:
    """Microseconds with at least one kernel running (overlaps counted once)."""
    spans = sorted((k["ts"], k["ts"] + k["dur"]) for k in kernels)
    total, cur_start, cur_end = 0.0, None, None
    for start, end in spans:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def summarize(kernels: list[dict]) -> dict:
    if not kernels:
        return {"kernels": 0}
    window = max(k["ts"] + k["dur"] for k in kernels) - min(k["ts"] for k in kernels)
    kernel_time = sum(k["dur"] for k in kernels)
    by_cat, by_name = {}, {}
    for k in kernels:
        by_cat[categorize(k["name"])] = by_cat.get(categorize(k["name"]), 0) + k["dur"]
        by_name[k["name"]] = by_name.get(k["name"], 0) + k["dur"]
    return {
        "kernels": len(kernels),
        "window_ms": window / 1000,
        "gpu_busy_ratio": busy_time(kernels) / window if window else None,
        "kernel_time_ms": kernel_time / 1000,
        "by_category": {c: {"ms": t / 1000, "share": t / kernel_time}
                        for c, t in sorted(by_cat.items(), key=lambda x: -x[1])},
        "top_kernels": [{"name": n[:160], "ms": t / 1000, "share": t / kernel_time, "category": categorize(n)}
                        for n, t in sorted(by_name.items(), key=lambda x: -x[1])[:15]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--label", default="profile")
    parser.add_argument("--phase", default="05_profiling_and_tuning")
    args = parser.parse_args()

    summary = summarize(load_kernels(args.trace_dir))
    path = save_run(args.phase, f"profile_{args.label}", metrics=summary,
                    config={"trace_dir": str(args.trace_dir)}, load={"label": args.label})
    if summary["kernels"]:
        print(f"{summary['kernels']:,} kernels over {summary['window_ms']:.0f} ms, GPU busy {summary['gpu_busy_ratio']:.0%}")
        for cat, v in summary["by_category"].items():
            print(f"  {cat:<20}{v['ms']:>9.1f} ms  {v['share']:>6.1%}")
    else:
        print("no GPU kernels found in the trace")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
