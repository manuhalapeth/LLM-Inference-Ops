"""Turn a Locust step run into one row per step, and find where the GPU broke.

Client side, from the per request log Locust wrote: requests, errors, and
TTFT / end to end / inter token percentiles. Server side, from Prometheus over
the same window: requests running and waiting in vLLM, KV cache usage,
preemptions, tokens per second, GPU utilization and power.

The first part of every step is skipped (WARMUP_S) so each row describes the
system after it has settled at that user count.

    python3 loadtest/analyze_load.py --label baseline

Standard library only.
"""

import argparse
import json
import math
import re
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import RESULTS_DIR, save_run, summarize  # noqa: E402

PHASE = "04_breaking_one_gpu"


class Prometheus:
    def __init__(self, url: str, vllm_selector: str = ""):
        self.url = url
        # e.g. 'model_name="llm"': limit every vLLM query to one model when several share a cluster
        self.vllm_selector = vllm_selector

    def _get(self, path: str, **params) -> list:
        if self.vllm_selector and "query" in params:
            params["query"] = re.sub(r"(vllm:[A-Za-z_:]+)(?![A-Za-z_:{])", r"\1{" + self.vllm_selector + "}", params["query"])
        with urllib.request.urlopen(f"{self.url}{path}?{urllib.parse.urlencode(params)}", timeout=30) as resp:
            return json.load(resp)["data"]["result"]

    def at(self, query: str, t: float) -> float | None:
        result = self._get("/api/v1/query", query=query, time=t)
        return float(result[0]["value"][1]) if result else None

    def over(self, query: str, start: float, end: float) -> dict:
        result = self._get("/api/v1/query_range", query=query, start=start, end=end, step=5)
        values = [float(v) for r in result for _, v in r["values"] if v not in ("NaN", "+Inf", "-Inf")]
        if not values:
            return {"avg": None, "max": None}
        return {"avg": sum(values) / len(values), "max": max(values)}

    def per_series(self, query: str, t: float, label: str) -> dict:
        """One value per series, keyed by a label (e.g. per vLLM server)."""
        return {r["metric"].get(label, "?"): float(r["value"][1]) for r in self._get("/api/v1/query", query=query, time=t)}

    def increase(self, counter: str, start: float, end: float) -> float | None:
        a, b = self.at(f"sum({counter})", start), self.at(f"sum({counter})", end)
        return b - a if a is not None and b is not None else None


def _ratio(a, b):
    return a / b if a is not None and b else None


def pct(values: list[float]) -> dict:
    s = summarize([v for v in values if v is not None])
    return {k: s.get(k) for k in ("count", "p50", "p95", "p99", "max")}


def analyze_step(rows: list[dict], prom: Prometheus | None, start: float, end: float) -> dict:
    ok = [r for r in rows if r["status"] == 200 and not r["error"]]
    duration = end - start
    step = {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "error_rate": (len(rows) - len(ok)) / len(rows) if rows else 0.0,
        "requests_per_s": len(ok) / duration,
        "ttft_s": pct([r["ttft_s"] for r in ok]),
        "e2e_s": pct([r["e2e_s"] for r in ok]),
        "itl_s": pct([r["itl_s"] for r in ok]),
        "by_category": {c: {"requests": len(rs), "ttft_p95_s": pct([r["ttft_s"] for r in rs])["p95"],
                            "e2e_p95_s": pct([r["e2e_s"] for r in rs])["p95"]}
                        for c in sorted({r["category"] for r in ok}) for rs in [[r for r in ok if r["category"] == c]]},
    }
    if prom:
        gen = prom.increase("vllm:generation_tokens_total", start, end)
        prompt = prom.increase("vllm:prompt_tokens_total", start, end)
        step["server"] = {
            "running": prom.over("sum(vllm:num_requests_running)", start, end),
            "waiting": prom.over("sum(vllm:num_requests_waiting)", start, end),
            "kv_cache_usage": prom.over("sum(vllm:kv_cache_usage_perc)", start, end),
            "preemptions": prom.increase("vllm:num_preemptions_total", start, end),
            "output_tokens_per_s": gen / duration if gen is not None else None,
            "prompt_tokens_per_s": prompt / duration if prompt is not None else None,
            "gpu_utilization": prom.over("max(nvidia_smi_utilization_gpu_ratio)", start, end),
            "gpu_power_w": prom.over("max(nvidia_smi_power_draw_watts)", start, end),
            "gpu_memory_bytes": prom.over("max(nvidia_smi_memory_used_bytes)", start, end),
            "prefix_cache_hit_rate": _ratio(prom.increase("vllm:prefix_cache_hits_total", start, end),
                                            prom.increase("vllm:prefix_cache_queries_total", start, end)),
            "external_cache_hit_rate": _ratio(prom.increase("vllm:external_prefix_cache_hits_total", start, end),
                                              prom.increase("vllm:prefix_cache_queries_total", start, end)),
            # How evenly the load was spread: average requests running on each vLLM server.
            "running_per_replica": prom.per_series("avg_over_time(vllm:num_requests_running[1m])", end, "replica"),
        }
    return step


def breaking_points(steps: list[dict], ttft_slo_s: float) -> dict:
    """The first user count where each kind of trouble shows up."""
    def first(pred):
        return next((s["users"] for s in steps if pred(s)), None)

    def server(s, key, field="max"):
        v = s.get("server", {}).get(key)
        return v.get(field) if isinstance(v, dict) else v

    peak = max(steps, key=lambda s: server(s, "output_tokens_per_s") or s["requests_per_s"])
    return {
        "requests_queue": first(lambda s: (server(s, "waiting") or 0) >= 1),
        "kv_cache_95pct": first(lambda s: (server(s, "kv_cache_usage") or 0) >= 0.95),
        "preemptions": first(lambda s: (server(s, "preemptions") or 0) > 0),
        "ttft_p95_over_slo": first(lambda s: (s["ttft_s"]["p95"] or 0) > ttft_slo_s),
        "errors_over_1pct": first(lambda s: s["error_rate"] > 0.01),
        "peak_throughput_users": peak["users"],
        "peak_output_tokens_per_s": server(peak, "output_tokens_per_s"),
        "ttft_slo_s": ttft_slo_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--phase", default=PHASE, help="results/<phase>/<label>/ holds the Locust logs")
    parser.add_argument("--vllm-selector", default="", help='only this model\'s vLLM metrics, e.g. model_name="llm"')
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--warmup-s", type=float, default=15, help="skip this much at the start of every step")
    parser.add_argument("--ttft-slo-s", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.phase / args.label
    meta = json.loads((run_dir / "run_meta.json").read_text())
    rows = [json.loads(line) for f in sorted(run_dir.glob("requests_*.jsonl")) for line in f.read_text().splitlines() if line]
    try:
        prom = Prometheus(args.prometheus, args.vllm_selector)
        prom.at("up", meta["start"])
    except OSError:
        prom = None
        print("Prometheus not reachable: client side numbers only")

    steps = []
    for i, users in enumerate(meta["steps"]):
        start = meta["start"] + i * meta["step_seconds"] + args.warmup_s
        end = meta["start"] + (i + 1) * meta["step_seconds"]
        in_step = [r for r in rows if start <= r["t_start"] < end]
        steps.append({"users": users, "window": [start, end], **analyze_step(in_step, prom, start, end)})

    breaks = breaking_points(steps, args.ttft_slo_s)
    path = save_run(args.phase, f"load_sweep_{args.label}",
                    metrics={"steps": steps, "breaking_points": breaks, "total_requests": len(rows)},
                    config={"label": args.label, "model": os.getenv("MODEL_NAME"), "vllm_image": os.getenv("VLLM_IMAGE"),
                            "vllm_config": os.getenv("VLLM_CONFIG", "baseline"), "topology": os.getenv("TOPOLOGY"),
                            "load_mode": os.getenv("LOAD_MODE", "mix"), "gateway_workers": os.getenv("GATEWAY_WORKERS", "1")},
                    load={**meta, "warmup_s": args.warmup_s, "closed_loop": True})

    print(f"{'users':>6}{'req/s':>8}{'out tok/s':>11}{'TTFT p50':>10}{'TTFT p95':>10}{'e2e p95':>9}{'ITL p50':>9}"
          f"{'running':>9}{'waiting':>9}{'KV max':>8}{'preempt':>9}{'GPU':>6}{'errors':>8}")
    def cell(value, fmt: str, width: int, scale: float = 1, unit: str = "") -> str:
        text = "-" if value is None else format(value * scale, fmt) + unit
        return text.rjust(width)

    for s in steps:
        sv = s.get("server", {})
        print(cell(s["users"], "d", 6) + cell(s["requests_per_s"], ".2f", 8)
              + cell(sv.get("output_tokens_per_s"), ".0f", 11)
              + cell(s["ttft_s"]["p50"], ".0f", 10, 1000, "ms") + cell(s["ttft_s"]["p95"], ".0f", 10, 1000, "ms")
              + cell(s["e2e_s"]["p95"], ".1f", 9, 1, "s") + cell(s["itl_s"]["p50"], ".1f", 9, 1000, "ms")
              + cell((sv.get("running") or {}).get("avg"), ".0f", 9) + cell((sv.get("waiting") or {}).get("max"), ".0f", 9)
              + cell((sv.get("kv_cache_usage") or {}).get("max"), ".0%", 8) + cell(sv.get("preemptions"), ".0f", 9)
              + cell((sv.get("gpu_utilization") or {}).get("avg"), ".0%", 6) + cell(s["errors"], "d", 8))
    print(f"\nbreaking points: {json.dumps(breaks)}\nsaved {path}")


if __name__ == "__main__":
    main()
