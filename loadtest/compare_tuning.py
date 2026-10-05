"""Compare every tuning run against the baseline, on the same metrics.

For each config's load sweep: peak throughput, latency at fixed user counts,
how many users it serves within each latency target, and what an output token
costs at that load. Saves one summary to results/05_profiling_and_tuning/.

    python3 loadtest/compare_tuning.py

Standard library only.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import load_runs, save_run  # noqa: E402

PHASE = "05_profiling_and_tuning"

# A request is "good" when the first token arrives quickly and the answer streams
# at a readable pace. Each target is (name, max TTFT p95 seconds, max ITL p50 seconds).
TARGETS = [
    ("interactive", 0.5, 0.040),  # first token within 0.5 s for 95% of requests, 25+ tokens/s each
    ("relaxed", 1.0, 0.060),      # within 1 s, 16+ tokens/s each
]


def server(step: dict, key: str, field: str = "max"):
    v = step.get("server", {}).get(key)
    return v.get(field) if isinstance(v, dict) else v


def at_users(steps: list[dict], users: int) -> dict | None:
    return next((s for s in steps if s["users"] == users), None)


def summarize_sweep(run: dict, price_per_hour: float) -> dict:
    steps = run["metrics"]["steps"]
    peak = max(steps, key=lambda s: server(s, "output_tokens_per_s") or 0)
    out = {
        "label": run["config"]["label"],
        "peak_output_tokens_per_s": server(peak, "output_tokens_per_s"),
        "peak_at_users": peak["users"],
        "max_running": max(server(s, "running") or 0 for s in steps),
        "max_kv_cache_usage": max(server(s, "kv_cache_usage") or 0 for s in steps),
        "preemptions": sum(server(s, "preemptions") or 0 for s in steps),
        "errors": sum(s["errors"] for s in steps),
        "breaking_points": run["metrics"]["breaking_points"],
        "at_users": {},
        "capacity": {},
    }
    for users in sorted({128, 256, 512} | {s["users"] for s in steps}):
        s = at_users(steps, users)
        if s:
            out["at_users"][users] = {
                "output_tokens_per_s": server(s, "output_tokens_per_s"),
                "ttft_p95_s": s["ttft_s"]["p95"],
                "itl_p50_s": s["itl_s"]["p50"],
                "waiting_max": server(s, "waiting"),
            }
    for name, ttft, itl in TARGETS:
        ok = [s for s in steps if s["ttft_s"]["p95"] is not None and s["ttft_s"]["p95"] <= ttft
              and s["itl_s"]["p50"] is not None and s["itl_s"]["p50"] <= itl]
        best = max(ok, key=lambda s: s["users"]) if ok else None
        tps = server(best, "output_tokens_per_s") if best else None
        out["capacity"][name] = {
            "users": best["users"] if best else 0,
            "output_tokens_per_s": tps,
            "usd_per_million_output_tokens": price_per_hour / (tps * 3600) * 1e6 if tps else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--price-per-hour", type=float, default=float(os.getenv("GPU_HOURLY_PRICE_USD") or 1.327))
    parser.add_argument("--phase", default=PHASE)
    parser.add_argument("--baseline", default="baseline", help="label the others are compared with")
    parser.add_argument("--labels", nargs="*", help="only these runs (default: every load sweep in the phase)")
    args = parser.parse_args()

    runs = {}
    for run in load_runs(args.phase):  # oldest first, so the latest run of each config wins
        label = run["config"]["label"]
        if run["name"].startswith("load_sweep_") and not label.startswith("latency_") and (not args.labels or label in args.labels):
            runs[label] = run
    rows = [summarize_sweep(r, args.price_per_hour) for r in runs.values()]
    base = next((r for r in rows if r["label"] == args.baseline), None)

    def rel(row, value):
        b = base and base["peak_output_tokens_per_s"]
        return f"{(value / b - 1):+.0%}" if b and value else ""

    print(f"{'config':<28}{'peak tok/s':>11}{'vs base':>9}{'@users':>8}{'TTFT p95 @256':>15}{'ITL p50 @256':>14}"
          f"{'running':>9}{'KV max':>8}{'preempt':>9}   interactive users ($/1M)   relaxed users ($/1M)")
    for r in rows:
        a = r["at_users"].get(256, {})
        cap = lambda k: f"{r['capacity'][k]['users']:>4} (${r['capacity'][k]['usd_per_million_output_tokens']:.3f})" \
            if r["capacity"][k]["usd_per_million_output_tokens"] else f"{r['capacity'][k]['users']:>4}"
        print(f"{r['label']:<28}{r['peak_output_tokens_per_s'] or 0:>11.0f}{rel(r, r['peak_output_tokens_per_s']):>9}"
              f"{r['peak_at_users']:>8}{(a.get('ttft_p95_s') or 0) * 1000:>13.0f}ms{(a.get('itl_p50_s') or 0) * 1000:>12.1f}ms"
              f"{r['max_running']:>9.0f}{r['max_kv_cache_usage']:>8.0%}{r['preemptions']:>9.0f}   {cap('interactive'):<24}{cap('relaxed')}")

    path = save_run(args.phase, "tuning_summary", metrics={"configs": rows, "targets": TARGETS},
                    config={"price_per_hour_usd": args.price_per_hour}, load={"configs": list(runs)})
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
