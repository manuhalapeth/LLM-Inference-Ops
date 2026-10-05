"""How tenant B (a small model) fared while tenant A's load grew.

Splits B's probe log into windows: B alone (before A's load starts), then one
window per Locust step of A. For each window: B's latency percentiles, and A's
throughput and latency from the matching load sweep.

    python3 loadtest/analyze_tenants.py --phase 07_gpu_slicing_hami --label shared

Standard library only.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import RESULTS_DIR, load_runs, save_run, summarize  # noqa: E402


def window_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == 200 and not r["error"]]
    return {
        "requests": len(rows), "errors": len(rows) - len(ok),
        "ttft_s": {k: v for k, v in summarize([r["ttft_s"] for r in ok if r["ttft_s"] is not None]).items() if k in ("p50", "p95", "p99")},
        "e2e_s": {k: v for k, v in summarize([r["e2e_s"] for r in ok]).items() if k in ("p50", "p95", "p99")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="07_gpu_slicing_hami")
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup-s", type=float, default=10)
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.phase / args.label
    meta = json.loads((run_dir / "run_meta.json").read_text())
    probe_meta = json.loads((run_dir / "probe_meta.json").read_text())
    probe = [json.loads(l) for l in (run_dir / "probe.jsonl").read_text().splitlines() if l]
    sweep = next((r for r in reversed(load_runs(args.phase)) if r["name"] == f"load_sweep_{args.label}"), None)
    a_steps = {s["users"]: s for s in sweep["metrics"]["steps"]} if sweep else {}

    windows = [{"window": "B alone", "a_users": 0,
                "start": probe_meta["start"] + args.warmup_s, "end": meta["start"]}]
    for i, users in enumerate(meta["steps"]):
        windows.append({"window": f"A at {users} users", "a_users": users,
                        "start": meta["start"] + i * meta["step_seconds"] + args.warmup_s,
                        "end": meta["start"] + (i + 1) * meta["step_seconds"]})

    out = []
    for w in windows:
        rows = [r for r in probe if w["start"] <= r["t_start"] < w["end"]]
        a = a_steps.get(w["a_users"], {})
        out.append({**w, "b": window_stats(rows),
                    "a": {"output_tokens_per_s": a.get("server", {}).get("output_tokens_per_s"),
                          "ttft_p95_s": a.get("ttft_s", {}).get("p95"), "itl_p50_s": a.get("itl_s", {}).get("p50"),
                          "errors": a.get("errors")} if a else None})

    path = save_run(args.phase, f"tenants_{args.label}", metrics={"windows": out},
                    config={"label": args.label}, load={"probe": probe_meta, "locust": meta})
    print(f"{'window':<18}{'B TTFT p50':>11}{'B TTFT p95':>11}{'B e2e p95':>10}{'B err':>6}   {'A tok/s':>8}{'A TTFT p95':>11}{'A ITL p50':>10}")
    for w in out:
        b, a = w["b"], w["a"] or {}
        f = lambda x, s=1000, u="ms": "-" if x is None else f"{x * s:.0f}{u}"
        print(f"{w['window']:<18}{f(b['ttft_s'].get('p50')):>11}{f(b['ttft_s'].get('p95')):>11}{f(b['e2e_s'].get('p95')):>10}{b['errors']:>6}   "
              f"{f(a.get('output_tokens_per_s'), 1, ''):>8}{f(a.get('ttft_p95_s')):>11}{f(a.get('itl_p50_s'), 1000, 'ms'):>10}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
