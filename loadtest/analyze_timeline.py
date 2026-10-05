"""What happened second by second: for the failover drill, where a vLLM server
is killed under load and started again.

Buckets every request Locust logged by the time it ended, and marks the
events (kill, restart) recorded by the run script.

    python3 loadtest/analyze_timeline.py --phase 06_scaling_out --label failover

Standard library only.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import RESULTS_DIR, save_run, summarize  # noqa: E402


def timeline(rows: list[dict], start: float, bucket_s: float) -> list[dict]:
    buckets = {}
    for r in rows:
        b = int((r["t_start"] + r["e2e_s"] - start) // bucket_s)
        buckets.setdefault(b, []).append(r)
    out = []
    for b in sorted(buckets):
        rs = buckets[b]
        ok = [r for r in rs if r["status"] == 200 and not r["error"]]
        out.append({"t_s": b * bucket_s, "finished": len(rs), "ok": len(ok), "errors": len(rs) - len(ok),
                    "ttft_p95_s": summarize([r["ttft_s"] for r in ok if r["ttft_s"] is not None]).get("p95"),
                    "error_samples": sorted({(r["error"] or str(r["status"]))[:80] for r in rs if r not in ok})[:3]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="06_scaling_out")
    parser.add_argument("--label", default="failover")
    parser.add_argument("--bucket-s", type=float, default=10)
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.phase / args.label
    meta = json.loads((run_dir / "run_meta.json").read_text())
    events = json.loads((run_dir / "events.json").read_text()) if (run_dir / "events.json").exists() else []
    rows = [json.loads(l) for f in sorted(run_dir.glob("requests_*.jsonl")) for l in f.read_text().splitlines() if l]
    tl = timeline(rows, meta["start"], args.bucket_s)
    for e in events:
        e["t_s"] = e["t"] - meta["start"]
    total_err = sum(b["errors"] for b in tl)
    path = save_run(args.phase, f"timeline_{args.label}",
                    metrics={"timeline": tl, "events": events, "requests": len(rows), "errors": total_err},
                    config={"label": args.label}, load=meta)
    marks = {int(e["t_s"] // args.bucket_s): e["event"] for e in events}
    for b in tl:
        mark = marks.get(int(b["t_s"] // args.bucket_s), "")
        print(f"{b['t_s']:>5.0f}s  ok {b['ok']:>4}  errors {b['errors']:>3}  TTFT p95 "
              f"{(b['ttft_p95_s'] or 0) * 1000:>6.0f} ms  {mark}  {'; '.join(b['error_samples'])}")
    print(f"\n{len(rows)} requests, {total_err} errors\nsaved {path}")


if __name__ == "__main__":
    main()
