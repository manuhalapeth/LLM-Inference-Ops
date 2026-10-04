"""Save traces from Jaeger into results/, so they outlive the GPU box.

Collects trace IDs from this phase's results (eval cases, failure drills,
agent runs) and saves each full trace in OpenTelemetry JSON, as returned by
Jaeger's API (/api/v3/traces/<id>).

    python3 scripts/export_traces.py

Standard library only.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import RESULTS_DIR, load_runs  # noqa: E402

PHASE = "03_harnesses_and_traces"


def wanted_traces() -> dict[str, str]:
    """name -> trace ID for the traces worth keeping."""
    traces = {}
    for run in load_runs(PHASE):
        m = run["metrics"]
        if run["name"].startswith("evals_"):
            label = run["config"]["label"]
            for case in m["cases"]:
                if case["result"].get("trace_id"):
                    traces[f"eval_{label}_{case['id']}"] = case["result"]["trace_id"]
        elif run["name"] == "failure_drills":
            for drill in m["drills"]:
                if drill.get("trace_id"):
                    traces[f"drill_{drill['drill']}"] = drill["trace_id"]
        elif run["name"] == "agent_run" and m.get("trace_id"):
            traces[f"agent_{m['run_id']}"] = m["trace_id"]
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaeger", default="http://localhost:16686")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / PHASE / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for name, trace_id in wanted_traces().items():
        try:
            with urllib.request.urlopen(f"{args.jaeger}/api/v3/traces/{trace_id}", timeout=10) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as err:
            print(f"skip {name}: HTTP {err.code}")
            continue
        services = sorted({a["value"]["stringValue"] for r in data["result"]["resourceSpans"]
                           for a in r["resource"]["attributes"] if a["key"] == "service.name"})
        spans = sum(len(s["spans"]) for r in data["result"]["resourceSpans"] for s in r["scopeSpans"])
        (out_dir / f"{name}.json").write_text(json.dumps({"name": name, "trace_id": trace_id, **data}) + "\n")
        saved += 1
        print(f"saved {name:<48} {spans:3d} spans  {', '.join(services)}")
    print(f"\n{saved} traces in {out_dir}")


if __name__ == "__main__":
    main()
