"""Concurrent agents: many CrewAI crews at once, the way a real product is used.

Locust sends single requests. A real agent product sends tasks, and every task
is a chain of LLM calls where each waits for the last. This starts N crews at
the same moment (each in its own process), for N in --concurrency, and
measures how long a whole task takes as more of them share the GPU.

Each crew gets a unique tag at the top of its document, so concurrent crews
are separate tasks rather than copies served from the prefix cache.

Run on the GPU box from the repo root, with the stack up:

    .venv/bin/python scripts/agent_load.py
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from analyze_load import Prometheus  # noqa: E402
from results_logger import save_run, summarize  # noqa: E402


def one_crew(args: tuple) -> dict:
    url, model, document, tag = args
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    sys.path.insert(0, str(ROOT / "agent"))
    from app import run_crew  # imported here: every process loads CrewAI once

    try:
        run = run_crew(url, model, f"(task {tag})\n{document}", run_id=f"agentload-{tag}")
        return {"ok": True, "total_s": run["total_s"], "tasks": run["tasks"]}
    except Exception as exc:  # one failed crew shouldn't stop the sweep
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "llm"))
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--document", default=str(ROOT / "agent" / "documents" / "incident_report.md"))
    parser.add_argument("--phase", default="04_breaking_one_gpu")
    args = parser.parse_args()

    document = Path(args.document).read_text()
    prom = Prometheus(args.prometheus)
    ctx = mp.get_context("spawn")
    levels = []
    for n in args.concurrency:
        jobs = [(args.url, args.model, document, uuid.uuid4().hex[:10]) for _ in range(n)]
        start = time.time()
        with ctx.Pool(n) as pool:
            runs = pool.map(one_crew, jobs)
        end = time.time()
        ok = [r for r in runs if r["ok"]]
        try:
            server = {"output_tokens_per_s": (prom.increase("vllm:generation_tokens_total", start, end) or 0) / (end - start),
                      "running": prom.over("sum(vllm:num_requests_running)", start, end),
                      "kv_cache_usage": prom.over("sum(vllm:kv_cache_usage_perc)", start, end)}
        except OSError:
            server = None
        level = {"concurrent_crews": n, "wall_s": end - start, "failed": len(runs) - len(ok),
                 "task_s": summarize([r["total_s"] for r in ok]),
                 "tasks_per_minute": len(ok) / (end - start) * 60,
                 "per_agent_p50_s": {a: summarize([t["duration_s"] for r in ok for t in r["tasks"] if t["agent"] == a])["p50"]
                                     for a in sorted({t["agent"] for r in ok for t in r["tasks"]})},
                 "server": server, "errors": [r["error"] for r in runs if not r["ok"]][:3]}
        levels.append(level)
        t = level["task_s"]
        print(f"{n:>3} crews: task p50 {t.get('p50', 0):5.1f} s  p95 {t.get('p95', 0):5.1f} s  "
              f"{level['tasks_per_minute']:5.1f} tasks/min  failed {level['failed']}"
              + (f"  {server['output_tokens_per_s']:.0f} tok/s" if server else ""))
        time.sleep(5)

    path = save_run(args.phase, "agent_load", metrics={"levels": levels},
                    config={"model": os.getenv("MODEL_NAME"), "vllm_config": os.getenv("VLLM_CONFIG", "baseline"),
                            "crewai": "1.15.23"},
                    load={"concurrency": args.concurrency, "document": Path(args.document).name, "agents_per_crew": 3})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
