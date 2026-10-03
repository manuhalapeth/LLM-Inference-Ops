"""Save every benchmark run to results/<phase>/<run_id>.json.

Notebooks read results from these files, never from memory, so every number
in a notebook can be traced back to a saved run with its exact config,
hardware and git commit.

Standard library only, so it runs on a fresh GPU box without installs.
"""

import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_environment() -> dict:
    """Hardware and code version the run was made with."""
    gpus = []
    smi = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    for line in (smi or "").splitlines():
        name, memory_mib, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "memory_mib": int(memory_mib), "driver": driver})

    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "gpus": gpus,
        "git_commit": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
    }


def summarize(values: list[float]) -> dict:
    """Count, mean, P50/P95/P99 and max of a list of latencies (or any values)."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        # Nearest-rank percentile: the smallest value with at least p% of samples at or below it.
        return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "max": ordered[-1],
    }


def save_run(
    phase: str,
    name: str,
    metrics: dict,
    config: dict | None = None,
    load: dict | None = None,
    notes: str = "",
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write one run to results/<phase>/<run_id>.json and return the path.

    phase:   e.g. "00_setup", "04_breaking_one_gpu"
    name:    short label for the run, e.g. "baseline_20_users"
    metrics: measured numbers (latency summaries, throughput, errors, ...)
    config:  what the system was running (engine args, model, image, ...)
    load:    what was sent (users, spawn rate, prompt set, duration, ...)
    """
    started = datetime.now(timezone.utc)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")
    run_id = f"{started:%Y%m%dT%H%M%SZ}_{slug}"

    record = {
        "run_id": run_id,
        "phase": phase,
        "name": name,
        "timestamp": started.isoformat(),
        "config": config or {},
        "load": load or {},
        "metrics": metrics,
        "environment": capture_environment(),
        "notes": notes,
    }

    path = Path(results_dir) / phase / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_runs(phase: str | None = None, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """All saved runs (optionally for one phase), oldest first."""
    pattern = f"{phase}/*.json" if phase else "*/*.json"
    runs = [json.loads(p.read_text()) for p in Path(results_dir).glob(pattern)]
    return sorted(runs, key=lambda r: r["timestamp"])
