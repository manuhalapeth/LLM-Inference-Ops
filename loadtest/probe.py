"""A steady trickle of short requests to one model, logged per request.

Tenant B in Phase 7: a latency-sensitive service sharing a GPU with a busy
neighbour. Runs a fixed number of concurrent clients for a fixed time and
writes every request to a JSON lines file, for loadtest/analyze_tenants.py.

    python3 loadtest/probe.py --url http://localhost:8001 --model small --duration 420 --out probe.jsonl

Standard library only.
"""

import argparse
import json
import random
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_test import stream_one_request  # noqa: E402

PROMPTS = [p for p in json.loads((ROOT / "loadtest" / "prompts.json").read_text()) if p["category"] == "short_qa"]


def worker(args, deadline: float, out, lock: threading.Lock) -> None:
    while time.time() < deadline:
        prompt = random.choice(PROMPTS)
        messages = [dict(m) for m in prompt["messages"]]
        messages[-1]["content"] = f"(probe {uuid.uuid4().hex[:10]}) {messages[-1]['content']}"
        t = time.time()
        try:
            r = stream_one_request(args.url, args.model, args.max_tokens, messages=messages)
            row = {"t_start": t, "status": 200, "error": None, "ttft_s": r["ttft_s"], "e2e_s": r["e2e_s"],
                   "output_tokens": r["output_tokens"]}
        except Exception as exc:  # the probe keeps going; errors are data
            row = {"t_start": t, "status": None, "error": f"{type(exc).__name__}: {exc}"[:200],
                   "ttft_s": None, "e2e_s": time.time() - t, "output_tokens": 0}
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, required=True, help="seconds")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.duration
    lock = threading.Lock()
    with open(args.out, "w") as out:
        threads = [threading.Thread(target=worker, args=(args, deadline, out, lock)) for _ in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()
