"""Does a client hanging up during the KV handover kill the decode server?

Phase 6's disaggregated run died on `assert req_id in self.requests` in the
decode server's scheduler (vLLM v0.30.0). My reading of the code: when a
request is cancelled while its KV is still arriving from the prefill server,
v0.30.0 frees it at once, the transfer then reports "finished" for a request
the scheduler no longer has, and the assertion fires. v0.34.0 keeps such
requests in `_aborted_kv_loads` until the transfer settles.

This script tests that reading. It sends long, unique prompts through the
gateway (so they go prefill -> KV transfer -> decode), in parts:

    control  every client waits for a 64 token answer   (should never crash)
    short    every client asks for ONE token: the prefill server already made
             it, so the decode side can finish the request while its KV is
             still arriving (the same free-before-transfer-settles path)
    abort    every client hangs up after a short, random delay, so many
             cancellations land while the KV is in flight

After each part it checks the decode server's /health directly and stops at
the first part that kills it. (In Phase 9 the v0.30.0 decode server already
died on the readiness probe, a single max_tokens=1 request, before this ran.)

    python3 scripts/pd_abort_repro.py --phase 09_disaggregated_serving --label abort_v0.30.0

Standard library only.
"""

import argparse
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import save_run  # noqa: E402

FILLER = ("The incident review covers the timeline, the customer impact, the root cause and the "
          "follow up actions agreed by the teams involved. ")


def prompt(words: int) -> list[dict]:
    text = f"(request {uuid.uuid4().hex}) " + FILLER * (words // len(FILLER.split()) + 1)
    return [{"role": "user", "content": text + "\nSummarize the review above in three sentences."}]


def one(url: str, model: str, words: int, hang_up_after_s: float | None, max_tokens: int = 64) -> str:
    body = json.dumps({"model": model, "messages": prompt(words), "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "x-user-id": "pd-abort-repro"})
    try:
        # The timeout closes the socket: from the server's side, the client hung up.
        with urllib.request.urlopen(req, timeout=hang_up_after_s or 300) as resp:
            for _ in resp:
                pass
        return "ok"
    except (socket.timeout, TimeoutError):
        return "hung_up"
    except urllib.error.HTTPError as exc:
        return f"http_{exc.code}"
    except Exception as exc:  # noqa: BLE001 (connection refused, reset, ...)
        return type(exc).__name__


def healthy(decode_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{decode_url}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def run(args, part: str) -> dict:
    outcomes, lock = [], threading.Lock()
    abort = part == "abort"
    max_tokens = 1 if part == "short" else 64

    def worker(n: int):
        for _ in range(n):
            delay = random.uniform(args.min_hang_up_s, args.max_hang_up_s) if abort else None
            result = one(args.url, args.model, args.words, delay, max_tokens)
            with lock:
                outcomes.append(result)

    start = time.time()
    threads = [threading.Thread(target=worker, args=(args.requests // args.concurrency,))
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(args.settle_s)  # let in-flight transfers finish (or crash the server)
    counts = {k: outcomes.count(k) for k in sorted(set(outcomes))}
    return {"part": part, "max_tokens": max_tokens, "hang_up": abort,
            "seconds": round(time.time() - start, 1), "outcomes": counts,
            "decode_healthy_after": healthy(args.decode_url)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="09_disaggregated_serving")
    parser.add_argument("--label", required=True, help="e.g. abort_v0.30.0")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--decode-url", default="http://localhost:8001", help="the decode server, directly")
    parser.add_argument("--model", default="llm")
    parser.add_argument("--words", type=int, default=3000, help="prompt length (about 1.3 tokens per word)")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--min-hang-up-s", type=float, default=0.05)
    parser.add_argument("--max-hang-up-s", type=float, default=0.6)
    parser.add_argument("--settle-s", type=float, default=10)
    parser.add_argument("--parts", default="control,short,abort", help="which parts, in order")
    args = parser.parse_args()

    if not healthy(args.decode_url):
        raise SystemExit(f"decode server at {args.decode_url} is not healthy before the test")
    parts, crashed_in = {}, None
    for part in args.parts.split(","):
        parts[part] = run(args, part)
        print(f"{part}:", json.dumps(parts[part]))
        if not parts[part]["decode_healthy_after"]:
            crashed_in = part
            break
    crashed = crashed_in is not None
    print("==> decode server", f"CRASHED in part '{crashed_in}'" if crashed else "survived")
    path = save_run(args.phase, f"pd_abort_{args.label}",
                    {"parts": parts, "decode_crashed": crashed, "crashed_in": crashed_in},
                    config={"topology": os.getenv("TOPOLOGY", ""), "vllm_image": os.getenv("VLLM_IMAGE", "")},
                    load=vars(args))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
