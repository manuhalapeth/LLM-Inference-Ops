"""GSM8K: grade school math word problems, a bigger quality check than the 22 cases.

Phase 5 found the FP8 KV cache broke arithmetic, on a handful of eval cases.
22 cases can't tell a 2% drop from noise, so Phase 11 also runs the first N
problems of the GSM8K test set (1,319 in all) through the gateway, greedy,
zero shot, and checks the final number against the reference answer.

    python3 evals/gsm8k.py --phase 11_fp8_kv_calibration --label fp8w_kv_bf16 --n 500

The test set is downloaded once from OpenAI's repository and cached in
evals/.cache/. Standard library only.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
from results_logger import save_run  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
CACHE = ROOT / "evals" / ".cache" / "gsm8k_test.jsonl"
INSTRUCTION = "Solve this problem step by step. End your reply with a line of the form: The answer is <number>."
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load(n: int) -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
            CACHE.write_bytes(resp.read())
    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    return [{"i": i, "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(rows[:n])]


def final_number(text: str) -> str | None:
    tail = text.split("answer is")[-1] if "answer is" in text else text
    nums = _NUMBER.findall(tail) or _NUMBER.findall(text)
    if not nums:
        return None
    value = nums[-1].replace(",", "").rstrip(".")
    try:  # "1250.00" and "1250" are the same answer
        number = float(value)
        return str(int(number)) if number.is_integer() else value
    except ValueError:
        return value


def ask(url: str, model: str, question: str, max_tokens: int) -> tuple[str, float]:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "x-user-id": "gsm8k"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", default="11_fp8_kv_calibration")
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "llm"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    problems = load(args.n)
    results, lock, queue = [], threading.Lock(), list(problems)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                p = queue.pop()
            try:
                text, seconds = ask(args.url, args.model, p["question"], args.max_tokens)
                got, error = final_number(text), None
            except Exception as exc:  # noqa: BLE001
                text, seconds, got, error = "", 0.0, None, f"{type(exc).__name__}: {exc}"[:200]
            with lock:
                results.append({**p, "reply": text, "got": got, "correct": got == p["answer"],
                                "seconds": round(seconds, 2), "error": error})

    start = time.time()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results.sort(key=lambda r: r["i"])
    correct = sum(r["correct"] for r in results)
    errors = sum(r["error"] is not None for r in results)
    n = len(results)
    acc = correct / n if n else 0.0
    # 95% interval for the accuracy (normal approximation): how far apart two configs
    # must be before the difference means anything.
    half = 1.96 * (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    print(f"{args.label}: {correct}/{n} correct ({acc:.1%} +- {half:.1%}), {errors} errors, {time.time() - start:.0f} s")
    path = save_run(args.phase, f"gsm8k_{args.label}",
                    metrics={"correct": correct, "n": n, "accuracy": acc, "ci95_half_width": half,
                             "errors": errors, "results": results},
                    config={"label": args.label, "vllm_config": os.getenv("VLLM_CONFIG", ""),
                            "model": os.getenv("MODEL_NAME", "")},
                    load={"n": args.n, "concurrency": args.concurrency, "max_tokens": args.max_tokens})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
