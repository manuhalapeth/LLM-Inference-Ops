"""Send one streaming request through the gateway and report its timings.

Phase 0 exit check: proves gateway -> NGINX -> vLLM works end to end.

    python3 scripts/smoke_test.py            # print timings
    python3 scripts/smoke_test.py --save     # also save to results/00_setup/

Standard library only, so it runs on a fresh GPU box.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loadtest"))
from results_logger import save_run  # noqa: E402

PROMPT = "In two sentences, explain what a KV cache is in LLM inference."


def wait_for_gateway(base_url: str, timeout_s: float) -> None:
    """Block until vLLM answers through the gateway (the model may still be loading)."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() > deadline:
            sys.exit(f"Gateway at {base_url} did not become ready within {timeout_s:.0f}s")
        time.sleep(5)


def stream_one_request(base_url: str, model: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_at = None
    text, usage = [], {}
    with urllib.request.urlopen(request, timeout=300) as resp:
        request_id = resp.headers.get("x-request-id")
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: "):])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text.append(content)
    end = time.perf_counter()

    output_tokens = usage.get("completion_tokens", 0)
    decode_s = end - first_token_at if first_token_at else 0
    return {
        "request_id": request_id,
        "ttft_s": first_token_at - start if first_token_at else None,
        "e2e_s": end - start,
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": output_tokens,
        # Time per output token after the first one.
        "itl_s": decode_s / (output_tokens - 1) if output_tokens > 1 else None,
        "output_tokens_per_s": output_tokens / (end - start) if output_tokens else 0,
        "text": "".join(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "llm"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--wait", type=float, default=0, help="seconds to wait for the gateway to be ready")
    parser.add_argument("--save", action="store_true", help="save the result to results/00_setup/")
    args = parser.parse_args()

    if args.wait:
        wait_for_gateway(args.url, args.wait)

    result = stream_one_request(args.url, args.model, args.max_tokens)

    print(f"request_id     {result['request_id']}")
    print(f"TTFT           {result['ttft_s'] * 1000:.0f} ms")
    print(f"end-to-end     {result['e2e_s'] * 1000:.0f} ms")
    print(f"tokens         {result['prompt_tokens']} in / {result['output_tokens']} out")
    if result["itl_s"]:
        print(f"inter-token    {result['itl_s'] * 1000:.1f} ms")
    print(f"throughput     {result['output_tokens_per_s']:.1f} output tokens/s")
    print(f"\n{result['text']}")

    if args.save:
        metrics = {k: v for k, v in result.items() if k != "text"}
        config = {
            "model": os.getenv("MODEL_NAME"),
            "vllm_image": os.getenv("VLLM_IMAGE"),
            "vllm_config": os.getenv("VLLM_CONFIG", "baseline"),
        }
        load = {"requests": 1, "concurrency": 1, "prompt": PROMPT, "max_tokens": args.max_tokens}
        path = save_run("00_setup", "smoke_test", metrics, config=config, load=load)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
