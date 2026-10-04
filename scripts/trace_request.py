"""Follow single requests through every layer and measure where the time goes.

For each prompt, sends requests one at a time (nothing else running) and
records the time spent:

    client ── gateway ── NGINX ── vLLM API server ── engine (queue, prefill, decode)

Client times come from the client itself, gateway and NGINX times from their
logs, and engine times from vLLM's metrics before and after each request.

Run on the GPU box from the repo root, with the stack up:

    python3 scripts/trace_request.py                         # default prompts
    python3 scripts/trace_request.py --prompts short_qa_05 --repeats 10

Standard library only.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
sys.path.insert(0, str(ROOT / "scripts"))
import stack_logs  # noqa: E402
import vllm_metrics  # noqa: E402
from results_logger import save_run, summarize  # noqa: E402
from smoke_test import stream_one_request  # noqa: E402

# One short, one prefill-heavy, one decode-heavy prompt from loadtest/prompts.json.
DEFAULT_PROMPTS = ["short_qa_05", "summarize_01", "long_gen_02"]


def layer_breakdown(client: dict, logs: dict, engine: dict) -> dict:
    """Split one request's end-to-end time into the time each layer added."""
    client_e2e = client["e2e_s"]
    gateway = logs.get("gateway_total_s")
    nginx = logs.get("nginx_total_s")
    upstream = logs.get("upstream_total_s")
    vllm_e2e = engine["e2e_s"]

    def gap(outer, inner):
        return outer - inner if outer is not None and inner is not None else None

    engine_parts = engine["queue_s"] + engine["prefill_s"] + engine["decode_s"]
    return {
        "client_to_gateway_s": gap(client_e2e, gateway),
        "gateway_s": gap(gateway, nginx),
        "nginx_s": gap(nginx, upstream),
        "vllm_api_server_s": gap(upstream, vllm_e2e),
        "engine_queue_s": engine["queue_s"],
        "engine_prefill_s": engine["prefill_s"],
        "engine_decode_s": engine["decode_s"],
        "engine_other_s": vllm_e2e - engine_parts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--vllm-metrics", default="http://localhost:8000/metrics")
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "llm"))
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="prompt IDs from loadtest/prompts.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="untimed requests before measuring")
    parser.add_argument("--phase", default="01_end_to_end", help="results/<phase>/ to save into")
    parser.add_argument("--label", default="", help="appended to the run name, e.g. baseline or mooncake")
    args = parser.parse_args()

    prompts = {p["id"]: p for p in json.loads((ROOT / "loadtest" / "prompts.json").read_text())}
    started = datetime.now(timezone.utc) - timedelta(seconds=2)

    for _ in range(args.warmup):
        stream_one_request(args.url, args.model, 16)

    samples = []
    for prompt_id in args.prompts:
        prompt = prompts[prompt_id]
        for i in range(args.repeats):
            request_id = uuid.uuid4().hex
            before = vllm_metrics.scrape(args.vllm_metrics)
            client = stream_one_request(
                args.url, args.model, prompt["max_tokens"],
                messages=prompt["messages"], headers={"x-request-id": request_id},
            )
            time.sleep(0.5)  # let vLLM finish recording the request's metrics
            engine = vllm_metrics.delta(before, vllm_metrics.scrape(args.vllm_metrics))
            if engine["requests_observed"] != 1:
                print(f"warning: vLLM saw {engine['requests_observed']:.0f} requests during {prompt_id} #{i}; is something else sending traffic?")
            client.pop("text")
            samples.append({"prompt_id": prompt_id, "category": prompt["category"],
                            "request_id": request_id, "client": client, "engine": engine})
            print(f"{prompt_id:<14} #{i + 1}  TTFT {client['ttft_s'] * 1000:6.0f} ms   "
                  f"e2e {client['e2e_s'] * 1000:6.0f} ms   {client['prompt_tokens']} in / {client['output_tokens']} out")

    time.sleep(1)  # let the gateway and NGINX flush their log lines
    logs = stack_logs.requests_by_id(started)
    for sample in samples:
        sample["logs"] = logs.get(sample["request_id"], {})
        sample["layers"] = layer_breakdown(sample["client"], sample["logs"], sample["engine"])

    summary = {}
    for prompt_id in args.prompts:
        rows = [s for s in samples if s["prompt_id"] == prompt_id]
        summary[prompt_id] = {
            "ttft_s": summarize([s["client"]["ttft_s"] for s in rows]),
            "e2e_s": summarize([s["client"]["e2e_s"] for s in rows]),
            "layers_p50_s": {
                layer: summarize([s["layers"][layer] for s in rows if s["layers"][layer] is not None]).get("p50")
                for layer in rows[0]["layers"]
            },
        }

    path = save_run(
        args.phase, "trace_single_requests" + (f"_{args.label}" if args.label else ""),
        metrics={"summary": summary, "samples": samples},
        config={"label": args.label, "model": os.getenv("MODEL_NAME"), "vllm_image": os.getenv("VLLM_IMAGE"),
                "vllm_config": os.getenv("VLLM_CONFIG", "baseline")},
        load={"concurrency": 1, "prompts": args.prompts, "repeats": args.repeats, "warmup": args.warmup},
    )
    print(f"\nsaved {path}")
    for prompt_id, s in summary.items():
        print(f"\n{prompt_id}: median time per layer")
        for layer, value in s["layers_p50_s"].items():
            print(f"  {layer:<22}{value * 1000:8.1f} ms" if value is not None else f"  {layer:<22}     n/a")


if __name__ == "__main__":
    main()
