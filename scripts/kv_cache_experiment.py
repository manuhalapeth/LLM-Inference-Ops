"""What happens when the working set of prompts is bigger than the GPU's KV cache.

Sends N distinct long documents (each ~6k tokens), one at a time, in three rounds:

    fill     every document once, cold: nothing is cached anywhere yet
    revisit  the same documents again, in the same order
    hot      the last few documents once more

With 48 documents (~286k tokens) and a GPU KV cache of ~209k tokens, the GPU's
prefix cache can't hold them all. By the time a document is revisited, the GPU
has evicted it (least recently used goes first), so without an extra tier the
whole prompt is computed again. With Mooncake, evicted KV can come back from
CPU memory instead. The hot round shows the third tier: KV still on the GPU.

Each request is measured on its own: client TTFT, plus vLLM's prefill time,
GPU prefix cache hits, Mooncake (external) hits and Mooncake transfers.

Run on the GPU box from the repo root, with the stack up:

    python3 scripts/kv_cache_experiment.py --label baseline
    python3 scripts/kv_cache_experiment.py --label mooncake

Standard library only.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
sys.path.insert(0, str(ROOT / "scripts"))
import vllm_metrics  # noqa: E402
from long_docs import make_messages  # noqa: E402
from results_logger import save_run, summarize  # noqa: E402
from smoke_test import stream_one_request  # noqa: E402


def measure(args, doc_id: int, round_name: str) -> dict:
    before = vllm_metrics.scrape(args.vllm_metrics)
    client = stream_one_request(args.url, args.model, args.max_tokens,
                                messages=make_messages(doc_id, args.words))
    time.sleep(0.5)  # let vLLM record the request and finish any async store writes
    engine = vllm_metrics.delta(before, vllm_metrics.scrape(args.vllm_metrics))
    client.pop("text")
    prompt = engine["prompt_tokens"] or 1
    return {
        "round": round_name,
        "doc_id": doc_id,
        "client": client,
        "engine": engine,
        "gpu_hit_ratio": engine["prefix_cache_hits"] / prompt,
        "mooncake_hit_ratio": engine["external_prefix_cache_hits"] / prompt,
    }


def round_summary(rows: list[dict]) -> dict:
    mooncake = {op: {k: sum(r["engine"]["mooncake"][op][k] for r in rows) for k in ("calls", "bytes", "time_s")}
                for op in vllm_metrics.MOONCAKE_OPERATIONS}
    return {
        "requests": len(rows),
        "prompt_tokens": sum(r["engine"]["prompt_tokens"] for r in rows),
        "ttft_s": summarize([r["client"]["ttft_s"] for r in rows]),
        "prefill_s": summarize([r["engine"]["prefill_s"] for r in rows]),
        "gpu_hit_ratio": sum(r["engine"]["prefix_cache_hits"] for r in rows) / max(1, sum(r["engine"]["prompt_tokens"] for r in rows)),
        "mooncake_hit_ratio": sum(r["engine"]["external_prefix_cache_hits"] for r in rows) / max(1, sum(r["engine"]["prompt_tokens"] for r in rows)),
        "preemptions": sum(r["engine"]["preemptions"] for r in rows),
        "mooncake": mooncake,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="e.g. baseline or mooncake")
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--vllm-metrics", default="http://localhost:8000/metrics")
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "llm"))
    parser.add_argument("--docs", type=int, default=48)
    parser.add_argument("--words", type=int, default=5900, help="~1 token per word")
    parser.add_argument("--hot", type=int, default=5, help="documents in the final hot round")
    parser.add_argument("--max-tokens", type=int, default=8, help="short answers: this experiment is about prefill")
    args = parser.parse_args()

    stream_one_request(args.url, args.model, 8)  # warmup, not measured

    rows = []
    plan = ([("fill", d) for d in range(args.docs)]
            + [("revisit", d) for d in range(args.docs)]
            + [("hot", d) for d in range(args.docs - args.hot, args.docs)])
    for i, (round_name, doc_id) in enumerate(plan, 1):
        row = measure(args, doc_id, round_name)
        rows.append(row)
        e = row["engine"]
        print(f"[{i:3d}/{len(plan)}] {round_name:<8} doc {doc_id:03d}  TTFT {row['client']['ttft_s'] * 1000:7.1f} ms  "
              f"prefill {e['prefill_s'] * 1000:7.1f} ms  tokens {e['prompt_tokens']:5.0f}  "
              f"GPU hit {row['gpu_hit_ratio']:4.0%}  Mooncake hit {row['mooncake_hit_ratio']:4.0%}")
        if e["requests_observed"] != 1:
            print(f"    warning: vLLM saw {e['requests_observed']:.0f} requests; is something else sending traffic?")

    summary = {name: round_summary([r for r in rows if r["round"] == name]) for name in ("fill", "revisit", "hot")}

    config = {"label": args.label, "model": os.getenv("MODEL_NAME"), "vllm_image": os.getenv("VLLM_IMAGE"),
              "vllm_config": os.getenv("VLLM_CONFIG", "baseline")}
    mooncake_config = ROOT / "vllm" / "mooncake" / "mooncake_config.json"
    if os.getenv("VLLM_CONFIG", "").startswith("mooncake") and mooncake_config.exists():
        config["mooncake_config"] = mooncake_config.read_text()
    path = save_run(
        "02_kv_cache_mooncake", f"kv_experiment_{args.label}",
        metrics={"summary": summary, "requests": rows},
        config=config,
        load={"concurrency": 1, "docs": args.docs, "words_per_doc": args.words, "hot": args.hot,
              "max_tokens": args.max_tokens, "rounds": ["fill", "revisit", "hot"]},
    )

    print(f"\nsaved {path}\n")
    print(f"{'round':<9}{'TTFT p50':>10}{'TTFT p95':>10}{'prefill p50':>13}{'GPU hits':>10}{'Mooncake hits':>15}")
    for name, s in summary.items():
        print(f"{name:<9}{s['ttft_s']['p50'] * 1000:>7.0f} ms{s['ttft_s']['p95'] * 1000:>7.0f} ms"
              f"{s['prefill_s']['p50'] * 1000:>10.0f} ms{s['gpu_hit_ratio']:>10.0%}{s['mooncake_hit_ratio']:>15.0%}")


if __name__ == "__main__":
    main()
