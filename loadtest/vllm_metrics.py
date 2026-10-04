"""Read vLLM's Prometheus metrics and work out what happened between two snapshots.

vLLM only exposes totals (counters and histogram sums), not per-request
numbers. But if you take a snapshot, send requests one at a time, and take
another snapshot, the difference is exactly what those requests cost inside
the engine: time queued, time in prefill, time in decode, tokens, and so on.

Metric names are from vLLM 0.30.0 (vllm/v1/metrics/loggers.py).
Standard library only.
"""

import re
import urllib.request

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")

# Histogram name (without _sum/_count) -> short name used in results.
HISTOGRAMS = {
    "vllm:request_queue_time_seconds": "queue",
    "vllm:request_prefill_time_seconds": "prefill",
    "vllm:request_decode_time_seconds": "decode",
    "vllm:request_inference_time_seconds": "inference",
    "vllm:e2e_request_latency_seconds": "e2e",
    "vllm:time_to_first_token_seconds": "ttft",
}

COUNTERS = {
    "vllm:request_success_total": "requests",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
}


def scrape(url: str = "http://localhost:8000/metrics") -> dict[str, float]:
    """Fetch /metrics and sum each metric over its labels."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode()
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE.match(line)
        if match and not line.startswith("#"):
            name, _, value = match.groups()
            try:
                totals[name] = totals.get(name, 0.0) + float(value)
            except ValueError:
                pass
    return totals


def delta(before: dict[str, float], after: dict[str, float]) -> dict:
    """What happened inside vLLM between two snapshots.

    Times are totals across all requests in the window, in seconds, so for a
    single request they are that request's times.
    """
    def diff(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    result = {short: diff(name) for name, short in COUNTERS.items()}
    for name, short in HISTOGRAMS.items():
        result[f"{short}_s"] = diff(f"{name}_sum")
    result["requests_observed"] = diff("vllm:e2e_request_latency_seconds_count")
    return result
