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
    # Hits served by an external KV store through a KV connector (Mooncake).
    "vllm:external_prefix_cache_queries_total": "external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits_total": "external_prefix_cache_hits",
}

# Mooncake store operations (vllm/.../mooncake/store/metrics.py), kept per
# operation: save_put writes KV to the store, save_exists checks before
# writing, load_get reads KV back into the GPU.
MOONCAKE_OPERATIONS = ("save_put", "save_exists", "load_get")
MOONCAKE_SERIES = {
    "vllm:mooncake_store_operation_total": "calls",
    "vllm:mooncake_store_operation_bytes_total": "bytes",
    "vllm:mooncake_store_operation_time_seconds_sum": "time_s",
}
_OPERATION = re.compile(r'operation="([^"]+)"')


def scrape(url: str = "http://localhost:8000/metrics") -> dict[str, float]:
    """Fetch /metrics and sum each metric over its labels.

    Mooncake store metrics are also kept per operation, as "name[operation]".
    """
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode()
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE.match(line)
        if match and not line.startswith("#"):
            name, labels, value = match.groups()
            try:
                number = float(value)
            except ValueError:
                continue
            totals[name] = totals.get(name, 0.0) + number
            if name in MOONCAKE_SERIES and labels and (op := _OPERATION.search(labels)):
                key = f"{name}[{op.group(1)}]"
                totals[key] = totals.get(key, 0.0) + number
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
    result["mooncake"] = {
        op: {short: diff(f"{name}[{op}]") for name, short in MOONCAKE_SERIES.items()}
        for op in MOONCAKE_OPERATIONS
    }
    return result
