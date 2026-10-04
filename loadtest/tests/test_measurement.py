import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "loadtest"))
sys.path.insert(0, str(ROOT / "scripts"))

import stack_logs  # noqa: E402
import vllm_metrics  # noqa: E402
from trace_request import layer_breakdown  # noqa: E402

METRICS_BEFORE = """\
# HELP vllm:request_prefill_time_seconds Histogram of time spent in PREFILL phase.
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_sum{engine="0",model_name="llm"} 1.5
vllm:request_prefill_time_seconds_count{engine="0",model_name="llm"} 3.0
vllm:request_decode_time_seconds_sum{engine="0",model_name="llm"} 4.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="llm"} 0.01
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="llm"} 6.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="llm"} 3.0
vllm:generation_tokens_total{engine="0",model_name="llm"} 300.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="llm"} 2.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="llm"} 1.0
"""

METRICS_AFTER = """\
vllm:request_prefill_time_seconds_sum{engine="0",model_name="llm"} 1.55
vllm:request_prefill_time_seconds_count{engine="0",model_name="llm"} 4.0
vllm:request_decode_time_seconds_sum{engine="0",model_name="llm"} 4.6
vllm:request_queue_time_seconds_sum{engine="0",model_name="llm"} 0.012
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="llm"} 6.7
vllm:e2e_request_latency_seconds_count{engine="0",model_name="llm"} 4.0
vllm:generation_tokens_total{engine="0",model_name="llm"} 363.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="llm"} 3.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="llm"} 1.0
"""


def parse(text: str) -> dict:
    totals = {}
    for line in text.splitlines():
        match = vllm_metrics._LINE.match(line)
        if match and not line.startswith("#"):
            name, _, value = match.groups()
            totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def test_delta_isolates_one_request():
    d = vllm_metrics.delta(parse(METRICS_BEFORE), parse(METRICS_AFTER))
    assert round(d["prefill_s"], 6) == 0.05
    assert round(d["decode_s"], 6) == 0.6
    assert round(d["e2e_s"], 6) == 0.7
    assert d["generation_tokens"] == 63
    assert d["requests"] == 1  # summed across finished_reason labels
    assert d["requests_observed"] == 1


GATEWAY_LINES = [
    "INFO:     Started server process [1]",
    '{"event": "request", "request_id": "abc", "run_id": null, "path": "/v1/chat/completions", "status": 200, "ttfb_ms": 282.0, "total_ms": 879.5}',
    '{"event": "request", "request_id": "def", "run_id": "agent-1", "path": "/v1/chat/completions", "status": 200, "ttfb_ms": 900.0, "total_ms": 900.1}',
]
NGINX_LINES = [
    '172.18.0.6 "POST /v1/chat/completions HTTP/1.1" 200 rid=abc run=- rt=0.877 uct=0.000 uht=0.003 urt=0.876 upstream=172.18.0.3:8000',
    '172.18.0.6 "POST /v1/chat/completions HTTP/1.1" 200 rid=def run=agent-1 rt=0.899 uct=0.001 uht=0.898 urt=0.898 upstream=172.18.0.3:8000',
    '172.18.0.6 "GET /health HTTP/1.1" 200 rid=- run=- rt=0.001 uct=- uht=- urt=0.001 upstream=172.18.0.3:8000',
]


def test_logs_merge_by_request_id(monkeypatch):
    monkeypatch.setattr(stack_logs, "_compose_logs",
                        lambda service, since: GATEWAY_LINES if service == "gateway" else NGINX_LINES)
    merged = stack_logs.requests_by_id(datetime.now(timezone.utc))

    assert merged["abc"]["gateway_total_s"] == 0.8795
    assert merged["abc"]["nginx_total_s"] == 0.877
    assert merged["abc"]["upstream_total_s"] == 0.876
    assert merged["abc"]["run_id"] is None
    assert merged["def"]["run_id"] == "agent-1"
    assert merged["-"]["upstream_connect_s"] is None
    assert list(merged)[:2] == ["abc", "def"]  # log order is kept


def test_layer_breakdown_adds_up():
    client = {"e2e_s": 0.890}
    logs = {"gateway_total_s": 0.8795, "nginx_total_s": 0.877, "upstream_total_s": 0.876}
    engine = {"e2e_s": 0.860, "queue_s": 0.001, "prefill_s": 0.020, "decode_s": 0.830}
    layers = layer_breakdown(client, logs, engine)

    assert round(sum(layers.values()), 9) == client["e2e_s"]
    assert round(layers["vllm_api_server_s"], 6) == 0.016
    assert round(layers["engine_other_s"], 6) == 0.009


def test_layer_breakdown_without_logs():
    layers = layer_breakdown({"e2e_s": 1.0}, {}, {"e2e_s": 0.9, "queue_s": 0, "prefill_s": 0.1, "decode_s": 0.8})
    assert layers["gateway_s"] is None
    assert layers["engine_decode_s"] == 0.8
