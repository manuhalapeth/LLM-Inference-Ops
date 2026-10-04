import json

import httpx
from fastapi.testclient import TestClient

from app.main import create_app


def make_client(handler) -> TestClient:
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://upstream")
    return TestClient(create_app(client=upstream))


def test_forwards_request_and_returns_upstream_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["request_id"] = request.headers["x-request-id"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    with make_client(handler) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "llm", "messages": [{"role": "user", "content": "hello"}]},
            headers={"x-request-id": "abc123"},
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["messages"][0]["content"] == "hello"
    assert seen["request_id"] == "abc123"
    assert resp.headers["x-request-id"] == "abc123"


def test_generates_request_id_when_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"request_id": request.headers["x-request-id"]})

    with make_client(handler) as client:
        resp = client.get("/v1/models")

    assert len(resp.headers["x-request-id"]) == 32
    assert resp.json()["request_id"] == resp.headers["x-request-id"]


def test_streams_server_sent_events_unchanged():
    sse = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    with make_client(handler) as client:
        resp = client.post("/v1/chat/completions", json={"stream": True})

    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == sse


def test_passes_through_upstream_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    with make_client(handler) as client:
        resp = client.post("/v1/chat/completions", json={})

    assert resp.status_code == 400


def test_returns_502_when_upstream_is_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with make_client(handler) as client:
        resp = client.post("/v1/chat/completions", json={})

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"


def test_metrics_are_recorded():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with make_client(handler) as client:
        client.post("/v1/chat/completions", json={})
        client.post("/v1/not-a-real-route", json={})
        metrics = client.get("/metrics").text

    assert 'gateway_requests_total{path="/v1/chat/completions",status="200"}' in metrics
    assert 'gateway_requests_total{path="other",status="200"}' in metrics
    assert "gateway_time_to_first_byte_seconds_bucket" in metrics


def test_forwards_run_id_and_logs_it(caplog):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["run_id"] = request.headers.get("x-run-id")
        return httpx.Response(200, json={})

    with caplog.at_level("INFO", logger="gateway"), make_client(handler) as client:
        client.post("/v1/chat/completions", json={}, headers={"x-run-id": "run-42"})

    assert seen["run_id"] == "run-42"
    logged = [json.loads(r.getMessage()) for r in caplog.records if r.name == "gateway"]
    assert logged[-1]["run_id"] == "run-42"
