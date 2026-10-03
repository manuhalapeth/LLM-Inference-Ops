"""API gateway: the front door every request passes before it reaches the GPU.

Phase 0: a transparent pass-through to NGINX -> vLLM that tags every request
with an ID and records how long it took. Harnesses (Phase 3) and tracing
(Phase 3) are added here, before the request is forwarded.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://nginx:80")
UPSTREAM_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "600"))

# Known OpenAI-compatible routes. Anything else is labelled "other" so a
# client can't blow up metric cardinality by inventing paths.
KNOWN_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/models"}

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)

REQUESTS = Counter(
    "gateway_requests_total", "Requests handled by the gateway", ["path", "status"]
)
REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds",
    "Time from request received to last byte sent",
    ["path"],
    buckets=LATENCY_BUCKETS,
)
TIME_TO_FIRST_BYTE = Histogram(
    "gateway_time_to_first_byte_seconds",
    "Time from request received to first byte from upstream (approximates TTFT when streaming)",
    ["path"],
    buckets=LATENCY_BUCKETS,
)

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # the gateway logs its own line per request


def _metric_path(path: str) -> str:
    return path if path in KNOWN_PATHS else "other"


def _log(**fields) -> None:
    logger.info(json.dumps(fields))


def create_app(client: httpx.AsyncClient | None = None) -> FastAPI:
    """Build the app. Tests inject a client with a mock transport."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client or httpx.AsyncClient(
            base_url=UPSTREAM_URL,
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=10.0),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(title="LLM Inference Ops gateway", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.api_route("/v1/{rest:path}", methods=["GET", "POST"])
    async def proxy(rest: str, request: Request):
        path = f"/v1/{rest}"
        metric_path = _metric_path(path)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()

        upstream_request = app.state.client.build_request(
            request.method,
            path,
            params=request.query_params,
            content=await request.body(),
            headers={
                "content-type": request.headers.get("content-type", "application/json"),
                "x-request-id": request_id,
            },
        )
        try:
            upstream = await app.state.client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            elapsed = time.perf_counter() - start
            REQUESTS.labels(metric_path, "502").inc()
            REQUEST_DURATION.labels(metric_path).observe(elapsed)
            _log(event="upstream_error", request_id=request_id, path=path,
                 error=type(exc).__name__, total_ms=round(elapsed * 1000, 1))
            return JSONResponse(
                {"error": {"type": "upstream_error", "message": f"{type(exc).__name__}: {exc}"}},
                status_code=502,
                headers={"x-request-id": request_id},
            )

        async def stream_body():
            first_byte_at = None
            try:
                async for chunk in upstream.aiter_bytes():
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()
                        TIME_TO_FIRST_BYTE.labels(metric_path).observe(first_byte_at - start)
                    yield chunk
            finally:
                await upstream.aclose()
                elapsed = time.perf_counter() - start
                REQUESTS.labels(metric_path, str(upstream.status_code)).inc()
                REQUEST_DURATION.labels(metric_path).observe(elapsed)
                _log(
                    event="request",
                    request_id=request_id,
                    path=path,
                    status=upstream.status_code,
                    ttfb_ms=round((first_byte_at - start) * 1000, 1) if first_byte_at else None,
                    total_ms=round(elapsed * 1000, 1),
                )

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers={"x-request-id": request_id},
        )

    return app


app = create_app()
