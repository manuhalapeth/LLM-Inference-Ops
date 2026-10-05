"""API gateway: the front door every request passes before it reaches the GPU.

Every request gets an ID and is timed (Phase 0). Before it's forwarded to
NGINX and vLLM, it goes through the harnesses (Phase 3): validation, a
prompt injection check, token budgets, per user rate limits and a cap on
requests in flight. Forwarded requests have a time limit, and if the client
disconnects, the request upstream is cancelled so the GPU stops working on it.
Each request is a trace that continues into NGINX and vLLM.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest, multiprocess

from . import tracing
from .config import Settings
from .harness import Harness, Rejection

# Known OpenAI-compatible routes. Anything else is labelled "other" so a
# client can't blow up metric cardinality by inventing paths.
KNOWN_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/models"}
GENERATION_PATHS = {"/v1/chat/completions", "/v1/completions"}

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
REJECTIONS = Counter(
    "gateway_rejections_total", "Requests stopped by a harness before reaching the GPU", ["reason"]
)
REJECTED_TOKENS = Counter(
    "gateway_rejected_prompt_tokens_total", "Prompt tokens that never reached the GPU", ["reason"]
)
HARNESS_SECONDS = Histogram(
    "gateway_harness_seconds", "Time spent running the harness checks",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
)
# "livesum" adds the gauge up across worker processes when there are several.
IN_FLIGHT = Gauge("gateway_in_flight_requests", "Requests forwarded upstream and not finished yet",
                  multiprocess_mode="livesum")
ENDED_EARLY = Counter(
    "gateway_requests_ended_early_total", "Forwarded requests that did not finish normally", ["why"]
)

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # the gateway logs its own line per request


def _metric_path(path: str) -> str:
    return path if path in KNOWN_PATHS else "other"


def _log(**fields) -> None:
    logger.info(json.dumps(fields))


def _error(status: int, kind: str, message: str, headers: dict) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message, "code": status}},
                        status_code=status, headers=headers)


def create_app(client: httpx.AsyncClient | None = None, settings: Settings | None = None,
               harness: Harness | None = None) -> FastAPI:
    """Build the app. Tests inject a client with a mock transport and their own settings."""
    settings = settings or Settings()
    harness = harness or Harness(settings)
    upstream_client = client or httpx.AsyncClient(
        base_url=settings.upstream_url,
        timeout=httpx.Timeout(settings.max_timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = upstream_client
        app.state.in_flight = 0
        yield
        await upstream_client.aclose()

    app = FastAPI(title="LLM Inference Ops gateway", lifespan=lifespan)
    tracing.setup(app, upstream_client)
    _log(event="startup", harnesses=settings.harnesses_enabled, exact_token_count=harness.counter.exact)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        if os.getenv("PROMETHEUS_MULTIPROC_DIR"):  # several worker processes: merge their metrics
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.api_route("/v1/{rest:path}", methods=["GET", "POST"])
    async def proxy(rest: str, request: Request):
        path = f"/v1/{rest}"
        metric_path = _metric_path(path)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        # Groups the several LLM calls one agent task makes. Set by the client.
        run_id = request.headers.get("x-run-id")
        user = request.headers.get("x-user-id") or "anonymous"
        start = time.perf_counter()
        trace_id = tracing.current_trace_id()
        response_headers = {"x-request-id": request_id, **({"x-trace-id": trace_id} if trace_id else {})}
        span = tracing.trace.get_current_span()
        if span.is_recording():
            span.update_name(f"gateway {request.method} {path}")
        span.set_attributes({"request.id": request_id, "user.id": user, **({"run.id": run_id} if run_id else {})})

        raw = await request.body()
        prompt_tokens = None

        def reject(r: Rejection) -> JSONResponse:
            REQUESTS.labels(metric_path, str(r.status)).inc()
            REJECTIONS.labels(r.reason).inc()
            if prompt_tokens:
                REJECTED_TOKENS.labels(r.reason).inc(prompt_tokens)
            span.set_attributes({"harness.result": "rejected", "harness.reason": r.reason})
            _log(event="rejected", request_id=request_id, run_id=run_id, user=user, path=path,
                 status=r.status, reason=r.reason, prompt_tokens=prompt_tokens, trace_id=trace_id,
                 total_ms=round((time.perf_counter() - start) * 1000, 2))
            headers = {**response_headers, "x-harness-reason": r.reason}
            if r.retry_after_s is not None:
                headers["retry-after"] = str(max(1, round(r.retry_after_s)))
            return _error(r.status, r.reason, r.message, headers)

        # Harnesses: everything here happens before any GPU work.
        if request.method == "POST" and path in GENERATION_PATHS and settings.harnesses_enabled:
            with tracing.tracer.start_as_current_span("harness") as harness_span:
                checks_start = time.perf_counter()
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = None
                decision = harness.check(path, body, user)
                HARNESS_SECONDS.observe(time.perf_counter() - checks_start)
                prompt_tokens = decision.prompt_tokens
                harness_span.set_attributes({
                    "harness.prompt_tokens": prompt_tokens or 0,
                    "harness.result": "rejected" if decision.rejection else "passed",
                    **({"harness.reason": decision.rejection.reason} if decision.rejection else {}),
                })
            if decision.rejection:
                return reject(decision.rejection)
            if decision.body is not body:  # the harness filled in a default max_tokens
                raw = json.dumps(decision.body).encode()
            if app.state.in_flight >= settings.max_in_flight:
                return reject(Rejection(503, "overloaded",
                                        f"{settings.max_in_flight} requests already in flight", retry_after_s=1))

        try:
            timeout_s = min(float(request.headers.get("x-timeout-s", settings.default_timeout_s)),
                            settings.max_timeout_s)
        except ValueError:
            timeout_s = settings.default_timeout_s
        deadline = start + timeout_s

        upstream_request = app.state.client.build_request(
            request.method,
            path,
            params=request.query_params,
            content=raw,
            headers={
                "content-type": request.headers.get("content-type", "application/json"),
                "x-request-id": request_id,
                **({"x-run-id": run_id} if run_id else {}),
            },
        )
        app.state.in_flight += 1
        IN_FLIGHT.inc()

        def finished() -> None:
            app.state.in_flight -= 1
            IN_FLIGHT.dec()

        try:
            upstream = await asyncio.wait_for(app.state.client.send(upstream_request, stream=True),
                                              timeout=max(deadline - time.perf_counter(), 0.001))
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            finished()
            elapsed = time.perf_counter() - start
            timed_out = isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException))
            status = 504 if timed_out else 502
            if timed_out:
                ENDED_EARLY.labels("timeout").inc()
            REQUESTS.labels(metric_path, str(status)).inc()
            REQUEST_DURATION.labels(metric_path).observe(elapsed)
            _log(event="upstream_error", request_id=request_id, run_id=run_id, user=user, path=path,
                 status=status, error=type(exc).__name__, trace_id=trace_id, total_ms=round(elapsed * 1000, 1))
            kind = "upstream_timeout" if timed_out else "upstream_error"
            return _error(status, kind, f"{type(exc).__name__}: {exc}", response_headers)

        is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")

        async def stream_body():
            first_byte_at, outcome = None, "completed"
            chunks = upstream.aiter_bytes()
            try:
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        chunk = await asyncio.wait_for(chunks.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()
                        TIME_TO_FIRST_BYTE.labels(metric_path).observe(first_byte_at - start)
                    yield chunk
            except asyncio.TimeoutError:
                # Stop reading: closing the upstream connection makes vLLM abort the request.
                outcome = "timeout"
                ENDED_EARLY.labels("timeout").inc()
                if is_sse:
                    yield (b'data: {"error": {"type": "timeout", "message": "request exceeded its '
                           + f"{timeout_s:g}".encode() + b's time limit"}}\n\n')
            except (asyncio.CancelledError, GeneratorExit):
                # The client went away. Closing upstream cancels the request in vLLM.
                outcome = "client_disconnected"
                ENDED_EARLY.labels("client_disconnected").inc()
                raise
            finally:
                await upstream.aclose()
                finished()
                elapsed = time.perf_counter() - start
                REQUESTS.labels(metric_path, str(upstream.status_code)).inc()
                REQUEST_DURATION.labels(metric_path).observe(elapsed)
                _log(
                    event="request",
                    request_id=request_id,
                    run_id=run_id,
                    user=user,
                    path=path,
                    status=upstream.status_code,
                    outcome=outcome,
                    prompt_tokens=prompt_tokens,
                    trace_id=trace_id,
                    ttfb_ms=round((first_byte_at - start) * 1000, 1) if first_byte_at else None,
                    total_ms=round(elapsed * 1000, 1),
                )

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=response_headers,
        )

    return app


app = create_app()
