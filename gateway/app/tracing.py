"""OpenTelemetry setup for the gateway.

When OTEL_EXPORTER_OTLP_ENDPOINT is set, every request becomes a trace that
continues into NGINX and vLLM through the traceparent header. When it isn't
(tests, local runs), tracing is a no-op.
"""

import os

import httpx
from fastapi import FastAPI
from opentelemetry import trace

tracer = trace.get_tracer("gateway")


def setup(app: FastAPI, client: httpx.AsyncClient) -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create())  # service name from OTEL_SERVICE_NAME
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(insecure=True)))
    trace.set_tracer_provider(provider)
    # Skip the per-message "http receive/send" spans: they add noise, not information.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz", exclude_spans=["receive", "send"])
    # Adds traceparent to upstream calls. The client is async, so the hook is too.
    HTTPXClientInstrumentor.instrument_client(client, request_hook=_name_upstream_span)


async def _name_upstream_span(span, request) -> None:
    if span.is_recording():
        method = request.method.decode() if isinstance(request.method, bytes) else request.method
        span.update_name(f"upstream {method} {request.url.path}")


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
