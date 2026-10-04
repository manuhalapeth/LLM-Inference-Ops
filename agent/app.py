"""The application: a CrewAI crew that summarizes a document.

Three agents work in sequence, each making its own LLM calls through the
gateway, so one task becomes several requests to vLLM:

    analyst  ->  extracts the key facts from the document
    writer   ->  writes an executive summary from those facts
    reviewer ->  checks the summary against the facts and fixes it

Every LLM call carries the same x-run-id header, so the gateway and NGINX
logs can be grouped by run. After the run, this script collects those logs
and vLLM's metrics and saves everything to results/01_end_to_end/.

Run on the GPU box from the repo root, with the stack up:

    .venv/bin/python agent/app.py
    .venv/bin/python agent/app.py --document agent/documents/incident_report.md
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# CrewAI's own usage telemetry is off. Our tracing (below) is separate.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loadtest"))
import stack_logs  # noqa: E402
import vllm_metrics  # noqa: E402
from results_logger import save_run  # noqa: E402


def setup_tracing():
    """Send this run's trace to Jaeger, if OTEL_EXPORTER_OTLP_ENDPOINT is set.

    The run is the root span; every LLM call is an HTTP span under it, and the
    traceparent header carries the trace on through the gateway, NGINX and vLLM.
    """
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None, None
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "agent")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(insecure=True)))
    trace.set_tracer_provider(provider)
    # The OpenAI client CrewAI uses is built on httpx, so each LLM call becomes a span.
    HTTPXClientInstrumentor().instrument(request_hook=lambda span, request: span.is_recording()
                                         and span.update_name(f"llm call {request.url.path}"))
    return provider, trace.get_tracer("agent")


def build_crew(llm: LLM) -> Crew:
    analyst = Agent(
        role="Incident analyst",
        goal="Pull the facts that matter out of an incident report",
        backstory="You are a site reliability engineer who reads incident reports and extracts "
                  "exactly what happened, without opinions or guesses.",
        llm=llm,
        allow_delegation=False,
    )
    writer = Agent(
        role="Technical writer",
        goal="Turn incident facts into a clear summary for engineering leadership",
        backstory="You write short, precise summaries that a busy engineering director can read in a minute.",
        llm=llm,
        allow_delegation=False,
    )
    reviewer = Agent(
        role="Reviewer",
        goal="Make sure the summary is accurate and complete",
        backstory="You check every claim in a summary against the source facts and fix anything wrong or missing.",
        llm=llm,
        allow_delegation=False,
    )

    extract = Task(
        description="Read this incident report and list the key facts: timeline, root cause, "
                    "impact (with numbers), how it was resolved, and the follow-up actions.\n\n"
                    "Incident report:\n{document}",
        expected_output="A bullet list of facts grouped under: Timeline, Root cause, Impact, Resolution, Follow-ups.",
        agent=analyst,
    )
    summarize = Task(
        description="Using the extracted facts, write an executive summary of the incident.",
        expected_output="A summary of at most 150 words: what happened, why, the impact, and what will change.",
        agent=writer,
        context=[extract],
    )
    review = Task(
        description="Check the summary against the extracted facts. Fix any inaccurate or missing "
                    "numbers, times or actions, and return the final summary.",
        expected_output="The final, corrected executive summary (at most 150 words).",
        agent=reviewer,
        context=[extract, summarize],
    )
    return Crew(agents=[analyst, writer, reviewer], tasks=[extract, summarize, review],
                process=Process.sequential, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document", default=str(ROOT / "agent" / "documents" / "incident_report.md"))
    parser.add_argument("--url", default=os.getenv("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--vllm-metrics", default="http://localhost:8000/metrics")
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "llm"))
    parser.add_argument("--no-collect", action="store_true", help="skip collecting logs and metrics (local testing)")
    parser.add_argument("--phase", default="01_end_to_end", help="results/<phase>/ to save into")
    args = parser.parse_args()

    provider, tracer = setup_tracing()
    run_id = f"agent-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    llm = LLM(
        model=args.model,
        custom_openai=True,
        base_url=f"{args.url}/v1",
        api_key="not-needed",
        temperature=0,
        max_tokens=512,
        default_headers={"x-run-id": run_id},
    )
    crew = build_crew(llm)

    task_finished_at = []
    crew.task_callback = lambda output: task_finished_at.append((output.agent, time.perf_counter()))

    document = Path(args.document).read_text()
    started_utc = datetime.now(timezone.utc) - timedelta(seconds=2)
    before = None if args.no_collect else vllm_metrics.scrape(args.vllm_metrics)
    start = time.perf_counter()
    trace_id = None
    if tracer is not None:
        with tracer.start_as_current_span("agent_run", attributes={"run.id": run_id, "agents": 3}) as span:
            trace_id = format(span.get_span_context().trace_id, "032x")
            result = crew.kickoff(inputs={"document": document})
        provider.force_flush()
    else:
        result = crew.kickoff(inputs={"document": document})
    total_s = time.perf_counter() - start

    tasks, previous = [], start
    for agent, finished in task_finished_at:
        tasks.append({"agent": agent.strip(), "duration_s": finished - previous})
        previous = finished

    print(f"\nrun {run_id}: {total_s:.1f} s, {len(tasks)} tasks" + (f", trace {trace_id}" if trace_id else ""))
    for task in tasks:
        print(f"  {task['agent']:<20}{task['duration_s']:6.1f} s")
    print(f"\n{result.raw}\n")
    if args.no_collect:
        return

    time.sleep(1)  # let vLLM, the gateway and NGINX record the last call
    engine = vllm_metrics.delta(before, vllm_metrics.scrape(args.vllm_metrics))
    # In log order, which is the order the crew made them.
    calls = [row for row in stack_logs.requests_by_id(started_utc).values() if row.get("run_id") == run_id]

    path = save_run(
        args.phase, "agent_run",
        metrics={
            "run_id": run_id,
            "trace_id": trace_id,
            "total_s": total_s,
            "tasks": tasks,
            "llm_calls": len(calls),
            "calls": calls,
            "engine": engine,
            "final_summary": result.raw,
        },
        config={"model": os.getenv("MODEL_NAME"), "vllm_image": os.getenv("VLLM_IMAGE"),
                "vllm_config": os.getenv("VLLM_CONFIG", "baseline"), "crewai": "1.15.23"},
        load={"concurrency": 1, "document": Path(args.document).name, "agents": 3, "process": "sequential"},
    )
    print(f"{len(calls)} LLM calls, {engine['prompt_tokens']:.0f} prompt / {engine['generation_tokens']:.0f} generated tokens")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
