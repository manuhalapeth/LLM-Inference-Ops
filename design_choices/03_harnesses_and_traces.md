# Phase 3 design choices: harnesses, traces and evals

Phase 3 was about trust. The system worked; now it had to refuse bad requests cheaply, explain itself when something went wrong, and prove that a change didn't make it worse. Here are the calls I made and why.

## 1. Every check runs in the gateway, before the GPU

Once a request reaches vLLM, it gets compute whether it deserves it or not. So every check lives in the gateway, in front of NGINX and vLLM. A rejected request costs a fraction of a millisecond of CPU instead of seconds of GPU time.

The numbers backed this up. The checks took 0.27 ms at the median. Without them, the four requests that should have been blocked used almost 4 seconds of GPU time between them.

## 2. Counting tokens exactly, with the model's own tokenizer

A token budget is only useful if the count is right. Guessing from characters can be off by a lot, so the gateway uses Qwen's real tokenizer, built into its image so it never needs the internet at startup.

The tokenizer only counts the message text, though, and the model also sees its chat template. I measured the difference against vLLM's own counts: 5 tokens per message, 3 for the assistant turn, and a 21 token default system prompt when a request has none. With those rules, the gateway's count matched vLLM's exactly on every request I checked.

## 3. A simple, explainable injection check

The prompt injection check is a set of phrase patterns, like asking the model to ignore its previous instructions or to reveal its system prompt. It only looks at user messages, because the system prompt comes from the application and is trusted.

I know this only catches common phrasings, not a determined attacker. I picked it because it's fast, it's easy to explain, and every rejection says exactly which rule fired. The eval set includes a harmless question about instructions to make sure it doesn't block normal users. A small classifier model would be the next step, at the cost of real latency on every request.

The strongest argument for the check came from turning it off. Asked to reveal a secret discount code, the model refused, and printed the code in the refusal. Twice. Its safety training wasn't enough on its own.

## 4. Rejecting clearly instead of quietly fixing

When a request asks for too many output tokens or sends too long a prompt, the gateway rejects it with a clear reason instead of trimming it behind the client's back. A silently shortened answer looks like a model problem; a clear error tells the caller exactly what to change.

The one thing the gateway does fill in is a missing output limit. Without it, vLLM lets a response run until the context is full. In the eval, a request with no limit was capped at 1,024 tokens; with the harness off, the same request ran to 1,309 tokens.

## 5. Rate limits per user, on requests and on tokens

Each user (identified by a user ID header) gets two token buckets: one for requests per minute and one for tokens per minute. Counting only requests would let one user send a few enormous prompts; counting only tokens would let them flood the server with tiny ones.

A rejected request gets a 429 and a retry after header saying how long to wait. In the burst drill, a little over 60 requests got through instead of exactly 60, because the bucket refills while the burst is still running. That's correct behaviour, and I had to fix my own drill to expect it.

## 6. Time limits and hang ups both cancel the work on the GPU

A request with a time limit is cut off when the limit passes, and a client that disconnects early is noticed straight away. Either way, the gateway closes its connection to NGINX, NGINX closes its connection to vLLM, and vLLM aborts the request. Closing the connection is the cancellation signal all the way down the chain.

The drills showed it working: a 2 second limit ended the stream at 2.00 seconds with vLLM stopping at 206 of 1,000 tokens, and a client hanging up after 1 second stopped vLLM at 103 tokens. Nobody pays for tokens nobody will read.

## 7. One trace from the agent into vLLM

Every layer reports to Jaeger through OpenTelemetry. The agent starts the trace, the gateway adds its own span plus a span for the harness, NGINX adds one, and vLLM adds an `llm_request` span with its queue, prefill and decode times. The trace context travels in the standard traceparent header, so every hop joins the same trace.

NGINX needed its image swapped for the official one with the OpenTelemetry module built in. vLLM 0.30.0 already had tracing built in; it just needed an endpoint, which is one config change (`baseline_tracing.yaml`). The gateway also returns the trace ID to the client, so any failed request can be looked up directly.

The result is one picture per agent task: 19 spans across 4 services, showing at a glance that almost all of the time is inside vLLM.

## 8. Evals scored by rules, not by another model

The eval set has 22 cases: correctness, following format instructions, summarizing without losing facts, refusing harmful requests, and every harness. Each is scored by a simple rule, like matching text, a regular expression or parsing JSON. I avoided using another LLM as the judge, because then the judge needs evaluating too, and its verdicts can change between runs.

Running the same set with the harnesses on and off turned it into a before and after: 20 of 22 passing with them, 16 without, plus two leaked secrets.

The evals also taught me to check the checker. One failure was my own scoring bug: the model answered in three words, but one of them was a compound word joined by a hyphen, and my counter split it in two. The other was real: the model's incident summary dropped the impact numbers and claimed "minimal impact" for an outage that hit thousands of customers. That's the kind of quiet mistake no one notices without an automatic check.

## 9. Failure drills as code

Each failure type has a drill that triggers it on purpose and checks the outcome: injection, oversized prompt, rate limit, timeout, client hang up, and vLLM going down. Each drill records what the client saw, the trace ID, and how the gateway's metrics changed, so every failure is visible on the dashboard as well as in the logs. Writing them as code means they can run again after any change.

## 10. NGINX finds vLLM again on its own

In Phase 2, NGINX looked up vLLM's address once at startup, so a restarted vLLM container could leave it pointing at nothing. NGINX 1.27.5 can look the address up again every few seconds, so I switched that on and added a 5 second connect timeout. In the drill, clients got a clean error within 5 seconds while vLLM was down, and traffic came back by itself 69 seconds after vLLM restarted, with no NGINX restart.

## 11. Testing the whole stack before renting a GPU

This phase had more moving parts than any before it, so I ran the full stack in Docker on my laptop with a small stand in for vLLM: real gateway image, real NGINX with tracing, real Jaeger, the evals, the drills and the trace export. It caught several real problems before any GPU time was spent, including a conflict between CrewAI and my tracing setup, the wrong hook for naming spans, and Jaeger having changed its query API.

## 12. Switching off automatic updates on rented machines

The session had a real incident. A few minutes after the machine booted, Ubuntu's automatic updates upgraded about 200 packages, including the NVIDIA driver, underneath the driver that was already running. vLLM kept working because it already had the GPU open. But when the vLLM down drill restarted it, the new container couldn't start: the driver files on disk no longer matched the driver in memory.

I fixed it on the spot by unloading and reloading the NVIDIA kernel modules, with no reboot, then reran everything on the new driver. The eval results matched the earlier run case for case. The permanent fix is that the setup script now switches off automatic updates before it touches the GPU. These machines live for an hour; an update in the middle of a measurement is never worth it. In hindsight, the GPU exporter failure in Phase 1 may well have had the same trigger.
