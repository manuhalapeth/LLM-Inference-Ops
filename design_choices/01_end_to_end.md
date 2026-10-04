# Phase 1 design choices: end to end system and Grafana

Phase 1 was about putting a real application on the stack and being able to say exactly where every millisecond of a request goes. Most of the decisions here are about measurement: how to get honest numbers out of layers that weren't built to hand them to me.

## 1. A three agent crew instead of a single prompt

The point of the agent is to send traffic that looks like a real product, not a benchmark. A single prompt would have been easier, but real agent apps make several dependent LLM calls for one user task, and each call carries more context than the last.

So the crew has three agents working in sequence: an analyst pulls the facts out of an incident report, a writer turns them into an executive summary, and a reviewer checks the summary against the facts. One task becomes three LLM calls with about 1,850 prompt tokens and 800 generated tokens in total. That shape (repeated system prompts, growing context, a mix of reading and writing) is exactly what prefix caching and batching will care about later.

I set the temperature to zero so every run produces the same tokens. That makes runs comparable: if a later run is slower, it's the system, not a longer answer.

## 2. Grouping calls with a run ID header

To understand an agent task, I need to know which LLM calls belong to it. The crew sends the same run ID header on every call, the gateway passes it along and logs it, and NGINX logs it too. After a run, I filter both logs by that ID and get every call the task made, in order.

Full distributed tracing with OpenTelemetry is coming in Phase 3. A header and two log lines got me per call timings now, with almost no code, and it's the same idea a trace ID is built on.

## 3. Measuring inside vLLM by comparing snapshots

vLLM only publishes totals: how much time all requests have spent queued, in prefill and in decode, ever. It doesn't tell you those numbers for one request.

The trick is to send requests one at a time with nothing else running, read the totals before and after each request, and subtract. The difference is exactly that one request's queue, prefill and decode time. The tracer checks that vLLM saw exactly one request in each window, and all 15 traced requests passed.

This only works with one request at a time, which is fine for this phase. Under load in Phase 4, the dashboard's percentiles take over.

## 4. Splitting a request into layers by nesting timers

Each layer records how long a request spent inside it: the client measures the whole thing, the gateway logs its total, NGINX logs its total and how long vLLM took, and vLLM's metrics give the engine time. Because each timer sits inside the one before, subtracting neighbours gives the time each layer added on its own.

The catch is resolution. NGINX logs to the millisecond, so its own share shows up as zero. That's still a useful answer: NGINX costs less than a millisecond.

## 5. Running the agent on the GPU machine, outside Docker

The agent and the tracer run as plain Python on the GPU machine, not as containers in the stack. That lets them read vLLM's metrics and the container logs directly, and it's also more realistic: the application is a client of the inference stack, not part of it.

That meant publishing vLLM's port, which I only did on localhost, the same way as everything else. Nothing new is reachable from the internet.

## 6. An nvidia smi based exporter instead of DCGM

DCGM is NVIDIA's standard tool for GPU metrics, but it's built for data center cards and its support for GeForce cards like the RTX 5090 is limited. The nvidia_gpu_exporter just runs nvidia smi, which works on any NVIDIA GPU, and gives utilization, memory, power and temperature.

I ran it in its demo mode on my laptop first to get the exact metric names, because the documentation summary I'd read had them wrong. Small check, but it meant the dashboard worked the first time it saw a real GPU.

## 7. A dashboard defined in code and checked before it ran

The Grafana dashboard is a JSON file in the repo that Grafana loads at startup, so it comes back identically on every new machine. I took the metric names straight from vLLM 0.30.0's source code and ran all 32 queries through Prometheus's own checker before renting a GPU. On the real machine, I confirmed every query returned data.

One thing I learned: dashboard percentiles are estimates. Prometheus works them out from histogram buckets, so the inter token latency panel showed about 5.5 ms while the exact measurement was 9.55 ms. The dashboard is for watching trends live; the notebook uses the exact traced numbers.

## 8. Warming up, and reporting cold and cached numbers separately

Phase 0's first request had a time to first token of 281 ms. In Phase 1, a warmed up server answered in 14 to 17 ms. The first request after startup pays one time costs, so the tracer now always sends a warmup request that isn't measured.

Repeats of the same prompt hit vLLM's prefix cache, which skips most of the prefill. For the 477 token document, prefill dropped from 33 ms on the first request to 12.5 ms on repeats. If I'd only reported the median, the cache would have hidden the real prefill cost, so the notebook shows the first request and the repeats separately.

## 9. Same GPU, different machine

The Hungary machine from Phase 0 wasn't available, and no RTX 5090 was listed on Secure Cloud. Switching to a different GPU would have broken the comparison with Phase 0, so I kept the RTX 5090 and picked a verified host in Denmark with 99.12% reliability.

The differences are on the host side: a desktop Ryzen CPU instead of a server EPYC, and PCIe 5.0 x8 instead of x16. Neither matters much for serving a model that's already loaded, and the results agree: the KV cache came out at exactly the same 209,472 tokens. The specification records both machines.

## 10. Installing packages before starting containers

Partway through the session, the GPU panels went blank. The exporter's container had lost access to the GPU and kept failing with "Failed to initialize NVML: Unknown Error". Installing python3 venv on the host had triggered a systemd reload, and on this kind of setup that cuts running containers off from the GPU for any new process. vLLM had already opened the GPU, so it carried on and every measurement was fine.

Restarting the exporter fixed it on the spot. The real fix is in the setup script, which now installs everything the host needs before any container starts, so the reload happens before there's anything to break.
