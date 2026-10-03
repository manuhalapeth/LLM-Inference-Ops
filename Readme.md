# Project Overview: LLM Inference Ops

## Summary

This project builds a small-scale version of a production LLM serving stack: an agent application backed by a self-hosted LLM on a rented GPU, with the infrastructure that surrounds it in production. The system is then loaded until it breaks, profiled to find out why, tuned, and scaled, with every step measured.

The application itself is not the focus. The agent could do anything. The goal is to show how an LLM system is run, measured and improved: getting requests in, generating responses quickly and cheaply, and keeping the system stable under load.

## Architecture: one request through the system

Example request: "Summarize this document"

```
User / Locust
│
▼
CrewAI agent (app.py) ← the application: an agent that calls the LLM, possibly several times
│
▼
API gateway ← front door: auth, rate limits, checks ("harnesses"), logging
│
▼
NGINX (load balancer) ← decides which vLLM server gets the request
│
▼
vLLM on a rented GPU ← the model runs here, producing tokens
│
▼
Grafana dashboard ← charts: latency, throughput, GPU memory, errors, over time
```

## Components

- **CrewAI agent:** the application layer. It generates realistic, multi-step LLM traffic so the system is tested with real-looking workloads, not just single requests.
- **API gateway:** the checkpoint every request passes before it reaches the GPU. Once a request reaches vLLM, it consumes GPU time whether it is valid or not, so checks such as "is this prompt too long?", "is this user over their limit?" and "is this malicious?" run here, before inference. These checks are the **harnesses**.
- **NGINX:** distributes traffic across vLLM servers. With a single server it passes requests through, but the structure is ready for multiple replicas.
- **vLLM:** the inference engine. It loads the model into GPU memory, batches requests from many users together, and manages the KV cache.
- **Grafana:** observability. Dashboards for latency, throughput, GPU memory, KV cache usage and errors show what the system is actually doing.

## Phases

1. **Make it work.** Get a request through every layer end to end, with dashboards on top.
2. **KV cache management (Mooncake).** Under load, the resource that runs out first is GPU memory for the KV cache: each active conversation holds memory for every token it has seen. Mooncake stores and moves that cache more efficiently across GPUs.
3. **Harnesses and traces.** Add checks at the gateway and record every request's path (a trace), so failures can be located and explained. Use traces and evaluation results to change the system.
4. **Load testing and profiling.** Use Locust and a traffic-generating agent to send increasing numbers of concurrent requests until latency explodes or GPU memory runs out. Find that point, identify the cause, change one vLLM setting (such as the maximum number of concurrent sequences or the GPU memory fraction), re-run and compare. With multiple GPUs, GPU slicing (HAMi) lets several workloads share a GPU.
5. **Scale.** Once one GPU is understood, move to multiple GPUs and multiple vLLM instances, where Mooncake and GPU slicing (HAMi) pay off fully.

**The notebook runs alongside every phase from day one.** A Jupyter notebook documents each change, what happened at each layer, the input and output, and the exact function called. It presents the results in numbers: one request traced through each layer with its timings; behavior at 10, 50 and 100 users; time to first token, prefill latency, and P95/P99 latency; where the system broke, what was changed and how much it improved; and the SLOs the system can meet (for example, "95% of requests get their first token within 500 ms").

The working method throughout: **profile → identify the bottleneck → change one thing → benchmark → optimize.**

See [roadmap.md](roadmap.md) for the detailed phase-by-phase plan, from one GPU to a multi-GPU cluster.

## Scope: inference only

No training or fine-tuning is involved. The project uses an existing open model (around 8B parameters, such as Llama or Qwen) and focuses entirely on serving it. This area is often called LLMOps or inference infrastructure.
