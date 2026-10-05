# Specification

The hardware, software, time and money behind each phase, with the key numbers it produced. Measured numbers come from the files in `results/`; hardware and billing details come from the Vast.ai console.

---

## Phase 0: Foundations

### Hardware

| | |
|---|---|
| Provider | Vast.ai, Secure Cloud (verified data center), on demand |
| Location | Hungary (data center 81036, machine 30024) |
| Host reliability | 99.60% |
| GPU | 1× NVIDIA GeForce RTX 5090 |
| GPU memory | 32 GB GDDR7 (31.8 GB usable) |
| GPU memory bandwidth | 1,455 GB/s |
| GPU compute | 108.1 TFLOPS (as reported by Vast.ai) |
| GPU driver / max CUDA | 580.95.05 / 13.3 |
| GPU link | PCIe 5.0 ×16 (54.2 GB/s) |
| CPU | AMD EPYC 9654 (96 cores), 48 of 384 vCPUs allocated |
| System RAM | 96.7 GB |
| Disk | 130 GB NVMe (KIOXIA), ~19 GB/s |
| Network | ~6.4 Gbps down / ~5.3 Gbps up |
| Machine image | Vast.ai Ubuntu 22.04 VM template (`vastai/kvm:cuda-12.9.1-auto`) |

### Software

| | |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct (BF16) |
| Inference engine | vLLM 0.30.0 (`vllm/vllm-openai`, digest `sha256:8a69ffad015f…`) |
| Engine config | `vllm/configs/baseline.yaml`: GPU memory utilization 0.90, max context 8,192 tokens |
| Stack | FastAPI gateway → NGINX 1.27 → vLLM, Prometheus 2.55.1, Grafana 11.3.0 |
| Code version | git `acce6ba` |

### Time and money

| | |
|---|---|
| GPU price | $1.327/hr (storage only while booting: $0.061/hr) |
| VM boot time | ~8 min (rent → running) |
| Setup script to "Ready" | ~5 min (includes the model download and vLLM warm up) |
| Total instance lifetime | ~24 min (rent → destroy) |
| Credit used | ~$0.40 ($25.00 → $24.61 at 23 min) |

### vLLM startup

From `results/00_setup/vllm_startup.log`.

| | |
|---|---|
| Model weights on disk | 14.19 GiB |
| Weights download time | 35 s (~410 MB/s) |
| Model load time | 43 s |
| Weights in GPU memory | 14.29 GiB |
| CUDA graph capture | 63 s + 59 s |
| Memory left for KV cache | 11.19 GiB |
| KV cache capacity | 209,472 tokens |
| Max concurrent requests at 8,192 tokens each | 25.6 |

### Results: one streaming request (smoke test)

From `results/00_setup/20261003T034035Z_smoke_test.json`. One request, no other load: gateway → NGINX → vLLM.

| Metric | Value |
|---|---|
| Time to first token (TTFT) | 281 ms |
| End to end latency | 878 ms |
| Inter token latency | 9.6 ms (~104 tokens/s while decoding) |
| Tokens | 44 in / 63 out |
| Overall throughput | 71.7 output tokens/s |

### Phase 0 takeaway

On an RTX 5090, the 7B model takes 14.3 GiB, leaving 11.2 GiB (~209k tokens) of KV cache. A single request decodes at ~104 tokens/s. KV cache capacity is the first limit later phases will hit.

---

## Phase 1: End to end system and Grafana

### Hardware

Same GPU model as Phase 0 (the Phase 0 machine was no longer available on Secure Cloud).

| | |
|---|---|
| Provider | Vast.ai, verified host (not Secure Cloud), on demand |
| Location | Denmark (host 542029, machine 150656) |
| Host reliability | 99.12% |
| GPU | 1× NVIDIA GeForce RTX 5090 |
| GPU memory | 32 GB GDDR7 (31.8 GB usable) |
| GPU memory bandwidth | 1,458 GB/s |
| GPU compute | 111.8 TFLOPS (as reported by Vast.ai) |
| GPU driver / max CUDA | 580.95.05 / 13.4 |
| GPU power limit | 600 W |
| GPU link | PCIe 5.0 ×8 (27.5 GB/s), half the Phase 0 host's ×16 |
| CPU | AMD Ryzen 9 9950X (16 cores), 32 vCPUs |
| System RAM | 126 GB |
| Disk | 130 GB NVMe (Samsung 9100 PRO), ~7 GB/s |
| Network | ~7.6 Gbps down / ~6.2 Gbps up |
| Machine image | Vast.ai Ubuntu 22.04 VM template (`vastai/kvm:cuda-12.9.1-auto`) |

### Software

| | |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct (BF16) |
| Inference engine | vLLM 0.30.0 (pinned) |
| Engine config | `vllm/configs/baseline.yaml` (unchanged from Phase 0) |
| Agent | CrewAI 1.15.23, 3 agents in sequence (analyst → writer → reviewer) |
| GPU metrics | nvidia_gpu_exporter 1.15.1 |
| Stack | FastAPI gateway → NGINX 1.27 → vLLM, Prometheus 2.55.1, Grafana 11.3.0 |
| Code version | git `72bb78e` |

### Time and money

| | |
|---|---|
| GPU price | $0.985/hr |
| Setup script to "Ready" | ~6 min |
| Phase 1 measurements | ~1 min (15 traced requests + 3 agent runs) |
| Total instance lifetime | ~20 min (rent → destroy) |
| Credit used | ~$0.26 ($24.61 → $24.35 at 18 min) |

### vLLM startup

From `results/01_end_to_end/vllm_startup.log`. Identical memory layout to Phase 0.

| | |
|---|---|
| Weights download time | 22 s (Phase 0: 35 s) |
| Model load time | 27 s (Phase 0: 43 s) |
| Weights in GPU memory | 14.29 GiB |
| Memory left for KV cache | 11.19 GiB |
| KV cache capacity | 209,472 tokens |

### Results: single requests, one at a time

From `results/01_end_to_end/20261004T015742Z_trace_single_requests.json`. Medians of 5 requests each, after 1 warmup request.

| | short_qa_05 | summarize_01 | long_gen_02 |
|---|---|---|---|
| Prompt / output tokens | 32 / 64 | 477 / 172 | 45 / 768 |
| TTFT (cold → cached) | 14.1 → 16.3 ms | 39.0 → 17.1 ms | 16.1 → 16.4 ms |
| Prefill (cold → cached) | 11.3 → 12.3 ms | 32.9 → 12.5 ms | 12.0 → 12.2 ms |
| Decode | 602 ms | 1,634 ms | 7,331 ms |
| End to end | 618 ms | 1,651 ms | 7,347 ms |
| Time per output token | 9.55 ms | 9.55 ms | 9.56 ms |
| Decode share of end to end | 97.3% | 99.0% | 99.8% |
| Time outside the engine | 2.7 ms | 2.7 ms | 3.0 ms |

Time outside the engine = client ↔ gateway (~1.0 ms) + gateway (~0.6 to 1.1 ms) + NGINX (<1 ms, below its log resolution) + vLLM API server (~0.7 to 1.2 ms).

### Results: CrewAI agent

From the 3 `results/01_end_to_end/*_agent_run.json` files.

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Total time | 8.16 s | 8.07 s | 8.05 s |
| LLM calls | 3 | 3 | 3 |
| Prompt / generated tokens | 1,852 / 805 | 1,852 / 805 | 1,852 / 805 |
| Prompt tokens from prefix cache | 0% | 98.5% | 98.5% |
| Total prefill | 125 ms | 38 ms | 37 ms |
| Total decode | 7.67 s | 7.67 s | 7.67 s |
| Waiting on the LLM | 7.81 s (96%) | 7.72 s (96%) | 7.72 s (96%) |
| CrewAI overhead | 0.35 s | 0.35 s | 0.33 s |

Per task: analyst 4.3 s, writer 1.5 s, reviewer 2.2 s.

### GPU (from Grafana)

| | |
|---|---|
| GPU memory used with vLLM loaded | ~28 GiB of 32 GiB |
| GPU utilization while generating | 100% |
| Power while generating | ~465 to 470 W (limit 600 W) |
| Temperature | 24 to 40 °C |
| KV cache usage, one request at a time | <0.5% |

### Phase 1 takeaway

One request at a time, decode is everything: 9.55 ms per output token, 97 to 99.8% of latency. The gateway, NGINX and HTTP layers add under 3 ms in total. The agent spends 96% of its time waiting on token generation. The GPU is busy but nowhere near full: the KV cache is under 0.5% used, so the open question is how much more it can serve at once.

---

## Phase 2: KV cache management with Mooncake

### Hardware

The same machine as Phase 0 (Hungary, machine 30024), so Phase 2 compares directly with the Phase 0 baseline.

| | |
|---|---|
| Provider | Vast.ai, Secure Cloud (verified data center), on demand |
| Location | Hungary (data center 81036, machine 30024) |
| Host reliability | 99.60% |
| GPU | 1× NVIDIA GeForce RTX 5090, 32 GB GDDR7 (31.8 GB usable), 1,457 GB/s |
| GPU driver / max CUDA | 580.95.05 / 13.3 |
| GPU link | PCIe 5.0 ×16 (54.2 GB/s) |
| CPU | AMD EPYC 9654 (96 cores), 48 of 384 vCPUs allocated |
| System RAM | 96.7 GB (32 GB of it given to the Mooncake store) |
| Disk | 130 GB NVMe (KIOXIA) |
| Network | ~6.7 Gbps down / ~5.4 Gbps up |
| Machine image | Vast.ai Ubuntu 22.04 VM template (`vastai/kvm:cuda-12.9.1-auto`) |

### Software

| | |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct (BF16) |
| Inference engine | vLLM 0.30.0 (pinned); `llm-inference-ops/vllm-mooncake:v0.30.0` = the same image plus Mooncake |
| Mooncake | mooncake-transfer-engine-cuda13 0.3.13.post1, `MooncakeStoreConnector`, embedded mode, TCP |
| Mooncake store | 32 GB CPU memory pool + 2 GB local buffer (`vllm/mooncake/mooncake_config.json`) |
| Engine configs | `baseline.yaml` vs `mooncake_store.yaml` (baseline + the Mooncake connector) |
| Code version | git `a568902` |

### Time and money

| | |
|---|---|
| GPU price | $1.327/hr |
| Setup script to "Ready" | ~8 min |
| Baseline measurements | ~3 min |
| Mooncake image build + vLLM restart | ~4 min (vLLM reload from cached weights: 7 s) |
| Mooncake measurements | ~3 min |
| Total instance lifetime | ~26 min |
| Credit used | ~$0.60 |

### KV cache tiers

| Tier | Size | Tokens of KV (57,344 bytes per token) |
|---|---|---|
| GPU (vLLM KV cache) | 11.19 GiB | 209,472 |
| Mooncake store (CPU memory) | 32 GB | ~558,000 |

### Results: working set bigger than the GPU cache

From `results/02_kv_cache_mooncake/*_kv_experiment_{baseline,mooncake}.json`. 48 documents × 5,949 tokens (286k tokens) sent one at a time, 8 output tokens each. Medians.

| Round | | TTFT | Queue | Prefill | From GPU | From Mooncake | Computed |
|---|---|---|---|---|---|---|---|
| fill | baseline | 438 ms | 0 ms | 410 ms | 0% | 0% | 100% |
| fill | Mooncake | 670 ms | 0 ms | 629 ms | 0% | 0% | 100% |
| revisit | baseline | 438 ms | 0 ms | 411 ms | 0% | 0% | 100% |
| revisit | Mooncake | **172 ms** | 109 ms (KV load) | 22 ms | 0% | **99.5%** | 0.5% |
| hot | baseline | 61 ms | 0 ms | 23 ms | 100% | 0% | 0% |
| hot | Mooncake | 55 ms | 0 ms | 28 ms | 100% | 0% | 0% |

### Results: Mooncake transfers

| Operation | Calls | Data | Time | Speed |
|---|---|---|---|---|
| save_put (fill round) | 144 | 15.18 GiB (339 MB per document) | 25.65 s (534 ms per document) | 0.59 GiB/s |
| load_get (revisit round) | 48 | 15.18 GiB (339 MB per document) | 4.65 s (97 ms per document) | 3.26 GiB/s |
| Failed keys | | 0 | | |

### Results: break-even

| | |
|---|---|
| Extra TTFT on a first visit | +231 ms |
| Saved TTFT per evicted revisit | −266 ms |
| Break-even | 0.87 revisits per document |

### Results: overhead when prompts fit on the GPU

From `results/02_kv_cache_mooncake/*_trace_single_requests_{baseline,mooncake}.json`. Medians of 5.

| | short_qa_05 | summarize_01 | long_gen_02 |
|---|---|---|---|
| TTFT baseline → Mooncake | 29.3 → 31.8 ms | 32.3 → 37.6 ms | 28.9 → 32.4 ms |
| End to end baseline → Mooncake | 637 → 638 ms | 1,684 → 1,687 ms | 7,438 → 7,434 ms |

### Same GPU, different host CPU (Phase 1 vs Phase 2 baseline)

| | Phase 1 (Ryzen 9 9950X) | Phase 2 (EPYC 9654) |
|---|---|---|
| TTFT, short prompt | 16 ms | 29 ms |
| Time outside the engine | ~2.7 ms | ~7.5 ms |
| Prefill, cached prompt | ~12 ms | ~18 ms |
| Decode per token | 9.55 ms | ~9.6 ms |

### Phase 2 takeaway

When the working set is bigger than the GPU's KV cache, prompts that come back get no reuse at all. Mooncake fixes that: evicted prompts reload from CPU memory 2.6× faster than recomputing (172 vs 438 ms TTFT). The cost is the write path, which adds 53% to a prompt's first visit over TCP, so Mooncake pays off once a long prompt is reused about once. When everything fits on the GPU, it costs 2 to 5 ms of TTFT.

---

## Phase 3: Harnesses, traces and evals

### Hardware

The same machine as Phases 0 and 2 (Hungary, machine 30024): RTX 5090 32 GB, AMD EPYC 9654 (48 vCPUs), 96.7 GB RAM, PCIe 5.0 ×16, Secure Cloud, 99.60% reliability.

| | |
|---|---|
| GPU driver | 580.95.05 at boot; **580.178.04** after Ubuntu's automatic updates upgraded it mid session (see below) |

### Software

| | |
|---|---|
| Model / engine | Qwen/Qwen2.5-7B-Instruct, vLLM 0.30.0 |
| Engine config | `vllm/configs/baseline_tracing.yaml` (baseline + OTLP traces to Jaeger) |
| Gateway | FastAPI + harnesses, tokenizers 0.23.2 (Qwen tokenizer baked into the image), OpenTelemetry SDK 1.45.0 |
| NGINX | 1.27.5 with the OpenTelemetry module (`nginx:1.27-alpine-otel`), re-resolves vLLM's address every 5 s |
| Tracing | Jaeger 2.21.0, OTLP over gRPC |
| Agent | CrewAI 1.15.23 + OpenTelemetry httpx instrumentation |
| Code version | git `060eb04` |

### Harness limits

| | |
|---|---|
| Max prompt tokens | 6,000 (and prompt + output ≤ 8,192) |
| Max output tokens | 1,024 (also the default when a request sets none) |
| Rate limit per user | 60 requests and 200,000 tokens per minute |
| Max requests in flight | 256 |
| Default / max time limit | 120 s / 600 s |

### Time and money

| | |
|---|---|
| GPU price | $1.327/hr |
| Session | ~40 min, including diagnosing and fixing the driver upgrade |
| Credit used | ~$0.85 |
| Credit used, Phases 0 to 3 | $2.09 ($25.00 → $22.91) |

### Results: evals

From `results/03_harnesses_and_traces/*_evals_harnesses_{on,off}.json`. 22 cases, temperature 0, scored automatically.

| | Harnesses on | Harnesses off |
|---|---|---|
| Passed | 20 / 22 | 16 / 22 |
| Correctness | 7 / 7 | 7 / 7 |
| Instructions | 4 / 5 (scoring bug, fixed) | 4 / 5 |
| Summarization | 0 / 1 (dropped impact numbers, claimed "minimal impact") | 0 / 1 |
| Safety (refusals) | 3 / 3 | 3 / 3 |
| Harness cases | 6 / 6 | 2 / 6 |
| Secrets leaked | 0 | 2 (leaked inside a refusal) |
| GPU time, all 22 cases | 17.3 s | 24.1 s |

The harnesses-on run was done twice, before and after the driver upgrade, with identical results case for case.

### Results: what the harnesses saved

| Blocked request | Rejected in | GPU time without the harness |
|---|---|---|
| Prompt injection (secret) | ~2 ms | 443 ms, and leaked the secret |
| Prompt injection (variant) | ~2 ms | 548 ms, and leaked the secret |
| 7,000 word prompt | ~18 ms | 1,760 ms |
| max_tokens 4,000 | ~2 ms | 1,184 ms (the poem ended at 122 tokens; the cap allowed up to 4,000) |
| **Total** | | **3.93 s** |
| No max_tokens (allowed, capped) | 1,024 tokens, 9.8 s | 1,309 tokens, 12.7 s |

Harness check time over 31 traced requests: **p50 0.27 ms, p95 16.4 ms, max 19.4 ms** (the slow end is counting the 7,000 word prompt exactly).

### Results: failure drills

From `results/03_harnesses_and_traces/*_failure_drills.json`. 6 / 6 passed.

| Drill | Result |
|---|---|
| Prompt injection | 400 in 1.8 ms, no GPU work |
| Oversized prompt | 413 in 17.9 ms, no GPU work |
| Rate limit burst (70 requests) | 61 allowed, 9 rejected with 429 and Retry-After: 1 s |
| 2 s time limit on a 1,000 token answer | stream ended at 2.00 s with a timeout event; vLLM stopped at 206 tokens |
| Client hangs up after 1 s | vLLM stopped at 103 tokens, nothing left running |
| vLLM stopped | client got 504 after 5.0 s (NGINX connect timeout); traffic back 69 s after vLLM restarted, no NGINX restart |

### Results: tracing

| | |
|---|---|
| Traces saved | 51 (`results/03_harnesses_and_traces/traces/`) |
| Traced agent run | 8.2 s, 3 LLM calls, 19 spans across agent, gateway, NGINX, vLLM |
| vLLM span attributes | queue, prefill, decode, inference, TTFT, end to end, prompt and completion tokens |

### Incident: driver upgraded during the session

Ubuntu's unattended upgrades started ~6 minutes after boot and upgraded ~200 packages, including the NVIDIA driver (580.95.05 → 580.178.04), while the old kernel module stayed loaded. Running GPU processes were unaffected; any new one failed with `driver/library version mismatch`, which surfaced when the vLLM-down drill restarted vLLM. Fixed on the spot by unloading and reloading the NVIDIA kernel modules (no reboot). `setup_gpu_box.sh` now disables automatic updates before anything touches the GPU.

### Phase 3 takeaway

Checks in front of the GPU cost about a third of a millisecond and stopped every bad request in the eval set. Without them, the same requests used ~4 s of GPU time and the model leaked a secret while refusing to reveal it. Every failure type is caught quickly and shows up in metrics, and one trace follows an agent task from the agent into vLLM's scheduler.

---

## Phase 4: Breaking one GPU

### Hardware

The same machine as Phases 0, 2 and 3 (Hungary, machine 30024): RTX 5090 32 GB, AMD EPYC 9654 (48 vCPUs), 96.7 GB RAM, PCIe 5.0 ×16, Secure Cloud, 99.60% reliability. A fresh VM, driver 580.95.05 (automatic updates switched off at boot).

### Software

| | |
|---|---|
| Model / engine | Qwen/Qwen2.5-7B-Instruct, vLLM 0.30.0, `vllm/configs/baseline.yaml` (defaults, incl. max 256 sequences at once) |
| Load generator | Locust 2.46.6 (`locustio/locust` image, 8 processes), closed loop |
| Agent load | CrewAI 1.15.23, 1 to 32 crews at once |
| Gateway | All harnesses on except rate limits (raised to 1M requests/min) and load shedding (raised to 4,096 in flight) |
| Code version | git `4d530d1` |

### Time and money

| | |
|---|---|
| GPU price | $1.327/hr |
| Session | ~62 min, of which ~25 min lost to stalled Docker Hub downloads (see below) |
| Credit used | ~$1.36 |
| Credit used, Phases 0 to 4 | ~$3.45 ($25.00 → ~$21.55) |

### Load

| | |
|---|---|
| Steps | 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512 users, 60 s each (first 15 s of each step skipped) |
| Mix | 35% short questions, 25% multi-turn, 25% summaries (~500 to 1,100 prompt tokens), 15% long answers (up to 768 tokens) |
| Prefix cache | each request tagged uniquely; hit rate ~8% (shared system prompts only) |
| Requests | 11,175, **0 errors** |

### Results: the sweep

From `results/04_breaking_one_gpu/*_load_sweep_baseline.json`.

| Users | Output tokens/s | TTFT p50 | TTFT p95 | ITL p50 | E2E p95 | Running | Waiting (max) | KV cache (max) | Power |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 102 | 24 ms | 62 ms | 9.7 ms | 3.8 s | 1 | 0 | 0% | 475 W |
| 8 | 767 | 41 ms | 88 ms | 10.2 ms | 7.9 s | 8 | 0 | 2% | 473 W |
| 32 | 2,551 | 54 ms | 113 ms | 12.2 ms | 9.4 s | 32 | 0 | 8% | 523 W |
| 64 | 4,054 | 73 ms | 135 ms | 15.5 ms | 11.7 s | 63 | 0 | 14% | 567 W |
| 96 | 4,746 | 104 ms | 182 ms | 20.0 ms | 15.3 s | 94 | 0 | 20% | 575 W |
| 128 | 5,461 | 169 ms | 328 ms | 23.3 ms | 18.3 s | 125 | 0 | 28% | 575 W |
| **192** | **5,951** | 339 ms | 499 ms | 31.3 ms | 25.6 s | 184 | 0 | 40% | 575 W |
| 256 | 5,918 | 467 ms | 666 ms | 41.4 ms | 32.3 s | 244 | 0 | 52% | 575 W |
| 384 | 5,137 | **5.75 s** | 6.22 s | 48.9 ms | 42.3 s | 255 | 127 | 53% | 575 W |
| 512 | 5,341 | **11.2 s** | 11.7 s | 47.8 ms | 28.8 s* | 255 | 256 | 52% | 575 W |

\* Requests still running when Locust stopped were not logged, which flatters the last step's client side numbers.

### Results: breaking points

| | |
|---|---|
| Peak throughput | **5,951 output tokens/s at 192 users** |
| Requests start queueing | 384 users (vLLM capped at 256 running) |
| TTFT p95 over 1 s | 384 users |
| KV cache ≥ 95% | never (max 53%) |
| Preemptions | none |
| Errors | none |

### Results: capacity per target

| Target | Users | Output tokens/s |
|---|---|---|
| TTFT p95 < 200 ms, ITL p50 < 20 ms | 96 | 4,746 |
| TTFT p95 < 500 ms, ITL p50 < 40 ms | 192 | 5,951 |
| TTFT p95 < 1 s | 256 | 5,918 |

### Results: CPU of each container (max % of one core)

| Users | Gateway | NGINX | vLLM | Locust |
|---|---|---|---|---|
| 64 | 56% | 24% | 116% | 99% |
| 256 | 84% | 47% | 100% | 65% |
| 512 | 81% | 46% | 98% | 72% |

Locust's own worker view at 512 users: 8 workers × 64 users, 5 to 7% CPU each, ~45 MB memory each. The load generator was never the limit.

### Results: concurrent CrewAI crews

From `results/04_breaking_one_gpu/*_agent_load.json`.

| Crews at once | Task p50 | Task p95 | Tasks per minute | Output tokens/s |
|---|---|---|---|---|
| 1 | 8.5 s | 8.5 s | 5.8 | 55 |
| 4 | 8.8 s | 8.9 s | 21.7 | 182 |
| 8 | 8.9 s | 9.0 s | 43.1 | 272 |
| 16 | 9.9 s | 10.3 s | 77.0 | 825 |
| 32 | 10.3 s | 11.0 s | 138.7 | 1,760 |

### Incident: Docker Hub downloads stalled

On this VM, every download served through Cloudflare (Docker Hub's image layers, Cloudflare's own speed test) stalled indefinitely, while other services (Hugging Face, Google Cloud Storage at ~20 MB/s) were fine. Not a rate limit (96 of 100 anonymous pulls left). The box also lacked the NVIDIA Container Toolkit, which the setup script installs, but it never got that far. Fixed by restarting Docker and pointing it at Google's Docker Hub mirror (`mirror.gcr.io`); the 21.6 GB vLLM image then downloaded normally. `setup_gpu_box.sh` now always uses the mirror and puts time limits on the download and GPU checks.

### Phase 4 takeaway

One RTX 5090 peaks at ~5,950 output tokens/s around 192 concurrent users, at its 575 W power limit. The break at 384 users is vLLM's default cap of 256 running requests, not memory: the KV cache never passed 53% and nothing was preempted. Time to first token goes from under 0.5 s to 5.8 s the moment the queue forms. Nothing errors; it just gets slow.

---

## Phase 5: Profiling and tuning one GPU

### Hardware

A different machine with the same hardware as Phases 0, 2, 3 and 4: **United Kingdom, host 166946, machine 33260**, RTX 5090 32 GB, **AMD EPYC 9654** (48 vCPUs), PCIe 5.0 ×16, verified host, 99.73% reliability, driver 580.95.05, $1.211/hr. Phase 5 reruns its own baseline on this machine, and it reproduced Phase 4 within 3% (peak 5,794 vs 5,951 tokens/s, same break at 384 users).

### Software

| | |
|---|---|
| Model / engine | Qwen/Qwen2.5-7B-Instruct, vLLM 0.30.0 |
| Configs | `vllm/configs/tune_*.yaml`, each the baseline plus one change |
| Load | Phase 4's Locust mix, unique tags, closed loop; 64, 128, 192, 256, 384, 512, 768 users, 45 s per step |
| Code version | git `6896b27` |

### Why the defaults were worth testing

vLLM 0.30.0 sets its batch limits by GPU memory. GPUs with 70 GB or more get up to 1,024 sequences and 8,192 batched tokens; anything smaller, like this 32 GB RTX 5090, falls through to an untuned fallback (marked TODO in vLLM's code) of 256 sequences and 2,048 tokens.

### Results: one change at a time

From `results/05_profiling_and_tuning/*_load_sweep_*.json` and `*_tuning_summary.json`.

| Config | Weights | KV cache capacity | Peak tokens/s | vs baseline | TTFT p95 @256 | ITL p50 @256 | KV max | Preemptions |
|---|---|---|---|---|---|---|---|---|
| baseline | 14.29 GiB | 209,472 | 5,794 | | 713 ms | 43.0 ms | 57% | 0 |
| max-num-seqs 512 | 14.29 GiB | 209,472 | 6,013 | +4% | 702 ms | 43.1 ms | 100% | 238 |
| max-num-seqs 1024 | 14.29 GiB | 209,472 | 5,835 | +1% | 657 ms | 43.9 ms | 100% | 348 |
| max-num-batched-tokens 8192 | 14.29 GiB | 209,472 | 5,949 | +3% | 743 ms | 41.4 ms | 55% | 0 |
| prefix caching off | 14.29 GiB | 209,472 | 5,942 | +3% | 709 ms | 42.8 ms | 56% | 0 |
| **FP8 KV cache** | 14.29 GiB | **404,480** | **7,064** | **+22%** | 748 ms | 34.9 ms | 30% | 0 |
| **FP8 weights** | **8.17 GiB** | **323,824** | **8,662** | **+50%** | 745 ms | 29.0 ms | 35% | 0 |

### Results: raising the sequence cap, step by step

| Users | Baseline TTFT p95 / ITL p50 | max-num-seqs 512 TTFT p95 / ITL p50 |
|---|---|---|
| 384 | 6.39 s / 51 ms (127 waiting) | **1.20 s** / 65 ms (23 waiting) |
| 512 | 12.99 s / 51 ms (256 waiting) | **1.48 s** / 90 ms, KV 99%, preemptions begin |
| 768 | 24.18 s / 51 ms | 13.80 s / 104 ms, KV 100% |

### Results: capacity and cost per target ($1.211/hr)

Interactive: TTFT p95 ≤ 0.5 s and ITL p50 ≤ 40 ms. Relaxed: TTFT p95 ≤ 1 s and ITL p50 ≤ 60 ms. Capacity is limited to the steps tested.

| Config | Interactive users | $/1M output tokens | Relaxed users | $/1M output tokens |
|---|---|---|---|---|
| baseline | 128 | $0.066 | 256 | $0.064 |
| FP8 KV cache | 128 | $0.061 | 256 | $0.053 |
| FP8 weights | 128 | $0.045 | 256 | $0.044 |

### Results: speculative decoding (n-gram) at low load

| Users | Baseline ITL p50 / tokens/s | N-gram speculative ITL p50 / tokens/s |
|---|---|---|
| 4 | 9.9 ms / 395 | 11.7 ms / 298 |
| 16 | 11.4 ms / 1,369 | 13.5 ms / 1,202 |
| 64 | 14.9 ms / 4,250 | 23.3 ms / 2,757 |

At 1 user, one multi-turn request took 58.9 s under speculation (under 1 s normally), so no request finished in that step's measured window. GPU utilization fell to 64 to 82%.

### Results: profile of the baseline at 128 users

60 engine steps, 1,310 ms, 24,306 GPU kernels. **GPU busy 98%** of the window.

| Work | Share of GPU kernel time |
|---|---|
| Matrix multiplies (weights) | 73.0% |
| Attention | 23.6% |
| Activation, normalization, copies, sampling, other | 3.4% |

The single largest kernel, a BF16 matrix multiply, was 63.6% of all GPU time.

### Results: the combined config

`vllm/configs/tuned.yaml`: FP8 weights + FP8 KV cache + max-num-seqs 512. KV cache capacity 625,184 tokens, weights 8.17 GiB.

| Users | Output tokens/s | TTFT p95 | ITL p50 | Running | KV max |
|---|---|---|---|---|---|
| 64 | 6,602 | 118 ms | 9.4 ms | 63 | 5% |
| 128 | 9,821 | 421 ms | 11.9 ms | 112 | 9% |
| 256 | 10,773 | 1,052 ms | 21.7 ms | 226 | 18% |
| 512 | 11,300 | 2,770 ms | 38.6 ms | 444 | 36% |
| 768 | **11,513** | 6,902 ms | 44.9 ms | 496 | 40% |

### Results: answer quality

The 22-case eval set (harnesses on), same machine, temperature 0. From `results/05_profiling_and_tuning/*_evals_quality_*.json`.

| Config | Evals passed | Same answer as BF16 | Peak tokens/s |
|---|---|---|---|
| baseline (BF16) | 21 / 22 | 22 / 22 | 5,794 |
| **FP8 weights** | **21 / 22** | 14 / 22 | **8,662** |
| FP8 KV cache | 17 / 22 | 7 / 22 | 7,064 |
| tuned (both + 512) | 15 / 22 | 6 / 22 | 11,513 |

FP8 weights passed exactly the cases the baseline passed. The FP8 KV cache answered 17 × 23 = "253", 100 °C = "239 °F", broke JSON output, and the combined config also failed to refuse a phishing request. The one case every config fails is the incident summary (Phase 3).

### Recommended config and SLOs

**FP8 weights** (`vllm/configs/tune_fp8_weights.yaml`): +50% throughput, same answer quality as BF16. The FP8 KV cache is not usable without calibrated scales.

| Target | Users | Output tokens/s | Cost per 1M output tokens ($1.211/hr) |
|---|---|---|---|
| Interactive: TTFT p95 ≤ 0.5 s, ITL p50 ≤ 40 ms | 128 | 8,264 | $0.045 (baseline $0.066) |
| Relaxed: TTFT p95 ≤ 1 s, ITL p50 ≤ 60 ms | 256 | 8,363 | $0.044 (baseline $0.064) |

### Time and money

| | |
|---|---|
| GPU price | $1.211/hr |
| Session | 2 h 53 min (single-change sweeps ~80 min, tuned run ~15 min, quality checks ~6 min, setup and a restart) |
| Credit used | $2.86 ($21.57 → $18.71) |
| Credit used, Phases 0 to 5 | $6.29 ($25.00 → $18.71) |

### Phase 5 takeaway

The GPU spends three quarters of its time multiplying by the model weights, so halving the weights' size with FP8 is the one change that matters: +50% throughput and 31% lower cost per token, with answers as good as BF16. The FP8 KV cache and the combined config looked even faster (+22%, +99%) but got arithmetic, unit conversions and JSON wrong; without the eval set, the broken config would have looked like the answer. Raising the sequence cap mostly trades queueing for slower streaming; batched tokens, prefix caching and n-gram speculation didn't help this traffic.
