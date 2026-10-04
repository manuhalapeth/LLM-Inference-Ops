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
