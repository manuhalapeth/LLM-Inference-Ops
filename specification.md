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
