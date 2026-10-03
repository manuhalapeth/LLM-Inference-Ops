# Roadmap: LLM Inference Ops

## Guiding principles

1. **The notebook is the proof.** It runs alongside the project from day one, and every phase ends with a notebook chapter.
2. **Start small, then scale aggressively.** 1 GPU → working system → KV cache → harnesses → break it → fix it → many GPUs → many vLLMs → shared GPUs in a cluster.
3. **One change at a time.** Every experiment follows the same loop: **profile → identify bottleneck → change one thing → benchmark → compare**.
4. **Everything is reproducible.** Configs live in the repo, results are saved as CSV/JSON, and the notebook reads from those files instead of from memory.
5. **Use the full toolset**: CrewAI, API gateway + harnesses, NGINX, vLLM, Grafana, traces, Locust, a traffic-generating agent, Mooncake, HAMi, and Jupyter.

---

## The notebook structure

One notebook per phase along with a final summary notebook:

```
notebooks/
  00_setup.ipynb
  01_single_request_walkthrough.ipynb
  02_kv_cache_mooncake.ipynb
  03_harnesses_and_traces.ipynb
  04_breaking_one_gpu.ipynb
  05_profiling_and_tuning.ipynb
  06_scaling_out.ipynb
  07_gpu_slicing_hami.ipynb
  99_final_report.ipynb          ← summary of all results
```

Every experiment in every notebook uses the same template:

| Section | Contents |
|---|---|
| **Hypothesis** | "Raising `max-num-seqs` will increase throughput but hurt P99 TTFT." |
| **Setup** | Hardware, model, exact config/engine args (loaded from the repo) |
| **Load** | Locust profile: users, spawn rate, prompt set, duration |
| **Results** | TTFT, prefill latency, P50/P95/P99 end-to-end latency, tokens/sec, req/sec, GPU memory, KV cache usage, errors |
| **What happened** | Grafana screenshots + trace of one representative request |
| **Conclusion** | Was the hypothesis right? What's the next bottleneck? |

**Metrics to collect in every phase** (so results can be compared across phases):
- Latency: TTFT, prefill latency, inter-token latency, end-to-end, at P50/P95/P99
- Throughput: requests/sec, output tokens/sec
- Resources: GPU utilization, GPU memory, KV cache usage %, requests running vs. waiting, preemptions
- Reliability: error rate, rejected requests (by harness), timeouts
- Cost: $ per 1M output tokens, based on the GPU's hourly price

---

## Phase 0: Foundations (1 GPU)

**Hardware:** 1× 24–48 GB GPU (e.g. A10, L4, A6000, L40S)

**Goal:** Set up the repo, the tooling and the measurement conventions.

- Repo layout: `agent/`, `gateway/`, `nginx/`, `vllm/`, `observability/`, `loadtest/`, `notebooks/`, `results/`
- `docker-compose.yml` that brings up the whole stack
- Gateway: **a custom FastAPI gateway**, because harnesses, logging and tracing are fully visible and easy to explain in the notebook. (Alternatives such as LiteLLM proxy or Envoy AI Gateway can be compared later.)
- Model: an ~8B instruct model (Llama 3.1 8B or Qwen 8B)
- Write 20+ synthetic prompts of varied lengths (short Q&A, long summarization, multi-turn) and save them in `loadtest/prompts.json`
- Write a results logger that saves every benchmark run to `results/<phase>/<run_id>.json`
- Write a setup script that builds a rented GPU box from scratch in one command

**Notebook (`00`):** Architecture diagram, tool choices and why, and the metric definitions (what TTFT, P95, etc. mean and how each one is measured).

**Exit criteria:** `docker compose up` brings up the full stack on the GPU box.

---

## Phase 1: End-to-end system + Grafana (1 GPU)

**Goal:** Get one request all the way through every layer, and make it visible.

```
CrewAI agent → API gateway → NGINX → vLLM (1 instance) → response
                                        ↓
                          Prometheus → Grafana
```

- CrewAI agent (`app.py`) with a real task, e.g. "Summarize this document", that makes several LLM calls
- Gateway: forwards requests, assigns a request ID, logs timing
- NGINX: one upstream (vLLM), with the config already shaped for several upstreams
- vLLM: serves the 8B model through its OpenAI-compatible API, sent requests directly (no cluster yet)
- Prometheus scrapes vLLM's `/metrics`, the gateway and NGINX; GPU metrics come from DCGM exporter or nvidia-smi exporter
- Grafana dashboard: latency, throughput, GPU memory, KV cache usage, running/waiting requests, errors

**Notebook (`01`):** Follow one request through the system: the input, each layer's output, the exact function called at each step, and how long each layer took. Show a breakdown of where the time went (agent overhead vs. gateway vs. NGINX vs. prefill vs. decode).

**Exit criteria:** One request is fully explained with per-layer timings, and the dashboard is live.

---

## Phase 2: KV cache management with Mooncake (1 GPU)

**Goal:** Hook up KV cache management early, so the system is shaped like a production inference workload before it is load tested.

- Connect Mooncake to vLLM through vLLM's KV connector integration (directly or through LMCache; check the current vLLM docs for the supported path)
- On one GPU: use Mooncake to store and reuse KV cache beyond GPU memory (CPU/DRAM pool)
- Add KV cache metrics to Grafana: cache usage, hit rate, offload/reload activity
- Workloads that show it off: long shared prefixes, multi-turn conversations, the agent resending context
- Mooncake pays off most with multiple GPUs; that part comes back in Phase 6

**Notebook (`02`):** How the KV cache works and why it is the resource that runs out first. Compare with and without Mooncake at moderate load: KV cache hit rate, TTFT for repeated prefixes, GPU memory headroom.

**Exit criteria:** Mooncake is connected and visible on the dashboard, with a first with/without comparison.

---

## Phase 3: Harnesses, traces and evaluation (1 GPU)

**Goal:** Stop bad requests before they use GPU time, make every failure traceable, and use that to change the system.

- **Harnesses in the gateway** (before vLLM, because once a request reaches vLLM it is generated as long as compute is available):
  - Prompt-length / token-budget check
  - Per-user rate limit and quota
  - Malicious-prompt / prompt-injection check
  - Request timeout and maximum output tokens
- **Tracing:** OpenTelemetry across agent → gateway → NGINX → vLLM, viewed in Jaeger or Grafana Tempo. One trace ID per user request with child spans for each LLM call the agent makes
- **Evaluation:** a small eval set (correct output, refusal when expected, harness triggered when expected); run it after every change
- **Failure catalog:** deliberately trigger failures (oversized prompt, rate limit, vLLM down, timeout) and confirm that each one is caught, traced and visible on the dashboard

**Notebook (`03`):** For each harness, show requests that were blocked and how much GPU time that saved. Show a trace of a failure, how the trace found it, and what was changed. Include the eval results before and after.

**Exit criteria:** Every failure type has a trace, a dashboard signal and a harness or a fix.

---

## Phase 4: Break one GPU (1 GPU)

**Goal:** Find exactly where a single GPU runs out of memory, and why.

- **Locust:** the synthetic prompt set, ramped step by step (e.g. "20 new requests per minute" increasing over time, or 1 → 5 → 10 → 20 → 50 → 100 users)
- **Traffic-generating agent:** a second agent that sends concurrent, realistic multi-step workloads to the CrewAI app, as opposed to Locust's raw requests. Compare the two kinds of load
- Watch for the breaking point: KV cache usage hits 100%, requests queue up (waiting > running), preemptions start, TTFT and P99 explode, out-of-memory errors or timeouts appear
- Record the exact concurrency level and the metrics at the moment of failure

**Notebook (`04`):** Charts of latency vs. concurrency and throughput vs. concurrency, with the breaking point marked. Show what was happening at each point ("at 20 users KV cache was at 60%; at 40 users it hit 100% and preemptions began"). This becomes the **baseline** that every later phase is compared against.

**Exit criteria:** A documented breaking point with a root cause, supported by metrics.

---

## Phase 5: Profile, tune, set SLOs (1 GPU)

**Goal:** Fix the single-GPU bottlenecks one variable at a time, and define what the system can promise.

Engine arguments to test, one at a time, re-running the Phase 4 load each time:
- `--max-num-seqs` (concurrent sequences)
- `--gpu-memory-utilization` (how much memory goes to KV cache)
- `--max-model-len` (shorter context means more room for concurrent requests)
- `--max-num-batched-tokens` and chunked prefill (trading TTFT against throughput)
- Prefix caching (big gains for agents that resend the same system prompt)
- Quantization: FP8 / AWQ weights, FP8 KV cache (more room for KV cache)
- Speculative decoding (lower decode latency)

Profiling: vLLM's built-in profiler / PyTorch profiler, plus Nsight Systems for one representative run, to see where GPU time actually goes.

**Define SLOs**, for example:
- 95% of requests get their first token within 500 ms
- P99 end-to-end latency under X seconds at Y concurrent users
- Error rate under 0.1%
- Then: **maximum users per GPU while meeting the SLOs**, and the cost per 1M tokens at that load

**Notebook (`05`):** A results table with one row per change: TTFT, P95, P99, throughput and the new breaking point vs. the baseline. Show the best config and why it wins, the SLOs, and the capacity of one GPU.

**Exit criteria:** A tuned single-GPU config with measured improvement, plus SLOs and a capacity number.

---

## Phase 6: Scale out: multiple GPUs, multiple vLLMs, Mooncake at scale

**Hardware:** One machine with 2–4 GPUs, then 2 nodes

**Goal:** Prove that the system scales, measure how well, and use Mooncake where it matters most.

- Run N vLLM replicas (one per GPU) behind NGINX
- Compare NGINX load-balancing strategies: round robin vs. `least_conn` vs. hash-based (sticky per user/session, for prefix-cache hits)
- Compare **replicas vs. tensor parallelism**: 2 replicas on 1 GPU each vs. 1 instance using `--tensor-parallel-size 2`. Which wins on throughput, and which wins on latency?
- Larger model comparison: e.g. a 70B model (quantized) across all GPUs using TP
- **Mooncake across GPUs:** shared KV cache between replicas, then prefill/decode disaggregation (separate prefill and decode instances, with Mooncake transferring KV cache between them)
- **Multiple nodes:** two machines connected with fast networking (RDMA if the provider offers it). Measure KV transfer overhead
- Failure testing: kill one replica under load. Does NGINX route around it? What happens to P99?
- Re-run the Phase 4/5 load at higher levels to find the new breaking point

**Notebook (`06`):** A scaling curve (1 → 2 → 4 GPUs → 2 nodes) with throughput and maximum users while meeting the SLOs. Is scaling linear? If not, what's the overhead? Include the load-balancing comparison, the failover result, and disaggregated vs. combined serving on TTFT and inter-token latency.

**Exit criteria:** A measured scaling factor, the best routing strategy, proven failover and measured Mooncake gains at scale.

---

## Phase 7: Cluster and GPU slicing with HAMi (Kubernetes)

**Goal:** Move from "my machine" to "a cluster," and share GPUs efficiently. (The cluster step is deliberately deferred until the single-machine system is understood.)

- Move the stack to Kubernetes (k3s on rented nodes is enough): Deployments for gateway, vLLM and NGINX (or a K8s ingress), with Prometheus and Grafana through Helm
- Install **HAMi** to slice GPUs: run several smaller vLLM instances (or different models, e.g. a small model for classification/routing and the 8B model for generation) on a shared GPU, each with a memory limit
- Test isolation: does one noisy tenant hurt another's P99?
- Autoscaling: scale vLLM replicas based on queue depth or KV cache usage

**Notebook (`07`):** Compare a whole GPU per model vs. sliced GPUs on utilization, cost per 1M tokens, and isolation (P99 under a noisy neighbor). Include the cluster architecture diagram.

**Exit criteria:** The stack runs on K8s, with measured results for GPU sharing.

---

## Phase 8: Final report


`99_final_report.ipynb`:
1. Architecture: final system diagram and what each layer does
2. One request, explained: per-layer timings (from Phase 1)
3. KV cache: what Mooncake changed (Phases 2 and 6)
4. Reliability: harnesses, traces, the failures that were caught (Phase 3)
5. The journey in one chart: max users within the SLOs and cost per 1M tokens at each stage:
   baseline 1 GPU → tuned 1 GPU → 4 replicas → + Mooncake at scale → + HAMi
6. Key lessons: the 3–5 biggest findings ("prefix caching cut TTFT by X%", "the bottleneck moved from KV memory to decode compute at N users")
7. SLOs and capacity plan: "to serve N users at these SLOs you need X GPUs, costing $Y/month"
8. What I'd do next

Also: a README with a 2-minute summary, the headline numbers and a link to the notebook.

---

## Hardware plan

| Phase | Hardware |
|---|---|
| 0–5 | 1× 24–48 GB GPU |
| 6 | 1 machine with 2–4 GPUs, then 2 nodes (check RDMA support before going multi-node) |
| 7 | 1–2 GPU nodes on k3s |

Rule: **script everything.** Setup, load tests and results collection are scripted, so every run is repeatable.

---

## Progression at a glance

```
Phase 0  1 GPU          → foundations, stack comes up
Phase 1  1 GPU          → it works, one request explained, Grafana live
Phase 2  1 GPU          → KV cache management (Mooncake hooked up)
Phase 3  1 GPU          → harnesses + traces + evals, failures are visible
Phase 4  1 GPU          → BREAK IT (baseline)
Phase 5  1 GPU          → FIX IT (tuning, profiling, SLOs)
Phase 6  multi-GPU/node → SCALE IT (replicas, LB, TP, Mooncake P/D)
Phase 7  K8s cluster    → GPU slicing (HAMi), cluster ops
Phase 8  —              → final report notebook = the proof
```
