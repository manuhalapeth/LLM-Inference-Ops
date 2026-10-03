# Phase 0 design choices: foundations

Phase 0 is where most of the "boring" decisions got made. However, they're the ones that quietly shape everything after. I created this file to talk about what I picked, what else I looked at, and why.

## 1. Writing my own gateway instead of using an off the shelf one

The gateway is the front door. Every request passes through it before it touches the GPU. Later this is where the harnesses (rate limits, prompt checks, timeouts) and tracing will live.

I could have used LiteLLM's proxy or Envoy's AI gateway. Both are solid and come with a lot built in. But the whole point of this project is to show and explain what happens at each layer. I felt like a lot of that happens inside someone else's code when we're dealing with a ready made gateway. A small FastAPI app keeps every check, every timer and every log line in plain view, which makes it easy to walk through in the notebook.

The cost is that I have to build features those tools already have. That's fine for now. If the gateway ever becomes the bottleneck or the feature list gets long, comparing it against LiteLLM or Envoy would make a good experiment of its own.

## 2. Qwen2.5 7B Instruct as the model

I wanted something in the 8B class: big enough to behave like a real production model, small enough to fit on one GPU with plenty of memory left over for the KV cache.

Llama 3.1 8B was the obvious other choice. The problem is that it's gated: you need a Hugging Face account and have to accept Meta's license before you can download it. That's one more thing that can break on a fresh machine. Qwen2.5 7B Instruct is openly downloadable, well supported by vLLM, and about the same size.

The model is just one line in the environment file. Switching to Llama later is easy if I want to compare them.

## 3. Vast.ai virtual machines instead of RunPod

I wanted cheap GPUs, decent availability, and something that's easy to shut off so I don't get surprise bills.

RunPod was my first pick on price and ease of use. Then I found out RunPod pods are containers. You can't run Docker or Docker Compose inside them. My whole stack is a compose file. Phase 7 needs Kubernetes, so that ruled it out.

I found out that Vast.ai has full virtual machine instances where Docker and Kubernetes work normally. It bills by the second, runs on prepaid credit (when the balance hits zero, instances stop), and has a Secure Cloud option that only shows verified data centers.

The tradeoff is that fewer machines support VMs, and they take a few minutes longer to boot. One gotcha worth remembering: on Vast.ai, stopping an instance still charges for its disk. Only destroying it brings the cost to zero.

## 4. An RTX 5090, kept for all the single GPU phases

I went looking for a 48 GB card like the RTX A6000 or L40S, so there'd be plenty of room for the KV cache. With Secure Cloud and VM support both turned on, none were available. A Secure Cloud RTX 5090 with 99.6% reliability was.

The 5090 has 32 GB. After loading the model, that leaves about 11 GB for the KV cache, roughly 209,000 tokens. That's less room than a 48 GB card would give, but it's actually handy for this project: the GPU will run out of memory at a lower load, which makes the breaking point in Phase 4 easier to reach and study.

The more important decision is to stick with this exact GPU for Phases 0 through 5. Every later result gets compared against the earlier ones, and if the hardware changes halfway through, I can't tell whether an improvement came from my tuning or from a better card. When I scale out in Phase 6, it'll be more 5090s.

## 5. Nothing exposed to the internet

The gateway, Prometheus and Grafana only listen on the GPU machine itself. I reach them from my laptop through an SSH tunnel.

The easier option would be to open the ports publicly. But these are rented machines with public IP addresses, and an open LLM endpoint is something people actively scan for and abuse, both for free GPU time and to run up your bill. Grafana also starts with a default password.

The SSH tunnel costs one extra command at the start of a session, and in exchange nothing is reachable without my SSH key.

## 6. vLLM settings live in a config file, one change at a time

The engine settings (how much GPU memory vLLM can use, the maximum context length, and later batch sizes and caching options) live in `vllm/configs/baseline.yaml` (not on the command line).

To run an experiment, I copy the baseline file, change exactly one value, and point the stack at the new file. That gives me a record of exactly what each run used, and it enforces the rule the whole project runs on: change one thing, measure, compare. If I change two settings at once and things get faster. I can't say which one helped.

## 7. Pinning the exact vLLM version

The first run used the `latest` vLLM image, which turned out to be version 0.30.0. I checked that the image that ran has the same digest as the published 0.30.0 tag, then pinned that version.

vLLM moves fast, and new releases change performance, defaults and even metric names. If the version drifted between phases, a change in my numbers might come from vLLM and not from anything I did. Pinning keeps the baseline honest. When I do upgrade, I'll do it on purpose and rerun the baseline.

## 8. Results are saved to files and the notebook only reads files

Every benchmark run gets saved as a JSON file under `results/`, along with the config, the load, the GPU model and the git commit it ran on. The notebooks read those files. No number in a notebook is ever typed in by hand.

This is what makes the notebook believable as proof. Anyone can trace a number back to the run that produced it and see exactly what it ran on. It also means I can rerun the notebook on my laptop long after the GPU machine is gone.

The results logger and the smoke test only use Python's standard library, so they run on a fresh GPU machine without installing anything. Percentiles use the nearest rank method, which always reports a latency that actually happened rather than an interpolated one.

## 9. Streaming all the way through

LLM responses are streamed token by token, and time to first token is one of the most important numbers in this project. Any layer that buffers the response would hide it.

NGINX buffers responses by default, so I turned that off. The gateway also forwards each chunk as soon as it arrives, without waiting for the full response. The smoke test confirms it works: against a test server with a 200 ms delay, the first token came through the gateway in 209 ms.

## 10. Clients call the model by a fixed name

vLLM serves the model under the name "llm", regardless of which model is actually loaded. The agent, the load tests and the smoke test all ask for "llm".

That way, switching from Qwen to Llama (or anything else) is a change to the environment file and nothing else. No client code has to know which model is running.
