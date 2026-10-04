# Phase 2 design choices: KV cache management with Mooncake

Phase 2 was about the resource that limits every inference server: KV cache memory. The goal wasn't just to get Mooncake running, it was to find out honestly what it buys on one GPU and what it costs.

## 1. The store connector

vLLM 0.30.0 ships two Mooncake connectors. MooncakeConnector moves KV directly from one vLLM instance to another, which is how you split prefill and decode across machines. MooncakeStoreConnector writes KV into a shared memory pool and reads it back on a cache miss.

With one GPU there's no second instance to send KV to, so the point to point connector has nothing to do. The store connector is the one that changes anything on a single machine: it turns CPU memory into a second, bigger tier of KV cache. LMCache can also sit in front of Mooncake, but the native connector does the job with one config change and one less moving part. The point to point connector comes back in Phase 6, when there are several GPUs.

## 2. Turning Mooncake on is one config change

The Mooncake run uses `mooncake_store.yaml`, which is the baseline config with exactly one addition: the KV transfer setting that enables the connector. Same model, same memory settings, same context length. Any difference in the results comes from Mooncake and nothing else.

## 3. Building my own image on top of the pinned vLLM

The official vLLM image is built without the optional KV connector packages, so Mooncake isn't in it. I wrote a short Dockerfile that starts from the exact vLLM 0.30.0 image and installs Mooncake on top, so vLLM itself doesn't change.

One detail mattered: the default Mooncake package is built for CUDA 12, and this vLLM image uses CUDA 13. The Dockerfile checks which CUDA version PyTorch was built with and installs the matching package. I also made the build check lenient, because Mooncake may need the GPU driver just to import, and Docker builds run without a GPU. The real check is vLLM starting up, and it did.

## 4. TCP, embedded mode and a 32 GB store

Mooncake is built for RDMA, the fast networking used in big GPU clusters. Rented machines don't have it, so the store uses plain TCP. That turned out to be the most important limitation in the results.

In embedded mode, vLLM's own process contributes the memory to the pool, and a small mooncake master service just keeps track of where everything is. The store gets 32 GB of the machine's RAM, about 558,000 tokens of KV. That's about twice what the experiment needs, while leaving plenty of memory for the host.

## 5. An experiment built to overflow the GPU

To see Mooncake do anything, the working set has to be bigger than the GPU cache. The GPU holds 209,472 tokens of KV, so I used 48 documents of about 5,950 tokens each, 286,000 tokens in total.

Each document starts with its own ID and uses its own random word sequence, because prefix caching matches blocks from the start of the prompt. If two documents shared an opening, they'd share cache and muddy the result. I checked the token counts with Qwen's real tokenizer instead of guessing.

The documents go in the same order twice. That's the worst case for a least recently used cache: by the time document zero comes back, it's the oldest thing in the cache and has already been evicted. Without an extra tier, the second round gets no reuse at all, which makes the effect easy to see.

Each request asks for only 8 output tokens, because this experiment is about the prompt. Long answers would add seconds of decode and bury the difference.

## 6. Three rounds to show three tiers

The fill round computes everything cold. The revisit round shows what happens to evicted prompts: recomputed without Mooncake, loaded from CPU memory with it. The hot round resends the last five documents, which are still on the GPU, to show the fastest tier for comparison. Together they show all three: GPU cache at about 60 ms, Mooncake at about 170 ms, recomputing at about 440 ms.

## 7. Baseline and Mooncake on the same machine, back to back

Phase 1 ran on a different machine than Phase 0, so comparing them would mix hardware differences into the results. This time both configurations ran in the same session on the same machine, a few minutes apart.

That paid off. This machine has a server CPU with slower individual cores than Phase 1's desktop CPU, and with the same GPU, CPU bound work was noticeably slower: about 29 ms to first token for a short prompt instead of 16 ms. If I'd compared Mooncake against Phase 1's numbers, some of that would have looked like Mooncake's fault.

## 8. Checking the cost when Mooncake isn't needed

A new tier is only worth it if it doesn't slow down the common case. I reran Phase 1's single request trace with and without Mooncake. Those prompts all fit on the GPU, so Mooncake can only add work, writing every new block to the store. It added 2 to 5 ms to the time to first token and nothing to the total time.

## 9. Reading vLLM's numbers carefully

Two things in the data needed interpreting. First, on a Mooncake revisit, vLLM's queue time jumped to about 109 ms. That isn't waiting for a slot: vLLM holds a request while its KV loads from the store, and counts that wait as queue time. So the queue number on revisits is really the load time.

Second, the KV cache usage panel on the dashboard stayed at 2 to 3% the whole time, even while the cache was evicting. vLLM's gauge only counts blocks held by running requests. Finished prompts sit in the cache as free, reusable blocks that the gauge doesn't count. I'd expected it to climb to 100% and said so, so it's worth writing down: the per request hit rates are what show eviction, not that panel.

## 10. Restarting NGINX after swapping vLLM

Switching to the Mooncake image recreates the vLLM container, and a recreated container can get a new address on the Docker network. NGINX looks up the address of each upstream server once, when it starts, so it could keep sending requests to an address that no longer exists. The session script restarts NGINX right after the switch. When Phase 6 adds and removes vLLM instances, NGINX will need to look addresses up on its own.

## What I'd change

The weak point is writing to the store: about 0.6 GiB/s over TCP, and part of that write delays the first response. RDMA, or writing KV completely in the background, would shrink that first visit cost, and Mooncake would come out ahead even for prompts that are only reused occasionally.
