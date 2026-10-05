# Phase 6 design choices: scaling out

Phase 6 took the tuned single GPU setup and asked what changes when there's more than one GPU. Most of the interesting lessons turned out to be about the parts between the GPUs: the load balancer, the network and the way the servers share memory.

## 1. Generating every setup from one script

There were six different shapes to test: one, two or four copies, a model split across GPUs, a shared Mooncake store, and separate prefill and decode servers. Writing a compose file for each by hand would have meant six chances to get a GPU number or a port wrong. Instead, one small script takes a description like "2 copies, least connections" and writes the Compose services, the NGINX server list and the Prometheus targets together, so they always agree.

## 2. Testing the whole session without a GPU first

The script was tested end to end on my laptop with fake vLLM servers standing in for the real ones. That run caught two bugs that would have wasted the paid session: the script only waited for the first server to be ready, so requests reached servers that were still loading, and the prefill/decode proxy didn't answer the readiness check. Both were fixed before renting anything.

## 3. Two GPUs instead of waiting for four

No machine with four RTX 5090s was available as a virtual machine. Two GPUs still answer every core question: does throughput double, which balancer wins, what happens when a server dies, and does splitting a model beat running copies. What's missing is the four GPU point on the scaling curve. Rather than wait an unknown amount of time, I ran the two GPU version and kept the script able to run four whenever a machine turns up.

## 4. Giving the gateway more workers

In Phase 4 one gateway process used most of a CPU core at about 6,000 tokens a second, and two GPUs make about 16,000. So the gateway now runs several worker processes, with their Prometheus metrics merged through a shared directory, so the dashboard still shows one set of numbers.

## 5. A conversation workload for testing routing

The Phase 4 traffic sends independent requests, and with independent requests it doesn't matter which server gets what. To test sticky routing and a shared cache fairly, I added a conversation mode: each simulated user discusses a long document for six turns, so every request repeats that user's growing prefix. That's the traffic where keeping a user on one server, or sharing cache between servers, could pay off.

## 6. Least connections as the default balancer

Round robin hands out requests evenly by count. But requests differ a lot in length, so once one server falls slightly behind, it keeps getting its full share and piles up. Near capacity, round robin had one server running about 190 requests and the other about 90, and tail latency hit 12 seconds. Least connections sends each request to whichever server is less busy and kept them even, with tail latency of 1.2 seconds. Sticky routing didn't help either: with only two servers, round robin already sends a user back to the same server half the time, and the imbalance cost far more than the cache saved.

## 7. Health checks that don't cause outages

The first two copy run had over 1,500 errors with both servers perfectly healthy. NGINX had decided both were dead. vLLM's web server closes idle connections after five seconds, NGINX kept them for up to a minute, so under heavy load NGINX sometimes reused a connection vLLM had already closed. With NGINX set to bench a server after a single failure, a handful of those resets benched both.

The fix has two parts: NGINX now closes idle connections after four seconds, before vLLM does, and a server is only benched after three failures. Dead servers are still noticed quickly, because a stopped container disappears from Docker's DNS within seconds. I reran the affected experiments and kept the original results, because a balancer causing the outage it exists to prevent is worth showing.

## 8. A failover drill under real load

To test failover, one server is stopped instantly, like a crash, while the load test is running, and started again 90 seconds later. The only errors were the requests already streaming from that server at the moment it stopped; an answer that's half sent can't be moved to another server. Everything after that succeeded on the remaining server, and NGINX picked the restarted server back up without being restarted itself.

## 9. Copies, not tensor parallel, on consumer GPUs

Splitting one model across two GPUs was about thirteen times slower than running two copies, and slower than a single GPU. The reason shows up in NVIDIA's own topology report: these GPUs talk through the PCIe host bridge with peer to peer access not supported, so the two all reduces in every layer go through host memory. Tensor parallel only makes sense with fast GPU to GPU links, or when a model doesn't fit on one GPU. A 7B model fits easily, so one copy per GPU is the answer here.

## 10. Measuring Mooncake honestly, including when it hurts

The first Mooncake store run looked identical to plain round robin, and Prometheus showed why: zero Mooncake operations. My script had added the Mooncake image but kept the normal engine config, which doesn't turn the connector on. I fixed the script so the store option picks the Mooncake config, and reran.

With Mooncake really on, it made things much worse: up to two thirds less throughput, and hundreds of requests evicted a minute. vLLM keeps each block of KV in GPU memory until it has been saved to the store, and saving over TCP can't keep up with many prefills at once, so blocks waiting to be saved crowd out the requests that are running. In Phase 2, Mooncake helped a long prompt that came back after being evicted, at low load. Under concurrent load without RDMA, it's a cost, not a benefit.

## 11. Stopping at the vLLM bug

Separate prefill and decode servers came up correctly, but the decode server's engine crashed on an assertion in vLLM's own scheduler while handing over KV. That's a bug in vLLM 0.30.0's disaggregated path, not something to debug at an hourly rate. I saved the crash log and recorded it as a finding.

## The deployment I'd recommend for this hardware

One FP8 weights copy per GPU, behind NGINX with least connections, idle connections closed before vLLM closes them, and health checks that need several failures. No tensor parallel, and no shared Mooncake store until there's RDMA.
