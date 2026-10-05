# Phase 7 design choices: Kubernetes, GPU sharing and autoscaling

Phase 7 moved the stack from Docker Compose onto Kubernetes and asked two questions a platform team would ask. Can two models share one GPU without hurting each other? And can the cluster add a GPU when load rises, then give it back?

## 1. k3s on one node instead of a managed cluster

A managed Kubernetes service with GPU nodes would have cost more than the whole project so far, and it hides the parts I wanted to show: the container runtime, the GPU device plugin, the scheduler. k3s is real Kubernetes in a single binary, and it runs fine inside the same rented VM as earlier phases. I made the NVIDIA runtime the default for every pod, so the GPU exporter can see both GPUs without asking the scheduler for one.

## 2. One script that builds the whole cluster

Just like the earlier phases, everything the GPU machine needs is one script: switch off automatic updates (they replaced the NVIDIA driver in the middle of the Phase 3 session), install the NVIDIA container toolkit, k3s, Helm, HAMi and KEDA, build the gateway image and load it straight into k3s, pull the 22 GB vLLM image once, and deploy the monitoring stack. It took about 15 minutes and worked the first time on these new GPUs.

## 3. Generating the Kubernetes resources from Python

There are four scenarios, and each one changes the vLLM pods, their GPU limits and which GPU they land on, while everything around them stays the same. Rather than keep four sets of YAML in sync, one Python script writes the resources for the base stack and for each scenario. The NGINX config is the same file as in Compose, with the server list pointed at a Kubernetes service that lists every vLLM pod, and Prometheus finds vLLM pods by asking the Kubernetes API instead of reading a file of targets.

## 4. HAMi for sharing, not time slicing or MIG

The RTX 5090 doesn't support MIG, NVIDIA's way of cutting a datacenter GPU into hardware partitions. NVIDIA's time slicing lets several pods use a GPU but gives them no memory limits, so one vLLM server would grab 90% of the memory and the second would fail to start. HAMi sits between the program and the CUDA driver and enforces a memory limit per pod, plus an optional compute limit, which is exactly the control a shared GPU needs.

I wasn't sure vLLM would respect it, since vLLM sizes its cache from what it thinks the GPU's total memory is. It did: with a 20,000 MiB limit, the 7B model used 17.8 GB instead of 29 GB, and the small model fit in 8.5 GB.

## 5. Two tenants with different jobs

Sharing only means something if someone could get hurt. So tenant A is the 7B model taking the real traffic, pushed from 32 to 256 users, and tenant B is a small 1.5B model playing a latency sensitive internal service: four clients asking short questions the whole time, starting a minute before A's load so there's a clean "alone" baseline. The question is what B's latency does as A gets busier.

## 6. Pinning pods to physical GPUs

For the isolated scenario to mean isolated, the two pods have to land on different GPUs, and for the shared scenario they have to land on the same one. HAMi lets a pod name the exact GPU it wants by its ID, so the run script reads both GPU IDs from the machine and pins each pod. The placement and an nvidia smi snapshot are saved with every scenario, so the notebook proves where things ran.

## 7. What sharing costs, and who pays

Sharing halved tenant A's throughput, from 7,873 to 3,964 tokens a second. But per GPU it's the same number, because the isolated setup spent an entire RTX 5090 and 29 GB of memory on a small model serving four clients. Sharing doesn't create capacity, it stops a small model from wasting a big GPU.

Tenant B paid too: its time to first token went from about 10 to about 22 milliseconds as soon as A had any load. What surprised me is that it didn't get any worse from there. Going from 32 to 256 users on A changed nothing for B, and B had zero errors. For an internal service that's an easy trade: a few extra milliseconds for half the hardware.

## 8. Testing compute limits, and finding they didn't matter

I ran sharing a second time with a 70/30 compute split, expecting it to protect B. It changed nothing measurable. HAMi's compute limit is a ceiling, enforced by holding back kernel launches when a pod goes over its share, not a reservation. Under contention the two pods were most likely already splitting the GPU about evenly, so A never reached its 70% and B never wanted more than its 30%. A ceiling like that is useful against a neighbour that would take the whole GPU, but it isn't a way to buy the quiet tenant lower latency. I kept the run because a control that doesn't do what you expect is worth knowing about before you rely on it.

## 9. Autoscaling on requests inside vLLM

GPU utilization is a poor signal for scaling an LLM server: it sits near 100% from light load to overload. The number that actually tracks pressure is how many requests vLLM is holding, running plus waiting. KEDA reads that from Prometheus and adds a pod when it passes 180 per pod, which is below the point where requests start queueing. KEDA asked for the second pod 12 seconds after the load jumped, and removed it once the load stayed low past a 90 second wait, so a short dip doesn't throw away a loaded model.

## 10. The real cost of scaling out is the cold start

The second pod needed 139 seconds to load the model and pass its health check, even with the weights already on local disk. For those two and a half minutes, about 125 requests waited and tail latency to first token sat near 5 seconds. Once it was ready, NGINX found it on its own and tail latency fell to 0.15 seconds with more than twice the throughput. On GPUs, scaling out isn't instant the way it is for web servers, so the threshold should fire early, or the cluster should keep one warm spare when spikes matter.

## 11. Scaling in dropped 32 streams

All 32 errors in the autoscale run came at the same moment, 30 seconds after the scale down. Kubernetes told the removed pod to stop, waited its default 30 seconds, then killed it while long answers were still streaming. The fix is a longer grace period with a short wait before shutdown, so the pod drops out of the service first and finishes what it started. I didn't add it after the fact, because it wasn't in the run that was measured, but it's the first change I'd make.

## 12. Fixing the dashboard during the session

The two HAMi panels came up empty. HAMi 2.10 renamed all its metrics to start with hami, and my panels used the old names. Prometheus had been collecting the new ones all along, so I fixed the queries, reloaded Grafana on the cluster, and the whole hour of data showed up. Nothing was lost, and the repo has the fixed dashboard.

## What I'd recommend for this hardware

Put small models on a shared GPU with HAMi memory limits, and keep a large model on a whole GPU when it's taking real traffic. Skip compute limits unless a neighbour is actively misbehaving. Autoscale on requests inside vLLM, plan for about two and a half minutes of cold start per new pod, and give vLLM pods a shutdown grace period longer than the longest answer.
