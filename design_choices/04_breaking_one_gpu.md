# Phase 4 design choices: breaking one GPU

Phase 4 was about finding the edge: how many users one RTX 5090 can serve, what gives way first, and why. Most of the decisions were about making the load realistic and the measurements honest, so the breaking point would mean something.

## 1. Locust, closed loop, in steps

Each simulated user sends its next request the moment the last one finishes. That way the number of users is exactly the number of requests in flight, which makes every step easy to reason about: at 192 users, vLLM has about 192 requests to juggle.

The user count goes up in steps (1, 2, 4 and so on up to 512), each held for a minute, and the analysis skips the first 15 seconds of every step so each row describes the system after it settled rather than the jump itself.

The other way to load test is an open loop, sending a fixed number of new requests per second whatever happens. That's closer to how real traffic arrives, and it would show the queue growing without bound once the GPU falls behind. I chose the closed loop for this first sweep because it finds the saturation point cleanly; an open loop test at a fixed arrival rate is a good follow up once the capacity is known.

## 2. A realistic mix, and a unique tag on every request

The traffic is weighted like a chat product: mostly short questions, a quarter multi turn conversations, a quarter long documents to summarize, and some long answers. The long documents stress prefill, the long answers stress decode.

Every request gets a unique tag at the start of its user message. Without it, the same 24 prompts sent thousands of times would mostly be served from vLLM's prefix cache, and the test would measure the cache instead of the GPU. System prompts stay identical, as they would in a real app, so they are still shared. The prefix cache hit rate during the run was about 8%, which is what shared system prompts alone should give.

## 3. Moving the rate limits out of the way, and only those

The gateway allows each user 60 requests a minute and sheds load beyond 256 requests in flight. Left on, those limits would have been what broke, not the GPU. For this test I raised exactly those two, and left every other harness on: token budgets, the injection check, time limits. The question was what the GPU can do; how to protect it with limits comes after knowing that.

## 4. Measuring on both sides

The client measures what a user feels: time to first token, time between tokens and total time, for every request, written to a log as it happens. Prometheus measures what the server is doing over the same windows: requests running and waiting inside vLLM, KV cache usage, preemptions, tokens per second, GPU power.

Neither side alone explains a breaking point. The client side showed time to first token jumping from half a second to almost six seconds at 384 users. The server side showed why: running requests flat at 256, a queue of 127, and the KV cache only half full.

## 5. Running Locust in Docker

Locust 2.46 needs Python 3.11 or newer, and the GPU machines run Ubuntu 22.04 with Python 3.10. Rather than pin an old Locust, I run the official Locust image on the same Docker network as the gateway, with 8 worker processes so the load generator itself can't become the bottleneck. As a bonus, Locust's own web UI is available live through the tunnel, and its HTML report is saved with the results.

## 6. Watching for bottlenecks outside the GPU

The gateway is a single Python process that handles every streamed token. If it had run out of CPU first, the GPU numbers would have been meaningless. So the session sampled every container's CPU every 10 seconds. At peak, the gateway used 84% of one core. It wasn't the limit, but it was closer than I expected, and it will need more workers once there's more than one GPU behind it.

## 7. Concurrent agents as a second kind of load

Locust sends independent requests. An agent product sends tasks, and each task is a chain of calls that wait on each other. So I also ran 1 to 32 CrewAI crews at the same moment, each on its own process with its own unique document tag, and measured how long a whole task takes. 32 crews at once took 21% longer per task and got 24 times as much work done per minute.

## 8. Reading the results carefully

Three things in the data needed care. GPU utilization showed 100% even with one user, because that metric only means a kernel was running, not that the GPU was fully used; power draw was the better signal, sitting at its 575 W limit from 96 users on. The KV cache gauge counts only blocks held by running requests, which is exactly the number that matters when asking whether memory is the limit, and it never passed 53%. And the last step's client side numbers are flattered, because Locust stops at the end of the step and requests still running at that moment never get logged; the server side numbers for that step are unaffected.

## 9. Defining "broken" more than one way

There isn't a single breaking point, so the analysis reports several: when requests start queueing, when the KV cache fills, when preemptions start, when time to first token passes a target, and when errors appear. It also reports the capacity at several latency targets, because "how many users" depends entirely on how fast is fast enough. With a 200 ms target this GPU serves 96 users; with a 1 second target, 256.

## 10. A mirror for Docker images

This session's machine couldn't download anything served through Cloudflare, which is where Docker Hub's image layers come from, while everything else on the internet was fine. Restarting Docker and pointing it at Google's public mirror of Docker Hub fixed it, and the 21.6 GB vLLM image came down normally. The setup script now always uses the mirror, with Docker falling back to Docker Hub on its own, and it puts time limits on the download and the GPU check so a stall can never again cost 25 minutes in silence.
