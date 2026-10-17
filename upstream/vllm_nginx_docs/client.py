# Client for the NGINX + vLLM reproduction. Talks to NGINX at $LB (default http://nginx).
#   race SECONDS WORKERS   requests with idle gaps near vLLM's 5 s keep-alive; counts statuses
#   ttfb N                 streaming requests; time to the first SSE event and to the end
#   spread SECONDS RPS     steady requests; which server answered, per 5 s bucket
import asyncio, collections, json, os, random, sys, time

import httpx

LB = os.environ.get("LB", "http://nginx")
URL = LB + "/v1/chat/completions"


async def race(seconds, workers):
    statuses = collections.Counter()
    end = time.monotonic() + seconds
    async with httpx.AsyncClient(timeout=30) as c:
        async def worker():
            await asyncio.sleep(random.uniform(0, 5))
            while time.monotonic() < end:
                try:
                    r = await c.post(URL, json={"max_tokens": 4, "token_delay": 0.01})
                    statuses[r.status_code] += 1
                except httpx.HTTPError as e:
                    statuses[type(e).__name__] += 1
                await asyncio.sleep(random.uniform(4.8, 5.2))
        await asyncio.gather(*(worker() for _ in range(workers)))
    total = sum(statuses.values())
    print(json.dumps({"mode": "race", "requests": total, "statuses": dict(statuses)}))


async def bursts(seconds, size):
    # Bursts of `size` concurrent requests, spaced so NGINX's pooled connections have been
    # idle for about vLLM's 5 s keep-alive when the next burst reuses them.
    statuses = collections.Counter()
    end = time.monotonic() + seconds
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=size * 2)) as c:
        async def one():
            try:
                r = await c.post(URL, json={"max_tokens": 4, "token_delay": 0.01})
                statuses[r.status_code] += 1
            except httpx.HTTPError as e:
                statuses[type(e).__name__] += 1
        while time.monotonic() < end:
            await asyncio.gather(*(one() for _ in range(size)))
            await asyncio.sleep(random.uniform(4.94, 5.06))
    print(json.dumps({"mode": "bursts", "requests": sum(statuses.values()), "statuses": dict(statuses)}))


async def ttfb(n, tokens=64):
    firsts, totals = [], []
    async with httpx.AsyncClient(timeout=30) as c:
        for _ in range(n):
            t0 = time.monotonic()
            first = None
            async with c.stream("POST", URL, json={"stream": True, "max_tokens": tokens, "token_delay": 0.02}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data:") and first is None:
                        first = time.monotonic() - t0
            firsts.append(first)
            totals.append(time.monotonic() - t0)
    firsts.sort(); totals.sort()
    print(json.dumps({"mode": "ttfb", "n": n, "tokens": tokens,
                      "first_token_ms_p50": round(firsts[n // 2] * 1000),
                      "full_response_ms_p50": round(totals[n // 2] * 1000)}))


async def spread(seconds, rps):
    buckets = collections.defaultdict(collections.Counter)
    t_start = time.monotonic()
    async with httpx.AsyncClient(timeout=30) as c:
        async def one():
            b = int((time.monotonic() - t_start) // 5) * 5
            try:
                r = await c.post(URL, json={"max_tokens": 4, "token_delay": 0.01})
                key = r.json()["server"] if r.status_code == 200 else str(r.status_code)
            except Exception as e:  # noqa: BLE001
                key = type(e).__name__
            buckets[b][key] += 1
        tasks = []
        while time.monotonic() - t_start < seconds:
            tasks.append(asyncio.create_task(one()))
            await asyncio.sleep(1 / rps)
        await asyncio.gather(*tasks)
    for b in sorted(buckets):
        print(f"t={b:>3}s " + " ".join(f"{k}={v}" for k, v in sorted(buckets[b].items())))


if __name__ == "__main__":
    mode, args = sys.argv[1], [float(a) for a in sys.argv[2:]]
    asyncio.run({"race": race, "bursts": bursts, "ttfb": ttfb, "spread": spread}[mode](*[int(a) if a.is_integer() else a for a in args]))
