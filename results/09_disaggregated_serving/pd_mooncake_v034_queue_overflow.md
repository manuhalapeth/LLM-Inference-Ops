# Mooncake over TCP under the crash test's load (vLLM v0.34.0, defaults)

Recorded live during the v0.34.0 crash test (16 clients, ~3,500 token prompts,
1 prefill + 1 decode, Mooncake TCP), 10:35 to 10:41 UTC. The container logs
were replaced when the next setup started, so these are the lines copied at the time.

* Prefill server: `tcp_transport_lane_impl.h:1330] TCP lane queue-full rejection count`
  climbing 16, 32, ... 32768 between 10:35:52 and 10:36:01.
* Decode server: `pulling kv_caches for [...] failed: Mooncake transfer engine returned -1`,
  then `Failing N request(s) due to KV load failure (failure_policy=fail, ~3,500 tokens affected)`:
  93 such failures by 10:38, 280 requests answered.
* After that, no request reached either server for minutes. Metrics at 10:40
  (pd_mooncake_v034_queue_overflow_metrics.txt): prefill server 97 failed
  transfers, 0 running, 198 waiting, KV cache 99.85% full; decode server 97 failed
  receives, KV cache 99.85% full. Blocks of failed transfers stay held until
  VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT (480 s by default), so both servers filled
  up and stopped admitting new prompts.
* No engine crash: v0.34.0 fails the request instead of asserting.
