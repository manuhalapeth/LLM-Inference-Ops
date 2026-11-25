"""Write the config for a multi-GPU setup: which vLLM servers run on which GPUs,
how NGINX balances between them, and which servers Prometheus scrapes.

    python3 scripts/topology.py --replicas 4                 # 4 copies, one per GPU
    python3 scripts/topology.py --replicas 1 --tp 2          # one copy split across 2 GPUs
    python3 scripts/topology.py --replicas 4 --lb user_hash  # sticky routing per user
    python3 scripts/topology.py --replicas 4 --mooncake-store
    python3 scripts/topology.py --pd 1,1                     # 1 prefill + 1 decode server
    python3 scripts/topology.py --pd 1,1 --pd-connector nixl # the same, KV moved by NIXL
    python3 scripts/topology.py --replicas 4 --router cache_aware  # vLLM's router picks the server

Writes generated/compose.json, generated/upstream.conf, generated/prom_targets/vllm.json
and generated/env. Then:

    set -a; . generated/env; set +a
    docker compose -f docker-compose.yml -f generated/compose.json up -d

Paths in the generated compose file are relative to the repo root, because
Compose resolves every -f file's paths against the first file.
Standard library only.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "generated"
# Phases 6 to 8 ran v0.30.0. Phase 9 on sets VLLM_TAG=v0.34.0 (and VLLM_IMAGE to match).
VLLM_IMAGE = "${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}"
MOONCAKE_IMAGE = "llm-inference-ops/vllm-mooncake:${VLLM_TAG:-v0.30.0}"
MOONCAKE_BUILD = {"context": "./vllm/mooncake", "args": {"VLLM_IMAGE": VLLM_IMAGE}}

ROUTER_IMAGE = "llm-inference-ops/vllm-router:0.1.17"
ROUTER_POLICIES = ["round_robin", "power_of_two", "consistent_hash", "cache_aware", "kv_aware"]

LB_METHODS = {
    "round_robin": "",                                   # NGINX's default: each server in turn
    "least_conn": "least_conn;",                         # the server with the fewest open requests
    "user_hash": "hash $http_x_user_id consistent;",     # the same user always goes to the same server
}


def vllm_service(name: str, gpus: list[int], config: str, tp: int, port: int, image: str,
                 extra_env: dict | None = None, fake_dir: str | None = None) -> dict:
    if fake_dir:  # local testing: a small stand-in for vLLM, no GPU
        return {
            "image": "python:3.12-slim",
            "entrypoint": ["sh", "-c", "pip install -q fastapi==0.142.2 uvicorn==0.54.0 && python /fake/fake_vllm_local.py"],
            "volumes": [f"{fake_dir}:/fake:ro"],
            "environment": {"FAKE_NAME": name},
            "ports": [f"127.0.0.1:{port}:8000"],
            "healthcheck": {"test": ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"],
                            "interval": "5s", "timeout": "5s", "retries": 3, "start_period": "60s"},
            "restart": "unless-stopped",
        }
    command = ["${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}", "--served-model-name", "${SERVED_MODEL_NAME:-llm}",
               "--config", f"/configs/{config}.yaml"]
    if tp > 1:
        command += ["--tensor-parallel-size", str(tp)]
    service = {
        "image": image,
        "entrypoint": ["vllm", "serve"],
        "command": command,
        "environment": {"HF_TOKEN": "${HF_TOKEN:-}", "OTEL_SERVICE_NAME": name, **(extra_env or {})},
        "volumes": ["./vllm/configs:/configs:ro", "./vllm/mooncake:/mooncake:ro",
                    "${HF_CACHE_DIR:-${HOME}/.cache/huggingface}:/root/.cache/huggingface"],
        "ipc": "host",
        "ports": [f"127.0.0.1:{port}:8000"],
        "deploy": {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "device_ids": [str(g) for g in gpus], "capabilities": ["gpu"]}]}}},
        "healthcheck": {"test": ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"],
                        "interval": "10s", "timeout": "5s", "retries": 3, "start_period": "15m"},
        "restart": "unless-stopped",
    }
    if image == MOONCAKE_IMAGE:
        service["build"] = MOONCAKE_BUILD
    return service


def build(args) -> tuple[dict, str, list, dict]:
    services, backends = {}, []
    image = MOONCAKE_IMAGE if (args.mooncake_store or args.pd) else VLLM_IMAGE

    if args.mooncake_store and args.config == "tune_fp8_weights":
        args.config = "fp8_weights_mooncake_store"  # same engine settings plus the Mooncake connector
    if args.pd:
        n_prefill, n_decode = (int(x) for x in args.pd.split(","))
        names = [f"vllm-prefill-{i}" for i in range(n_prefill)] + [f"vllm-decode-{i}" for i in range(n_decode)]
        suffix = {"nixl": "_nixl", "nixl_hostpid": "_nixl", "mooncake_tuned": "_mooncake_tuned"}.get(args.pd_connector, "")
        configs = [args.prefill_config + suffix] * n_prefill + [args.decode_config + suffix] * n_decode
    else:
        names = [f"vllm-{i}" for i in range(args.replicas)]
        configs = [args.config] * args.replicas

    if (len(names) * args.tp) > args.gpus:
        raise SystemExit(f"{len(names)} servers x {args.tp} GPUs each needs {len(names) * args.tp} GPUs; only {args.gpus}")

    extra_env = {"MOONCAKE_CONFIG_PATH": "/mooncake/mooncake_config_shared.json"} if args.mooncake_store else {}
    for i, (name, config) in enumerate(zip(names, configs)):
        gpus = list(range(i * args.tp, (i + 1) * args.tp))
        env = dict(extra_env)
        if config.endswith("_kv_events"):  # full 32 byte block hashes, as vLLM's router expects
            env["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] = "0"
        if args.pd and args.pd_connector in ("nixl", "nixl_hostpid"):
            # The prefill server tells the decode server where to fetch the KV from:
            # its own name on the compose network, not the default "localhost".
            env.update({"VLLM_NIXL_SIDE_CHANNEL_HOST": name, "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600",
                        "UCX_NET_DEVICES": "all", "UCX_LOG_LEVEL": "info"})
        if args.pd and args.pd_connector == "mooncake_tuned":
            # A deeper TCP send queue (default overflowed at 16 concurrent long prompts), and
            # give back a prefill server's KV blocks 60 s after a transfer stalls, not 480 s.
            env.update({"MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER": "65536", "MC_TCP_LANES_PER_PEER": "8",
                        "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": "60"})
        services[name] = vllm_service(name, gpus, config, args.tp, 8000 + i, image, env, args.fake_dir)
        if args.pd and args.pd_connector == "nixl_hostpid":
            # UCX's shared memory transports (cma, posix) need to see the other process,
            # which separate PID namespaces hide; with them hidden, NIXL fell back to TCP.
            services[name]["pid"] = "host"
        if args.mooncake_store and not args.fake_dir:
            services[name]["depends_on"] = ["mooncake-master"]

    if args.mooncake_store and not args.fake_dir:
        services["mooncake-master"] = {"image": MOONCAKE_IMAGE, "build": MOONCAKE_BUILD,
                                       "entrypoint": ["mooncake_master"], "command": ["--port", "50051"],
                                       "restart": "unless-stopped"}

    if args.pd and args.pd_connector in ("nixl", "nixl_hostpid"):
        # vLLM's NIXL proxy: the prefill server returns where its KV lives, the
        # decode server pulls it with NIXL and streams the answer.
        cmd = ["python3", "/mooncake/nixl_proxy.py", "--host", "0.0.0.0", "--port", "8000",
               "--prefiller-hosts", *names[:n_prefill], "--prefiller-ports", *["8000"] * n_prefill,
               "--decoder-hosts", *names[n_prefill:], "--decoder-ports", *["8000"] * n_decode]
    elif args.pd:
        # vLLM's example proxy: sends each request to a prefill server (1 token,
        # KV written via Mooncake), then streams the answer from a decode server.
        bootstrap = "8000" if args.fake_dir else "8998"
        cmd = ["python3", "/mooncake/pd_proxy.py", "--host", "0.0.0.0", "--port", "8000"]
        for n in names[:n_prefill]:
            cmd += ["--prefill", f"http://{n}:8000", bootstrap]
        for n in names[n_prefill:]:
            cmd += ["--decode", f"http://{n}:8000"]
    if args.pd:
        if args.fake_dir:
            proxy = {"image": "python:3.12-slim", "volumes": ["./vllm/mooncake:/mooncake:ro"],
                     "entrypoint": ["sh", "-c", "pip install -q fastapi==0.142.2 uvicorn==0.54.0 httpx==0.28.1 && exec " + " ".join(cmd)]}
        else:
            proxy = {"image": MOONCAKE_IMAGE, "build": MOONCAKE_BUILD, "volumes": ["./vllm/mooncake:/mooncake:ro"],
                     "entrypoint": cmd}
        services["pd-proxy"] = {**proxy, "depends_on": names, "restart": "unless-stopped"}
        backends = ["pd-proxy"]
    elif args.router:
        # vLLM's router does the balancing; NGINX keeps one upstream (the router), so
        # the gateway, NGINX and the traces stay as they were.
        cmd = ["--host", "0.0.0.0", "--port", "8000", "--policy", args.router,
               "--worker-urls", *[f"http://{n}:8000" for n in names],
               "--prometheus-host", "0.0.0.0", "--prometheus-port", "29000",
               "--request-timeout-secs", "600", "--log-level", "warning",
               # NGINX doesn't retry a streaming request either; a failure should count.
               "--disable-retries"]
        if args.router == "kv_aware":
            # Tell it where each server publishes its KV cache events (set in
            # tune_fp8_weights_kv_events.yaml) rather than relying on discovery.
            for n in names:
                cmd += ["--kv-event-source", f"http://{n}:8000=tcp://{n}:5557",
                        "--kv-event-replay-source", f"http://{n}:8000=tcp://{n}:5558"]
            # Static sources also need the served model's name ("model identity").
            cmd += ["--kv-event-block-size", "16", "--kv-event-model", "${SERVED_MODEL_NAME:-llm}"]
        if args.router == "cache_aware":
            cmd += ["--cache-threshold", str(args.cache_threshold),
                    "--balance-abs-threshold", str(args.balance_abs_threshold)]
        services["vllm-router"] = {"image": ROUTER_IMAGE, "build": {"context": "./router"}, "command": cmd,
                                   "depends_on": names, "restart": "unless-stopped"}
        backends = ["vllm-router"]
    else:
        backends = names

    # The single-server vLLM from docker-compose.yml is not part of these setups.
    services["vllm"] = {"profiles": ["single"]}
    if args.fake_dir:  # no GPU locally, so no GPU metrics exporter either
        services["gpu-exporter"] = {"profiles": ["gpu"]}
    services["nginx"] = {"depends_on": ["jaeger"] + backends}

    upstream = "\n".join([
        f"# Generated by scripts/topology.py: {describe(args)}",
        "upstream vllm_backends {",
        "    zone vllm_backends 64k;",
        *([f"    {LB_METHODS[args.lb]}"] if LB_METHODS[args.lb] else []),
        # A server is benched after 3 failed requests in 10 s. (With max_fails=1, two
        # stale connection resets under load benched BOTH servers: see Phase 6.)
        *[f"    server {b}:8000 resolve max_fails=3 fail_timeout=10s;" for b in backends],
        "    keepalive 64;",
        # vLLM's web server drops idle connections after 5 s; close them first, so
        # NGINX never reuses a connection vLLM has already closed.
        "    keepalive_timeout 4s;",
        "}",
        "",
    ])
    targets = [{"targets": [f"{n}:8000"], "labels": {"replica": n}} for n in names]
    if args.router:
        targets.append({"targets": ["vllm-router:29000"], "labels": {"replica": "router"}})
    env = {
        "NGINX_UPSTREAM": "./generated/upstream.conf",
        "PROM_TARGETS_DIR": "./generated/prom_targets",
        "GATEWAY_WORKERS": str(args.gateway_workers),
        "TOPOLOGY": describe(args),
    }
    return {"services": services}, upstream, targets, env


def describe(args) -> str:
    if args.pd:
        p, d = args.pd.split(",")
        return f"pd {p}P+{d}D {args.pd_connector} tp{args.tp} lb={args.lb}"
    lb = f"router={args.router}" if args.router else f"lb={args.lb}"
    return f"{args.replicas}x tp{args.tp} {args.config} {lb}" + (" mooncake_store" if args.mooncake_store else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1, help="GPUs per vLLM server (tensor parallel)")
    parser.add_argument("--gpus", type=int, default=4, help="GPUs on the machine")
    parser.add_argument("--config", default="tune_fp8_weights", help="vllm/configs/<name>.yaml for every server")
    parser.add_argument("--lb", choices=list(LB_METHODS), default="round_robin")
    parser.add_argument("--router", choices=ROUTER_POLICIES, help="put vLLM's router behind NGINX with this policy")
    parser.add_argument("--cache-threshold", type=float, default=0.3, help="cache_aware: prefix match needed to follow the cache")
    parser.add_argument("--balance-abs-threshold", type=int, default=64,
                        help="cache_aware: requests of imbalance before it balances load instead")
    parser.add_argument("--mooncake-store", action="store_true", help="share KV cache between servers through Mooncake")
    parser.add_argument("--pd", help="prefill,decode server counts, e.g. 1,1 (disaggregated serving)")
    parser.add_argument("--pd-connector", choices=["mooncake", "mooncake_tuned", "nixl", "nixl_hostpid"], default="mooncake",
                        help="what moves the KV from prefill to decode")
    parser.add_argument("--prefill-config", default="pd_prefill")
    parser.add_argument("--decode-config", default="pd_decode")
    parser.add_argument("--gateway-workers", type=int, default=4)
    parser.add_argument("--fake-dir", help="local testing only: folder with fake_vllm_local.py")
    args = parser.parse_args()

    compose, upstream, targets, env = build(args)
    (OUT / "prom_targets").mkdir(parents=True, exist_ok=True)
    (OUT / "compose.json").write_text(json.dumps(compose, indent=2) + "\n")
    (OUT / "upstream.conf").write_text(upstream)
    (OUT / "prom_targets" / "vllm.json").write_text(json.dumps(targets, indent=2) + "\n")
    (OUT / "env").write_text("".join(f'{k}="{v}"\n' for k, v in env.items()))
    print(f"{describe(args)}: {', '.join(t['labels']['replica'] for t in targets)}")


if __name__ == "__main__":
    main()
