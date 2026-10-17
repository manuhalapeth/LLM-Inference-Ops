#!/bin/sh
# lab.sh up CONF   start two fake vLLM servers and NGINX with nginx/CONF.conf
# lab.sh down      remove everything
# lab.sh client ARGS...   run client.py inside the network
set -e
cd "$(dirname "$0")"
NET=vllm_nginx_repro
case "$1" in
  up)
    docker build -q -t vllm-fake . >/dev/null
    docker network create $NET >/dev/null 2>&1 || true
    for s in vllm0 vllm1; do docker run -d --rm -e BUSY_MS=${BUSY_MS:-0} --name $s --hostname $s --network $NET vllm-fake >/dev/null; done
    docker run -d --rm --name nginx --network $NET -v "$PWD/nginx/$2.conf:/etc/nginx/conf.d/default.conf:ro" nginx:1.27 >/dev/null
    sleep 3 ;;
  down)
    docker rm -f nginx vllm0 vllm1 >/dev/null 2>&1 || true ;;
  client)
    shift; docker run --rm --network $NET vllm-fake python /app/client.py "$@" ;;
esac
