# vLLM's router (Rust, a fork of SGLang's model gateway), for Phase 10.
# It replaces NGINX's balancing: NGINX still sits in front, with one upstream,
# this router, which picks the vLLM server for each request.
FROM python:3.12-slim
ARG VLLM_ROUTER_VERSION=0.1.17
RUN pip install --no-cache-dir "vllm-router==${VLLM_ROUTER_VERSION}" && vllm-router --help > /dev/null
ENTRYPOINT ["vllm-router"]
