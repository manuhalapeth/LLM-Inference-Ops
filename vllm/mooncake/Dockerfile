# The pinned vLLM image plus Mooncake. The official image is built without the
# optional KV connector packages, so this adds the Mooncake store client and the
# mooncake_master binary.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.30.0
FROM ${VLLM_IMAGE}

ARG MOONCAKE_VERSION=0.3.13.post1

# The default Mooncake wheel links CUDA 12; images built on CUDA 13 need the
# cuda13 variant (both install the same `mooncake` package).
RUN CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])") && \
    if [ "$CUDA_MAJOR" = "13" ]; then PKG=mooncake-transfer-engine-cuda13; else PKG=mooncake-transfer-engine; fi && \
    echo "CUDA $CUDA_MAJOR -> installing $PKG==$MOONCAKE_VERSION" && \
    (uv pip install --system --no-cache "$PKG==$MOONCAKE_VERSION" \
      || python3 -m pip install --no-cache-dir "$PKG==$MOONCAKE_VERSION") && \
    (python3 -c "from mooncake.store import MooncakeDistributedStore; print('mooncake store OK')" \
      || echo "mooncake import check skipped: it may need the GPU driver, which only exists at run time") && \
    which mooncake_master
