#!/usr/bin/env bash
# Phase 7: build a rented GPU VM into a one-node Kubernetes cluster (k3s) with
# HAMi (GPU sharing) and KEDA (autoscaling), then deploy the stack.
#
#   curl -fsSL https://raw.githubusercontent.com/manuhalapeth/LLM-Inference-Ops/main/scripts/setup_k3s.sh | bash
#
# Safe to re-run.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/manuhalapeth/LLM-Inference-Ops.git}"
REPO_DIR="${REPO_DIR:-$HOME/LLM-Inference-Ops}"
K3S_VERSION="${K3S_VERSION:-v1.34.12+k3s1}"
HAMI_VERSION="${HAMI_VERSION:-2.10.0}"
KEDA_VERSION="${KEDA_VERSION:-2.21.0}"
VLLM_IMAGE="docker.io/vllm/vllm-openai:v0.30.0"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

log "Switching off automatic package updates (they can replace the NVIDIA driver mid session)"
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
systemctl disable unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
while pgrep -x unattended-upgr >/dev/null; do sleep 10; done

log "Checking the GPUs"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv

log "NVIDIA Container Toolkit (k3s needs nvidia-container-runtime before it starts)"
if ! command -v nvidia-container-runtime >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
fi

log "Docker (to build the gateway image and run Locust)"
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
if ! grep -q mirror.gcr.io /etc/docker/daemon.json 2>/dev/null; then
  python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
cfg = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
cfg["registry-mirrors"] = ["https://mirror.gcr.io"]
json.dump(cfg, open(p, "w"), indent=2)
PY
  systemctl restart docker
fi
command -v python3 >/dev/null && python3 -c "import ensurepip" 2>/dev/null || apt-get install -y -qq python3-venv

log "k3s $K3S_VERSION, with NVIDIA as the default runtime"
mkdir -p /etc/rancher/k3s
# Pull Docker Hub images through Google's mirror (Docker Hub downloads stalled on one box in Phase 4).
cat > /etc/rancher/k3s/registries.yaml <<'EOF'
mirrors:
  docker.io:
    endpoint: ["https://mirror.gcr.io", "https://registry-1.docker.io"]
EOF
if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -s - server \
    --default-runtime nvidia --disable traefik --service-node-port-range 3000-32767 --write-kubeconfig-mode 644
fi
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
grep -q '^export KUBECONFIG' ~/.bashrc || echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc
for i in $(seq 1 60); do kubectl get nodes 2>/dev/null | grep -q " Ready" && break; sleep 5; done
kubectl get nodes -o wide
grep -q nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml* || die "k3s did not detect the NVIDIA runtime"

log "Helm"
command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

log "HAMi $HAMI_VERSION (GPU sharing)"
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl label node "$NODE" gpu=on --overwrite
helm repo add hami-charts https://project-hami.github.io/HAMi/ >/dev/null 2>&1 || true
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update >/dev/null
# The chart's default kube-scheduler image comes from a registry in China; use the upstream one.
helm upgrade --install hami hami-charts/hami --version "$HAMI_VERSION" -n kube-system \
  --set scheduler.kubeScheduler.image.registry=registry.k8s.io \
  --set scheduler.kubeScheduler.image.repository=kube-scheduler --wait --timeout 10m

log "KEDA $KEDA_VERSION (autoscaling)"
helm upgrade --install keda kedacore/keda --version "$KEDA_VERSION" -n keda --create-namespace --wait --timeout 10m

log "Getting the code"
if [ -d "$REPO_DIR/.git" ]; then git -C "$REPO_DIR" pull --ff-only; else git clone "$REPO_URL" "$REPO_DIR"; fi
cd "$REPO_DIR"
[ -f .env ] || cp .env.example .env

log "Gateway image (built with Docker, imported into k3s)"
docker build -q -t llmops/gateway:dev ./gateway >/dev/null
docker save llmops/gateway:dev | k3s ctr images import - >/dev/null

log "Pulling the vLLM image into k3s (~22 GB)"
k3s ctr images ls -q | grep -q "vllm-openai:v0.30.0" || k3s ctr images pull "$VLLM_IMAGE" >/dev/null

log "Deploying the base stack"
mkdir -p generated
python3 scripts/k8s_manifests.py base --node-ip "$(hostname -I | awk '{print $1}')" > generated/k8s_base.json
kubectl apply -f generated/k8s_base.json >/dev/null
kubectl -n llmops rollout status deploy --timeout 10m

log "Ready"
kubectl get pods -A -o wide
cat <<EOF
Cluster is up. Tunnels from your laptop: ~/llm-inference-ops/notes/tunnel.sh PORT IP
  Grafana http://localhost:3000   Prometheus http://localhost:9090   Jaeger http://localhost:16686
Next: ./scripts/run_phase7.sh
EOF
