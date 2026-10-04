#!/usr/bin/env bash
# Build a rented GPU box from scratch in one command.
#
# On a fresh Vast.ai VM (Ubuntu template with CUDA + Docker):
#   curl -fsSL https://raw.githubusercontent.com/manuhalapeth/LLM-Inference-Ops/main/scripts/setup_gpu_box.sh | bash
#
# Safe to re-run: every step checks before it changes anything.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/manuhalapeth/LLM-Inference-Ops.git}"
REPO_DIR="${REPO_DIR:-$HOME/LLM-Inference-Ops}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

log "Checking the GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this machine has no NVIDIA driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

log "Checking Docker"
if ! command -v docker >/dev/null; then
  log "Installing Docker"
  curl -fsSL https://get.docker.com | $SUDO sh
fi
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then DOCKER="$SUDO docker"; fi
$DOCKER compose version >/dev/null 2>&1 || die "Docker Compose plugin missing (install docker-compose-plugin)"

log "Checking that containers can see the GPU"
if ! $DOCKER run --rm --gpus all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1; then
  log "Installing NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker
  $DOCKER run --rm --gpus all ubuntu:22.04 nvidia-smi -L || die "containers still cannot see the GPU"
fi

# Install system packages BEFORE starting containers. Installing packages can
# trigger a systemd reload, which makes running containers lose access to the
# GPU for new processes ("Failed to initialize NVML: Unknown Error").
if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  log "Installing python3-venv"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq python3-venv
fi

log "Getting the code"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

if [ ! -f .env ]; then
  cp .env.example .env
  log "Created .env from .env.example (edit it to set GPU_HOURLY_PRICE_USD and GRAFANA_ADMIN_PASSWORD)"
fi
set -a; . ./.env; set +a

log "Starting the stack"
$DOCKER compose pull --quiet --ignore-buildable
$DOCKER compose up -d --build

log "Waiting for vLLM to load the model (first run downloads the weights)"
python3 scripts/smoke_test.py --wait "$READY_TIMEOUT_S" --save

log "Ready"
cat <<EOF
Stack is up. From your laptop, open an SSH tunnel:
  ssh -L 8080:localhost:8080 -L 3000:localhost:3000 -L 9090:localhost:9090 <this box>
Then:
  gateway     http://localhost:8080/v1/models
  Grafana     http://localhost:3000   (admin / GRAFANA_ADMIN_PASSWORD from .env)
  Prometheus  http://localhost:9090/targets

Before you leave: push code, copy results/ off the box, then DESTROY the instance on Vast.ai.
EOF
