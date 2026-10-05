"""Kubernetes manifests for Phase 7: the stack on k3s, with HAMi GPU sharing.

    python3 scripts/k8s_manifests.py base > generated/k8s_base.json
    python3 scripts/k8s_manifests.py isolated --gpu-uuids GPU-a,GPU-b > generated/k8s_scenario.json
    python3 scripts/k8s_manifests.py shared --gpu-uuids GPU-a,GPU-b
    python3 scripts/k8s_manifests.py shared_cores --gpu-uuids GPU-a,GPU-b
    python3 scripts/k8s_manifests.py autoscale

Scenarios (both models are OpenAI-compatible vLLM servers):

    tenant A  "llm"    Qwen2.5-7B-Instruct, FP8 weights (Phase 5's config), behind gateway and NGINX
    tenant B  "small"  Qwen2.5-1.5B-Instruct, reached directly on NodePort 8001

    isolated      A and B each get a whole GPU
    shared        A and B on the same GPU, HAMi memory limits (A 20,000 MB, B 9,000 MB)
    shared_cores  shared, plus HAMi compute limits (A 70%, B 30% of the GPU's cores)
    autoscale     A only, 1 to 2 whole-GPU pods, scaled by KEDA on requests in vLLM

Output is a JSON "List" that kubectl applies directly. Standard library only.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NS = "llmops"
VLLM_IMAGE = "docker.io/vllm/vllm-openai:v0.30.0"
KUBE_DNS = "10.43.0.10"  # k3s default cluster DNS

MODELS = {
    "large": {"model": "Qwen/Qwen2.5-7B-Instruct", "served": "llm", "extra": ["--quantization", "fp8"]},
    "small": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "served": "small", "extra": []},
}


def meta(name, labels=None, annotations=None, namespace=NS):
    m = {"name": name, "labels": {"app": name, **(labels or {})}}
    if namespace:
        m["namespace"] = namespace
    if annotations:
        m["annotations"] = annotations
    return m


def deployment(name, containers, labels=None, volumes=None, replicas=1, pod_annotations=None, extra_spec=None):
    pod_meta = {"labels": {"app": name, **(labels or {})}}
    if pod_annotations:
        pod_meta["annotations"] = pod_annotations
    return {
        "apiVersion": "apps/v1", "kind": "Deployment", "metadata": meta(name, labels),
        "spec": {"replicas": replicas, "selector": {"matchLabels": {"app": name}},
                 "template": {"metadata": pod_meta,
                              "spec": {"containers": containers, "volumes": volumes or [], **(extra_spec or {})}}},
    }


def service(name, port, node_port=None, target=None, headless=False, selector=None):
    spec = {"selector": selector or {"app": name},
            "ports": [{"name": "http", "port": port, "targetPort": target or port}]}
    if headless:
        spec["clusterIP"] = "None"
    elif node_port:
        spec["type"] = "NodePort"
        spec["ports"][0]["nodePort"] = node_port
    return {"apiVersion": "v1", "kind": "Service", "metadata": meta(name), "spec": spec}


def configmap(name, data):
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(name), "data": data}


# ---------------------------------------------------------------- base stack

def nginx_config() -> dict:
    conf = (ROOT / "nginx" / "nginx.conf").read_text()
    conf = conf.replace("resolver 127.0.0.11", f"resolver {KUBE_DNS}")  # Docker DNS -> cluster DNS
    conf = re.sub(r"endpoint jaeger:4317;", f"endpoint jaeger.{NS}.svc.cluster.local:4317;", conf)
    upstream = f"""# Phase 7: vLLM pods behind a headless Service; "resolve" picks up new pods (autoscaling).
upstream vllm_backends {{
    zone vllm_backends 64k;
    least_conn;
    server vllm-large.{NS}.svc.cluster.local:8000 resolve max_fails=3 fail_timeout=10s;
    keepalive 64;
    keepalive_timeout 4s;
}}
"""
    return {"nginx.conf": conf, "upstream.conf": upstream}


def prometheus_config(node_ip: str) -> str:
    return f"""global:
  scrape_interval: 5s
  evaluation_interval: 5s
scrape_configs:
  # Every vLLM pod, found through the Kubernetes API (pods come and go with autoscaling).
  - job_name: vllm
    kubernetes_sd_configs:
      - role: pod
        namespaces: {{names: [{NS}]}}
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_component]
        regex: vllm
        action: keep
      - source_labels: [__meta_kubernetes_pod_ip]
        target_label: __address__
        replacement: "$1:8000"
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: replica
      - source_labels: [__meta_kubernetes_pod_label_tenant]
        target_label: tenant
  - job_name: gateway
    static_configs: [{{targets: ["gateway.{NS}.svc.cluster.local:8080"]}}]
  - job_name: gpu
    static_configs: [{{targets: ["gpu-exporter.{NS}.svc.cluster.local:9835"]}}]
  # HAMi: device plugin (per container GPU memory and cores) and scheduler (allocations).
  - job_name: hami-device-plugin
    static_configs: [{{targets: ["{node_ip}:31992"]}}]
  - job_name: hami-scheduler
    static_configs: [{{targets: ["{node_ip}:31993"]}}]
"""


def base(args) -> list:
    items = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NS}}]

    # Prometheus needs to list pods to find vLLM.
    items += [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": meta("prometheus")},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole",
         "metadata": {"name": "llmops-prometheus"},
         "rules": [{"apiGroups": [""], "resources": ["pods", "nodes", "endpoints", "services"], "verbs": ["get", "list", "watch"]}]},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding",
         "metadata": {"name": "llmops-prometheus"},
         "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "llmops-prometheus"},
         "subjects": [{"kind": "ServiceAccount", "name": "prometheus", "namespace": NS}]},
        configmap("prometheus-config", {"prometheus.yml": prometheus_config(args.node_ip)}),
        deployment("prometheus", [{
            "name": "prometheus", "image": "prom/prometheus:v2.55.1",
            "args": ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus"],
            "ports": [{"containerPort": 9090}],
            "volumeMounts": [{"name": "config", "mountPath": "/etc/prometheus"}, {"name": "data", "mountPath": "/prometheus"}]}],
            volumes=[{"name": "config", "configMap": {"name": "prometheus-config"}}, {"name": "data", "emptyDir": {}}],
            extra_spec={"serviceAccountName": "prometheus"}),
        service("prometheus", 9090, node_port=9090),
    ]

    dashboard = (ROOT / "observability" / "grafana" / "dashboards" / "llm_inference.json").read_text()
    datasource = (f"apiVersion: 1\ndatasources:\n  - name: Prometheus\n    uid: prometheus\n    type: prometheus\n"
                  f"    access: proxy\n    url: http://prometheus.{NS}.svc.cluster.local:9090\n    isDefault: true\n")
    provider = ("apiVersion: 1\nproviders:\n  - name: llm-inference-ops\n    folder: LLM Inference Ops\n"
                "    type: file\n    options:\n      path: /var/lib/grafana/dashboards\n")
    items += [
        configmap("grafana-provisioning", {"datasource.yml": datasource, "dashboards.yml": provider}),
        configmap("grafana-dashboards", {"llm_inference.json": dashboard}),
        deployment("grafana", [{
            "name": "grafana", "image": "grafana/grafana:11.3.0",
            "env": [{"name": "GF_SECURITY_ADMIN_PASSWORD", "value": args.grafana_password},
                    {"name": "GF_USERS_ALLOW_SIGN_UP", "value": "false"}],
            "ports": [{"containerPort": 3000}],
            "volumeMounts": [
                {"name": "provisioning", "mountPath": "/etc/grafana/provisioning/datasources/datasource.yml", "subPath": "datasource.yml"},
                {"name": "provisioning", "mountPath": "/etc/grafana/provisioning/dashboards/dashboards.yml", "subPath": "dashboards.yml"},
                {"name": "dashboards", "mountPath": "/var/lib/grafana/dashboards"}]}],
            volumes=[{"name": "provisioning", "configMap": {"name": "grafana-provisioning"}},
                     {"name": "dashboards", "configMap": {"name": "grafana-dashboards"}}]),
        service("grafana", 3000, node_port=3000),
    ]

    items += [
        deployment("jaeger", [{"name": "jaeger", "image": "jaegertracing/jaeger:2.21.0",
                               "ports": [{"containerPort": 4317}, {"containerPort": 16686}]}]),
        {"apiVersion": "v1", "kind": "Service", "metadata": meta("jaeger"),
         "spec": {"type": "NodePort", "selector": {"app": "jaeger"},
                  "ports": [{"name": "otlp", "port": 4317, "targetPort": 4317, "nodePort": 4317},
                            {"name": "ui", "port": 16686, "targetPort": 16686, "nodePort": 16686}]}},
    ]

    items += [
        configmap("nginx-config", nginx_config()),
        deployment("nginx", [{
            "name": "nginx", "image": "nginx:1.27-alpine-otel", "ports": [{"containerPort": 80}],
            "volumeMounts": [{"name": "config", "mountPath": "/etc/nginx/nginx.conf", "subPath": "nginx.conf"},
                             {"name": "config", "mountPath": "/etc/nginx/upstream.conf", "subPath": "upstream.conf"}]}],
            volumes=[{"name": "config", "configMap": {"name": "nginx-config"}}]),
        service("nginx", 80),
    ]

    gateway_env = {
        "UPSTREAM_URL": f"http://nginx.{NS}.svc.cluster.local:80",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://jaeger.{NS}.svc.cluster.local:4317",
        "OTEL_SERVICE_NAME": "gateway", "GATEWAY_WORKERS": "8",
        # Load testing measures the GPUs, not per-user limits (as in Phases 4 to 6).
        "RATE_LIMIT_RPM": "1000000", "RATE_LIMIT_TPM": "1000000000", "MAX_IN_FLIGHT": "100000",
    }
    items += [
        deployment("gateway", [{
            "name": "gateway", "image": "llmops/gateway:dev", "imagePullPolicy": "Never",
            "env": [{"name": k, "value": v} for k, v in gateway_env.items()],
            "ports": [{"containerPort": 8080}]}]),
        service("gateway", 8080, node_port=8080),
    ]

    if not args.fake:
        # GPU metrics for both physical GPUs. No GPU resource request, so HAMi doesn't
        # schedule it; NVIDIA is the default runtime and exposes every GPU to it.
        items += [
            deployment("gpu-exporter", [{
                "name": "gpu-exporter", "image": "utkuozdemir/nvidia_gpu_exporter:1.15.1",
                "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "utility"}],
                "ports": [{"containerPort": 9835}]}]),
            service("gpu-exporter", 9835),
        ]
    return items


# ---------------------------------------------------------------- vLLM pods

def vllm(name, tenant, gpu_limits=None, annotations=None, replicas=1, fake=False, util=0.90) -> list:
    m = MODELS[tenant]
    labels = {"component": "vllm", "tenant": m["served"]}
    if fake:
        container = {
            "name": "vllm", "image": "python:3.12-slim",
            "command": ["sh", "-c", "pip install -q fastapi==0.142.2 uvicorn==0.54.0 && python /fake/fake_vllm_local.py"],
            "volumeMounts": [{"name": "fake", "mountPath": "/fake"}],
        }
        volumes = [{"name": "fake", "configMap": {"name": "fake-vllm"}}]
    else:
        container = {
            "name": "vllm", "image": VLLM_IMAGE,
            "command": ["vllm", "serve", m["model"], "--served-model-name", m["served"], "--host", "0.0.0.0",
                        "--port", "8000", "--max-model-len", "8192", "--gpu-memory-utilization", str(util), *m["extra"]],
            "env": [{"name": "HF_TOKEN", "value": ""}, {"name": "VLLM_NO_USAGE_STATS", "value": "1"}],
            "resources": {"limits": {k: str(v) for k, v in (gpu_limits or {}).items()}},
            "volumeMounts": [{"name": "hf-cache", "mountPath": "/root/.cache/huggingface"},
                             {"name": "shm", "mountPath": "/dev/shm"}],
        }
        volumes = [{"name": "hf-cache", "hostPath": {"path": "/root/.cache/huggingface", "type": "DirectoryOrCreate"}},
                   {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}]
    container["ports"] = [{"containerPort": 8000}]
    container["readinessProbe"] = {"httpGet": {"path": "/health", "port": 8000},
                                   "periodSeconds": 5, "failureThreshold": 3}
    container["startupProbe"] = {"httpGet": {"path": "/health", "port": 8000},
                                 "periodSeconds": 5, "failureThreshold": 360}  # up to 30 min to load
    items = [deployment(name, [container], labels=labels, volumes=volumes, replicas=replicas, pod_annotations=annotations),
             service(name, 8000, headless=True)]
    if tenant == "small":  # tenant B is reached directly, for the latency probe
        items.append({"apiVersion": "v1", "kind": "Service", "metadata": meta(f"{name}-np"),
                      "spec": {"type": "NodePort", "selector": {"app": name},
                               "ports": [{"name": "http", "port": 8000, "targetPort": 8000, "nodePort": 8001}]}})
    return items


def scenario(args) -> list:
    uuids = (args.gpu_uuids or "").split(",") if args.gpu_uuids else []
    on_gpu = lambda i: {"nvidia.com/use-gpuuuid": uuids[i]} if len(uuids) > i else None  # pin to a physical GPU
    whole = {"nvidia.com/gpu": 1, "nvidia.com/gpumem-percentage": 100}
    items = []
    if args.fake:
        fake = (Path(args.fake_dir) / "fake_vllm_local.py").read_text()
        items.append(configmap("fake-vllm", {"fake_vllm_local.py": fake}))

    if args.scenario == "isolated":
        items += vllm("vllm-large", "large", whole, on_gpu(0), fake=args.fake)
        items += vllm("vllm-small", "small", whole, on_gpu(1), fake=args.fake)
    elif args.scenario in ("shared", "shared_cores"):
        large = {"nvidia.com/gpu": 1, "nvidia.com/gpumem": 20000}
        small = {"nvidia.com/gpu": 1, "nvidia.com/gpumem": 9000}
        if args.scenario == "shared_cores":
            large["nvidia.com/gpucores"], small["nvidia.com/gpucores"] = 70, 30
        items += vllm("vllm-large", "large", large, on_gpu(0), fake=args.fake)
        items += vllm("vllm-small", "small", small, on_gpu(0), fake=args.fake)
    elif args.scenario == "autoscale":
        items += vllm("vllm-large", "large", whole, None, fake=args.fake)
        items.append({
            "apiVersion": "keda.sh/v1alpha1", "kind": "ScaledObject", "metadata": meta("vllm-large-scaler"),
            "spec": {
                "scaleTargetRef": {"name": "vllm-large"},
                "minReplicaCount": 1, "maxReplicaCount": args.max_replicas,
                "pollingInterval": 10, "cooldownPeriod": 60,
                "advanced": {"horizontalPodAutoscalerConfig": {"behavior": {
                    "scaleUp": {"stabilizationWindowSeconds": 0},
                    "scaleDown": {"stabilizationWindowSeconds": 90}}}},
                # Requests in vLLM (running + waiting) per pod. One FP8 pod meets the relaxed
                # SLO up to ~256 users (Phase 5), so add a pod above ~180 each.
                "triggers": [{"type": "prometheus", "metadata": {
                    "serverAddress": f"http://prometheus.{NS}.svc.cluster.local:9090",
                    "query": 'sum(vllm:num_requests_running{model_name="llm"}) + sum(vllm:num_requests_waiting{model_name="llm"})',
                    "threshold": str(args.scale_threshold)}}],
            }})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", choices=["base", "isolated", "shared", "shared_cores", "autoscale"])
    parser.add_argument("--gpu-uuids", help="comma separated, from nvidia-smi -L")
    parser.add_argument("--node-ip", default="127.0.0.1")
    parser.add_argument("--grafana-password", default="admin")
    parser.add_argument("--max-replicas", type=int, default=2)
    parser.add_argument("--scale-threshold", type=int, default=180)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--fake-dir", default="")
    args = parser.parse_args()
    items = base(args) if args.scenario == "base" else scenario(args)
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=1))


if __name__ == "__main__":
    main()
