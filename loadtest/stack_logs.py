"""Pull per-request timings out of the gateway and NGINX logs.

The gateway writes one JSON line per request, and NGINX writes one access log
line per request. Both carry the request ID (and the agent run ID, if any),
so one request can be followed through both layers.

Runs on the GPU box from the repo root (needs `docker compose`).
Standard library only.
"""

import json
import re
import subprocess
from datetime import datetime

_NGINX_FIELD = re.compile(r"(\w+)=(\S+)")


def _compose_logs(service: str, since: datetime) -> list[str]:
    out = subprocess.run(
        ["docker", "compose", "logs", "--no-color", "--no-log-prefix",
         "--since", since.strftime("%Y-%m-%dT%H:%M:%SZ"), service],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.splitlines()


def _seconds(value: str) -> float | None:
    # NGINX logs "-" when a value doesn't apply, and "a, b" if it retried upstreams.
    if value in ("-", ""):
        return None
    return float(value.split(",")[-1].strip())


def gateway_requests(since: datetime) -> list[dict]:
    """Gateway log lines for requests that finished after `since` (UTC)."""
    rows = []
    for line in _compose_logs("gateway", since):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") in ("request", "upstream_error"):
            rows.append(event)
    return rows


def nginx_requests(since: datetime) -> list[dict]:
    """NGINX access log lines after `since` (UTC), with times in seconds."""
    rows = []
    for line in _compose_logs("nginx", since):
        fields = dict(_NGINX_FIELD.findall(line))
        if "rid" not in fields:
            continue
        rows.append({
            "request_id": fields["rid"],
            "run_id": None if fields.get("run") == "-" else fields.get("run"),
            "nginx_total_s": _seconds(fields.get("rt", "-")),
            "upstream_connect_s": _seconds(fields.get("uct", "-")),
            "upstream_header_s": _seconds(fields.get("uht", "-")),
            "upstream_total_s": _seconds(fields.get("urt", "-")),
        })
    return rows


def requests_by_id(since: datetime) -> dict[str, dict]:
    """Gateway and NGINX timings merged per request ID."""
    merged: dict[str, dict] = {}
    for row in gateway_requests(since):
        merged.setdefault(row["request_id"], {}).update({
            "request_id": row["request_id"],
            "run_id": row.get("run_id"),
            "path": row.get("path"),
            "status": row.get("status"),
            "gateway_ttfb_s": row["ttfb_ms"] / 1000 if row.get("ttfb_ms") is not None else None,
            "gateway_total_s": row["total_ms"] / 1000 if row.get("total_ms") is not None else None,
        })
    for row in nginx_requests(since):
        merged.setdefault(row["request_id"], {"request_id": row["request_id"]}).update(
            {k: v for k, v in row.items() if k != "run_id" or v}
        )
    return merged
