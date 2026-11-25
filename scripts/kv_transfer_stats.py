"""How much KV moved between the prefill and decode servers, and how fast.

Snapshots the KV connector metrics (Mooncake or NIXL, vLLM v0.34.0 names) on
every server before a run and after it, and prints what happened in between:
transfers, bytes, average and 95th percentile transfer time, GiB/s, failures.

    python3 scripts/kv_transfer_stats.py snapshot before.json
    ... run the load ...
    python3 scripts/kv_transfer_stats.py snapshot after.json
    python3 scripts/kv_transfer_stats.py diff before.json after.json > kv_transfer.json

Servers default to the two published ports of a 1 prefill + 1 decode setup.
Standard library only.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")
_LE = re.compile(r'le="([^"]+)"')
PREFIXES = ("vllm:mooncake_", "vllm:nixl_")


def scrape(url: str) -> dict:
    """Sum every KV connector series over its labels, keeping histogram buckets by le."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m or line.startswith("#") or not m.group(1).startswith(PREFIXES):
            continue
        name, labels, value = m.groups()
        if name.endswith("_bucket") and labels and (le := _LE.search(labels)):
            name = f"{name}[{le.group(1)}]"
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out


def quantile(buckets: dict[float, float], q: float) -> float | None:
    """Prometheus style histogram_quantile over cumulative bucket counts."""
    if not buckets:
        return None
    edges = sorted(buckets)
    total = buckets[edges[-1]]
    if total <= 0:
        return None
    target, prev_edge, prev_count = q * total, 0.0, 0.0
    for edge in edges:
        if buckets[edge] >= target:
            if edge == float("inf"):
                return prev_edge
            span = buckets[edge] - prev_count
            return prev_edge + (edge - prev_edge) * ((target - prev_count) / span if span else 0)
        prev_edge, prev_count = edge, buckets[edge]
    return prev_edge


def diff(before: dict, after: dict) -> dict:
    result = {}
    for server in after:
        b, a = before.get(server, {}), after[server]
        d = {k: a[k] - b.get(k, 0.0) for k in a}
        for prefix in ("vllm:mooncake_", "vllm:nixl_"):
            count = d.get(f"{prefix}xfer_time_seconds_count", 0.0)
            if not count:
                continue
            seconds = d.get(f"{prefix}xfer_time_seconds_sum", 0.0)
            nbytes = d.get(f"{prefix}bytes_transferred_sum", 0.0)
            buckets = {float(k.split("[")[1].rstrip("]")): v for k, v in d.items()
                       if k.startswith(f"{prefix}xfer_time_seconds_bucket[")}
            result[server] = {
                "connector": prefix.split(":")[1].rstrip("_"),
                "transfers": count,
                "gib": nbytes / 2**30,
                "mib_per_transfer": nbytes / count / 2**20,
                "xfer_ms_mean": seconds / count * 1000,
                "xfer_ms_p95": (q * 1000) if (q := quantile(buckets, 0.95)) is not None else None,
                "gib_per_s": nbytes / seconds / 2**30 if seconds else None,
                "failed_transfers": d.get(f"{prefix}num_failed_transfers_total", d.get(f"{prefix}num_failed_transfers", 0.0)),
                "expired_requests": d.get(f"{prefix}num_kv_expired_reqs_total", d.get(f"{prefix}num_kv_expired_reqs", 0.0)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("out")
    snap.add_argument("--servers", nargs="+", default=["prefill=http://localhost:8000", "decode=http://localhost:8001"])
    dif = sub.add_parser("diff")
    dif.add_argument("before")
    dif.add_argument("after")
    args = parser.parse_args()

    if args.cmd == "snapshot":
        data = {}
        for spec in args.servers:
            name, url = spec.split("=", 1)
            try:
                data[name] = scrape(f"{url}/metrics")
            except Exception as exc:  # noqa: BLE001 (a crashed server still gets a row)
                print(f"{name}: {exc}", file=sys.stderr)
                data[name] = {}
        Path(args.out).write_text(json.dumps(data))
    else:
        print(json.dumps(diff(json.loads(Path(args.before).read_text()),
                              json.loads(Path(args.after).read_text())), indent=2))


if __name__ == "__main__":
    main()
