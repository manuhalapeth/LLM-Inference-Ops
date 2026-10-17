# Charts for the NGINX reproduction, drawn from the raw files in results/.
import json, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "charts")
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CURRENT, PROPOSED = "#eb6834", "#2a78d6"
LABELS = {"current": "Current docs config", "proposed": "Proposed config"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def ttfb(name):
    return [json.loads(l) for l in (RES / f"ttfb_{name}.txt").read_text().splitlines() if l.startswith("{")]


def spread(name):
    rows = []
    for line in (RES / name).read_text().splitlines():
        m = re.match(r"t=\s*(\d+)s (.*)", line)
        if m:
            counts = dict(kv.split("=") for kv in m.group(2).split())
            rows.append((int(m.group(1)), {k: int(v) for k, v in counts.items()}))
    return rows


def finish(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK2)


# 1. Time to first streamed token
cur, pro = ttfb("docs"), ttfb("proposed")
fig, ax = plt.subplots(figsize=(8, 4.2))
groups = [f"{r['tokens']} token answer\n(full answer {r['full_response_ms_p50'] / 1000:.1f} s)" for r in cur]
y = range(len(groups))
h = 0.36
for i, (rows, color, key) in enumerate([(cur, CURRENT, "current"), (pro, PROPOSED, "proposed")]):
    vals = [r["first_token_ms_p50"] for r in rows]
    pos = [v + (h / 2 + 0.02) * (-1 if i == 0 else 1) for v in y]
    ax.barh(pos, vals, height=h, color=color, label=LABELS[key], edgecolor=SURFACE, linewidth=2)
    for p, v in zip(pos, vals):
        ax.text(v + max(r["first_token_ms_p50"] for r in cur) * 0.01, p, f"{v:,} ms", va="center", fontsize=10, color=INK)
ax.set_yticks(list(y), groups)
ax.invert_yaxis()
ax.set_xlabel("Time to first streamed token, median (ms)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(r["first_token_ms_p50"] for r in cur) * 1.18)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
finish(ax, "Streaming: NGINX buffering delays the first token",
       "vLLM-shaped SSE chunks, one token every 20 ms, through NGINX 1.27")
fig.tight_layout()
fig.savefig(OUT / "ttfb.png", dpi=160)

# 2 and 3. Outage timelines
for kind, title, down, metric, ylabel in [
    ("one_down", "One server down for 10 s: does it get traffic again?", (10, 20),
     lambda c: c.get("vllm1", 0), "Requests answered by vllm1 per 5 s"),
    ("both_down", "Both servers down for 5 s: does the service recover?", (10, 15),
     lambda c: 100 * (c.get("vllm0", 0) + c.get("vllm1", 0)) / max(1, sum(c.values())), "Requests succeeding (%)"),
]:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axvspan(*down, color="#f0efec", zorder=0)
    ax.text(sum(down) / 2, 0.5, "vllm1 down" if kind == "one_down" else "both down", transform=ax.get_xaxis_transform(),
            ha="center", va="center", rotation=90, fontsize=9, color=INK2)
    for name, color, key in [("proposed", PROPOSED, "proposed"), ("docs", CURRENT, "current")]:
        rows = spread(f"{kind}_{name}.txt")
        xs = [t + 2.5 for t, _ in rows]
        ys = [metric(c) for _, c in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[key])
        ax.text(xs[-1] + 1, ys[-1], LABELS[key].replace(" config", ""), color=INK, fontsize=9, va="center")
    ax.set_xlabel("Seconds into the test (10 requests/s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(xs) + 14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    finish(ax, title, "max_fails=3 with fail_timeout=10000s (current) vs 10s (proposed)")
    fig.tight_layout()
    fig.savefig(OUT / f"{kind}.png", dpi=160)

print("charts written to", OUT)
