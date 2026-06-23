"""
Breakdown chart generation.

Produces a single PNG containing:
  (a) Horizontal bar -- percentage of wall-clock time per bucket (more
      readable than a pie for 9 buckets).
  (b) Stacked bar -- absolute time per bucket (single bar, since one run).
  (c) Top-N events per bucket -- which kernels dominate each bucket.

The chart is deliberately engineered to render correctly in any environment
(no LaTeX, no external fonts), so the artifact can be opened on a machine
without matplotlib installed.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .categorize import BUCKETS, BUCKET_STYLE, Breakdown


def _top_events(events, k: int = 5):
    """Return top-k events per bucket by total duration."""
    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ev in events:
        by_bucket[ev.bucket][ev.name] += ev.duration_us
    out = {}
    for bucket, names in by_bucket.items():
        ranked = sorted(names.items(), key=lambda kv: -kv[1])[:k]
        out[bucket] = ranked
    return out


def render(breakdown: Breakdown, title: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.45, wspace=0.20)

    # ---------- Panel (a): horizontal bar of percentages ---------------- #
    ax_pct = fig.add_subplot(gs[0, 0])
    labels = [BUCKET_STYLE[b]["label"] for b in BUCKETS]
    pcts = [breakdown.percent(b) for b in BUCKETS]
    colors = [BUCKET_STYLE[b]["color"] for b in BUCKETS]
    counts = [breakdown.per_bucket_count.get(b, 0) for b in BUCKETS]
    # Sort by descending percentage so the eye reads top-down.
    order = sorted(range(len(BUCKETS)), key=lambda i: -pcts[i])
    labels_s = [labels[i] for i in order]
    pcts_s = [pcts[i] for i in order]
    colors_s = [colors[i] for i in order]
    counts_s = [counts[i] for i in order]
    y = np.arange(len(labels_s))
    bars = ax_pct.barh(y, pcts_s, color=colors_s, edgecolor="white", linewidth=1.2)
    ax_pct.set_yticks(y)
    ax_pct.set_yticklabels(labels_s, fontsize=9)
    ax_pct.invert_yaxis()
    ax_pct.set_xlabel("% of wall-clock time")
    ax_pct.set_title("(a) Time-share by bucket", fontsize=12, pad=10)
    ax_pct.set_xlim(0, max(pcts_s + [1]) * 1.20)
    for i, (pct, cnt) in enumerate(zip(pcts_s, counts_s)):
        ax_pct.text(pct + 0.6, i, f"{pct:.1f}%  ({cnt} ev)",
                    va="center", fontsize=9)

    # ---------- Panel (b): absolute stacked bar ------------------------- #
    ax_bar = fig.add_subplot(gs[0, 1])
    values_us = [breakdown.per_bucket_us.get(b, 0.0) for b in BUCKETS]
    bottom = 0.0
    total_ms = sum(values_us) / 1000.0
    for b, val, cnt in zip(BUCKETS, values_us, counts):
        if val <= 0:
            continue
        label = BUCKET_STYLE[b]["label"]
        color = BUCKET_STYLE[b]["color"]
        ax_bar.bar(
            [""], [val / 1000.0], bottom=bottom / 1000.0,
            label=f"{label}  ({cnt} ev)",
            color=color, edgecolor="white", linewidth=1.2,
        )
        # in-bar label for big buckets
        frac = val / sum(values_us) if sum(values_us) else 0
        if frac > 0.10:
            ax_bar.text(
                0, bottom / 1000.0 + val / 2000.0,
                f"{val/1000.0:.1f} ms\n{cnt} ev",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )
        bottom += val
    ax_bar.set_ylabel("Time (ms)")
    ax_bar.set_title(f"(b) Absolute time per bucket  (total {total_ms:.1f} ms)",
                     fontsize=12, pad=10)
    ax_bar.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_bar.set_ylim(0, max(bottom, 1) / 1000.0 * 1.15)

    # ---------- Panel (c): top events per bucket ------------------------ #
    ax_top = fig.add_subplot(gs[1, :])
    top = _top_events(breakdown.events, k=4)
    rows = []
    for bucket in BUCKETS:
        for name, dur in top.get(bucket, []):
            rows.append((BUCKET_STYLE[bucket]["label"], name, dur / 1000.0,
                         BUCKET_STYLE[bucket]["color"]))
    if not rows:
        ax_top.text(0.5, 0.5, "no events", ha="center", va="center")
    else:
        labels = [f"{lbl}  —  {name}" for lbl, name, _, _ in rows]
        values = [v for _, _, v, _ in rows]
        colors = [c for _, _, _, c in rows]
        y = np.arange(len(rows))
        ax_top.barh(y, values, color=colors, edgecolor="white")
        ax_top.set_yticks(y)
        ax_top.set_yticklabels(labels, fontsize=8)
        ax_top.invert_yaxis()
        ax_top.set_xlabel("Time (ms)")
        ax_top.set_title("(c) Top events per bucket", fontsize=12, pad=10)
        for i, v in enumerate(values):
            ax_top.text(v, i, f" {v:.2f} ms", va="center", fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
