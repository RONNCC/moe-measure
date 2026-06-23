"""
Topology visualizations for expert placement analysis.

Two charts:
  * `render_transfer_matrix(matrix)` -- N x N heatmap of transfer times
  * `render_topology(matrix, placement, cost)` -- experts laid out on
    a grid of GPUs (one column per rack, one row per GPU slot) with
    edges drawn between communicating experts.  Intra-rack edges are
    green, inter-rack edges are red.
  * `render_placement_comparison(...)` -- side-by-side bars of total
    network time for each placement strategy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .topology import TransferMatrix, Placement, PlacementCost, PLACEMENT_STRATEGIES, evaluate


def render_transfer_matrix(matrix: TransferMatrix, out_path_base: str | Path,
                           title: str = "Expert-to-expert transfers") -> list[Path]:
    """Render BOTH time and data-volume heatmaps for the transfer matrix.

    Produces two PNGs:
      * `<base>_time.png`   -- log(us) per pair, total ms in title
      * `<base>_bytes.png`  -- log(MB) per pair, total GBs in title

    Returns both paths so the caller can list them.
    """
    out_base = Path(out_path_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # --- Time heatmap -----------------------------------------------------
    Mt = matrix.matrix_us
    log_Mt = np.log1p(Mt)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(log_Mt, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("destination expert")
    ax.set_ylabel("source expert")
    ax.set_title(
        f"{title} -- TIME\n"
        f"{matrix.num_experts} experts, total {matrix.total_us/1e3:.2f} ms "
        f"({matrix.num_experts * (matrix.num_experts - 1)} pairs)",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="log(1 + transfer time, us)")
    fig.tight_layout()
    path_time = out_base.with_name(out_base.stem + "_time.png")
    fig.savefig(path_time, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Bytes heatmap ----------------------------------------------------
    Mb = matrix.matrix_bytes.astype(np.float64)
    log_Mb = np.log1p(Mb)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(log_Mb, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("destination expert")
    ax.set_ylabel("source expert")
    ax.set_title(
        f"{title} -- DATA VOLUME\n"
        f"{matrix.num_experts} experts, total {matrix.total_gbs:.2f} GB "
        f"({matrix.num_experts * (matrix.num_experts - 1)} pairs)",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="log(1 + bytes transferred)")
    fig.tight_layout()
    path_bytes = out_base.with_name(out_base.stem + "_bytes.png")
    fig.savefig(path_bytes, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return [path_time, path_bytes]


def render_topology(matrix: TransferMatrix, placement: Placement, cost: PlacementCost,
                    out_path: str | Path,
                    title: str = "Expert placement topology",
                    top_edges: int = 200) -> Path:
    """Draw experts as nodes on a 2-D grid (rack x gpu_slot), with edges
    between high-traffic pairs.  Intra-rack edges in green, inter-rack
    in red, edge width proportional to transfer time."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = matrix.num_experts
    num_racks = placement.num_racks
    gpus_per_rack = placement.gpus_per_rack

    # Compute node positions: (rack, gpu_slot_in_rack)
    pos = {}
    for i in range(N):
        pos[i] = (placement.expert_to_rack[i],
                  placement.expert_to_gpu[i] % gpus_per_rack)

    # Collect the top edges by traffic for clarity.
    edges = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            t = matrix.matrix_us[i, j] + matrix.matrix_us[j, i]
            if t > 0:
                edges.append((t, i, j))
    edges.sort(reverse=True)
    edges = edges[:top_edges]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#fafafa")

    # Draw rack backgrounds.
    for r in range(num_racks):
        ax.add_patch(plt.Rectangle(
            (r - 0.5, -0.5), 1.0, gpus_per_rack,
            facecolor="#e8f0fe", edgecolor="#5B8FF9", linewidth=2,
            alpha=0.4, zorder=0,
        ))
        ax.text(r, gpus_per_rack + 0.3, f"Rack {r}", ha="center", fontsize=10,
                fontweight="bold")

    # Draw nodes.
    for i in range(N):
        r, s = pos[i]
        ax.scatter([r], [s], s=140, c="#F6BD16", edgecolor="black",
                   linewidth=0.8, zorder=3)
        ax.text(r, s, str(i), ha="center", va="center", fontsize=6,
                zorder=4, color="black")

    # Draw edges.
    max_t = max((t for t, _, _ in edges), default=1.0)
    for t, i, j in edges:
        ri, si = pos[i]
        rj, sj = pos[j]
        intra = (ri == rj)
        color = "#6DC354" if intra else "#E8684A"
        lw = max(0.3, 3.5 * (t / max_t))
        ax.plot([ri, rj], [si, sj], color=color, linewidth=lw, alpha=0.5, zorder=1)

    ax.set_xlim(-0.6, num_racks - 0.4)
    ax.set_ylim(-0.6, gpus_per_rack + 0.6)
    ax.invert_yaxis()
    ax.set_xticks(range(num_racks))
    ax.set_xticklabels([f"rack {r}" for r in range(num_racks)])
    ax.set_yticks(range(gpus_per_rack))
    ax.set_yticklabels([f"GPU {s}" for s in range(gpus_per_rack)])
    ax.set_title(f"{title}\n"
                 f"intra-rack: {cost.intra_rack_us/1e3:.2f} ms ({cost.as_dict()['intra_rack_pct']:.1f}%)  |  "
                 f"inter-rack: {cost.inter_rack_us/1e3:.2f} ms ({cost.as_dict()['inter_rack_pct']:.1f}%)  |  "
                 f"total: {cost.total_us/1e3:.2f} ms",
                 fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_placement_comparison(matrix: TransferMatrix,
                                placements: dict[str, Placement],
                                out_path: str | Path,
                                title: str = "Placement strategies: total network time") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, p in placements.items():
        cost = evaluate(matrix, p)
        rows.append((name, cost.intra_rack_us, cost.inter_rack_us, cost.total_us))
    rows.sort(key=lambda r: r[3])  # sort by total

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r[0] for r in rows]
    intra = np.array([r[1] for r in rows]) / 1e3
    inter = np.array([r[2] for r in rows]) / 1e3
    y = np.arange(len(names))
    ax.barh(y, intra, color="#6DC354", label="intra-rack")
    ax.barh(y, inter, left=intra, color="#E8684A", label="inter-rack")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Network time (ms)")
    ax.set_title(f"{title}\n(matrix total: {matrix.total_us/1e3:.2f} ms)",
                 fontsize=11)
    for i, (n, _, _, total) in enumerate(rows):
        ax.text(total / 1e3 + 0.05, i, f"{total/1e3:.2f} ms total", va="center", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
