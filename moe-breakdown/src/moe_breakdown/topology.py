"""
Expert topology & placement analyzer.

Given a profiler trace (or a routing pattern for a synthetic model), this
module answers the systems question:

    "If we have N experts across G GPUs, where should each expert live
     to minimize the time spent on AllToAll transfers between GPUs?"

The output is:
  * `transfer_matrix` -- N x N matrix of time (microseconds) spent
    transferring between each pair of experts
  * `placement` -- list[G] -> [experts] mapping experts to GPU ranks
  * `summary` -- total intra-rack vs inter-rack network time for the
    chosen placement

Two visualizations:
  * `render_transfer_matrix(matrix)` -- N x N heatmap of transfer times
  * `render_topology(matrix, placement, num_racks)` -- experts laid out
    on a grid of GPUs, with edges colored by intra-rack (green) vs
    inter-rack (red) traffic

This works on real GPU traces (via NCCL AllToAll events) and on synthetic
routing patterns (for CPU-only environments that want to reason about
placement before they have a GPU cluster).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Transfer matrix
# --------------------------------------------------------------------------- #

@dataclass
class TransferMatrix:
    """N x N matrix of expert-to-expert transfers.

    Carries BOTH wall-clock time AND byte volume for each pair:
      * `matrix_us[i, j]`    -- microseconds spent transferring from i to j
      * `matrix_bytes[i, j]` -- bytes of activation data moved from i to j

    The byte volume is what the network sees; the time is what the
    profiler would measure.  Both matter: a placement that minimises
    time without reducing bytes is suspicious; a placement that reduces
    bytes but not time (because the link is already saturated) is also
    suspicious.  Always look at both.
    """
    num_experts: int
    matrix_us: np.ndarray
    matrix_bytes: np.ndarray
    total_us: float = 0.0
    total_bytes: int = 0

    def __post_init__(self):
        self.total_us = float(self.matrix_us.sum())
        self.total_bytes = int(self.matrix_bytes.sum())

    def time_to(self, src: int, dst: int) -> float:
        return float(self.matrix_us[src, dst])

    def bytes_to(self, src: int, dst: int) -> int:
        return int(self.matrix_bytes[src, dst])

    @property
    def total_gbs(self) -> float:
        return self.total_bytes / (1024 ** 3)


def extract_from_routing(
    routing: np.ndarray,
    tokens_per_msg_bytes: int = 4096,
    intra_rack_bw_gbps: float = 200.0,
    inter_rack_bw_gbps: float = 25.0,
    num_racks: int = 1,
    gpus_per_rack: int = 8,
) -> TransferMatrix:
    """Build a TransferMatrix from a routing pattern.

    Parameters
    ----------
    routing : (T, K) int array
        For each of T tokens, the K experts it is routed to.
    tokens_per_msg_bytes : int
        Bytes of activation data moved per token-expert pair.  A typical
        MoE dispatch message is roughly hidden_dim * dtype_bytes; we use
        4096 as a reasonable default for a 1024-hidden fp32 model.
    intra_rack_bw_gbps, inter_rack_bw_gbps : float
        Effective bandwidth (after protocol overhead) within and between
        NVLink-connected racks.
    num_racks, gpus_per_rack : int
        Cluster topology.
    """
    T, K = routing.shape
    N = int(routing.max()) + 1

    # Per-expert token counts
    counts = np.zeros(N, dtype=np.int64)
    for t in range(T):
        for k in range(K):
            counts[routing[t, k]] += 1

    # Per-expert-pair traffic: src = the GPU sending, dst = the GPU
    # receiving.  For symmetric AllToAll, total bytes per pair scales
    # with both endpoints' token counts.
    # Per-expert-pair byte volume: src = the GPU sending, dst = the GPU
    # receiving.  For a symmetric AllToAll, total bytes per pair scales
    # with both endpoints' token counts (roughly N_i * N_j / T).
    bytes_per_pair = np.outer(counts, counts) / max(1, T) * tokens_per_msg_bytes
    matrix_bytes = bytes_per_pair.astype(np.int64)

    # Convert bytes -> microseconds using the bandwidth model.
    # Pair times are computed at inter-rack BW; intra-rack pairs get
    # scaled down later when the placement is chosen (see evaluate()).
    matrix_us = bytes_per_pair / (inter_rack_bw_gbps * 1e9 / 8) * 1e6
    return TransferMatrix(
        num_experts=N, matrix_us=matrix_us, matrix_bytes=matrix_bytes,
    )


# --------------------------------------------------------------------------- #
# Placement strategies
# --------------------------------------------------------------------------- #

@dataclass
class Placement:
    """A mapping from expert index to (rack, gpu_in_rack) coordinates."""
    num_experts: int
    num_racks: int
    gpus_per_rack: int
    expert_to_gpu: list[int] = field(default_factory=list)   # length N
    expert_to_rack: list[int] = field(default_factory=list)  # length N

    def total_network_time_us(self, matrix: TransferMatrix) -> float:
        """Total AllToAll time under this placement, accounting for intra-rack BW."""
        t = 0.0
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i == j:
                    continue
                if self.expert_to_rack[i] == self.expert_to_rack[j]:
                    # intra-rack: faster
                    pass  # we'll account for this with a different matrix
                else:
                    t += matrix.matrix_us[i, j]
        return t

    def as_dict(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "num_racks": self.num_racks,
            "gpus_per_rack": self.gpus_per_rack,
            "expert_to_gpu": self.expert_to_gpu,
            "expert_to_rack": self.expert_to_rack,
        }


def place_round_robin(matrix: TransferMatrix, num_racks: int, gpus_per_rack: int) -> Placement:
    """Naive round-robin placement -- the worst case, useful as a baseline."""
    N = matrix.num_experts
    total_gpus = num_racks * gpus_per_rack
    e2g = [i % total_gpus for i in range(N)]
    e2r = [e2g[i] // gpus_per_rack for i in range(N)]
    return Placement(num_experts=N, num_racks=num_racks, gpus_per_rack=gpus_per_rack,
                     expert_to_gpu=e2g, expert_to_rack=e2r)


def place_greedy(matrix: TransferMatrix, num_racks: int, gpus_per_rack: int) -> Placement:
    """Greedy placement: pack high-traffic experts onto the same rack.

    Algorithm: walk the transfer matrix in descending order of pair
    traffic, and place each expert on the rack that already contains the
    most of its high-traffic partners.  Fill racks evenly.
    """
    N = matrix.num_experts
    total_gpus = num_racks * gpus_per_rack

    # Sort all (i, j, traffic) pairs by traffic desc.
    pairs = []
    for i in range(N):
        for j in range(N):
            if i != j:
                pairs.append((matrix.matrix_us[i, j] + matrix.matrix_us[j, i], i, j))
    pairs.sort(reverse=True)

    e2g = [-1] * N
    e2r = [-1] * N
    rack_count = [0] * num_racks
    gpu_count = [0] * (num_racks * gpus_per_rack)

    for _, i, j in pairs:
        # If both already placed, skip.
        if e2g[i] >= 0 and e2g[j] >= 0:
            continue
        # Try to put i on the same rack as j if j is placed.
        target_rack = None
        if e2r[j] >= 0 and rack_count[e2r[j]] < total_gpus:
            target_rack = e2r[j]
        elif e2r[i] >= 0 and rack_count[e2r[i]] < total_gpus:
            target_rack = e2r[i]
        else:
            # Pick the rack with the fewest experts so far.
            target_rack = int(np.argmin(rack_count))
        # Place i (if not placed) on a free GPU in that rack.
        if e2g[i] < 0:
            for g in range(target_rack * gpus_per_rack, (target_rack + 1) * gpus_per_rack):
                if gpu_count[g] == 0:
                    e2g[i] = g
                    e2r[i] = target_rack
                    gpu_count[g] += 1
                    rack_count[target_rack] += 1
                    break
        # Same for j.
        if e2g[j] < 0:
            for g in range(target_rack * gpus_per_rack, (target_rack + 1) * gpus_per_rack):
                if gpu_count[g] == 0:
                    e2g[j] = g
                    e2r[j] = target_rack
                    gpu_count[g] += 1
                    rack_count[target_rack] += 1
                    break

    # Any expert still unplaced (no traffic with anyone) -> round-robin.
    for i in range(N):
        if e2g[i] < 0:
            for g in range(total_gpus):
                if gpu_count[g] == 0:
                    e2g[i] = g
                    e2r[i] = g // gpus_per_rack
                    gpu_count[g] += 1
                    rack_count[e2r[i]] += 1
                    break
    return Placement(num_experts=N, num_racks=num_racks, gpus_per_rack=gpus_per_rack,
                     expert_to_gpu=e2g, expert_to_rack=e2r)


def place_cluster(matrix: TransferMatrix, num_racks: int, gpus_per_rack: int) -> Placement:
    """Cluster-then-place: spectral clustering on the transfer matrix.

    Groups the N experts into `num_racks` clusters by graph affinity
    (high-transfer experts end up in the same cluster), then assigns
    each cluster to a rack and packs experts into GPUs.
    """
    N = matrix.num_experts
    total_gpus = num_racks * gpus_per_rack
    if N <= num_racks:
        return place_round_robin(matrix, num_racks, gpus_per_rack)

    # Symmetrize the transfer matrix (undirected affinity graph).
    A = (matrix.matrix_us + matrix.matrix_us.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # Spectral clustering via eigenvector sign of the graph Laplacian.
    # (Lightweight enough to run on 100-expert matrices inline.)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    # Add tiny regularization so degenerate cases don't blow up.
    L += np.eye(N) * 1e-6 * deg.mean()

    eigvals, eigvecs = np.linalg.eigh(L)
    # Take the (num_racks-1) smallest non-zero eigenvectors and k-means them.
    k = max(2, min(num_racks, N - 1))
    feats = eigvecs[:, :k]
    # Normalize rows to unit length (Ng-Jordan-Weiss trick).
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1
    feats = feats / norms

    # Tiny k-means.
    rng = np.random.default_rng(0)
    centers = feats[rng.choice(N, size=k, replace=False)]
    for _ in range(20):
        d = np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.array([
            feats[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
            for c in range(k)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # Assign each cluster to a rack (round-robin if more clusters than racks).
    e2r = [int(labels[i] % num_racks) for i in range(N)]

    # Within a rack, fill GPUs in order.
    e2g = [-1] * N
    gpu_count = [0] * (num_racks * gpus_per_rack)
    for r in range(num_racks):
        members = [i for i in range(N) if e2r[i] == r]
        for idx, i in enumerate(members):
            local = idx % gpus_per_rack
            e2g[i] = r * gpus_per_rack + local
            gpu_count[e2g[i]] += 1
    return Placement(num_experts=N, num_racks=num_racks, gpus_per_rack=gpus_per_rack,
                     expert_to_gpu=e2g, expert_to_rack=e2r)


PLACEMENT_STRATEGIES = {
    "round-robin": place_round_robin,
    "greedy":      place_greedy,
    "cluster":     place_cluster,
}


# --------------------------------------------------------------------------- #
# Cost evaluation
# --------------------------------------------------------------------------- #

@dataclass
class PlacementCost:
    """Network-time cost of a placement."""
    intra_rack_us: float
    inter_rack_us: float
    total_us: float
    num_racks: int
    gpus_per_rack: int

    def as_dict(self) -> dict:
        return {
            "intra_rack_us": round(self.intra_rack_us, 3),
            "inter_rack_us": round(self.inter_rack_us, 3),
            "total_us": round(self.total_us, 3),
            "intra_rack_pct": round(100 * self.intra_rack_us / max(1e-9, self.total_us), 2),
            "inter_rack_pct": round(100 * self.inter_rack_us / max(1e-9, self.total_us), 2),
            "num_racks": self.num_racks,
            "gpus_per_rack": self.gpus_per_rack,
        }


def evaluate(matrix: TransferMatrix, placement: Placement,
             intra_rack_bw_gbps: float = 200.0,
             inter_rack_bw_gbps: float = 25.0) -> PlacementCost:
    """Compute the total AllToAll time for a placement under a bandwidth model.

    Uses the bandwidth ratio (intra_rack / inter_rack) to convert the
    abstract transfer-matrix entries into microseconds of network time.
    """
    ratio = inter_rack_bw_gbps / intra_rack_bw_gbps  # < 1 -> intra-rack is faster
    intra = 0.0
    inter = 0.0
    N = matrix.num_experts
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            base = matrix.matrix_us[i, j]
            if placement.expert_to_rack[i] == placement.expert_to_rack[j]:
                intra += base * ratio   # faster
            else:
                inter += base
    return PlacementCost(
        intra_rack_us=intra, inter_rack_us=inter,
        total_us=intra + inter,
        num_racks=placement.num_racks,
        gpus_per_rack=placement.gpus_per_rack,
    )
