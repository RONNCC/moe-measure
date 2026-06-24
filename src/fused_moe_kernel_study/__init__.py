"""MoE kernel characterization benchmarks.

These benchmarks measure raw kernel latency, NOT end-to-end model
forward time. The point is to characterize how the fused MoE kernel
responds to individual knobs, holding everything else constant.

Use cases:

  * Find the latency vs. (batch, imbalance, TP) operating surface.

  * Decide where the sweet spot is for your hardware before deploying.

  * Compare kernels (custom Triton vs. torch._grouped_mm vs. naive).
"""

from .moe_kernel import (
    KernelMeasurement,
    MoEKernelConfig,
    measure_moe_kernel_latency,
    run_sweep,
)

__all__ = [
    "MoEKernelConfig",
    "KernelMeasurement",
    "measure_moe_kernel_latency",
    "run_sweep",
]
