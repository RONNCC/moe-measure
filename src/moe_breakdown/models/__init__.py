"""Built-in models used by the framework's backends.

These are small enough to run on CPU in a few seconds, so you can smoke-test
the profiler pipeline without a GPU or large checkpoint download.

To profile a real (quantized) MoE model, use the `transformers` or `vllm`
backend with --model <hf-id> instead.
"""

from .tiny_moe import TinyMoE, TinyMoEConfig, build

__all__ = ["TinyMoE", "TinyMoEConfig", "build"]
