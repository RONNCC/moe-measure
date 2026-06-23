"""Backend registry.  Each backend is a callable that runs a profiled
inference pass and returns a list of event dicts compatible with
categorize.categorize_dicts().
"""

from __future__ import annotations

import importlib

_BACKENDS: dict[str, str] = {
    "synthetic":     "moe_breakdown.backends.synthetic:profile",
    "tiny":          "moe_breakdown.backends.tiny:profile",
    "transformers":  "moe_breakdown.backends.transformers:profile",
    "vllm":          "moe_breakdown.backends.vllm:profile",
}


def available() -> list[str]:
    return list(_BACKENDS.keys())


def get(name: str):
    """Import a backend by name and return its profile() function."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}.  Available: {list(_BACKENDS)}")
    mod_path, attr = _BACKENDS[name].split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)
