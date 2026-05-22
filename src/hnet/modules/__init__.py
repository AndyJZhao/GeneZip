from __future__ import annotations

"""HNet modules subpackage.

This package contains optional CUDA/Triton-backed components (e.g. FlashAttention).
Avoid importing them eagerly from `__init__` so that lightweight utilities like
`src.hnet.modules.dc` remain importable without those deps.
"""

__all__ = [
    "Block",
    "CausalMHA",
    "SwiGLU",
    "get_seq_idx",
]


def __getattr__(name: str):
    if name == "Block":
        from .block import Block

        return Block
    if name == "CausalMHA":
        from .mha import CausalMHA

        return CausalMHA
    if name == "SwiGLU":
        from .mlp import SwiGLU

        return SwiGLU
    if name == "get_seq_idx":
        from .utils import get_seq_idx

        return get_seq_idx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

