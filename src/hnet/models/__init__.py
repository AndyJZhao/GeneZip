from __future__ import annotations

"""HNet model subpackage.

Keep this `__init__` lightweight: importing `src.hnet.models.*` should not
eagerly import CUDA / Triton kernels.
"""

__all__ = [
    "AttnConfig",
    "SSMConfig",
    "HNetConfig",
    "HNetState",
    "HNet",
    "HNetForCausalLM",
]


def __getattr__(name: str):
    if name in {"AttnConfig", "SSMConfig", "HNetConfig"}:
        from .config_hnet import AttnConfig, HNetConfig, SSMConfig

        return {"AttnConfig": AttnConfig, "SSMConfig": SSMConfig, "HNetConfig": HNetConfig}[name]

    if name in {"HNet", "HNetState"}:
        from .hnet import HNet, HNetState

        return {"HNet": HNet, "HNetState": HNetState}[name]

    if name == "HNetForCausalLM":
        from .mixer_seq import HNetForCausalLM

        return HNetForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

