from __future__ import annotations

"""HNet package, modified from https://github.com/goombalab/hnet

Avoid importing optional GPU kernel dependencies at package import time.
Import submodules explicitly (e.g. `src.hnet.models.config_hnet`) or access
exports via `src.hnet.<name>` which are resolved lazily.
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
        from .models.config_hnet import AttnConfig, HNetConfig, SSMConfig

        return {"AttnConfig": AttnConfig, "SSMConfig": SSMConfig, "HNetConfig": HNetConfig}[name]

    if name in {"HNet", "HNetState"}:
        from .models.hnet import HNet, HNetState

        return {"HNet": HNet, "HNetState": HNetState}[name]

    if name == "HNetForCausalLM":
        from .models.mixer_seq import HNetForCausalLM

        return HNetForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

