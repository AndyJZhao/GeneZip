from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
from torch import nn


def _import_flash_rmsnorm() -> Any | None:
    for module_path in (
        "flash_attn.ops.triton.layer_norm",
        "flash_attn.ops.rms_norm",
        "flash_attn.ops.layer_norm",
    ):
        try:
            module = import_module(module_path)
        except Exception:
            continue
        rmsnorm = getattr(module, "RMSNorm", None)
        if rmsnorm is not None:
            return rmsnorm
    return None


_FlashRMSNorm = _import_flash_rmsnorm()


if _FlashRMSNorm is not None:
    RMSNorm = _FlashRMSNorm
else:

    class RMSNorm(nn.Module):
        """RMSNorm compatible with flash-attn's residual+prenorm interface.

        This is a lightweight fallback when flash-attn isn't available or
        doesn't ship the expected module paths.
        """

        def __init__(
            self,
            dim: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
            self.eps = float(eps)

        def forward(  # type: ignore[override]
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
            *,
            prenorm: bool = True,
            residual_in_fp32: bool = False,
        ):
            if residual is None:
                residual_out = x.float() if residual_in_fp32 else x
            elif residual_in_fp32:
                residual_out = residual.float() + x.float()
            else:
                residual_out = residual + x

            norm_input = residual_out.float() if residual_out.dtype != torch.float32 else residual_out
            inv_rms = torch.rsqrt(norm_input.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            y = norm_input * inv_rms
            y = y * self.weight.float()
            y = y.to(dtype=x.dtype)

            if prenorm:
                return y, residual_out
            return y

