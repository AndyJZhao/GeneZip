from __future__ import annotations

import os
from typing import Any, Dict, Optional

import rootutils
import torch
from omegaconf import OmegaConf
from src.genezip.config import parse_routing_ceiling_config, parse_routing_floor_config
from src.genezip.region_aware_hnet import (
    HNetForCausalLMWithRoutingCeiling,
    HNetForCausalLMWithRoutingFloor,
    HNetForCausalLMWithRoutingFloorAndCeiling,
)
from src.hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from src.hnet.models.mixer_seq import HNetForCausalLM
from src.utils import RunConfig, resolve_hydra_cfg_path
from src.utils.tqdm_progress import TqdmProgress, build_tqdm_progress

root = rootutils.find_root(__file__)


def build_progress(mininterval_sec: Optional[float] = None) -> TqdmProgress:
    return build_tqdm_progress(mininterval_sec)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    if dtype_name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def prepare_hnet_config_kwargs(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    valid_fields = set(HNetConfig.__dataclass_fields__.keys())
    cfg = {k: v for k, v in dict(cfg_dict).items() if k in valid_fields}
    if "ssm_cfg" in cfg and not isinstance(cfg["ssm_cfg"], SSMConfig):
        cfg["ssm_cfg"] = SSMConfig(**cfg["ssm_cfg"])
    if "attn_cfg" in cfg and not isinstance(cfg["attn_cfg"], AttnConfig):
        cfg["attn_cfg"] = AttnConfig(**cfg["attn_cfg"])
    cfg.setdefault("ssm_cfg", SSMConfig())
    cfg.setdefault("attn_cfg", AttnConfig())
    return cfg


def load_state_dict(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    def _map_location(storage, location):
        location = str(location)
        if location.startswith("cpu"):
            return storage.cpu()
        if device.type == "cuda":
            index = device.index if device.index is not None else 0
            return storage.cuda(index)
        return storage

    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load .safetensors checkpoints") from exc
        return load_file(checkpoint_path, device=device)
    try:
        from torch.serialization import safe_globals
        from omegaconf import ListConfig

        with safe_globals([ListConfig]):
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        try:
            state = torch.load(checkpoint_path, map_location=device)
        except Exception:
            state = torch.load(checkpoint_path, map_location=_map_location)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def build_model(
    checkpoint_dir: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
    disable_routing: bool = False,
    use_routing_ceiling: bool = False,
    k_max_list: Optional[list[int]] = None,
    use_routing_floor: Optional[bool] = None,
) -> tuple[torch.nn.Module, RunConfig]:
    cfg_path = resolve_hydra_cfg_path(checkpoint_dir)
    prev_cfg = RunConfig.from_path(cfg_path)
    model_cfg = OmegaConf.to_container(prev_cfg.model_cfg, resolve=True)
    hnet_kwargs = prepare_hnet_config_kwargs(model_cfg)
    if not hnet_kwargs.get("d_model"):
        raise ValueError(
            "Model config missing d_model. Provide a resolved model config "
            "(for example outputs/.../.hydra/config.yaml or a model JSON)."
        )
    hnet_cfg = HNetConfig(**hnet_kwargs)
    if use_routing_floor is None:
        use_routing_floor = bool(prev_cfg.use_routing_floor)
    if disable_routing:
        use_routing_floor = False
        use_routing_ceiling = False
    routing_floor = parse_routing_floor_config(model_cfg)
    routing_ceiling = parse_routing_ceiling_config({"k_max_list": k_max_list})
    if use_routing_floor and routing_floor.enabled() and use_routing_ceiling and routing_ceiling.enabled():
        model = HNetForCausalLMWithRoutingFloorAndCeiling(
            hnet_cfg, routing_floor, routing_ceiling, device=device, dtype=dtype
        )
    elif use_routing_floor and routing_floor.enabled():
        model = HNetForCausalLMWithRoutingFloor(hnet_cfg, routing_floor, device=device, dtype=dtype)
    elif use_routing_ceiling and routing_ceiling.enabled():
        model = HNetForCausalLMWithRoutingCeiling(hnet_cfg, routing_ceiling, device=device, dtype=dtype)
    else:
        model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, prev_cfg
