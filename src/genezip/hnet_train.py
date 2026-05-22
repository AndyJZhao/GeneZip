from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf
import torch

from src.genezip.config import parse_routing_ceiling_config, parse_routing_floor_config
from src.genezip.region_aware_hnet import (
    HNetForCausalLMWithRoutingCeiling,
    HNetForCausalLMWithRoutingFloor,
    HNetForCausalLMWithRoutingFloorAndCeiling,
)
from src.genezip.tokenizer import (
    FastSingleNucleotideTokenizer,
    SingleNucleotideTokenizer,
    SixMerTokenizer,
)
from src.hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from src.hnet.models.mixer_seq import HNetForCausalLM


def resolve_model_arch(cfg) -> str:
    model_cfg = getattr(cfg, "model", None)
    arch = getattr(model_cfg, "arch", None) if model_cfg is not None else None
    if arch is None:
        arch = getattr(cfg, "arch", None)
    if arch is None:
        raise ValueError("Model arch not provided. Set `model.arch` (preferred) or `arch`.")
    return str(arch)


def _log_info(logger, message: str) -> None:
    if logger is None:
        print(message, flush=True)
    else:
        logger.print(message)


def _load_model_cfg_container(cfg) -> dict[str, Any]:
    model_cfg_inline = getattr(cfg, "model_cfg", None)
    if model_cfg_inline is None:
        model = getattr(cfg, "model", None)
        model_cfg_inline = getattr(model, "config", None) if model is not None else None
    if model_cfg_inline is None:
        raise ValueError("model_cfg not provided. Set `model_cfg` in the Hydra config.")
    cfg_dict = OmegaConf.to_container(model_cfg_inline, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"model_cfg must be a dict, got {type(cfg_dict)}")
    if "dtype" in cfg_dict and "torch_dtype" not in cfg_dict:
        cfg_dict["torch_dtype"] = cfg_dict["dtype"]
    return cfg_dict


def _prepare_hnet_config_kwargs(cfg_dict: Mapping[str, Any]) -> dict[str, Any]:
    valid_fields = set(HNetConfig.__dataclass_fields__.keys())
    cfg = {k: v for k, v in dict(cfg_dict).items() if k in valid_fields}
    if "ssm_cfg" in cfg and not isinstance(cfg["ssm_cfg"], SSMConfig):
        cfg["ssm_cfg"] = SSMConfig(**cfg["ssm_cfg"])
    if "attn_cfg" in cfg and not isinstance(cfg["attn_cfg"], AttnConfig):
        cfg["attn_cfg"] = AttnConfig(**cfg["attn_cfg"])
    cfg.setdefault("ssm_cfg", SSMConfig())
    cfg.setdefault("attn_cfg", AttnConfig())
    return cfg


def build_model(cfg, device: torch.device, logger=None):
    arch = resolve_model_arch(cfg)
    if arch != "hnet":
        raise ValueError(f"Only arch='hnet' is supported in this release, got arch={arch!r}")

    model_config_dict = _load_model_cfg_container(cfg)
    model_config_kwargs = _prepare_hnet_config_kwargs(model_config_dict)
    model_config = HNetConfig(**model_config_kwargs)

    use_floor = bool(getattr(cfg, "use_routing_floor", False))
    use_ceiling = bool(getattr(cfg, "use_routing_ceiling", False))

    if use_floor and use_ceiling:
        routing_floor = parse_routing_floor_config(model_config_dict)
        routing_ceiling = parse_routing_ceiling_config(model_config_dict)
        if routing_floor.enabled() or routing_ceiling.enabled():
            model = HNetForCausalLMWithRoutingFloorAndCeiling(
                model_config,
                routing_floor,
                routing_ceiling,
            ).to(device)
            if not routing_floor.enabled():
                _log_info(logger, "[hnet] use_routing_floor=true but floor disabled in config.")
            if not routing_ceiling.enabled():
                _log_info(logger, "[hnet] use_routing_ceiling=true but ceiling disabled in config.")
        else:
            model = HNetForCausalLM(model_config).to(device)
            _log_info(logger, "[hnet] routing floor+ceiling requested but both disabled in config.")
    elif use_ceiling:
        routing_ceiling = parse_routing_ceiling_config(model_config_dict)
        if routing_ceiling.enabled():
            model = HNetForCausalLMWithRoutingCeiling(model_config, routing_ceiling).to(device)
        else:
            model = HNetForCausalLM(model_config).to(device)
            _log_info(logger, "[hnet] use_routing_ceiling=true but ceiling disabled in config.")
    elif use_floor:
        routing_floor = parse_routing_floor_config(model_config_dict)
        if routing_floor.enabled():
            model = HNetForCausalLMWithRoutingFloor(model_config, routing_floor).to(device)
        else:
            model = HNetForCausalLM(model_config).to(device)
            _log_info(logger, "[hnet] use_routing_floor=true but floor disabled in config.")
    else:
        model = HNetForCausalLM(model_config).to(device)
    return model, model_config_dict


def build_tokenizer(cfg):
    tokenizer_name = cfg.get("tokenizer", "fast")
    if tokenizer_name == "singlenucleotide":
        rc_aug = bool(cfg.get("RC_augmentation", False))
        return SingleNucleotideTokenizer(RC_augmentation=rc_aug)
    if tokenizer_name == "sixmer":
        return SixMerTokenizer()
    if tokenizer_name == "fast":
        return FastSingleNucleotideTokenizer()
    raise ValueError(f"Unknown tokenizer choice: {tokenizer_name}")


def count_model_parameters(model) -> tuple[int, int]:
    total_params = 0
    effective_params = 0
    seen_data_ptrs: set[int] = set()

    try:
        params_iter = model.named_parameters(remove_duplicate=False)
    except TypeError:
        def _iter_params():
            for module in model.modules():
                for param in getattr(module, "_parameters", {}).values():
                    if param is not None:
                        yield param

        params_iter = ((None, p) for p in _iter_params())

    for _, param in params_iter:
        if param is None:
            continue
        numel = param.numel()
        total_params += numel

        data_ptr = param.data_ptr()
        if data_ptr in seen_data_ptrs:
            continue

        effective_params += numel
        seen_data_ptrs.add(data_ptr)

    return total_params, effective_params


def load_training_config(cfg) -> dict[str, Any]:
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    if not isinstance(training_cfg, dict):
        raise TypeError(f"cfg.training must be a dict-like config, got {type(training_cfg)}")

    training_cfg.pop("output_dir", None)
    training_cfg.pop("use_lr_multiplier", None)
    training_cfg.pop("hnet_lr_multiplier", None)
    training_cfg.pop("hnet_initializer_range", None)

    overrides = training_cfg.pop("overrides", {}) or {}
    if overrides:
        if not isinstance(overrides, dict):
            raise TypeError(f"training.overrides must be a dict, got {type(overrides)}")
        training_cfg.update(overrides)

    max_train_steps = training_cfg.pop("max_train_steps", None)
    if training_cfg.get("max_steps", None) is None and max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps > 0:
            training_cfg["max_steps"] = max_train_steps

    training_cfg["report_to"] = "none"
    training_cfg.setdefault("run_name", cfg.alias)
    training_cfg.setdefault("metric_for_best_model", "eval_ppl")
    training_cfg.setdefault("greater_is_better", False)
    return training_cfg
