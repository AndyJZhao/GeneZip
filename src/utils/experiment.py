"""Experiment bootstrap utilities shared with RaftPPI."""

from collections.abc import Mapping, Sequence
import numpy as np
import os
import random
import sys
import torch
import wandb

from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import RunConfig, _resolve_wandb_url, get_important_cfg
from .log import ExpLogger, logger
from .distributed import initialize_distributed

import rootutils

root = rootutils.find_root(__file__)

_DEPRECATED_CFG_KEYS: dict[str, str] = {
    # GeneZip: SRL/soft-RAR removal (hard RAR only).
    "strictness_max": "Removed. Soft RAR is no longer supported; use hard RAR only.",
    "strictness_exp": "Removed. Soft RAR is no longer supported; use hard RAR only.",
    # GeneZip: routing schedule removal (bounded routing is constant now).
    "r_warm_up_start": "Removed. Routing schedules are no longer supported; use constant bounded routing.",
    "r_warm_up_end": "Removed. Routing schedules are no longer supported; use constant bounded routing.",
    # GeneZip: ratio-based routing bounds removal (paper uses per-stage K_min/K_max budgets now).
    "r_hi": "Removed. Use per-stage `k_min_list` / `k_max_list` token budgets instead.",
    "r_low": "Removed. Use per-stage `k_min_list` / `k_max_list` token budgets instead.",
    "min_routing_tokens": "Removed. Use per-stage `k_min_list` token budgets instead.",
    "max_routing_tokens": "Removed. Use per-stage `k_max_list` token budgets instead.",
    # GeneZip: RAR alpha schedule removal.
    "alpha_max": "Removed. Use `alpha` (constant RAR loss weight) instead.",
    "alpha_exp": "Removed. Alpha schedule is no longer supported; use constant `alpha` instead.",
    # Training: early-stop callback removal.
    "stop_steps": "Removed. Early-stop callback is no longer supported; use max_train_steps instead.",
}

_DEPRECATED_CFG_PATHS: dict[str, str] = {
    # NOTE: warmup_steps still exists under cfg.training.warmup_steps (LR schedule).
    # We only deprecate the top-level warmup_steps that previously controlled RAR alpha warmup.
    "warmup_steps": "Removed. Alpha warmup is no longer supported; use constant `alpha` instead.",
}


def _find_deprecated_cfg_paths(cfg: object, deprecated_keys: set[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def walk(node: object, prefix: str) -> None:
        if isinstance(node, (DictConfig, dict)):
            for key, value in node.items():
                key_str = str(key)
                path = f"{prefix}.{key_str}" if prefix else key_str
                if key_str in deprecated_keys:
                    found.append((path, key_str))
                walk(value, path)
            return

        if isinstance(node, (ListConfig, list, tuple)):
            for idx, value in enumerate(node):
                path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                walk(value, path)
            return

        if isinstance(node, Mapping):
            for key, value in node.items():
                key_str = str(key)
                path = f"{prefix}.{key_str}" if prefix else key_str
                if key_str in deprecated_keys:
                    found.append((path, key_str))
                walk(value, path)
            return

        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for idx, value in enumerate(node):
                path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                walk(value, path)
            return

    walk(cfg, "")
    return found


def _assert_no_deprecated_cfg_keys(cfg: DictConfig) -> None:
    deprecated = set(_DEPRECATED_CFG_KEYS.keys())
    hits = _find_deprecated_cfg_paths(cfg, deprecated)
    sentinel = object()
    for path in _DEPRECATED_CFG_PATHS:
        value = OmegaConf.select(cfg, path, default=sentinel)
        if value is not sentinel:
            hits.append((path, path))

    if not hits:
        return

    lines = ["Deprecated config keys detected (no longer supported):"]
    for path, key in hits:
        reason = _DEPRECATED_CFG_KEYS.get(key) or _DEPRECATED_CFG_PATHS.get(key) or "Removed."
        lines.append(f"  - {path}: {reason}")
    lines.append("Update your command/config and remove these keys.")
    raise ValueError("\n".join(lines))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_experiment(
    cfg: RunConfig, init_wandb: bool = True
) -> tuple[RunConfig, ExpLogger]:
    run_cfg = cfg if isinstance(cfg, RunConfig) else RunConfig(cfg)
    OmegaConf.set_struct(run_cfg.cfg, False)
    _assert_no_deprecated_cfg_keys(run_cfg.cfg)

    run_cfg.cmd = "python " + " ".join(sys.argv)
    if torch.cuda.is_available():
        run_cfg.device_type = "GPU"
    else:
        run_cfg.device_type = "XPU" if hasattr(torch, "xpu") and torch.xpu.is_available() else "CPU"

    initialize_distributed(run_cfg.cfg)
    if "pd_batch_size" in run_cfg.cfg:
        run_cfg.eq_batch_size = (
            run_cfg.cfg.pd_batch_size * run_cfg.cfg.world_size * run_cfg.cfg.get("grad_acc_steps", 1)
        )

    run_cfg._sync_attrs()
    if init_wandb:
        wandb_init(run_cfg)

    run_cfg.logger = ExpLogger(run_cfg)
    run_cfg.save()

    run_cfg.logger.print(str(run_cfg))
    run_cfg.logger.warning(
        f"Running\n{run_cfg.cmd}\n Running on {run_cfg.device_type}\n"
        f"{run_cfg.rank=} output_dir={run_cfg.dirs.output}"
    )

    set_seed(run_cfg.rank + run_cfg.seed)
    return run_cfg, run_cfg.logger


def finish_experiment(
    cfg: RunConfig, exp_logger: ExpLogger, effective_params: int | None = None
) -> None:
    if exp_logger is None:
        return
    run_summary = _build_run_summary(effective_params)
    if run_summary:
        exp_logger.update_summary(run_summary)
    result_summary = _summarize_results(exp_logger.results)
    cfg.summary = dict(getattr(exp_logger, "summary", {}) or {})
    for key, value in {**cfg.summary, **result_summary}.items():
        attr_key = key if isinstance(key, str) and key.isidentifier() else str(key).replace("/", "_")
        try:
            setattr(cfg, attr_key, value)
        except Exception:
            continue

    cfg.save()

    combined_summary = {**cfg.summary, **result_summary}
    if hasattr(exp_logger, "wandb_finish"):
        exp_logger.wandb_finish(combined_summary)
    if combined_summary:
        exp_logger.print(exp_logger.clean_metrics_for_display(combined_summary))
        if cfg.use_wandb:
            exp_logger.print(f'Experiment finished, check the run at {cfg.wandb.url}')

def _format_param_count_human_readable(count: int) -> str:
    """Format parameter counts into K/M/B with one decimal place."""
    for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if count >= scale:
            value = count / scale
            trimmed = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{trimmed}{suffix}"
    return str(int(count))


def _wandb_peak_gpu_mem_gb():
    run = wandb.run
    if run is None:
        return None
    try:
        history_df = run.history()
    except Exception:
        return None
    if history_df is None:
        return None
    try:
        max_bytes = history_df["gpu.0.memoryAllocatedBytes"].max()
    except Exception:
        return None
    if max_bytes is None:
        return None
    if isinstance(max_bytes, (float, np.floating)) and np.isnan(max_bytes):
        return None
    return float(max_bytes) / (1024 ** 3)


def _torch_peak_gpu_mem_gb():
    if not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.current_device()
        max_bytes = torch.cuda.max_memory_allocated(device)
    except Exception:
        return None
    if not max_bytes:
        return None
    return float(max_bytes) / (1024 ** 3)


def _build_run_summary(effective_params: int | None) -> dict:
    summary = {}
    if effective_params is not None:
        summary["model_size"] = _format_param_count_human_readable(effective_params)
    peak_gpu_mem = _wandb_peak_gpu_mem_gb()
    if peak_gpu_mem is None:
        peak_gpu_mem = _torch_peak_gpu_mem_gb()
    if peak_gpu_mem is not None:
        summary["max_gpu_mem"] = round(peak_gpu_mem, 3)
    return summary


def _summarize_results(results) -> dict:
    if results is None:
        return {}
    if isinstance(results, list):
        return results[-1] if results else {}
    if isinstance(results, dict):
        summary = {}
        for key, values in results.items():
            if isinstance(values, list):
                if values:
                    summary[key] = values[-1]
            else:
                summary[key] = values
        return summary
    return {}


def get_device(cfg: RunConfig) -> str:
    if cfg.device_type == "XPU" and torch.xpu.is_available():
        if cfg.is_distributed:
            device = f"xpu:{cfg.local_rank}"
            logger.info(f"Using XPU: {device} (local_rank: {cfg.local_rank}, global_rank: {cfg.rank})")
        else:
            device = "xpu:0"
            logger.info("Using single XPU: xpu:0")
    elif cfg.device_type == "GPU" and torch.cuda.is_available():
        if cfg.is_distributed:
            device = f"cuda:{cfg.local_rank}"
            logger.info(f"Using GPU: {device} (local_rank: {cfg.local_rank}, global_rank: {cfg.rank})")
        else:
            device = "cuda:0"
            logger.info("Using single GPU: cuda:0")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def clean_up_cfg_for_wandb(cfg: dict | DictConfig) -> dict:
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict):
        return cfg

    return {key: value for key, value in cfg.items() if not isinstance(value, dict)}


def wandb_init(cfg: RunConfig) -> None:
    os.environ["WANDB_WATCH"] = "false"
    rank = int(cfg.get("rank", 0) or 0)
    if cfg.use_wandb and rank == 0 and int(os.getenv("OMPI_COMM_WORLD_RANK", 0)) == 0:
        wandb_tags = cfg.wandb.tags
        mode = cfg.wandb.get("mode", "online")
        imp_cfg = get_important_cfg(cfg)
        imp_cfg = clean_up_cfg_for_wandb(imp_cfg)
        logger.critical(f"Creating WANDB session at rank={rank}, mode={mode}")
        if cfg.wandb.id is None:
            os.makedirs(cfg.dirs.wandb_cache, exist_ok=True)
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                dir=cfg.dirs.wandb_cache,
                reinit=True,
                config=imp_cfg,
                name=cfg.wandb.name,
                tags=wandb_tags,
                mode=mode,
            )
        else:
            logger.critical(f"Resume from previous wandb run {cfg.wandb.id}")
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                reinit=True,
                resume="must",
                id=cfg.wandb.id,
                mode=mode,
            )
            cfg.wandb.is_master_process = False
        if wandb.run is not None:
            cfg.wandb.id = wandb.run.id
            cfg.wandb.name = wandb.run.name
            cfg.wandb.url = _resolve_wandb_url(cfg, wandb.run)
        elif cfg.wandb.get("url") is None:
            cfg.wandb.url = _resolve_wandb_url(cfg)
        if mode == "offline":
            logger.critical(
                f"Wandb local mode use command to sync:\n"
                f"wandb sync --include-offline {cfg.dirs.wandb_cache}wandb/offline-*"
            )
        step_metric = cfg.wandb.get("step_metric", None)
        if step_metric:
            wandb.run.define_metric(step_metric)
        wandb.run.define_metric("*", step_metric=step_metric, step_sync=True)
    else:
        os.environ["WANDB_DISABLED"] = "true"
        cfg.use_wandb = False
        cfg.wandb.id, cfg.wandb.name, cfg.wandb.url = None, None, None
