"""Minimal distributed helpers used by init_experiment.

This is a slimmed-down copy of the RaftPPI utilities. For DNAFM we only need
basic rank/local_rank bookkeeping plus a couple of cache helpers.
"""

import os
import socket

import torch
from torch import distributed as dist


def patch_accelerate_env() -> None:
    """Map accelerate env vars to torch.distributed keys when missing."""
    mapping = {
        "RANK": "ACCELERATE_PROCESS_INDEX",
        "LOCAL_RANK": "ACCELERATE_LOCAL_PROCESS_INDEX",
        "WORLD_SIZE": "ACCELERATE_NUM_PROCESSES",
    }
    for target_key, source_key in mapping.items():
        if os.environ.get(target_key) is None and os.environ.get(source_key) is not None:
            os.environ[target_key] = os.environ[source_key]


def _is_accelerate_launch() -> bool:
    return any(
        os.environ.get(key) is not None
        for key in (
            "ACCELERATE_PROCESS_INDEX",
            "ACCELERATE_LOCAL_PROCESS_INDEX",
            "ACCELERATE_NUM_PROCESSES",
        )
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def empty_cache(device_type):
    if device_type == "XPU" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def initialize_distributed(cfg):
    """Populate rank/local_rank/world_size and set device when needed."""
    patch_accelerate_env()
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["LOCAL_RANK"] = os.getenv("LOCAL_RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    if "MASTER_PORT" not in os.environ:
        cfg.master_port = str(find_free_port())
        os.environ["MASTER_PORT"] = cfg.master_port
    else:
        cfg.master_port = os.getenv("MASTER_PORT")

    cfg.rank = int(os.getenv("RANK", "0"))
    cfg.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.world_size = int(os.getenv("WORLD_SIZE", "1"))
    cfg.is_distributed = cfg.world_size > 1

    if cfg.device_type != "CPU" and cfg.world_size > 1:
        if dist.is_available() and dist.is_initialized():
            cfg.rank = dist.get_rank()
            cfg.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            cfg.world_size = dist.get_world_size()
            cfg.is_distributed = True
        elif not _is_accelerate_launch():
            dist.init_process_group(backend="nccl", init_method="env://")
            cfg.rank = dist.get_rank()
            cfg.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            cfg.world_size = dist.get_world_size()
            cfg.is_distributed = True

    if cfg.device_type == "GPU" and torch.cuda.is_available():
        device = cfg.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
