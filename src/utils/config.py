from __future__ import annotations

"""Runtime config loading, saving, and display helpers."""

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Callable, Any
from uuid import uuid4
from .log import ExpLogger

import yaml
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pretty_repr

UNIMPORTANT_CFG = DictConfig(
    {'fields': ['gpus', 'debug', 'wandb', 'env', 'uid',
                'local_rank', 'file_prefix'],
     "prefix": ['_'],
     "postfix": ['_path', '_file', '_dir']}
)


def subset_dict_by_condition(d: dict, is_preserve: Callable[[str], bool] = lambda _: True) -> dict:
    if not isinstance(d, dict):
        return d
    filtered = {k: v for k, v in d.items() if is_preserve(str(k))}
    for key, value in list(filtered.items()):
        if isinstance(value, dict) and is_preserve(str(key)):
            filtered[key] = subset_dict_by_condition(value, is_preserve)
    return filtered


def get_important_cfg(
        cfg: DictConfig | "RunConfig", reserve_file_cfg=True, unimportant_cfg=UNIMPORTANT_CFG
):
    if isinstance(cfg, RunConfig):
        cfg = cfg.cfg
    imp_cfg = OmegaConf.to_object(cfg)

    def is_preserve(k: str):
        judge_file_setting = k == '_file_' and reserve_file_cfg
        prefix_allowed = (not any([k.startswith(_) for _ in unimportant_cfg.prefix])) or judge_file_setting
        postfix_allowed = not any([k.endswith(_) for _ in unimportant_cfg.postfix])
        field_allowed = k not in unimportant_cfg.fields
        return prefix_allowed and postfix_allowed and field_allowed

    imp_cfg = subset_dict_by_condition(imp_cfg, is_preserve)
    return imp_cfg


def save_config(cfg: DictConfig, path, as_global=True):
    OmegaConf.save(config=DictConfig(cfg), f=path)
    if as_global:
        with open(path, "r") as input_file:
            original_content = input_file.read()

        with open(path, "w") as output_file:
            output_file.write("# @package _global_\n" + original_content)
    return cfg


def load_config(path: str):
    # Load the configuration from the file
    cfg = OmegaConf.load(path)

    # If the config file has the global package declaration,
    # remove it so that it's compatible with normal OmegaConf usage
    with open(path, "r") as file:
        content = file.read()

    # Check if the file starts with the global package declaration
    if content.startswith("# @package _global_"):
        # Remove the declaration if present
        with open(path, "w") as file:
            file.write(content.replace("# @package _global_\n", "", 1))

    return cfg


def resolve_hydra_cfg_path(ckpt_dir: str) -> str:
    ckpt_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir)))
    if os.path.isfile(ckpt_dir):
        ckpt_dir = os.path.dirname(ckpt_dir)
    candidate = os.path.join(ckpt_dir, "hydra_cfg.yaml")
    if os.path.isfile(candidate):
        return candidate
    base_name = os.path.basename(ckpt_dir)
    if base_name == "last-checkpoint" or base_name.startswith("checkpoint-"):
        parent_dir = os.path.dirname(ckpt_dir)
        parent_candidate = os.path.join(parent_dir, "hydra_cfg.yaml")
        if os.path.isfile(parent_candidate):
            return parent_candidate
    raise FileNotFoundError(f"hydra_cfg.yaml not found under {ckpt_dir}")


@dataclass
class RunConfig:
    cfg: DictConfig
    logger: ExpLogger | None = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    device_type: str = "CPU"
    cmd: str = ""
    uid: str | None = None
    cfg_out_file: str | None = None
    _LOCAL_ONLY_FIELDS = {"cfg", "logger", "cfg_out_file"}

    def __init__(self, cfg: DictConfig) -> None:
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg
        self.logger = None
        self._sync_attrs()

    @classmethod
    def from_path(cls, cfg_path: str) -> "RunConfig":
        cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
        cfg = load_config(cfg_path)
        run_cfg = cls(cfg)
        run_cfg.cfg_out_file = cfg_path
        return run_cfg

    def __str__(self) -> str:
        return pretty_repr(self._important_cfg())

    def _important_cfg(self) -> dict:
        imp_cfg = get_important_cfg(self.cfg, reserve_file_cfg=False)
        if OmegaConf.is_config(imp_cfg):
            imp_cfg = OmegaConf.to_container(imp_cfg, resolve=True)
        return dict(imp_cfg) if isinstance(imp_cfg, dict) else {"config": imp_cfg}

    def _sync_attrs(self) -> None:
        self.rank = int(self.cfg.get("rank", 0) or 0)
        self.local_rank = int(self.cfg.get("local_rank", 0) or 0)
        self.world_size = int(self.cfg.get("world_size", 1) or 1)
        self.is_distributed = bool(self.cfg.get("is_distributed", False))
        self.device_type = str(self.cfg.get("device_type", "CPU"))
        self.cmd = str(self.cfg.get("cmd", ""))
        self.uid = self.cfg.get("uid")

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.cfg.get(key, default)

    def setdefault(self, key: str, default: Any | None = None) -> Any:
        if key in self.cfg:
            return self.cfg.get(key)
        self.cfg[key] = default
        return default

    def __getattr__(self, name: str) -> Any:
        cfg = self.__dict__.get("cfg")
        if cfg is None:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        return cfg.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self)._LOCAL_ONLY_FIELDS or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name in type(self).__dataclass_fields__:
            object.__setattr__(self, name, value)
            cfg = self.__dict__.get("cfg")
            if cfg is not None:
                cfg[name] = value
            return

        cfg = self.__dict__.get("cfg")
        if cfg is None:
            object.__setattr__(self, name, value)
            return
        cfg[name] = value

    def __getitem__(self, key: str) -> Any:
        return self.cfg[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.cfg[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.cfg

    def __iter__(self):
        return iter(self.cfg)

    def save(self) -> str:
        self.uid = generate_unique_id(self)
        if self.get("use_wandb", False) and self.wandb.get("url") is None:
            self.wandb.url = _resolve_wandb_url(self)
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)

        cfg_out_file = os.path.join(self.dirs.output, "hydra_cfg.yaml")
        save_config(self.cfg, cfg_out_file, as_global=True)
        self.cfg_out_file = cfg_out_file

        if cfg_out_file is not None and self.logger is not None:
            self.logger.save_file_to_wandb(
                self.cfg_out_file, base_path=self.dirs.output, policy="now"
            )
        return cfg_out_file


def generate_unique_id(cfg: RunConfig) -> str:
    """Generate a unique run id, optionally reusing wandb id."""
    if cfg.get("uid") is not None and cfg.wandb.id is not None:
        assert cfg.get("uid") == cfg.wandb.id, "Confliction: Wandb and uid mismatch!"
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.wandb.id or cfg.get("uid")
    uid = given_uid if given_uid else cur_time + str(uuid4()).split("-")[0]
    return uid


def _build_wandb_url(entity, project, run_id):
    if not entity or not project or not run_id:
        return None
    base_url = os.getenv("WANDB_BASE_URL", "https://wandb.ai").rstrip("/")
    return f"{base_url}/{entity}/{project}/runs/{run_id}"


def _resolve_wandb_url(cfg: RunConfig, run=None):
    if run is not None:
        url = getattr(run, "url", None)
        if url:
            return url
        entity = getattr(run, "entity", None)
        project = getattr(run, "project", None)
        run_id = getattr(run, "id", None)
    else:
        entity = None
        project = None
        run_id = None
    entity = entity or cfg.wandb.get("entity")
    project = project or cfg.wandb.get("project")
    run_id = run_id or cfg.wandb.get("id") or cfg.get("uid")
    return _build_wandb_url(entity, project, run_id)
