import os
import re
import posixpath
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from datasets import load_from_disk
from huggingface_hub import HfApi, HfFolder
from huggingface_hub import snapshot_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError
import torch
from transformers import TrainingArguments, TrainerCallback, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class EvalResumeCUDACleaner(TrainerCallback):
    """
    Clear CUDA cache after evaluation before training resumes to reduce
    fragmentation-related transient OOMs.
    """

    def __init__(self):
        self._pending_clear = False

    def on_evaluate(self, args, state, control, **kwargs):
        self._pending_clear = True

    def on_step_begin(self, args, state, control, **kwargs):
        if self._pending_clear and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._pending_clear = False


class PeriodicCUDACleaner(TrainerCallback):
    """
    Periodically clear CUDA cache to reduce fragmentation during long runs.
    """

    def __init__(self, every_steps: int = 10):
        self.every_steps = max(1, int(every_steps))

    def on_optimizer_step(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % self.every_steps == 0:
            torch.cuda.empty_cache()


class ExpLoggerCallback(TrainerCallback):
    """
    Route HF Trainer metrics into ExpLogger.log_metrics so we own wandb reporting.
    """

    def __init__(self, logger) -> None:
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.logger is None or not logs:
            return
        payload = dict(logs)
        step = payload.get("step")
        if step is None:
            step = payload.get("global_step")
        if step is None:
            step = getattr(state, "global_step", None)
        if step is not None:
            payload.setdefault("step", step)
        self.logger.log_metrics(payload, step=step)


def _remove_key_recursive(value: Any, key: str) -> bool:
    removed = False
    if isinstance(value, dict):
        if key in value:
            del value[key]
            removed = True
        for item in value.values():
            removed = _remove_key_recursive(item, key) or removed
    elif isinstance(value, list):
        for item in value:
            removed = _remove_key_recursive(item, key) or removed
    return removed


def anonymize_config(config_path: Path, logger=None) -> None:
    if not config_path.exists():
        msg = f"[share] hydra config not found: {config_path}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
    with config_path.open("rt", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        msg = f"[share] hydra config is not a dict: {config_path}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
    if "env" in config and isinstance(config["env"], dict) and "vars" in config["env"]:
        del config["env"]["vars"]
        msg = "[share] removed env.vars from hydra config."
        if logger:
            logger.info(msg)
        else:
            print(msg)
    else:
        msg = "[share] no env.vars section found in hydra config."
        if logger:
            logger.info(msg)
        else:
            print(msg)
    if _remove_key_recursive(config, "hf_token"):
        msg = "[share] removed hf_token from hydra config."
        if logger:
            logger.info(msg)
        else:
            print(msg)
    else:
        msg = "[share] no hf_token found in hydra config."
        if logger:
            logger.info(msg)
        else:
            print(msg)
    with config_path.open("wt", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False)


def resolve_hf_repo_id(cfg) -> str:
    hf_repo = cfg.get("hf_repo")
    if isinstance(hf_repo, str) and hf_repo.lower() == "none":
        hf_repo = None
    if hf_repo:
        return str(hf_repo)
    user = cfg.get("hf_user", None)
    if isinstance(user, str) and user.lower() == "none":
        user = None
    model_name = cfg.get("model_name", None)
    if isinstance(model_name, str) and model_name.lower() == "none":
        model_name = None
    if not model_name:
        raise ValueError("model_name must be set when hf_repo is not provided.")
    return f"{user}/{model_name}" if user else str(model_name)


class HFCheckpointUploader(TrainerCallback):
    def __init__(
        self,
        repo_id: str,
        token: str,
        private: bool,
        logger=None,
        ignore_patterns: Iterable[str] | None = None,
        config_filename: str | None = "hydra_cfg.yaml",
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.private = bool(private)
        self.logger = logger
        self.ignore_patterns = list(ignore_patterns) if ignore_patterns else None
        self.config_filename = config_filename

    def _maybe_upload_config(self, output_dir: str) -> None:
        if not self.config_filename:
            return
        config_path = Path(output_dir) / self.config_filename
        if not config_path.is_file():
            msg = f"[hf] config file not found for upload: {config_path}"
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)
            return
        anonymize_config(config_path, self.logger)
        try:
            upload_file_to_hub(
                str(config_path),
                self.repo_id,
                self.token,
                private=self.private,
                path_in_repo=self.config_filename,
                logger=self.logger,
            )
        except Exception as exc:
            msg = f"[hf] upload failed for {config_path}: {exc}"
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        step = getattr(state, "global_step", None)
        if step is None:
            return
        checkpoint_name = f"{PREFIX_CHECKPOINT_DIR}-{step}"
        checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
        if not os.path.isdir(checkpoint_dir):
            msg = f"[hf] checkpoint dir not found for upload: {checkpoint_dir}"
            self.logger.warning(msg)
            return
        try:
            upload_folder_to_hub(
                checkpoint_dir,
                self.repo_id,
                self.token,
                private=self.private,
                path_in_repo=checkpoint_name,
                ignore_patterns=self.ignore_patterns,
                logger=self.logger,
            )
        except Exception as exc:
            msg = f"[hf] upload failed for {checkpoint_dir}: {exc}"
            self.logger.warning(msg)
        self._maybe_upload_config(args.output_dir)


def _normalize_optional_str(value: Any):
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


CKPT_FILE_NAMES = (
    "pytorch_model.bin",
    "pytorch_model.safetensors",
    "model.safetensors",
    "model.bin",
)
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def resolve_hf_cache_dir(cfg) -> str | None:
    if cfg is None or not hasattr(cfg, "get"):
        return None
    dirs = cfg.get("dirs") or {}
    if hasattr(dirs, "get"):
        return dirs.get("hf_cache")
    return None


def _resolve_local_path(
    ckpt: str,
    path_resolver: Callable[[str], str] | None,
) -> str:
    expanded = os.path.expandvars(os.path.expanduser(ckpt))
    if path_resolver is not None:
        return path_resolver(expanded)
    return os.path.abspath(expanded)


def _parse_hf_repo_spec(ckpt: str, default_owner: str | None) -> tuple[str, str | None, str]:
    spec = str(ckpt).strip()
    if spec.startswith("hf://"):
        spec = spec[len("hf://") :]
    if spec.startswith("https://huggingface.co/"):
        spec = spec[len("https://huggingface.co/") :]
    spec = spec.strip("/")
    parts = [part for part in spec.split("/") if part]
    if not parts:
        raise ValueError(f"Invalid checkpoint spec: {ckpt}")
    default_owner = _normalize_optional_str(default_owner)
    if len(parts) == 1:
        if not default_owner:
            raise ValueError(f"ckpt='{ckpt}' requires hf_user to resolve.")
        owner = default_owner
        repo_part = parts[0]
        subfolder = ""
    elif default_owner and (
        parts[1].startswith("checkpoint-") or parts[1] == "last-checkpoint"
    ):
        owner = default_owner
        repo_part = parts[0]
        subfolder = "/".join(parts[1:])
    else:
        owner = parts[0]
        repo_part = parts[1]
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else ""
    if "@" in repo_part:
        repo_name, revision = repo_part.split("@", 1)
    else:
        repo_name, revision = repo_part, None
    repo_id = f"{owner}/{repo_name}"
    return repo_id, revision, subfolder


def _list_checkpoint_subfolders(
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision=revision, token=token)
    files_set = set(files)
    root_files = {name for name in files if "/" not in name}
    subfolders: list[str] = []
    if any(name in root_files for name in CKPT_FILE_NAMES):
        subfolders.append("")
    if any(f"last-checkpoint/{name}" in files_set for name in CKPT_FILE_NAMES):
        subfolders.append("last-checkpoint")
    step_to_dir: dict[int, str] = {}
    for name in files:
        if "/" not in name:
            continue
        dir_name, fname = name.split("/", 1)
        if fname not in CKPT_FILE_NAMES:
            continue
        match = _CHECKPOINT_DIR_RE.match(dir_name)
        if not match:
            continue
        step_to_dir[int(match.group(1))] = dir_name
    for step in sorted(step_to_dir, reverse=True):
        subfolders.append(step_to_dir[step])
    return subfolders


def _build_ckpt_allow_patterns(subfolder: str) -> list[str]:
    patterns = ["hydra_cfg.yaml", ".hydra/**"]
    if subfolder:
        patterns.append(f"{subfolder}/hydra_cfg.yaml")
        patterns.append(f"{subfolder}/.hydra/**")
        patterns.extend([f"{subfolder}/{name}" for name in CKPT_FILE_NAMES])
    else:
        patterns.extend(CKPT_FILE_NAMES)
    return patterns


def _is_recoverable_snapshot_error(exc: Exception) -> bool:
    if isinstance(exc, (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError)):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status == 404
    return False


def resolve_checkpoint_path(
    ckpt: str,
    default_owner: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    path_resolver: Callable[[str], str] | None = None,
) -> dict[str, str]:
    ckpt = str(ckpt)
    token = _normalize_optional_str(token)
    local_path = _resolve_local_path(ckpt, path_resolver)
    if os.path.exists(local_path):
        return _resolve_ckpt_file(local_path)

    repo_id, revision, subfolder = _parse_hf_repo_spec(ckpt, default_owner)
    explicit_subfolder = bool(subfolder)
    if explicit_subfolder:
        subfolders = [subfolder]
    else:
        subfolders = _list_checkpoint_subfolders(repo_id, revision, token)
    if not subfolders:
        raise FileNotFoundError(
            f"No checkpoint files found in repo={repo_id}. "
            f"Expected {CKPT_FILE_NAMES} in root/last-checkpoint/checkpoint-*/"
        )

    errors: list[tuple[str, Exception]] = []
    for candidate in subfolders:
        allow_patterns = _build_ckpt_allow_patterns(candidate)
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=allow_patterns,
            )
        except Exception as exc:
            if explicit_subfolder or not _is_recoverable_snapshot_error(exc):
                raise
            errors.append((candidate, exc))
            continue
        ckpt_dir = os.path.join(snapshot_path, candidate) if candidate else snapshot_path
        return _resolve_ckpt_file(ckpt_dir)

    attempted = ", ".join([c or "<root>" for c, _ in errors])
    msg = f"Failed to download checkpoints for repo={repo_id}. Tried: {attempted}"
    raise FileNotFoundError(msg) from errors[-1][1]


def _resolve_ckpt_file(path: str) -> dict[str, str]:
    if os.path.isdir(path):
        for name in CKPT_FILE_NAMES:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return {"file": candidate, "dir": path}
        raise FileNotFoundError(f"No checkpoint file found under {path}")
    if os.path.isfile(path):
        return {"file": path, "dir": os.path.dirname(path)}
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def _normalize_hf_subdir(value: Any) -> str:
    value = _normalize_optional_str(value)
    if value is None:
        return ""
    return str(value).strip("/")


def resolve_hf_dataset_root(
    data_root: str,
    *,
    repo_id: str,
    required_subdir: str,
    token: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    path_resolver: Callable[[str], str] | None = None,
) -> str:
    local_root = os.path.expandvars(os.path.expanduser(str(data_root)))
    if path_resolver is not None:
        local_root = path_resolver(local_root)
    else:
        local_root = os.path.abspath(local_root)
    required_subdir = _normalize_hf_subdir(required_subdir)
    if os.path.isdir(os.path.join(local_root, required_subdir)):
        return local_root

    repo_id = _normalize_optional_str(repo_id)
    if not repo_id:
        raise FileNotFoundError(f"Missing local release dataset directory: {os.path.join(local_root, required_subdir)}")

    os.makedirs(local_root, exist_ok=True)
    snapshot_download(
        repo_id=str(repo_id),
        repo_type="dataset",
        token=_normalize_optional_str(token),
        local_dir=local_root,
        local_files_only=local_files_only,
        allow_patterns=[f"{required_subdir}/**"],
    )
    return local_root


def load_hf_dataset_from_cfg(
    dataset_cfg,
    hf_token=None,
    path_resolver: Callable[[str], str] | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
):
    dataset_cfg = dataset_cfg or {}
    local_path = _normalize_optional_str(dataset_cfg.get("path"))
    hf_subdir = _normalize_hf_subdir(dataset_cfg.get("hf_subdir") or dataset_cfg.get("path_in_repo"))
    if local_path and path_resolver is not None:
        local_path = path_resolver(local_path)

    if local_path and os.path.exists(local_path):
        return load_from_disk(local_path)

    repo_id = _normalize_optional_str(dataset_cfg.get("hf_repo") or dataset_cfg.get("repo_id"))
    if repo_id and local_path:
        if hf_subdir:
            local_root = os.path.dirname(local_path)
            os.makedirs(local_root, exist_ok=True)
            snapshot_download(
                repo_id=str(repo_id),
                repo_type="dataset",
                token=_normalize_optional_str(hf_token),
                local_dir=local_root,
                local_files_only=local_files_only,
                allow_patterns=[f"{hf_subdir}/**"],
            )
        else:
            os.makedirs(local_path, exist_ok=True)
            snapshot_download(
                repo_id=str(repo_id),
                repo_type="dataset",
                token=_normalize_optional_str(hf_token),
                local_dir=local_path,
                local_files_only=local_files_only,
            )
        return load_from_disk(local_path)

    raise FileNotFoundError(f"Local release dataset not found at '{local_path}'.")


def downsample_dataset_splits(dataset, fraction, seed):
    if not fraction:
        return dataset
    if "validation" in dataset:
        dataset["validation"] = (
            dataset["validation"].train_test_split(test_size=fraction, seed=seed)["test"]
        )
    if "test" in dataset:
        dataset["test"] = dataset["test"].train_test_split(
            test_size=fraction, seed=seed
        )["test"]
    return dataset


def split_train_valid_test(dataset, seed, valid_size=0.05):
    if hasattr(dataset, "train_test_split"):
        ds_splits = dataset.train_test_split(test_size=valid_size, seed=seed)
        train_split = ds_splits["train"]
        valid_split = ds_splits["test"]
        test_split = None
    else:
        train_split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
        if "validation" in dataset:
            valid_split = dataset["validation"]
        elif "test" in dataset:
            valid_split = dataset["test"]
        else:
            split = train_split.train_test_split(test_size=valid_size, seed=seed)
            train_split, valid_split = split["train"], split["test"]
        test_split = dataset["test"] if "test" in dataset else None
    return train_split, valid_split, test_split


def subset_dataset(dataset, max_samples: int):
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(max_samples))
    return dataset[:max_samples]


def upload_folder_to_hub(
    local_model_path: str,
    repo_name: str,
    access_token: str,
    private: bool,
    path_in_repo: str | None = None,
    ignore_patterns: Iterable[str] | None = None,
    logger=None,
):
    # Authenticate with Hugging Face Hub
    repo_name = repo_name.replace(':', '_')
    path_in_repo = (path_in_repo or "").strip("/")
    repo_path = f"{repo_name}/{path_in_repo}" if path_in_repo else repo_name
    repo_url = f"https://huggingface.co/{repo_name}"
    if path_in_repo:
        repo_url = f"{repo_url}/tree/main/{path_in_repo}"
    HfFolder.save_token(access_token)
    api = HfApi()

    # Create the repository if it doesn't already exist
    api.create_repo(repo_id=repo_name, private=private, exist_ok=True, token=access_token)

    if hasattr(api, "upload_folder"):
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_name,
            path_in_repo=path_in_repo or "",
            token=access_token,
            ignore_patterns=ignore_patterns,
        )
        msg = f"Uploaded folder {local_model_path} to {repo_path} ({repo_url})"
        if logger:
            logger.critical(msg)
        else:
            print(msg)
        return

    # Iterate through each file in the local directory and upload it
    for root, _, files in os.walk(local_model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            repo_file_path = os.path.relpath(local_file_path, local_model_path)  # Preserve folder structure
            repo_file_path = repo_file_path.replace(os.sep, "/")
            if path_in_repo:
                repo_file_path = posixpath.join(path_in_repo, repo_file_path)

            # Upload each file to the repository
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=repo_file_path,
                repo_id=repo_name,
                token=access_token
            )
            msg = f"Uploaded {repo_file_path} to {repo_name}"
            if logger:
                logger.critical(msg)
            else:
                print(msg)
    msg = f"Uploaded folder {local_model_path} to {repo_path} ({repo_url})"
    if logger:
        logger.critical(msg)
    else:
        print(msg)


def upload_file_to_hub(
    local_file_path: str,
    repo_name: str,
    access_token: str,
    private: bool,
    path_in_repo: str | None = None,
    logger=None,
):
    # Authenticate with Hugging Face Hub
    repo_name = repo_name.replace(':', '_')
    path_in_repo = (path_in_repo or "").strip("/")
    repo_path = f"{repo_name}/{path_in_repo}" if path_in_repo else repo_name
    repo_url = f"https://huggingface.co/{repo_name}"
    if path_in_repo:
        repo_url = f"{repo_url}/blob/main/{path_in_repo}"
    HfFolder.save_token(access_token)
    api = HfApi()

    # Create the repository if it doesn't already exist
    api.create_repo(repo_id=repo_name, private=private, exist_ok=True, token=access_token)
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=path_in_repo or os.path.basename(local_file_path),
        repo_id=repo_name,
        token=access_token,
    )
    msg = f"Uploaded file {local_file_path} to {repo_path} ({repo_url})"
    if logger:
        logger.critical(msg)
    else:
        print(msg)
