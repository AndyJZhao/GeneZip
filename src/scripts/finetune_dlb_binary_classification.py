import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from bisect import bisect_left
import os
import shutil
from types import SimpleNamespace
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.downstream.dlb_data import load_data
from src.downstream.utils import prepare_hnet_config_kwargs, resolve_dtype, load_state_dict
from src.genezip.config import parse_routing_ceiling_config, parse_routing_floor_config
from src.genezip.embedding import coverage_aware_pooling
from src.genezip.region_aware_hnet import (
    HNetForCausalLMWithRoutingCeiling,
    HNetForCausalLMWithRoutingFloor,
    HNetForCausalLMWithRoutingFloorAndCeiling,
)
from src.hnet.models.config_hnet import HNetConfig
from src.hnet.models.mixer_seq import HNetForCausalLM
from src.genezip.tokenizer import FastSingleNucleotideTokenizer, SingleNucleotideTokenizer, SixMerTokenizer
from src.utils import RunConfig, finish_experiment, init_experiment, resolve_hydra_cfg_path
from src.utils.hf_utils import (
    ExpLoggerCallback,
    resolve_checkpoint_path,
    resolve_hf_cache_dir,
    resolve_hf_dataset_root,
)


TASK_SPECS: dict[str, dict[str, Any]] = {
    "eqtl_prediction": {"num_labels": 2, "metrics": ("auroc", "auprc")},
    "enhancer_target_gene_prediction": {"num_labels": 2, "metrics": ("auroc", "auprc")},
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    denom = exp.sum(axis=-1, keepdims=True)
    return exp / np.clip(denom, 1e-12, None)


def _auroc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    n = labels.size
    if n == 0:
        return float("nan")
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    rank = 1
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1))
        ranks[i:j] = avg_rank
        rank += (j - i)
        i = j

    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    n_pos = int(np.sum(labels == 1))
    if labels.size == 0 or n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(precision[y == 1].sum() / float(n_pos))
    return ap


def compute_metrics(eval_pred) -> dict[str, float]:
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    probs = _softmax(logits)
    pos = probs[:, 1]
    return {
        "auroc": _auroc_binary(labels, pos),
        "auprc": _average_precision_binary(labels, pos),
    }


class HNetClassificationTrainer(Trainer):
    def __init__(
        self,
        *args,
        global_target_bp: float,
        alpha: float = 1.0,
        **kwargs,
    ):
        self.global_target_bp = float(global_target_bp)
        self.alpha = float(alpha)
        super().__init__(*args, **kwargs)

    def _ratio_weight(self) -> float:
        return max(0.0, float(self.alpha))

    @staticmethod
    def _extract_all_boundary_outputs(outputs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(outputs, dict):
            bpred_output = outputs.get("bpred_output", None)
        else:
            bpred_output = getattr(outputs, "bpred_output", None)

        if bpred_output is None:
            return []
        if isinstance(bpred_output, tuple):
            bpred_output = list(bpred_output)
        elif not isinstance(bpred_output, list):
            bpred_output = [bpred_output]

        boundary_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for router_output in bpred_output:
            if isinstance(router_output, dict):
                boundary_mask = router_output.get("boundary_mask_pre")
                if boundary_mask is None:
                    boundary_mask = router_output.get("boundary_mask")
                boundary_prob = router_output.get("boundary_prob_pre")
                if boundary_prob is None:
                    boundary_prob = router_output.get("boundary_prob")
            else:
                boundary_mask = getattr(router_output, "boundary_mask_pre", None)
                if boundary_mask is None:
                    boundary_mask = getattr(router_output, "boundary_mask", None)
                boundary_prob = getattr(router_output, "boundary_prob_pre", None)
                if boundary_prob is None:
                    boundary_prob = getattr(router_output, "boundary_prob", None)

            if boundary_mask is None or boundary_prob is None:
                continue

            boundary_mask = boundary_mask.bool()
            boundary_prob = boundary_prob.float()
            if boundary_prob.dim() == boundary_mask.dim() + 1:
                boundary_prob = boundary_prob[..., -1]

            boundary_outputs.append((boundary_mask, boundary_prob))
        return boundary_outputs

    @staticmethod
    def _global_ratio_loss(
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        target_bp: float,
    ) -> torch.Tensor | None:
        target_bp = float(target_bp)
        if target_bp <= 1.0:
            return None

        if boundary_mask.dim() == 1:
            boundary_mask = boundary_mask.unsqueeze(0)
            boundary_prob = boundary_prob.unsqueeze(0)

        b = boundary_mask.float()
        p = boundary_prob.float()

        denom = torch.full((b.shape[0],), float(b.shape[-1]), device=b.device, dtype=b.dtype).clamp_min(1.0)
        f = b.sum(dim=-1) / denom
        g = p.sum(dim=-1) / denom

        n = target_bp
        if n <= 1.0:
            return None
        loss = (((n - 1.0) * f * g + (1.0 - f) * (1.0 - g)) * (n / (n - 1.0)))
        return loss.mean()

    def _ratio_loss_all_stages(self, outputs) -> tuple[torch.Tensor | None, dict[str, float]]:
        boundary_outputs = self._extract_all_boundary_outputs(outputs)
        if not boundary_outputs:
            return None, {}

        logs: dict[str, float] = {}
        for si, (boundary_mask, boundary_prob) in enumerate(boundary_outputs):
            logs[f"router/sel_tokens_s{si}"] = float(boundary_mask.sum(dim=-1).float().mean().detach().cpu())
            logs[f"router/prob_mean_s{si}"] = float(boundary_prob.mean().detach().cpu())
            logs[f"router/prob_std_s{si}"] = float(boundary_prob.std(unbiased=False).detach().cpu())

        num_stages = max(1, len(boundary_outputs))
        stage_target_bp = float(self.global_target_bp) ** (1.0 / float(num_stages))

        losses: list[torch.Tensor] = []
        for boundary_mask, boundary_prob in boundary_outputs:
            loss = self._global_ratio_loss(boundary_mask, boundary_prob, stage_target_bp)
            if loss is not None:
                losses.append(loss)

        if not losses:
            return None, logs
        return torch.stack(losses).sum(), logs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch

        outputs = model(**inputs)
        cls_loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        if cls_loss is None:
            raise ValueError("Model must return `loss` when labels are provided.")

        ratio_w = self._ratio_weight()
        ratio_loss, ratio_logs = self._ratio_loss_all_stages(outputs)

        loss = cls_loss
        logs: dict[str, float] = {
            "loss_cls": float(cls_loss.detach().item()),
            "comp/rl_weight": float(ratio_w),
        }
        logs.update(ratio_logs)

        if model.training and ratio_loss is not None and ratio_w > 0:
            scaled = ratio_loss * float(ratio_w)
            loss = loss + scaled
            logs["loss_ratio_raw"] = float(ratio_loss.detach().item())
            logs["loss_ratio"] = float(scaled.detach().item())
        else:
            logs["loss_ratio"] = 0.0

        logs["loss_total"] = float(loss.detach().item())
        self._last_loss_terms = logs

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        last_terms = getattr(self, "_last_loss_terms", None)
        if last_terms and "loss" in logs:
            logs.update(last_terms)
        super().log(logs, start_time=start_time)


def _ensure_dnalongbench_dataset(
    *,
    data_root: str,
    task_name: str,
    repo_id: str,
    hf_token: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> str:
    task_dir = {
        "eqtl_prediction": "eqtl",
        "enhancer_target_gene_prediction": "etgp",
    }.get(str(task_name))
    if not task_dir:
        raise ValueError(f"Unsupported task_name={task_name!r}")

    return resolve_hf_dataset_root(
        data_root,
        repo_id=repo_id,
        required_subdir=task_dir,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        path_resolver=to_absolute_path,
    )


def _auto_lora_targets(model: torch.nn.Module) -> list[str]:
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            candidates.append(name.split(".")[-1])
    priority = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "in_proj", "fc1", "fc2", "w1", "w2", "w3"]
    present = [p for p in priority if p in set(candidates)]
    if not present:
        from collections import Counter

        cnt = Counter(candidates)
        present = [k for k, _ in cnt.most_common(8)]
    return sorted(set(present))


def apply_lora_to_model(cfg: RunConfig, model: torch.nn.Module, logger) -> torch.nn.Module:
    if not bool(cfg.get("use_lora", False)):
        return model

    target_modules = str(cfg.get("lora_target_modules", "") or "").strip()
    if target_modules:
        target = [item.strip() for item in target_modules.split(",") if item.strip()]
    else:
        target = _auto_lora_targets(model)
        logger.print(f"[LoRA] auto target_modules={target}")

    task_type = TaskType.SEQ_CLS if str(cfg.get("lora_task_type", "SEQ_CLS")).upper() == "SEQ_CLS" else TaskType.CAUSAL_LM
    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=target,
        bias=str(cfg.get("lora_bias", "none")),
        task_type=task_type,
    )
    model = get_peft_model(model, lora_cfg)

    for name, param in model.named_parameters():
        if ("lora_" in name) or (".classifier." in name) or name.endswith("classifier.weight") or name.endswith("classifier.bias"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def _resolve_checkpoint_weight_path(checkpoint_path: str | None) -> str | None:
    if not checkpoint_path:
        return None
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    if not os.path.isdir(checkpoint_path):
        return None
    candidates = (
        "pytorch_model.bin",
        "pytorch_model.safetensors",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    )
    for name in candidates:
        path = os.path.join(checkpoint_path, name)
        if os.path.isfile(path):
            return path
    return None


def _load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
    *,
    strict: bool,
) -> bool:
    weight_path = _resolve_checkpoint_weight_path(checkpoint_path)
    if not weight_path:
        return False
    state_dict = load_state_dict(weight_path, device)
    model.load_state_dict(state_dict, strict=strict)
    return True


class CollatorForClassification:
    def __init__(self, pad: int) -> None:
        self.pad = int(pad)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.as_tensor(example["input_ids"], dtype=torch.long) for example in batch]
        labels = torch.stack([torch.as_tensor(example["labels"], dtype=torch.long) for example in batch])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad)
        attention_mask = input_ids.ne(self.pad)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class _HNetSequenceConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_return_dict = True
        self.return_dict = True
        self.problem_type = None
        self.tie_word_embeddings = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class HNetForSequenceClassification(torch.nn.Module):
    def __init__(
        self,
        *,
        backbone: torch.nn.Module,
        num_labels: int,
        pooling_method: str,
        pad_token_id: int,
        conjoin_test: bool = False,
        span_k: int = 1,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "backbone") or not hasattr(backbone, "embeddings"):
            raise ValueError("Expected an HNetForCausalLM-style backbone.")
        self.hnet = backbone
        self.num_labels = int(num_labels)
        self.pooling_method = str(pooling_method)
        self.pad_token_id = int(pad_token_id)
        self.conjoin_test = bool(conjoin_test)
        self.span_k = max(1, int(span_k))

        hidden_size = int(backbone.config.d_model[0])
        factory_kwargs = {
            "device": backbone.embeddings.weight.device,
            "dtype": backbone.embeddings.weight.dtype,
        }
        self.classifier = torch.nn.Linear(hidden_size, self.num_labels, **factory_kwargs)
        self.config = _HNetSequenceConfig(pad_token_id=self.pad_token_id, num_labels=self.num_labels)

    def freeze_encoder(self) -> None:
        for _name, param in self.hnet.embeddings.named_parameters():
            param.requires_grad = False
        for name, param in self.hnet.named_parameters():
            if ("routing_module" in name) or (".encoder." in name):
                param.requires_grad = False

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling_method == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hidden_states * mask).sum(dim=1) / denom
        if self.pooling_method == "last":
            if attention_mask is None:
                return hidden_states[:, -1]
            lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, lengths]
        if self.pooling_method == "coverage_aware_pooling":
            if attention_mask is None:
                attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
            batch_size, _, dim = hidden_states.shape
            outputs = []
            for idx in range(batch_size):
                valid = attention_mask[idx].to(dtype=torch.bool)
                sample_hidden = hidden_states[idx][valid]
                if sample_hidden.numel() == 0:
                    outputs.append(hidden_states.new_zeros((dim,)))
                    continue
                positions = torch.arange(sample_hidden.shape[0], device=sample_hidden.device, dtype=torch.long)
                start_bp = positions - (self.span_k - 1) if self.span_k > 1 else positions
                end_bp = positions + 1
                spans = torch.stack([start_bp, end_bp], dim=-1)
                seq_len = int(sample_hidden.shape[0])
                pooled = coverage_aware_pooling(sample_hidden, spans, bin_size=max(1, seq_len), num_bins=1)
                outputs.append(pooled.squeeze(0))
            return torch.stack(outputs, dim=0)
        raise ValueError(f"Unknown pooling_method: {self.pooling_method}")

    def _forward_once(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[list]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        mask = attention_mask.to(dtype=torch.bool)
        embeddings = self.hnet.embeddings(input_ids)
        hidden_states, bpred_output = self.hnet.backbone(embeddings, mask=mask, inference_params=None)
        pooled = self._pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        return logits, bpred_output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        bpred_output = None
        if self.conjoin_test and not self.training:
            input_ids_f, input_ids_rc = input_ids.chunk(2, dim=-1)
            input_ids_f = input_ids_f.squeeze(-1)
            input_ids_rc = input_ids_rc.squeeze(-1)
            if attention_mask is None:
                attention_mask = input_ids.ne(self.pad_token_id)
            mask_f, mask_rc = attention_mask.chunk(2, dim=-1)
            mask_f = mask_f.squeeze(-1)
            mask_rc = mask_rc.squeeze(-1)
            logits_f, _ = self._forward_once(input_ids=input_ids_f, attention_mask=mask_f)
            logits_rc, _ = self._forward_once(input_ids=input_ids_rc, attention_mask=mask_rc)
            logits = (logits_f + logits_rc) / 2.0
        else:
            logits, bpred_output = self._forward_once(input_ids=input_ids, attention_mask=attention_mask)

        if labels is None:
            return {"logits": logits}
        if self.num_labels == 1:
            labels = labels.to(dtype=logits.dtype)
            loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        out = {"logits": logits, "loss": loss}
        if self.training and bpred_output is not None:
            out["bpred_output"] = bpred_output
        return out


def build_hnet_backbone(
    *,
    checkpoint_dir: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
) -> tuple[torch.nn.Module, RunConfig]:
    cfg_path = resolve_hydra_cfg_path(checkpoint_dir)
    prev_cfg = RunConfig.from_path(cfg_path)
    cfg_dict = OmegaConf.to_container(prev_cfg.cfg, resolve=True) or {}
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}
    model_cfg = cfg_dict.get("model_cfg") or (cfg_dict.get("model") or {}).get("config") or cfg_dict
    hnet_kwargs = prepare_hnet_config_kwargs(model_cfg)
    if not hnet_kwargs.get("d_model"):
        raise ValueError("Model config missing d_model in checkpoint hydra_cfg.yaml")

    hnet_cfg = HNetConfig(**hnet_kwargs)
    use_floor = bool(cfg_dict.get("use_routing_floor", True))
    routing_floor = parse_routing_floor_config(cfg_dict) if use_floor else None

    # Default behavior matches the main-branch sweeps: always cap routing tokens to
    # avoid pathological memory blow-ups on 1.28M inputs.
    routing_ceiling = parse_routing_ceiling_config(cfg_dict)
    if not routing_ceiling.enabled():
        k_max = 40000 if dtype == torch.float32 else 70000
        num_stages = max(1, len(list(getattr(hnet_cfg, "d_model", []) or [])))
        forced_cfg = {"routing_ceiling": {"k_max_list": [k_max] * num_stages}}
        routing_ceiling = parse_routing_ceiling_config(forced_cfg)

    if routing_floor is not None and routing_floor.enabled() and routing_ceiling.enabled():
        model: torch.nn.Module = HNetForCausalLMWithRoutingFloorAndCeiling(
            hnet_cfg, routing_floor, routing_ceiling, device=device, dtype=dtype
        )
    elif routing_floor is not None and routing_floor.enabled():
        model = HNetForCausalLMWithRoutingFloor(hnet_cfg, routing_floor, device=device, dtype=dtype)
    elif routing_ceiling.enabled():
        model = HNetForCausalLMWithRoutingCeiling(hnet_cfg, routing_ceiling, device=device, dtype=dtype)
    else:
        model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)

    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=bool(strict))
    return model, prev_cfg


def _cleanup_checkpoints(output_dir: str, keep_dirs: set[str]) -> None:
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if os.path.abspath(path) in keep_dirs:
            continue
        shutil.rmtree(path, ignore_errors=True)


def _slice_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return str(value)


def _resolve_global_target_bp(cfg: RunConfig, prev_cfg: RunConfig | None) -> float:
    candidates = [
        ("cfg.bp_per_token", cfg.get("bp_per_token")),
        ("checkpoint hydra_cfg.yaml bp_per_token", prev_cfg.get("bp_per_token") if prev_cfg is not None else None),
        ("default", 128),
    ]
    for source, raw_value in candidates:
        value = _slice_optional_str(raw_value)
        if value is None:
            continue
        try:
            target = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {source}: {raw_value!r}") from exc
        if target <= 0:
            raise ValueError(f"{source} must be positive, got {target}")
        return target
    raise ValueError("Unable to resolve bp_per_token for compression regularization.")


@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg: RunConfig) -> None:
    cfg, logger = init_experiment(cfg, init_wandb=bool(cfg.get("use_wandb", True)))
    try:
        task_name = str(cfg.get("task_name"))
        if task_name not in TASK_SPECS:
            raise ValueError(f"Unsupported task_name={task_name!r}. Supported: {sorted(TASK_SPECS)}")

        tokenizer_name = str(cfg.get("tokenizer", "fast")).lower()
        if tokenizer_name == "singlenucleotide":
            tokenizer = SingleNucleotideTokenizer()
        elif tokenizer_name == "sixmer":
            tokenizer = SixMerTokenizer()
        elif tokenizer_name == "fast":
            tokenizer = FastSingleNucleotideTokenizer()
        else:
            raise ValueError(f"Unsupported tokenizer={tokenizer_name!r}. Use fast/singlenucleotide/sixmer.")

        span_k = int(getattr(tokenizer, "K", 1))
        if span_k < 1:
            raise ValueError("tokenizer K must be >= 1")

        data_root = _ensure_dnalongbench_dataset(
            data_root=str(cfg.get("data_root", "./data/dnalongbench")),
            task_name=task_name,
            repo_id=str(cfg.get("dnalongbench_repo", "andyjzhao/dnalongbench")),
            hf_token=_slice_optional_str(cfg.get("hf_token")),
            cache_dir=resolve_hf_cache_dir(cfg),
            local_files_only=bool(cfg.get("offline_mode", False)),
        )

        train_dataset, valid_dataset, test_dataset = load_data(
            root=data_root,
            task_name=task_name,
            cell_type=str(cfg.get("cell_type")),
            sequence_length=int(cfg.get("data_max_length", 1280000)),
            tokenizer=tokenizer,
            conjoin_test=bool(cfg.get("conjoin_test", False)),
            max_train_records=cfg.get("max_train_records"),
            max_valid_records=cfg.get("max_valid_records"),
            max_test_records=cfg.get("max_test_records"),
        )

        hnet_dtype = resolve_dtype(str(cfg.get("hnet_dtype", "bfloat16")))
        device_str = str(cfg.get("device", "cuda"))
        if device_str == "cpu" and hnet_dtype in (torch.float16, torch.bfloat16):
            hnet_dtype = torch.float32

        ckpt_value = _slice_optional_str(cfg.get("ckpt"))
        if not ckpt_value:
            raise ValueError("ckpt must be set to a local checkpoint path or a HF repo id.")
        ckpt_info = resolve_checkpoint_path(
            ckpt_value,
            default_owner=_slice_optional_str(cfg.get("hf_user")),
            token=_slice_optional_str(cfg.get("hf_token")),
            cache_dir=resolve_hf_cache_dir(cfg),
            path_resolver=to_absolute_path,
        )
        checkpoint_path = ckpt_info["file"]
        checkpoint_dir = ckpt_info["dir"]

        backbone, prev_cfg = build_hnet_backbone(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            device=torch.device(device_str),
            dtype=hnet_dtype,
            strict=bool(cfg.get("hnet_strict", False)),
        )
        global_target_bp = _resolve_global_target_bp(cfg, prev_cfg)

        pad_token_id = int(getattr(tokenizer, "pad", 0))
        spec = TASK_SPECS[task_name]
        model = HNetForSequenceClassification(
            backbone=backbone,
            num_labels=int(spec["num_labels"]),
            pooling_method=str(cfg.get("pooling_method", "mean")),
            pad_token_id=pad_token_id,
            conjoin_test=bool(cfg.get("conjoin_test", False)),
            span_k=span_k,
        ).to(device_str)

        model = apply_lora_to_model(cfg, model, logger)
        if bool(cfg.get("freeze_encoder", False)) and hasattr(model, "freeze_encoder"):
            model.freeze_encoder()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.print(f"Trainable params: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")
        logger.print(f"Compression target bp_per_token: {global_target_bp:g}")
        logger.print(f"Dataset sizes: {len(train_dataset)} train / {len(valid_dataset)} valid / {len(test_dataset)} test")

        training_cfg = OmegaConf.to_container(cfg.get("training") or {}, resolve=True) or {}
        if not isinstance(training_cfg, dict):
            raise ValueError("cfg.training must be a dict.")

        output_base = to_absolute_path(str(cfg.get("save_dir", cfg.dirs.output)))
        output_dir = os.path.join(output_base, f"{task_name}_{cfg.get('cell_type')}")
        os.makedirs(output_dir, exist_ok=True)

        training_cfg = dict(training_cfg)
        training_cfg.setdefault("per_device_train_batch_size", 1)
        training_cfg.setdefault("per_device_eval_batch_size", 1)
        training_cfg.setdefault("remove_unused_columns", False)
        training_cfg.setdefault("group_by_length", False)
        training_cfg.setdefault("save_total_limit", 20)

        training_cfg["report_to"] = []

        training_args = TrainingArguments(
            output_dir=output_dir,
            seed=int(cfg.get("seed", 0)),
            data_seed=int(cfg.get("seed", 0)),
            save_safetensors=False,
            **training_cfg,
        )

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_epochs", 10))),
            ExpLoggerCallback(logger),
        ]
        trainer = HNetClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=CollatorForClassification(pad=pad_token_id),
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            global_target_bp=global_target_bp,
            alpha=float(cfg.alpha),
        )

        if bool(training_cfg.get("do_train", True)):
            trainer.train()

        test_load = str(cfg.get("test_load", "auto")).lower()
        best_checkpoint = trainer.state.best_model_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        selected_checkpoint = None
        if test_load == "best":
            selected_checkpoint = best_checkpoint
        elif test_load == "last":
            selected_checkpoint = last_checkpoint
        elif test_load == "auto":
            selected_checkpoint = best_checkpoint or last_checkpoint
        elif test_load in {"current", "none", "skip"}:
            selected_checkpoint = None
        else:
            raise ValueError("test_load must be one of: auto, best, last, current, none, skip.")

        if selected_checkpoint:
            strict = not bool(cfg.get("use_lora", False))
            _load_checkpoint_into_model(model, selected_checkpoint, torch.device(device_str), strict=strict)

        valid_metrics = trainer.evaluate(valid_dataset, metric_key_prefix="valid")
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        logger.print({"valid": valid_metrics, "test": test_metrics})
        logger.update_summary({**valid_metrics, **test_metrics})

        if bool(cfg.get("cleanup_checkpoints", True)):
            keep = set()
            for ckpt in (best_checkpoint, last_checkpoint):
                if ckpt and os.path.isdir(ckpt):
                    keep.add(os.path.abspath(ckpt))
            _cleanup_checkpoints(output_dir, keep)
    finally:
        finish_experiment(cfg, logger)


if __name__ == "__main__":
    main()
