import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import io
import math

import hydra

from hydra.utils import to_absolute_path

from omegaconf import OmegaConf
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import Subset
from transformers import TrainingArguments, Trainer
from src.utils import timer

from src.genezip.hnet_train import (
    build_model,
    build_tokenizer,
    count_model_parameters,
    load_training_config,
)
from src.genezip.data import RegionAwareDNADataset
from src.genezip.region_aware_hnet import (
    build_region_target_bp_per_token,
    compute_hnet_lr_multipliers,
    RegionCompressionTracker,
    RegionLossTracker,
)
from src.utils.hf_utils import (
    ExpLoggerCallback,
    downsample_dataset_splits,
    load_hf_dataset_from_cfg,
    split_train_valid_test,
    EvalResumeCUDACleaner,
    PeriodicCUDACleaner,
    HFCheckpointUploader,
    resolve_checkpoint_path,
    resolve_hf_cache_dir,
    resolve_hf_repo_id,
)
from src.utils import finish_experiment, init_experiment, RunConfig, get_device
from src.utils.log import logger as base_logger


def load_state_dict(checkpoint_path: str, device: torch.device) -> dict[str, torch.Tensor]:
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
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


class GenomeLMStage1Trainer(Trainer):
    def __init__(
            self,
            *args,
            architecture: str,
            region_targets: dict[str, float],
            region_label_map: dict[str, int],
            global_target_bp: float,
            alpha: float = 1.0,
            reduction: str = "evo2",
            **kwargs,
    ):
        self.architecture = architecture
        self.reduction = reduction
        super().__init__(*args, **kwargs)
        if getattr(self.args, "metric_for_best_model", None) is None:
            self.args.metric_for_best_model = "eval_ppl"
            if getattr(self.args, "greater_is_better", None) is None:
                self.args.greater_is_better = False
        self._tokens_seen = None
        self.region_targets = region_targets
        self.region_label_map = region_label_map
        self.global_target_bp = float(global_target_bp)
        # Paper notation: constant alpha (no schedule).
        self.alpha = float(alpha)
        self._collect_region_stats = False
        self._region_tracker = RegionCompressionTracker(list(region_targets.keys()))
        self._collect_eval_stats = False
        self._eval_region_names = self._resolve_eval_region_names()
        self._region_loss_tracker = RegionLossTracker(self._eval_region_names)

    def _ensure_tokens_seen_initialized(self):
        if self._tokens_seen is not None:
            return
        self._tokens_seen = 0
        for record in reversed(self.state.log_history):
            if "trainer/tokens_seen" in record or "tokens_seen" in record:
                prior_tokens = record.get("trainer/tokens_seen") or record.get("tokens_seen")
                self._tokens_seen = prior_tokens
                break
            tokens_trained = record.get("tokens_trained")
            if tokens_trained is not None:
                try:
                    self._tokens_seen = float(tokens_trained) * 1e9
                except Exception:
                    self._tokens_seen = 0
                break

    def _count_target_tokens(self, inputs):
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return 0

        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            token_count = input_ids.numel()
        elif all(
                hasattr(tokenizer, attr)
                for attr in ("informative_min", "informative_max", "repetitive_offset")
        ):
            processed_ids = input_ids
            is_repetitive = processed_ids > tokenizer.informative_max
            processed_ids = torch.where(
                is_repetitive, processed_ids - tokenizer.repetitive_offset, processed_ids
            )
            labels = processed_ids[:, 1:]
            informative_positions = (labels >= tokenizer.informative_min) & (
                    labels <= tokenizer.informative_max
            )
            token_count = informative_positions.sum().item()
        else:
            token_count = input_ids.numel()

        return int(token_count * max(1, getattr(self.args, "world_size", 1)))

    def _region_mask(self, region_ids: torch.Tensor, region_name: str) -> torch.Tensor:
        if region_name not in self.region_label_map:
            raise KeyError(
                f"Unknown region_name '{region_name}' in region_label_map={list(self.region_label_map.keys())}"
            )
        target_id = self.region_label_map[region_name]
        return region_ids == target_id

    def _resolve_eval_region_names(self) -> list[str]:
        if not self.region_targets:
            return []
        return list(dict.fromkeys(self.region_targets.keys()))

    def _region_metric_name(self, region_name: str) -> str:
        return region_name

    def _rar_loss_weight(self) -> float:
        return max(0.0, float(self.alpha))

    def _compute_region_stats(self, boundary_mask, region_ids, boundary_prob=None, valid_mask=None):
        stats = {}
        if region_ids.dim() == 1:
            region_ids = region_ids.unsqueeze(0)
        if boundary_mask.dim() == 1:
            boundary_mask = boundary_mask.unsqueeze(0)
        if boundary_prob is not None and boundary_prob.dim() == 1:
            boundary_prob = boundary_prob.unsqueeze(0)
        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)
            valid_mask = valid_mask.bool()
        for region_name in self.region_targets:
            region_mask = self._region_mask(region_ids, region_name)
            if valid_mask is not None:
                region_mask = region_mask & valid_mask
            bp = region_mask.sum()
            tokens = (boundary_mask & region_mask).sum()
            if boundary_prob is not None:
                prob_sum = (boundary_prob * region_mask.float()).sum()
                stats[region_name] = (bp, tokens, prob_sum)
            else:
                stats[region_name] = (bp, tokens)
        return stats

    def _compute_region_bp(self, region_ids, valid_mask=None):
        stats = {}
        if region_ids.dim() == 1:
            region_ids = region_ids.unsqueeze(0)
        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)
            valid_mask = valid_mask.bool()
        for region_name in self.region_targets:
            region_mask = self._region_mask(region_ids, region_name)
            if valid_mask is not None:
                region_mask = region_mask & valid_mask
            stats[region_name] = region_mask.sum()
        return stats

    def _compute_unified_ratio_loss(
            self,
            boundary_mask,
            boundary_prob,
            region_ids,
            valid_mask=None,
            target_bp=None,
            region_targets=None,
    ):
        if target_bp is None:
            target_bp = self.global_target_bp
        if region_targets is None:
            region_targets = self.region_targets
        if target_bp <= 1.0 or not region_targets:
            return None

        if region_ids.dim() == 1:
            region_ids = region_ids.unsqueeze(0)
        if boundary_mask.dim() == 1:
            boundary_mask = boundary_mask.unsqueeze(0)
        if boundary_prob.dim() == 1:
            boundary_prob = boundary_prob.unsqueeze(0)
        if valid_mask is not None and valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        del target_bp

        losses = []
        for i in range(boundary_mask.shape[0]):
            region_losses = []
            region_weights = []
            b_mask = boundary_mask[i].float()
            p_prob = boundary_prob[i].float()
            v_mask = None
            if valid_mask is not None:
                v_mask = valid_mask[i].float()
                b_mask = b_mask * v_mask
                p_prob = p_prob * v_mask
            for region_name, region_target_bp in region_targets.items():
                n_c = float(region_target_bp)
                if n_c <= 1.0:
                    continue
                region_mask = self._region_mask(region_ids[i], region_name).float()
                if v_mask is not None:
                    region_mask = region_mask * v_mask
                l_c = region_mask.sum()
                if l_c <= 0:
                    continue
                f_c = (region_mask * b_mask).sum() / l_c
                g_c = (region_mask * p_prob).sum() / l_c
                loss_c = (
                        ((n_c - 1.0) * f_c * g_c + (1.0 - f_c) * (1.0 - g_c))
                        * (n_c / (n_c - 1.0))
                )
                region_losses.append(loss_c)
                region_weights.append(l_c)

            if not region_losses:
                continue
            weight_sum = torch.stack(region_weights).sum()
            if weight_sum <= 0:
                continue
            weighted = [
                loss * (weight / weight_sum) for loss, weight in zip(region_losses, region_weights)
            ]
            losses.append(torch.stack(weighted).sum())

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _extract_all_boundary_outputs(
            self,
            outputs,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.architecture != "hnet":
            return []
        bpred_output = getattr(outputs, "bpred_output", None)
        if not bpred_output:
            return []
        boundary_outputs = []
        for router_output in bpred_output:
            if isinstance(router_output, dict):
                boundary_mask_post = router_output.get("boundary_mask", None)
                boundary_prob_post = router_output.get("boundary_prob", None)
                boundary_mask_pre = router_output.get("boundary_mask_pre", None)
                boundary_prob_pre = router_output.get("boundary_prob_pre", None)
            else:
                boundary_mask_post = getattr(router_output, "boundary_mask", None)
                boundary_prob_post = getattr(router_output, "boundary_prob", None)
                boundary_mask_pre = getattr(router_output, "boundary_mask_pre", None)
                boundary_prob_pre = getattr(router_output, "boundary_prob_pre", None)

            if boundary_mask_post is None or boundary_prob_post is None:
                continue

            boundary_mask_loss = boundary_mask_pre if boundary_mask_pre is not None else boundary_mask_post
            boundary_prob_loss = boundary_prob_pre if boundary_prob_pre is not None else boundary_prob_post

            boundary_mask_loss = boundary_mask_loss.bool()
            boundary_mask_post = boundary_mask_post.bool()
            if boundary_prob_loss.dim() == boundary_mask_loss.dim() + 1:
                boundary_prob_loss = boundary_prob_loss[..., -1]
            boundary_prob_loss = boundary_prob_loss.float()

            boundary_outputs.append((boundary_mask_loss, boundary_prob_loss, boundary_mask_post))
        return boundary_outputs

    def _collect_router_metrics(self, outputs) -> dict[str, float]:
        if self.architecture != "hnet":
            return {}
        bpred_output = getattr(outputs, "bpred_output", None)
        if not bpred_output:
            return {}
        metrics: dict[str, float] = {}
        for stage_idx, router_output in enumerate(bpred_output):
            selected_tokens = None
            if isinstance(router_output, dict):
                boundary_mask = router_output.get("boundary_mask", None)
                selected_tokens = router_output.get("selected_tokens_pre", None)
            else:
                boundary_mask = getattr(router_output, "boundary_mask", None)
                selected_tokens = getattr(router_output, "selected_tokens_pre", None)

            if selected_tokens is not None:
                if torch.is_tensor(selected_tokens):
                    selected_tokens = float(selected_tokens.detach().cpu())
                else:
                    selected_tokens = float(selected_tokens)
            elif boundary_mask is not None:
                if boundary_mask.dim() == 1:
                    selected_tokens = float(boundary_mask.sum().item())
                else:
                    selected_tokens = float(boundary_mask.sum(dim=-1).float().mean().item())
            if selected_tokens is not None:
                metrics[f"router/selected_tokens_s{stage_idx}"] = selected_tokens

            floor_stats = getattr(router_output, "floor_stats", None)
            if not floor_stats:
                continue
            segments = float(floor_stats.get("segments", 0))
            triggers = float(floor_stats.get("triggers", 0))
            deficit = float(floor_stats.get("deficit", 0))
            trigger_rate = triggers / segments if segments > 0 else 0.0
            avg_deficit = deficit / max(triggers, 1.0)
            metrics[f"router/trigger_rate_s{stage_idx}"] = trigger_rate
            metrics[f"router/avg_deficit_s{stage_idx}"] = avg_deficit
        return metrics

    def _downsample_region_ids(self, region_ids, boundary_mask):
        if region_ids.dim() == 1:
            region_ids = region_ids.unsqueeze(0)
            boundary_mask = boundary_mask.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        num_tokens = boundary_mask.sum(dim=-1)
        max_tokens = int(num_tokens.max().item()) if num_tokens.numel() > 0 else 0
        if max_tokens <= 0:
            next_ids = region_ids.new_zeros((region_ids.shape[0], 0), dtype=region_ids.dtype)
            valid_mask = region_ids.new_zeros((region_ids.shape[0], 0), dtype=torch.bool)
        else:
            device = region_ids.device
            L = region_ids.shape[1]
            token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            next_ids = torch.gather(region_ids, dim=1, index=seq_sorted_indices[:, :max_tokens])
            valid_mask = torch.arange(max_tokens, device=device)[None, :] < num_tokens[:, None]
            if valid_mask.shape != next_ids.shape:
                valid_mask = valid_mask.expand_as(next_ids)
            next_ids = torch.where(valid_mask, next_ids, torch.zeros_like(next_ids))

        if squeeze:
            return next_ids.squeeze(0), valid_mask.squeeze(0)
        return next_ids, valid_mask

    def _compute_region_losses(self, outputs, inputs):
        boundary_outputs = self._extract_all_boundary_outputs(outputs)
        if not boundary_outputs:
            return None, {}

        region_ids = inputs.get("region_ids", None)
        if region_ids is None:
            return None, {}
        num_stages = max(1, len(boundary_outputs))
        stage_target_bp = float(self.global_target_bp) ** (1.0 / float(num_stages))
        stage_region_targets = {
            name: float(target) ** (1.0 / float(num_stages)) for name, target in self.region_targets.items()
        }

        losses = []
        stats = {}
        base_bp = None
        final_stats = None
        region_ids_stage = region_ids
        valid_mask_stage = None

        for stage_idx, (boundary_mask, boundary_prob, boundary_mask_post) in enumerate(boundary_outputs):
            # Align region ids length with boundary outputs (model operates on input_ids[:-1])
            if region_ids_stage.shape[-1] != boundary_mask.shape[-1]:
                region_ids_stage = region_ids_stage[..., : boundary_mask.shape[-1]]
                if valid_mask_stage is not None:
                    valid_mask_stage = valid_mask_stage[..., : boundary_mask.shape[-1]]

            if stage_idx == 0 and self._collect_region_stats:
                base_bp = self._compute_region_bp(region_ids_stage, valid_mask=valid_mask_stage)

            unified_loss = self._compute_unified_ratio_loss(
                boundary_mask,
                boundary_prob,
                region_ids_stage,
                valid_mask=valid_mask_stage,
                target_bp=stage_target_bp,
                region_targets=stage_region_targets,
            )
            if unified_loss is not None:
                losses.append(unified_loss)

            if stage_idx == len(boundary_outputs) - 1 and self._collect_region_stats:
                final_stats = self._compute_region_stats(
                    boundary_mask,
                    region_ids_stage,
                    boundary_prob=boundary_prob,
                    valid_mask=valid_mask_stage,
                )

            if stage_idx < len(boundary_outputs) - 1:
                downsample_mask = boundary_mask_post if boundary_mask_post is not None else boundary_mask
                region_ids_stage, valid_mask_stage = self._downsample_region_ids(region_ids_stage, downsample_mask)

        if self._collect_region_stats and base_bp is not None and final_stats is not None:
            merged = {}
            for region_name in self.region_targets:
                bp = base_bp.get(region_name, 0.0)
                final_values = final_stats.get(region_name)
                if final_values is None:
                    merged[region_name] = (bp, 0.0)
                    continue
                if len(final_values) > 2:
                    merged[region_name] = (bp, final_values[1], final_values[2])
                else:
                    merged[region_name] = (bp, final_values[1])
            stats = merged

        if not losses:
            return None, stats
        total_loss = torch.stack(losses).sum()
        return total_loss, stats

    def _extract_logits(self, outputs):
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            logits = outputs.get("logits", None)
            if logits is not None:
                return logits
        logits = getattr(outputs, "logits", None)
        if logits is not None:
            return logits
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return None

    def _prepare_eval_labels(self, inputs):
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None, None, None
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None or not all(
                hasattr(tokenizer, attr)
                for attr in ("informative_min", "informative_max", "repetitive_offset")
        ):
            labels = input_ids[:, 1:].clone()
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
            return labels, valid_mask, None

        is_repetitive = input_ids > tokenizer.informative_max
        adjusted_ids = torch.where(is_repetitive, input_ids - tokenizer.repetitive_offset, input_ids)
        labels = adjusted_ids[:, 1:].clone()
        valid_mask = (labels >= tokenizer.informative_min) & (labels <= tokenizer.informative_max)
        labels = labels.masked_fill(~valid_mask, -100)
        loss_weight = None
        if self.reduction == "evo2":
            is_repetitive = is_repetitive[:, 1:]
            loss_weight = 1.0 - is_repetitive.float() * 0.9
        return labels, valid_mask, loss_weight

    def _compute_eval_region_losses(self, outputs, inputs):
        if not self._eval_region_names:
            return {}
        logits = self._extract_logits(outputs)
        if logits is None:
            return {}
        region_ids = inputs.get("region_ids")
        if region_ids is None:
            return {}

        labels, valid_mask, loss_weight = self._prepare_eval_labels(inputs)
        if labels is None:
            return {}

        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if valid_mask is not None and valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if loss_weight is not None and loss_weight.dim() == 1:
            loss_weight = loss_weight.unsqueeze(0)
        if region_ids.dim() == 1:
            region_ids = region_ids.unsqueeze(0)

        seq_len = min(labels.shape[-1], logits.shape[-2])
        if seq_len <= 0:
            return {}

        logits = logits[..., :seq_len, :]
        labels = labels[..., :seq_len]
        if valid_mask is not None:
            valid_mask = valid_mask[..., :seq_len]
        if loss_weight is not None:
            loss_weight = loss_weight[..., :seq_len]

        if region_ids.shape[-1] == seq_len + 1:
            region_ids = region_ids[..., 1:]
        elif region_ids.shape[-1] != seq_len:
            region_ids = region_ids[..., :seq_len]

        per_token_loss = cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)
        if loss_weight is not None:
            per_token_loss = per_token_loss * loss_weight
        if valid_mask is None:
            valid_mask = labels.ne(-100)
        per_token_loss = per_token_loss * valid_mask.float()

        stats = {}
        for region_name in self._eval_region_names:
            region_mask = self._region_mask(region_ids, region_name)
            if region_mask.dim() == 1:
                region_mask = region_mask.unsqueeze(0)
            if region_mask.shape[-1] != seq_len:
                region_mask = region_mask[..., :seq_len]
            count = (valid_mask & region_mask).sum()
            if count.item() == 0:
                continue
            loss_sum = (per_token_loss * region_mask.float()).sum()
            stats[region_name] = (loss_sum, count)
        return stats

    def _compute_lm_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        tokenizer = self.processing_class
        input_ids = inputs["input_ids"]
        is_repetitive = input_ids > tokenizer.informative_max
        input_ids = torch.where(is_repetitive, input_ids - tokenizer.repetitive_offset, input_ids)

        labels = input_ids[:, 1:].clone()
        informative_positions = (labels >= tokenizer.informative_min) & (labels <= tokenizer.informative_max)
        labels[~informative_positions] = -100
        is_repetitive = is_repetitive[:, 1:]
        input_ids = input_ids[:, :-1]
        if self.architecture != "hnet":
            raise NotImplementedError(f"Unsupported architecture: {self.architecture}")

        # Provide a full True mask to avoid packed mode (seq_idx) which requires channel-last layout.
        mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        forward_code = getattr(model.forward, "__code__", None)
        if forward_code and "routing_step" in forward_code.co_varnames:
            routing_step = int(getattr(self.state, "global_step", 0) or 0)
            outputs = model(input_ids=input_ids, mask=mask, routing_step=routing_step)
        else:
            outputs = model(input_ids=input_ids, mask=mask)
        logits = outputs.logits

        if self.reduction == "none":
            loss = cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            )
            loss = loss.view(logits.shape[:-1])
            return (loss, outputs) if return_outputs else loss
        if self.reduction == "mean":
            loss = cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="mean",
            )
        elif self.reduction == "evo2":
            loss = cross_entropy(
                logits[informative_positions],
                labels[informative_positions],
                reduction="none",
            )
            loss_weight = 1 - is_repetitive * 0.9
            loss = loss * loss_weight[informative_positions]
            loss = loss.mean()
        else:
            raise NotImplementedError

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_count = None
        if model.training:
            self._ensure_tokens_seen_initialized()
            token_count = self._count_target_tokens(inputs)

        base_loss, outputs = self._compute_lm_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if model.training and token_count is not None:
            self._tokens_seen += token_count
        rl_weight = self._rar_loss_weight()
        loss_terms = {
            "loss_ce": float(base_loss.detach().item()) if hasattr(base_loss, "detach") else float(base_loss),
        }
        router_metrics = self._collect_router_metrics(outputs)
        if router_metrics:
            loss_terms.update(router_metrics)
        extra_loss, region_stats = self._compute_region_losses(outputs, inputs)
        loss = base_loss
        if model.training and extra_loss is not None and float(rl_weight) > 0.0:
            scaled_loss = extra_loss * float(rl_weight)
            loss = loss + scaled_loss
            loss_terms["loss_region"] = float(scaled_loss.detach().item())
        else:
            loss_terms["loss_region"] = 0.0

        loss_terms["comp/alpha"] = float(rl_weight)
        loss_terms["loss_total"] = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
        # cache for logging hook
        self._last_loss_terms = loss_terms

        if self._collect_region_stats and region_stats:
            self._region_tracker.update(region_stats)

        if self._collect_eval_stats and not model.training:
            eval_region_losses = self._compute_eval_region_losses(outputs, inputs)
            if eval_region_losses:
                self._region_loss_tracker.update(eval_region_losses)

        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._collect_region_stats = True
        self._collect_eval_stats = True
        self._region_tracker.reset()
        self._region_loss_tracker.reset()
        prev_drop_last = getattr(self.args, "dataloader_drop_last", False)
        self.args.dataloader_drop_last = True
        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.args.dataloader_drop_last = prev_drop_last
        region_metrics = self._region_tracker.metrics(prefix=f"{metric_key_prefix}_")
        extra_metrics = {}
        if region_metrics:
            extra_metrics.update(region_metrics)
        loss_metrics = self._region_loss_tracker.metrics(
            prefix=f"{metric_key_prefix}_",
            name_mapper=self._region_metric_name,
        )
        if loss_metrics:
            extra_metrics.update(loss_metrics)
        summary_metrics = self._region_tracker.summary_metrics(
            prefix=f"{metric_key_prefix}_",
            name_mapper=self._region_metric_name,
            region_names=self._eval_region_names,
        )
        if summary_metrics:
            extra_metrics.update(summary_metrics)
        if extra_metrics:
            metrics.update(extra_metrics)
            self.log(extra_metrics)
        self._collect_region_stats = False
        self._collect_eval_stats = False
        if torch.cuda.is_available():
            # Clear evaluation leftovers to reduce fragmentation in later train steps.
            torch.cuda.empty_cache()
        return metrics

    def log(self, logs, start_time=None):
        # Inject last recorded loss breakdown into logs when available.
        last_terms = getattr(self, "_last_loss_terms", None)
        if last_terms and "loss" in logs:
            logs.update(last_terms)

        self._ensure_tokens_seen_initialized()

        eval_loss = logs.pop("eval_loss", None)
        if eval_loss is not None:
            logs["eval_ppl"] = math.exp(eval_loss)

        for key in list(logs.keys()):
            if "samples_per_second" in key or "steps_per_second" in key:
                logs.pop(key, None)

        tokens_seen = logs.pop("trainer/tokens_seen", None)
        if tokens_seen is None:
            tokens_seen = logs.pop("tokens_seen", None)
        if tokens_seen is None and self._tokens_seen is not None:
            tokens_seen = self._tokens_seen
        if tokens_seen is not None:
            logs["tokens_trained"] = float(tokens_seen) / 1e9

        # Normalize key names for cleaner logging.
        if "learning_rate" in logs:
            logs["lr"] = logs.pop("learning_rate")

        super().log(logs, start_time=start_time)

    def _save(self, output_dir, **kwargs):
        # H-Net ties embedding and lm_head weights, which safetensors rejects.
        if self.architecture == "hnet" and getattr(self.args, "save_safetensors", True):
            prev = self.args.save_safetensors
            self.args.save_safetensors = False
            try:
                super()._save(output_dir=output_dir, **kwargs)
            finally:
                self.args.save_safetensors = prev
        else:
            super()._save(output_dir=output_dir, **kwargs)


class HNetStage1Trainer(GenomeLMStage1Trainer):
    """
    Trainer that uses HNet's native parameter grouping to honor stage-wise LR multipliers
    and weight decay settings.
    """

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        try:
            from src.hnet.utils.train import group_params
        except Exception as exc:
            base_logger.warning(f"[hnet] fallback to default optimizer; group_params unavailable: {exc}")
            return super().create_optimizer()

        base_lr = self.args.learning_rate
        beta1, beta2 = self.args.adam_beta1, self.args.adam_beta2
        eps = self.args.adam_epsilon
        weight_decay_default = self.args.weight_decay

        param_groups = group_params(self.model)
        for group in param_groups:
            lr_mult = group.pop("lr_multiplier", 1.0)
            try:
                lr_mult_val = float(lr_mult)
            except (TypeError, ValueError):
                lr_mult_val = 1.0
            group["lr"] = base_lr * lr_mult_val
            if "weight_decay" not in group or group["weight_decay"] is None:
                group["weight_decay"] = weight_decay_default

        import torch

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay_default,
        )
        return self.optimizer


def prepare_stage1_datasets(cfg, tokenizer, logger):
    dataset_cfg = cfg.dataset
    hf_token = cfg.hf_token
    valid_test_downsample = cfg.valid_test_downsample
    dataset = load_hf_dataset_from_cfg(
        dataset_cfg,
        hf_token=hf_token,
        path_resolver=to_absolute_path,
        cache_dir=resolve_hf_cache_dir(cfg),
        local_files_only=bool(cfg.get("offline_mode", False)),
    )
    dataset = downsample_dataset_splits(dataset, valid_test_downsample, seed=cfg.seed)

    data_source = dataset_cfg.type
    if data_source != "refseq":
        raise NotImplementedError(f"Stage-1 loader currently supports dataset.type=refseq, got {data_source}")

    max_length = cfg.max_length

    train_split, valid_split, _ = split_train_valid_test(
        dataset, seed=cfg.seed, valid_size=0.05
    )

    train_dataset = RegionAwareDNADataset(tokenizer, train_split, max_length)
    valid_dataset = RegionAwareDNADataset(tokenizer, valid_split, max_length)
    return train_dataset, valid_dataset


@timer()
@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg: RunConfig) -> None:
    cfg, logger = init_experiment(cfg)

    torch.serialization.add_safe_globals([io.BytesIO])

    device = get_device(cfg)
    model, model_config_dict = build_model(cfg, device)
    tokenizer = build_tokenizer(cfg)

    pretrained_ckpt = cfg.pretrained_ckpt

    if str(cfg.model.name).startswith("hnet"):
        lr_multipliers = compute_hnet_lr_multipliers(model, model_config_dict)
        for stage_idx, mult in enumerate(lr_multipliers):
            logger.print(f"[hnet] lr_multiplier[stage={stage_idx}]={mult}")
        if len(lr_multipliers) != len(model.config.d_model):
            raise ValueError(
                f"hnet_lr_multiplier length ({len(lr_multipliers)}) must match number of stages ("
                f"{len(model.config.d_model)})"
            )
        model.apply_lr_multiplier(lr_multipliers)

    if pretrained_ckpt:
        ckpt_info = resolve_checkpoint_path(
            str(pretrained_ckpt),
            default_owner=cfg.hf_user,
            token=cfg.hf_token,
            cache_dir=resolve_hf_cache_dir(cfg),
            path_resolver=to_absolute_path,
        )
        logger.print(f"[stage1] loading pretrained checkpoint: {ckpt_info['file']}")
        state_dict = load_state_dict(ckpt_info["file"], device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"[stage1] pretrained_ckpt load mismatch: missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )
        logger.print("[stage1] pretrained checkpoint loaded.")
    else:
        initializer_range = float(cfg.training.hnet_initializer_range)
        if hasattr(model, "init_weights"):
            model.init_weights(initializer_range=initializer_range)

    region_label_map = OmegaConf.to_container(cfg.region_label_map, resolve=True)
    region_targets, region_label_map = build_region_target_bp_per_token(
        region_info=cfg.region_info,
        bp_per_token=cfg.bp_per_token,
        region_label_map=region_label_map,
    )
    logger.print(f"[stage1] region targets (bp/token): {region_targets}")
    logger.print(f"[stage1] region label map: {region_label_map}")
    num_stages = max(1, len(model.config.d_model) - 1)
    bp_per_token = float(cfg.bp_per_token)
    per_stage_bp = bp_per_token ** (1.0 / float(num_stages))
    stage_targets = [per_stage_bp for _ in range(num_stages)]
    logger.print(
        f"[stage1] compression_stages={num_stages}, per_stage_bp_per_token={stage_targets}"
    )

    train_dataset, valid_dataset = prepare_stage1_datasets(cfg, tokenizer, logger)

    training_config_dict = load_training_config(cfg)
    # Remove non-HF argument to avoid passing into TrainingArguments.
    max_eval_samples = cfg.max_eval_samples

    raw_params, effective_params = count_model_parameters(model)
    logger.critical(
        f"Total Parameters (raw): {raw_params} / Effective (deduped ties): {effective_params}"
    )
    logger.print(f"Dataset sizes: {len(train_dataset)} train / {len(valid_dataset)} valid")

    def maybe_cap_eval(ds):
        cap = int(max_eval_samples)
        if cap <= 0 or len(ds) <= cap:
            return ds
        logger.print(f"[stage1] limiting eval dataset to {cap} samples (from {len(ds)})")
        return Subset(ds, range(cap))

    if max_eval_samples is not None and valid_dataset is not None:
        valid_dataset = maybe_cap_eval(valid_dataset)

    training_args = TrainingArguments(
        output_dir=cfg.dirs.output,
        **training_config_dict,
    )

    trainer = HNetStage1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        architecture=cfg.model.arch,
        region_targets=region_targets,
        region_label_map=region_label_map,
        global_target_bp=cfg.bp_per_token,
        alpha=cfg.alpha,
    )

    trainer.add_callback(ExpLoggerCallback(logger))
    trainer.add_callback(EvalResumeCUDACleaner())
    trainer.add_callback(PeriodicCUDACleaner(every_steps=10))

    if cfg.upload_to_hf:
        if not cfg.hf_token:
            raise ValueError(
                "hf_token is required when upload_to_hf=true; set HUGGINGFACE_HUB_TOKEN or cfg.hf_token."
            )
        repo_id = resolve_hf_repo_id(cfg)
        trainer.add_callback(HFCheckpointUploader(repo_id, cfg.hf_token, cfg.private, logger))
        logger.print(f"[hf] upload enabled: repo={repo_id}")

    trainer.train()
    finish_experiment(cfg, logger, effective_params=effective_params)


if __name__ == "__main__":
    main()
