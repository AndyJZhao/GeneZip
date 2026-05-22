import math
import os
from typing import Optional

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from hydra.utils import to_absolute_path
from torch.utils.data import ConcatDataset, DataLoader

from src.downstream.dlb_data import (
    TISP_SEQUENCE_LENGTH,
    TISP_WINDOW_STEP,
    TISPChromosomeSlidingWindowDataset,
    build_tisp_loaders,
)
from src.downstream.tokenize import build_tokenizer
from src.downstream.utils import build_model, resolve_dtype
from src.utils import RunConfig, finish_experiment, init_experiment
from src.utils.hf_utils import (
    resolve_checkpoint_path,
    resolve_hf_cache_dir,
    resolve_hf_dataset_root,
)


DNA_ALPHABET = np.asarray([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)
TISP_ASSAY_GROUPS = {
    "fc": (0, 9, "cage_like"),
    "ec": (1, 8, "cage_like"),
    "er": (2, 7, "cage_like"),
    "gc": (3, 6, "natlog_like"),
    "pc": (4, 5, "natlog_like"),
}
TISP_EVAL_TRIM_BP = (TISP_SEQUENCE_LENGTH - TISP_WINDOW_STEP) // 2
TISP_VALID_MASK_FLANK_BP = 500


class PearsonAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.reshape(-1).detach().to(device="cpu", dtype=torch.float64)
        y = y.reshape(-1).detach().to(device="cpu", dtype=torch.float64)
        valid = torch.isfinite(x) & torch.isfinite(y)
        if not bool(valid.any()):
            return
        x = x[valid]
        y = y[valid]
        self.count += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_x2 += float(x.square().sum().item())
        self.sum_y2 += float(y.square().sum().item())
        self.sum_xy += float((x * y).sum().item())

    def compute(self) -> float:
        if self.count <= 1:
            return float("nan")
        count = float(self.count)
        cov = self.sum_xy - (self.sum_x * self.sum_y / count)
        var_x = self.sum_x2 - (self.sum_x * self.sum_x / count)
        var_y = self.sum_y2 - (self.sum_y * self.sum_y / count)
        if var_x <= 0.0 or var_y <= 0.0:
            return float("nan")
        return float(cov / math.sqrt(var_x * var_y))


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def decode_one_hot_sequence(one_hot: object) -> str:
    array = _to_numpy(one_hot).astype(np.float32, copy=False)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected one-hot DNA array shaped [L, 4], got {array.shape}.")
    token_idx = array.argmax(axis=-1)
    token_score = array.max(axis=-1)
    chars = np.full(array.shape[0], ord("N"), dtype=np.uint8)
    confident = token_score > 0.5
    chars[confident] = DNA_ALPHABET[token_idx[confident]]
    return chars.tobytes().decode("ascii")


def _normalize_one_hot_length(one_hot: object, sequence_length: int) -> np.ndarray:
    array = _to_numpy(one_hot).astype(np.float32, copy=False)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected one-hot DNA array shaped [L, 4], got {array.shape}.")
    current_length = int(array.shape[0])
    if current_length == int(sequence_length):
        return array
    drift = current_length - int(sequence_length)
    if abs(drift) > 1:
        raise ValueError(
            f"TISP one-hot length drift {current_length} is too large for expected length {sequence_length}."
        )
    if drift > 0:
        return array[:sequence_length, :]
    pad = np.zeros((-drift, array.shape[1]), dtype=array.dtype)
    return np.concatenate((array, pad), axis=0)


def _normalize_raw_target_length(target: object, sequence_length: int) -> np.ndarray:
    array = _to_numpy(target).astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Expected raw TISP target shaped [10, L] or [L, 10], got {array.shape}.")
    if array.shape[0] == 10:
        axis = 1
    elif array.shape[1] == 10:
        axis = 0
    else:
        raise ValueError(f"Expected raw TISP target shaped [10, L] or [L, 10], got {array.shape}.")
    current_length = int(array.shape[axis])
    if current_length == int(sequence_length):
        return array
    drift = current_length - int(sequence_length)
    if abs(drift) > 1:
        raise ValueError(
            f"TISP raw target length drift {current_length} is too large for expected length {sequence_length}."
        )
    slicer = [slice(None)] * array.ndim
    if drift > 0:
        slicer[axis] = slice(0, sequence_length)
        return array[tuple(slicer)]
    pad_shape = list(array.shape)
    pad_shape[axis] = -drift
    pad = np.zeros(tuple(pad_shape), dtype=array.dtype)
    return np.concatenate((array, pad), axis=axis)


def collate_tisp_raw_batch(batch):
    inputs = [
        torch.as_tensor(_normalize_one_hot_length(sample[0], TISP_SEQUENCE_LENGTH), dtype=torch.float32)
        for sample in batch
    ]
    targets = [
        torch.as_tensor(_normalize_raw_target_length(sample[1], TISP_SEQUENCE_LENGTH), dtype=torch.float32)
        for sample in batch
    ]
    return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)


def _normalize_sequence_length(sequence: str, max_length: int) -> str:
    current_length = len(sequence)
    if current_length == int(max_length):
        return sequence
    drift = current_length - int(max_length)
    if abs(drift) > 1:
        raise ValueError(
            f"TISP sequence length drift {current_length} is too large for expected length {max_length}."
        )
    if drift > 0:
        return sequence[:max_length]
    return sequence + ("N" * (-drift))


def _normalize_label_length(labels: torch.Tensor, sequence_length: int) -> torch.Tensor:
    current_length = int(labels.shape[1])
    if current_length == int(sequence_length):
        return labels
    drift = current_length - int(sequence_length)
    if abs(drift) > 1:
        raise ValueError(
            f"TISP label length drift {current_length} is too large for expected length {sequence_length}."
        )
    if drift > 0:
        return labels[:, :sequence_length, :]
    pad = labels.new_zeros((labels.shape[0], -drift, labels.shape[2]))
    return torch.cat((labels, pad), dim=1)


def tokenize_sequence(tokenizer, sequence: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.as_tensor(tokenizer(sequence), dtype=torch.long)
    if int(input_ids.numel()) != int(max_length):
        raise ValueError(
            "TISP requires a length-preserving tokenizer. "
            f"Got {input_ids.numel()} tokens for sequence_length={max_length}."
        )
    return input_ids, torch.ones_like(input_ids, dtype=torch.bool)


def prepare_tisp_batch(
    raw_batch,
    *,
    tokenizer,
    sequence_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs, targets = raw_batch
    input_array = _to_numpy(inputs)
    if input_array.ndim == 2:
        input_array = input_array[None, ...]
    target_array = _to_numpy(targets)
    if target_array.ndim == 2:
        target_array = target_array[None, ...]

    sequences = [
        _normalize_sequence_length(decode_one_hot_sequence(sample), sequence_length)
        for sample in input_array
    ]
    encoded = [tokenize_sequence(tokenizer, sequence, sequence_length) for sequence in sequences]
    input_ids = torch.stack([item[0] for item in encoded], dim=0).to(device=device, dtype=torch.long)
    attention_mask = torch.stack([item[1] for item in encoded], dim=0).to(device=device, dtype=torch.bool)

    valid_lengths = {int(sequence_length) - 1, int(sequence_length), int(sequence_length) + 1}
    label_batches: list[torch.Tensor] = []
    for target in target_array:
        labels = torch.as_tensor(target, dtype=torch.float32, device=device)
        if labels.ndim == 2:
            labels = labels.unsqueeze(0)
        if labels.ndim != 3:
            raise ValueError(f"Expected TISP labels shaped [B, 10, L] or [B, L, 10], got {tuple(labels.shape)}.")
        if labels.shape[1] == 10 and labels.shape[2] in valid_lengths:
            labels = labels.transpose(1, 2)
        elif labels.shape[1] in valid_lengths and labels.shape[2] == 10:
            pass
        else:
            raise ValueError(
                f"Unexpected TISP label shape {tuple(labels.shape)} for sequence_length={sequence_length}."
            )
        label_batches.append(_normalize_label_length(labels, sequence_length).squeeze(0))
    labels = torch.stack(label_batches, dim=0)

    if input_ids.shape[1] != labels.shape[1]:
        raise ValueError(
            f"Token length {input_ids.shape[1]} does not match label length {labels.shape[1]}."
        )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _transform_tisp_assay_values(values: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "cage_like":
        return torch.log10(torch.pow(10.0, values) - 1.0 + 0.1)
    if transform_name == "natlog_like":
        return values / math.log(10.0)
    raise ValueError(f"Unknown TISP assay transform {transform_name!r}.")


def _new_tisp_assay_accumulators() -> dict[str, PearsonAccumulator]:
    return {name: PearsonAccumulator() for name in TISP_ASSAY_GROUPS}


def update_tisp_assay_accumulators(
    accumulators: dict[str, PearsonAccumulator],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    for assay_name, (plus_idx, minus_idx, transform_name) in TISP_ASSAY_GROUPS.items():
        for track_idx in (plus_idx, minus_idx):
            pred_values = _transform_tisp_assay_values(predictions[:, :, track_idx], transform_name)
            label_values = _transform_tisp_assay_values(labels[:, :, track_idx], transform_name)
            if valid_mask is not None:
                mask = valid_mask
                if mask.dim() == pred_values.dim() - 1:
                    mask = mask.unsqueeze(0)
                mask = mask.to(device=pred_values.device, dtype=torch.bool)
                pred_values = pred_values[mask]
                label_values = label_values[mask]
            accumulators[assay_name].update(pred_values, label_values)


def finalize_tisp_assay_metrics(accumulators: dict[str, PearsonAccumulator]) -> dict[str, float]:
    metrics = {f"{name}_pcc": accumulator.compute() for name, accumulator in accumulators.items()}
    assay_values = np.asarray(list(metrics.values()), dtype=np.float64)
    metrics["avg_pcc"] = float(np.nanmean(assay_values)) if np.isfinite(assay_values).any() else float("nan")
    return metrics


def compute_tisp_task_loss(log_rates: torch.Tensor, labels: torch.Tensor, loss_name: str) -> torch.Tensor:
    normalized = str(loss_name).strip().lower()
    if normalized in {"poisson", "poisson_nll", "poisson_nll_loss"}:
        return F.poisson_nll_loss(log_rates, labels, log_input=True, full=False, reduction="mean")
    if normalized in {"pseudo_poisson_kl", "pseudo-poisson-kl"}:
        rates = torch.exp(log_rates)
        return (labels * torch.log((labels + 1e-5) / (rates + 1e-5)) + rates - labels).mean()
    raise ValueError(f"Unsupported TISP loss {loss_name!r}.")


def evaluate_tisp_posthoc_metrics(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    tokenizer,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float]:
    dataset_specs = _build_tisp_official_eval_specs(data_loader.dataset)
    if dataset_specs is None:
        return _evaluate_tisp_window_posthoc_metrics(
            model=model,
            data_loader=data_loader,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            device=device,
        )

    was_training = model.training
    model.eval()
    accumulators = _new_tisp_assay_accumulators()
    chrom_n_positions_cache: dict[tuple[int, str, int], np.ndarray] = {}
    spec_index = 0
    current_chrom_key: Optional[tuple[int, str, int]] = None
    current_n_positions: Optional[np.ndarray] = None
    pending_segment: Optional[dict[str, object]] = None

    with torch.no_grad():
        for raw_batch in data_loader:
            batch = prepare_tisp_batch(
                raw_batch,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                device=device,
            )
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = torch.exp(output["logits"].float()).detach().cpu()
            labels = batch["labels"].detach().cpu()

            for batch_idx in range(predictions.shape[0]):
                if spec_index >= len(dataset_specs):
                    raise ValueError("TISP official eval saw more predictions than dataset windows.")
                spec = dataset_specs[spec_index]
                spec_index += 1

                chrom_key = spec["chrom_key"]
                if current_chrom_key != chrom_key:
                    if pending_segment is not None:
                        if current_n_positions is None:
                            raise ValueError("Missing N-position cache for pending TISP segment.")
                        _flush_tisp_official_segment(
                            accumulators=accumulators,
                            predictions=pending_segment["predictions"],
                            labels=pending_segment["labels"],
                            segment_start=int(pending_segment["segment_start"]),
                            segment_end=int(pending_segment["segment_end"]),
                            n_positions=current_n_positions,
                        )
                        pending_segment = None

                    current_chrom_key = chrom_key
                    current_n_positions = chrom_n_positions_cache.get(chrom_key)
                    if current_n_positions is None:
                        current_n_positions = _get_tisp_unknown_positions(
                            genome=spec["genome"],
                            chrom=spec["chrom"],
                            chrom_len=int(spec["chrom_len"]),
                        )
                        chrom_n_positions_cache[chrom_key] = current_n_positions

                window_start = int(spec["window_start"])
                segment_start = window_start + TISP_EVAL_TRIM_BP
                segment_end = window_start + TISP_SEQUENCE_LENGTH - TISP_EVAL_TRIM_BP
                center_slice = slice(TISP_EVAL_TRIM_BP, TISP_SEQUENCE_LENGTH - TISP_EVAL_TRIM_BP)
                current_segment = {
                    "predictions": predictions[batch_idx, center_slice, :],
                    "labels": labels[batch_idx, center_slice, :],
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                }

                if pending_segment is not None:
                    pending_end = int(pending_segment["segment_end"])
                    pending_start = int(pending_segment["segment_start"])
                    if segment_start > pending_end:
                        raise ValueError(
                            f"TISP official eval found a gap between segments: {pending_start}:{pending_end} "
                            f"then {segment_start}:{segment_end}."
                        )
                    flush_end = min(pending_end, segment_start)
                    if flush_end > pending_start:
                        flush_len = flush_end - pending_start
                        if current_n_positions is None:
                            raise ValueError("Missing N-position cache for TISP flush.")
                        _flush_tisp_official_segment(
                            accumulators=accumulators,
                            predictions=pending_segment["predictions"][:flush_len, :],
                            labels=pending_segment["labels"][:flush_len, :],
                            segment_start=pending_start,
                            segment_end=flush_end,
                            n_positions=current_n_positions,
                        )

                pending_segment = current_segment

    if spec_index != len(dataset_specs):
        raise ValueError(
            f"TISP official eval consumed {spec_index} predictions for {len(dataset_specs)} dataset windows."
        )
    if pending_segment is not None:
        if current_n_positions is None:
            raise ValueError("Missing N-position cache for final TISP segment.")
        _flush_tisp_official_segment(
            accumulators=accumulators,
            predictions=pending_segment["predictions"],
            labels=pending_segment["labels"],
            segment_start=int(pending_segment["segment_start"]),
            segment_end=int(pending_segment["segment_end"]),
            n_positions=current_n_positions,
        )

    if was_training:
        model.train()
    return finalize_tisp_assay_metrics(accumulators)


def _evaluate_tisp_window_posthoc_metrics(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    tokenizer,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    accumulators = _new_tisp_assay_accumulators()

    with torch.no_grad():
        for raw_batch in data_loader:
            batch = prepare_tisp_batch(
                raw_batch,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                device=device,
            )
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = torch.exp(output["logits"].float())
            update_tisp_assay_accumulators(accumulators, predictions, batch["labels"])

    if was_training:
        model.train()
    return finalize_tisp_assay_metrics(accumulators)


def _build_tisp_official_eval_specs(dataset) -> Optional[list[dict[str, object]]]:
    chrom_datasets: list[TISPChromosomeSlidingWindowDataset]
    if isinstance(dataset, TISPChromosomeSlidingWindowDataset):
        chrom_datasets = [dataset]
    elif isinstance(dataset, ConcatDataset) and all(
        isinstance(inner, TISPChromosomeSlidingWindowDataset) for inner in dataset.datasets
    ):
        chrom_datasets = list(dataset.datasets)
    else:
        return None

    specs: list[dict[str, object]] = []
    for chrom_dataset in chrom_datasets:
        expected_last_start = int(chrom_dataset.chrom_len) - int(chrom_dataset.window_size)
        window_starts = list(getattr(chrom_dataset, "window_starts", []))
        if not window_starts or len(window_starts) != len(chrom_dataset):
            return None
        if window_starts[-1] != expected_last_start:
            return None
        for window_start in window_starts:
            specs.append(
                {
                    "chrom": chrom_dataset.chrom,
                    "chrom_len": int(chrom_dataset.chrom_len),
                    "genome": chrom_dataset.genome,
                    "window_start": int(window_start),
                    "chrom_key": (id(chrom_dataset.genome), chrom_dataset.chrom, int(chrom_dataset.chrom_len)),
                }
            )
    return specs


def _get_tisp_unknown_positions(*, genome, chrom: str, chrom_len: int) -> np.ndarray:
    sequence = genome.get_sequence_from_coords(chrom, 0, chrom_len)
    if len(sequence) != int(chrom_len):
        raise ValueError(
            f"TISP genome returned sequence length {len(sequence)} for {chrom}:{chrom_len}, "
            f"expected {chrom_len}."
        )
    sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return np.flatnonzero((sequence_bytes == ord("N")) | (sequence_bytes == ord("n"))).astype(np.int64, copy=False)


def _make_tisp_segment_valid_mask(
    *,
    n_positions: np.ndarray,
    segment_start: int,
    segment_end: int,
) -> torch.Tensor:
    valid_mask = np.ones(segment_end - segment_start, dtype=np.bool_)
    if n_positions.size == 0:
        return torch.from_numpy(valid_mask)

    left = int(np.searchsorted(n_positions, segment_start - TISP_VALID_MASK_FLANK_BP, side="left"))
    right = int(np.searchsorted(n_positions, segment_end + TISP_VALID_MASK_FLANK_BP, side="left"))
    for n_position in n_positions[left:right]:
        invalid_start = max(segment_start, int(n_position) - TISP_VALID_MASK_FLANK_BP)
        invalid_end = min(segment_end, int(n_position) + TISP_VALID_MASK_FLANK_BP + 1)
        valid_mask[invalid_start - segment_start : invalid_end - segment_start] = False
    return torch.from_numpy(valid_mask)


def _flush_tisp_official_segment(
    *,
    accumulators: dict[str, PearsonAccumulator],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    segment_start: int,
    segment_end: int,
    n_positions: np.ndarray,
) -> None:
    if segment_end <= segment_start:
        return
    valid_mask = _make_tisp_segment_valid_mask(
        n_positions=n_positions,
        segment_start=segment_start,
        segment_end=segment_end,
    )
    if not bool(valid_mask.any()):
        return
    update_tisp_assay_accumulators(
        accumulators,
        predictions.unsqueeze(0),
        labels.unsqueeze(0),
        valid_mask=valid_mask.unsqueeze(0),
    )


class HNetForTISPRegression(nn.Module):
    def __init__(
        self,
        hnet: nn.Module,
        *,
        num_outputs: int,
        use_layer_norm: bool,
        head_dropout: float,
        routing_step: Optional[int],
    ) -> None:
        super().__init__()
        if not hasattr(hnet, "backbone") or not hasattr(hnet, "embeddings"):
            raise ValueError("Expected HNetForCausalLM-style checkpoint for GeneZip/HNet TISP.")
        self.hnet = hnet
        self.routing_step = routing_step
        hidden_dim = int(hnet.config.d_model[0])
        factory_kwargs = {
            "device": hnet.embeddings.weight.device,
            "dtype": hnet.embeddings.weight.dtype,
        }
        self.norm = nn.LayerNorm(hidden_dim, **factory_kwargs) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(float(head_dropout))
        self.head = nn.Linear(hidden_dim, int(num_outputs), **factory_kwargs)

    def freeze_encoder(self) -> None:
        for param in self.hnet.embeddings.parameters():
            param.requires_grad = False
        for name, param in self.hnet.named_parameters():
            if "routing_module" in name or ".encoder." in name:
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        mask = attention_mask.to(dtype=torch.bool)
        embeddings = self.hnet.embeddings(input_ids)
        if hasattr(self.hnet, "_set_routing_step"):
            self.hnet._set_routing_step(self.routing_step)
        hidden_states, bpred_output = self.hnet.backbone(embeddings, mask=mask, inference_params=None)
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                f"Backbone output shape {tuple(hidden_states.shape)} does not align with token ids {tuple(input_ids.shape)}."
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return {"logits": self.head(hidden_states), "bpred_output": bpred_output}


def build_tisp_model(cfg: RunConfig, device: torch.device) -> nn.Module:
    ckpt_info = resolve_checkpoint_path(
        str(cfg.ckpt),
        default_owner=cfg.get("hf_user"),
        token=cfg.get("hf_token"),
        cache_dir=resolve_hf_cache_dir(cfg),
        local_files_only=bool(cfg.get("offline_mode", False)),
        path_resolver=to_absolute_path,
    )
    dtype = resolve_dtype(str(cfg.get("dtype", "bfloat16")))
    hnet, _ = build_model(
        ckpt_info["dir"],
        ckpt_info["file"],
        device=device,
        dtype=dtype,
        strict=bool(cfg.get("strict", False)),
    )
    model = HNetForTISPRegression(
        hnet,
        num_outputs=int(cfg.get("num_outputs", 10)),
        use_layer_norm=bool(cfg.get("use_layer_norm", True)),
        head_dropout=float(cfg.get("head_dropout", 0.0)),
        routing_step=cfg.get("routing_step"),
    )
    if bool(cfg.get("freeze_encoder", False)):
        model.freeze_encoder()
    return model


@torch.no_grad()
def evaluate_tisp(
    *,
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    sequence_length: int,
    loss_name: str,
    accelerator: Accelerator,
    split: str,
) -> dict[str, float]:
    del loss_name
    metrics = evaluate_tisp_posthoc_metrics(
        model=accelerator.unwrap_model(model),
        data_loader=loader,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        device=accelerator.device,
    )
    return {f"{split}/{key}": value for key, value in metrics.items()}


def save_tisp_checkpoint(accelerator: Accelerator, model: nn.Module, output_path: str) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        accelerator.save(accelerator.get_state_dict(model), output_path)
    accelerator.wait_for_everyone()


def load_tisp_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)


def train_tisp(
    *,
    cfg: RunConfig,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    tokenizer,
    accelerator: Accelerator,
) -> dict[str, float]:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable TISP parameters found.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    model, optimizer, train_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
    )

    max_steps = int(cfg.train_steps)
    global_step = 0
    best_pcc = float("-inf")
    best_model_path = os.path.join(to_absolute_path(str(cfg.dirs.output)), "best_model.pt")
    last_metrics: dict[str, float] = {}
    if max_steps <= 0:
        accelerator.wait_for_everyone()
        metrics: dict[str, float] = {}
        if accelerator.is_main_process:
            metrics = evaluate_tisp(
                model=model,
                loader=test_loader,
                tokenizer=tokenizer,
                sequence_length=int(cfg.sequence_length),
                loss_name=str(cfg.tisp_loss),
                accelerator=accelerator,
                split="test",
            )
            metrics["best_pcc"] = metrics.get("test/avg_pcc", float("nan"))
        accelerator.wait_for_everyone()
        return metrics

    model.train()
    optimizer.zero_grad(set_to_none=True)
    while global_step < max_steps:
        produced_batch = False
        for raw_batch in train_loader:
            produced_batch = True
            with accelerator.accumulate(model):
                batch = prepare_tisp_batch(
                    raw_batch,
                    tokenizer=tokenizer,
                    sequence_length=int(cfg.sequence_length),
                    device=accelerator.device,
                )
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = output["logits"].float()
                labels = batch["labels"].float()
                loss = compute_tisp_task_loss(logits, labels, str(cfg.tisp_loss))
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, float(cfg.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                loss_value = float(accelerator.gather_for_metrics(loss.detach().reshape(1)).float().mean().item())
                if accelerator.is_main_process and global_step % int(cfg.log_every) == 0:
                    cfg.logger.log_metrics({"train/loss": loss_value, "step": global_step}, step=global_step)
                if int(cfg.eval_steps) > 0 and global_step % int(cfg.eval_steps) == 0:
                    accelerator.wait_for_everyone()
                    should_save = torch.zeros((), device=accelerator.device, dtype=torch.int32)
                    if accelerator.is_main_process:
                        last_metrics = evaluate_tisp(
                            model=model,
                            loader=valid_loader,
                            tokenizer=tokenizer,
                            sequence_length=int(cfg.sequence_length),
                            loss_name=str(cfg.tisp_loss),
                            accelerator=accelerator,
                            split="valid",
                        )
                        cfg.logger.log_metrics(last_metrics, step=global_step)
                        valid_pcc = float(last_metrics["valid/avg_pcc"])
                        if valid_pcc > best_pcc:
                            best_pcc = valid_pcc
                            should_save.fill_(1)
                    should_save = accelerator.reduce(should_save, reduction="max")
                    if int(should_save.item()) > 0:
                        save_tisp_checkpoint(accelerator, model, best_model_path)
                        if accelerator.is_main_process:
                            cfg.logger.print(f"[best] step={global_step} valid/avg_pcc={best_pcc:.4f}")
                    best_tensor = torch.tensor(best_pcc, device=accelerator.device, dtype=torch.float64)
                    best_tensor = accelerator.reduce(best_tensor, reduction="max")
                    best_pcc = float(best_tensor.item())
                    accelerator.wait_for_everyone()
                    model.train()
                if global_step >= max_steps:
                    break
        if not produced_batch:
            raise RuntimeError("TISP training loader produced no batches.")

    if not os.path.exists(best_model_path):
        save_tisp_checkpoint(accelerator, model, best_model_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        load_tisp_checkpoint(accelerator.unwrap_model(model), best_model_path, accelerator.device)
        valid_metrics = evaluate_tisp(
            model=model,
            loader=valid_loader,
            tokenizer=tokenizer,
            sequence_length=int(cfg.sequence_length),
            loss_name=str(cfg.tisp_loss),
            accelerator=accelerator,
            split="valid",
        )
        test_metrics = evaluate_tisp(
            model=model,
            loader=test_loader,
            tokenizer=tokenizer,
            sequence_length=int(cfg.sequence_length),
            loss_name=str(cfg.tisp_loss),
            accelerator=accelerator,
            split="test",
        )
        last_metrics.update(valid_metrics)
        last_metrics.update(test_metrics)
        last_metrics["best_pcc"] = (
            float(best_pcc) if math.isfinite(best_pcc) else float(valid_metrics.get("valid/avg_pcc", float("nan")))
        )
    accelerator.wait_for_everyone()
    return last_metrics


@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg: RunConfig) -> None:
    cfg, logger = init_experiment(cfg, init_wandb=bool(cfg.get("use_wandb", True)))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.get("grad_acc_steps", 1)),
        mixed_precision=str(cfg.get("mixed_precision", "bf16")),
    )
    if str(cfg.get("tokenizer", "fast")).lower() != "fast":
        raise ValueError("The release TISP path supports only tokenizer=fast.")

    data_root = resolve_hf_dataset_root(
        str(cfg.root),
        repo_id=str(cfg.get("dnalongbench_repo", "andyjzhao/dnalongbench")),
        required_subdir="tisp",
        token=cfg.get("hf_token"),
        cache_dir=resolve_hf_cache_dir(cfg),
        local_files_only=bool(cfg.get("offline_mode", False)),
        path_resolver=to_absolute_path,
    )
    train_loader, valid_loader, test_loader = build_tisp_loaders(
        root=data_root,
        batch_size=int(cfg.batch_size),
        sequence_length=int(cfg.sequence_length),
        num_train_samples=int(cfg.get("num_train_samples", 0)),
        num_valid_samples=int(cfg.get("num_valid_samples", 0)),
        num_test_samples=int(cfg.get("num_test_samples", 0)),
        random_strand=bool(cfg.get("tisp_random_strand", False)),
    )
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=int(cfg.batch_size),
        collate_fn=collate_tisp_raw_batch,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    valid_loader = DataLoader(
        valid_loader.dataset,
        batch_size=int(cfg.get("eval_batch_size", cfg.batch_size)),
        shuffle=False,
        collate_fn=collate_tisp_raw_batch,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=int(cfg.get("eval_batch_size", cfg.batch_size)),
        shuffle=False,
        collate_fn=collate_tisp_raw_batch,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    tokenizer = build_tokenizer("fast")
    model = build_tisp_model(cfg, accelerator.device)
    metrics = train_tisp(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        accelerator=accelerator,
    )
    if accelerator.is_main_process:
        logger.log_metrics(metrics, step=int(cfg.train_steps))
        logger.update_summary(metrics)
    accelerator.wait_for_everyone()
    finish_experiment(cfg, logger)


if __name__ == "__main__":
    main()
