import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import math
import os
import time
from contextlib import nullcontext
from typing import IO, Dict, Iterable, List, Optional, Set, Tuple

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from src.downstream.dlb_data import (
    CMPHbinsDataset,
    CMPTrainingDataset,
    EmbeddingProvider,
    SequenceProvider,
    collate_cmp_features,
    collate_cmp_hbins,
    resolve_cmp_split,
)
from src.downstream.utils import build_progress, resolve_dtype
from src.genezip.embedding import coverage_aware_pooling
from src.utils import finish_experiment, init_experiment, RunConfig, timer
from src.utils.hf_utils import (
    load_hf_dataset_from_cfg,
    resolve_checkpoint_path,
    resolve_hf_cache_dir,
    subset_dataset,
)


def upper_triangular_length(num_bins: int, offset: int) -> int:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    total = 0
    for dist in range(offset, num_bins):
        total += num_bins - dist
    return total


def upper_triangular_indices(
    num_bins: int,
    offset: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.triu_indices(num_bins, num_bins, offset=offset, device=device)


def _normalize_optional_str(value):
    if isinstance(value, str) and value.lower() in ("none", "null", ""):
        return None
    return value


def resolve_emb_position(emb_position: str) -> Tuple[str, str]:
    emb_position = str(emb_position).lower()
    mapping = {
        "encoder_dc0": ("dc0", "encoder"),
        "encoder_dc1": ("dc1", "encoder"),
        "decoder_dc0": ("dc0", "decoder"),
        "decoder_dc1": ("dc1", "decoder"),
        "before_lm_head": ("io", "before_lm_head"),
    }
    if emb_position not in mapping:
        valid = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unsupported emb_position={emb_position}. Expected one of: {valid}.")
    return mapping[emb_position]


def resolve_checkpoint_file(cfg: RunConfig) -> str:
    ckpt_value = _normalize_optional_str(cfg.get("ckpt"))
    if not ckpt_value:
        raise ValueError("ckpt must be set to a local checkpoint or a HF repo id.")
    ckpt_info = resolve_checkpoint_path(
        str(ckpt_value),
        default_owner=_normalize_optional_str(cfg.get("hf_user")),
        token=_normalize_optional_str(cfg.get("hf_token")),
        cache_dir=resolve_hf_cache_dir(cfg),
        path_resolver=to_absolute_path,
    )
    return ckpt_info["file"]


class CoveragePooler(nn.Module):
    # Coverage-aware pooling from token spans to fixed-size bins.
    def __init__(self, num_bins: int, bin_size: int) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.bin_size = int(bin_size)

    def forward(
        self,
        hidden: torch.Tensor,
        start_bp: torch.Tensor,
        end_bp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if start_bp.dim() == 1:
            start_bp = start_bp.unsqueeze(0)
        if end_bp.dim() == 1:
            end_bp = end_bp.unsqueeze(0)
        batch_size = hidden.shape[0]
        outputs = []
        for idx in range(batch_size):
            if mask is None:
                valid = slice(None)
            else:
                valid = mask[idx]
            pooled = self._coverage_pool_single(
                hidden[idx][valid],
                start_bp[idx][valid],
                end_bp[idx][valid],
            )
            outputs.append(pooled)
        return torch.stack(outputs, dim=0)

    def _coverage_pool_single(
        self,
        hidden: torch.Tensor,
        start_bp: torch.Tensor,
        end_bp: torch.Tensor,
    ) -> torch.Tensor:
        spans = torch.stack([start_bp, end_bp], dim=-1)
        return coverage_aware_pooling(
            hidden,
            spans,
            self.bin_size,
            num_bins=self.num_bins,
        )


def mean_pooling_num_den(
    embeddings: torch.Tensor,
    token_spans: torch.Tensor,
    bin_size: int,
    *,
    num_bins: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape (T, D) for mean pooling.")
    if token_spans.dim() != 2 or token_spans.shape[1] != 2:
        raise ValueError("token_spans must have shape (T, 2) for mean pooling.")
    if token_spans.shape[0] != embeddings.shape[0]:
        raise ValueError("token_spans length must match embeddings.")
    bin_size = int(bin_size)
    if bin_size <= 0:
        raise ValueError("bin_size must be positive for mean pooling.")

    if num_bins is None:
        if token_spans.numel() == 0:
            raise ValueError("num_bins is required when token_spans is empty.")
        max_end = int(token_spans[:, 1].max().item())
        if max_end < 0:
            max_end = 0
        num_bins = (max_end + bin_size - 1) // bin_size
    num_bins = int(num_bins)

    acc_dtype = torch.float32
    num = torch.zeros((num_bins, embeddings.shape[1]), device=embeddings.device, dtype=acc_dtype)
    den = torch.zeros((num_bins,), device=embeddings.device, dtype=acc_dtype)
    if embeddings.numel() == 0 or num_bins == 0:
        return num, den

    max_len = num_bins * bin_size
    spans = token_spans.to(device=embeddings.device)
    start = spans[:, 0].clamp(0, max_len)
    end = spans[:, 1].clamp(0, max_len)
    lengths = end - start
    valid = lengths > 0
    if not torch.any(valid):
        return num, den

    emb = embeddings[valid].to(acc_dtype)
    start = start[valid]
    end = end[valid]
    start_bin = torch.div(start, bin_size, rounding_mode="floor").to(torch.long)
    end_bin = torch.div(end - 1, bin_size, rounding_mode="floor").to(torch.long)

    same = start_bin == end_bin
    idx = start_bin[same]
    if idx.numel() > 0:
        num.index_add_(0, idx, emb[same])
        den.index_add_(0, idx, torch.ones_like(idx, dtype=acc_dtype))

    multi = ~same
    sb = start_bin[multi]
    eb = end_bin[multi]
    emb_multi = emb[multi]
    if sb.numel() > 0:
        counts = (eb - sb + 1).clamp(min=0).to(torch.long)
        if counts.numel() > 0:
            base = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
            if base.numel() > 0:
                offsets = torch.arange(base.numel(), device=embeddings.device) - base
                bins = sb.repeat_interleave(counts) + offsets
                emb_rep = emb_multi.repeat_interleave(counts, dim=0)
                num.index_add_(0, bins, emb_rep)
                den.index_add_(0, bins, torch.ones_like(bins, dtype=acc_dtype))

    return num, den


def mean_pooling(
    embeddings: torch.Tensor,
    token_spans: torch.Tensor,
    bin_size: int,
    *,
    num_bins: Optional[int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    num, den = mean_pooling_num_den(
        embeddings,
        token_spans,
        bin_size,
        num_bins=num_bins,
    )
    pooled = num / (den[:, None] + eps)
    return pooled.to(embeddings.dtype)


class MeanPooler(nn.Module):
    def __init__(self, num_bins: int, bin_size: int) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.bin_size = int(bin_size)

    def forward(
        self,
        hidden: torch.Tensor,
        start_bp: torch.Tensor,
        end_bp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if start_bp.dim() == 1:
            start_bp = start_bp.unsqueeze(0)
        if end_bp.dim() == 1:
            end_bp = end_bp.unsqueeze(0)
        batch_size = hidden.shape[0]
        outputs = []
        for idx in range(batch_size):
            if mask is None:
                valid = slice(None)
            else:
                valid = mask[idx]
            pooled = self._mean_pool_single(
                hidden[idx][valid],
                start_bp[idx][valid],
                end_bp[idx][valid],
            )
            outputs.append(pooled)
        return torch.stack(outputs, dim=0)

    def _mean_pool_single(
        self,
        hidden: torch.Tensor,
        start_bp: torch.Tensor,
        end_bp: torch.Tensor,
    ) -> torch.Tensor:
        spans = torch.stack([start_bp, end_bp], dim=-1)
        return mean_pooling(
            hidden,
            spans,
            self.bin_size,
            num_bins=self.num_bins,
        )


class BilinearHead(nn.Module):
    def __init__(
        self,
        num_bins: int,
        dim: int,
        dist_emb_dim: int,
        symmetrize: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)
        self.symmetrize = bool(symmetrize)
        if dist_emb_dim > 0:
            self.dist_embedding = nn.Embedding(num_bins, dist_emb_dim)
            self.dist_proj = nn.Linear(dist_emb_dim, 1, bias=False)
            idx = torch.arange(num_bins)
            dist_idx = (idx[None, :] - idx[:, None]).abs()
            self.register_buffer("dist_idx", dist_idx, persistent=False)
        else:
            self.dist_embedding = None
            self.dist_proj = None
            self.register_buffer("dist_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def _distance_bias(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.dist_embedding is None or self.dist_proj is None:
            return None
        idx = self.dist_idx.to(device=device)
        emb = self.dist_embedding(idx)
        bias = self.dist_proj(emb).squeeze(-1)
        return bias.to(dtype=dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hw = torch.matmul(h, self.weight)
        pred = torch.matmul(hw, h.transpose(-1, -2))
        bias = self._distance_bias(h.device, pred.dtype)
        if bias is not None:
            pred = pred + bias
        if self.symmetrize:
            pred = 0.5 * (pred + pred.transpose(-1, -2))
        return pred


class Conv1DBilinearHead(nn.Module):
    def __init__(
        self,
        num_bins: int,
        dim: int,
        dist_emb_dim: int,
        cnn_channels: int,
        cnn_layers: int,
        cnn_kernel: int,
        dropout: float,
        symmetrize: bool = True,
    ) -> None:
        super().__init__()
        padding = cnn_kernel // 2
        layers: List[nn.Module] = []
        for layer_idx in range(cnn_layers):
            in_ch = dim if layer_idx == 0 else cnn_channels
            layers.append(nn.Conv1d(in_ch, cnn_channels, kernel_size=cnn_kernel, padding=padding))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.cnn = nn.Sequential(*layers)
        self.head = BilinearHead(num_bins, cnn_channels, dist_emb_dim, symmetrize=symmetrize)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        return self.head(x)


class Conv2DHead(nn.Module):
    def __init__(
        self,
        num_bins: int,
        dim: int,
        dist_emb_dim: int,
        conv_channels: Iterable[int],
        conv_kernel: int,
        dropout: float,
        use_cnn1d: bool,
        cnn1d_channels: int,
        cnn1d_layers: int,
        cnn1d_kernel: int,
        cnn1d_dropout: float,
        symmetrize: bool = True,
    ) -> None:
        super().__init__()
        self.use_cnn1d = bool(use_cnn1d)
        if self.use_cnn1d:
            padding = cnn1d_kernel // 2
            layers: List[nn.Module] = []
            for layer_idx in range(cnn1d_layers):
                in_ch = dim if layer_idx == 0 else cnn1d_channels
                layers.append(nn.Conv1d(in_ch, cnn1d_channels, kernel_size=cnn1d_kernel, padding=padding))
                layers.append(nn.GELU())
                if cnn1d_dropout > 0:
                    layers.append(nn.Dropout(cnn1d_dropout))
            self.cnn1d = nn.Sequential(*layers)
            dim = cnn1d_channels
        else:
            self.cnn1d = None
        self.symmetrize = bool(symmetrize)
        if dist_emb_dim > 0:
            self.dist_embedding = nn.Embedding(num_bins, dist_emb_dim)
            idx = torch.arange(num_bins)
            dist_idx = (idx[None, :] - idx[:, None]).abs()
            self.register_buffer("dist_idx", dist_idx, persistent=False)
        else:
            self.dist_embedding = None
            self.register_buffer("dist_idx", torch.empty(0, dtype=torch.long), persistent=False)
        pair_dim = dim * 3 + (dist_emb_dim if dist_emb_dim > 0 else 0)
        conv_layers: List[nn.Module] = []
        in_ch = pair_dim
        padding = conv_kernel // 2
        for ch in conv_channels:
            conv_layers.append(nn.Conv2d(in_ch, ch, kernel_size=conv_kernel, padding=padding))
            conv_layers.append(nn.GELU())
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))
            in_ch = int(ch)
        conv_layers.append(nn.Conv2d(in_ch, 1, kernel_size=1))
        self.conv2d = nn.Sequential(*conv_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.cnn1d is not None:
            x = h.transpose(1, 2)
            x = self.cnn1d(x)
            h = x.transpose(1, 2)
        batch_size, num_bins, dim = h.shape
        hi = h.unsqueeze(2).expand(batch_size, num_bins, num_bins, dim)
        hj = h.unsqueeze(1).expand(batch_size, num_bins, num_bins, dim)
        pairwise = [hi, hj, hi * hj]
        if self.dist_embedding is not None:
            dist_idx = self.dist_idx.to(device=h.device)
            dist = self.dist_embedding(dist_idx).to(dtype=h.dtype)
            pairwise.append(dist.unsqueeze(0).expand(batch_size, -1, -1, -1))
        pair = torch.cat(pairwise, dim=-1)
        pair = pair.permute(0, 3, 1, 2)
        pred = self.conv2d(pair).squeeze(1)
        if self.symmetrize:
            pred = 0.5 * (pred + pred.transpose(-1, -2))
        return pred


def build_head(cfg: DictConfig, input_dim: int, num_bins: int) -> Tuple[nn.Module, nn.Module]:
    head_cfg = cfg.head or {}
    head_type = str(cfg.head_type).lower()
    head_dim = int(head_cfg.get("dim") or input_dim)
    dist_emb_dim = int(head_cfg.get("dist_emb_dim", 0) or 0)
    symmetrize = bool(head_cfg.get("symmetrize", True))

    proj: nn.Module
    if head_dim == input_dim:
        proj = nn.Identity()
    else:
        proj = nn.Linear(input_dim, head_dim)

    if head_type == "bilinear":
        head = BilinearHead(num_bins, head_dim, dist_emb_dim, symmetrize=symmetrize)
    elif head_type == "1dcnn_bilinear":
        cnn_cfg = head_cfg.get("cnn1d", {})
        head = Conv1DBilinearHead(
            num_bins=num_bins,
            dim=head_dim,
            dist_emb_dim=dist_emb_dim,
            cnn_channels=int(cnn_cfg.get("channels", head_dim)),
            cnn_layers=int(cnn_cfg.get("layers", 2)),
            cnn_kernel=int(cnn_cfg.get("kernel_size", 5)),
            dropout=float(cnn_cfg.get("dropout", 0.0)),
            symmetrize=symmetrize,
        )
    elif head_type in ("2dcnn", "pairwise"):
        cnn1d_cfg = head_cfg.get("cnn1d", {})
        cnn2d_cfg = head_cfg.get("cnn2d", {})
        head = Conv2DHead(
            num_bins=num_bins,
            dim=head_dim,
            dist_emb_dim=dist_emb_dim,
            conv_channels=cnn2d_cfg.get("channels", [64, 64, 64]),
            conv_kernel=int(cnn2d_cfg.get("kernel_size", 3)),
            dropout=float(cnn2d_cfg.get("dropout", 0.0)),
            use_cnn1d=bool(cnn1d_cfg.get("enabled", True)),
            cnn1d_channels=int(cnn1d_cfg.get("channels", head_dim)),
            cnn1d_layers=int(cnn1d_cfg.get("layers", 2)),
            cnn1d_kernel=int(cnn1d_cfg.get("kernel_size", 5)),
            cnn1d_dropout=float(cnn1d_cfg.get("dropout", 0.0)),
            symmetrize=symmetrize,
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    return proj, head


def build_strata_indices(num_bins: int, offset: int) -> List[np.ndarray]:
    tri = upper_triangular_indices(num_bins, offset, device=torch.device("cpu"))
    dist = (tri[1] - tri[0]).numpy()
    strata = []
    for d in range(offset, num_bins):
        idx = np.where(dist == d)[0]
        if idx.size:
            strata.append(idx)
    return strata


def scc_from_upper(pred: np.ndarray, true: np.ndarray, strata: List[np.ndarray]) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"SCC shape mismatch: {pred.shape} vs {true.shape}")
    weighted_sum = 0.0
    weight_total = 0.0
    for idx in strata:
        x = pred[idx]
        y = true[idx]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        x = x[mask]
        y = y[mask]
        x_var = x - x.mean()
        y_var = y - y.mean()
        denom = np.sqrt(np.dot(x_var, x_var) * np.dot(y_var, y_var))
        if denom == 0:
            continue
        corr = np.dot(x_var, y_var) / denom
        weight = mask.sum()
        weighted_sum += corr * weight
        weight_total += weight
    if weight_total == 0:
        return float("nan")
    return float(weighted_sum / weight_total)


def pearson_from_upper(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"Pearson shape mismatch: {pred.shape} vs {true.shape}")
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return float("nan")
    x = pred[mask] - pred[mask].mean()
    y = true[mask] - true[mask].mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


class CacheStatusTracker:
    def __init__(self, cache_dir: str, stage: str, rank: int, kind: str = "tokens") -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.stage = str(stage)
        self.rank = int(rank)
        self.kind = str(kind or "tokens").lower()
        self.meta_dir = os.path.join(self.cache_dir, "meta_info")
        if self.kind == "tokens":
            filename = f"status_{self.stage}_rank{self.rank}.jsonl"
        else:
            filename = f"status_{self.stage}_{self.kind}_rank{self.rank}.jsonl"
        self.status_path = os.path.join(self.meta_dir, filename)
        self.done_ids: Set[str] = set()
        self.has_meta = False
        self._loaded = False
        self._handle: Optional[IO[str]] = None

    def _match_status_file(self, name: str) -> bool:
        if not (name.startswith("status_") and name.endswith(".jsonl")):
            return False
        if self.kind == "tokens":
            return (
                name.startswith(f"status_{self.stage}_rank")
                or name.startswith(f"status_{self.stage}_tokens_rank")
            )
        return name.startswith(f"status_{self.stage}_{self.kind}_rank")

    def load(self) -> None:
        if self._loaded:
            return
        if os.path.isdir(self.meta_dir):
            for name in os.listdir(self.meta_dir):
                if not self._match_status_file(name):
                    continue
                self.has_meta = True
                self.done_ids.update(self._read_status(os.path.join(self.meta_dir, name)))
        self._loaded = True

    def _read_status(self, path: str) -> Iterable[str]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample_id = payload.get("sample_id")
                    if sample_id is not None:
                        yield str(sample_id)
        except FileNotFoundError:
            return

    def is_done(self, sample_id: str) -> bool:
        return str(sample_id) in self.done_ids

    def mark_done(self, sample_id: str) -> None:
        sample_id = str(sample_id)
        if sample_id in self.done_ids:
            return
        if self._handle is None:
            os.makedirs(self.meta_dir, exist_ok=True)
            self._handle = open(self.status_path, "a", encoding="utf-8")
        payload = {"sample_id": sample_id, "stage": self.stage}
        self._handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._handle.flush()
        self.done_ids.add(sample_id)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def precompute_cmp_embeddings(
    stage: str,
    split,
    provider: EmbeddingProvider,
    accelerator: Accelerator,
    progress_mininterval_sec: Optional[float] = None,
    cache_dir: Optional[str] = None,
    logger=None,
    cache_kind: str = "tokens",
) -> None:
    if split is None or len(split) == 0:
        return
    if not getattr(provider, "cache_dir", None):
        return
    if "sample_id" not in split.column_names:
        raise ValueError("CMPDataset missing sample_id column for precompute.")
    sample_ids = list(split["sample_id"])
    sample_ids = sample_ids[accelerator.process_index :: accelerator.num_processes]
    tracker: Optional[CacheStatusTracker] = None
    cache_root = cache_dir or getattr(provider, "cache_dir", None)
    if cache_root:
        tracker = CacheStatusTracker(
            cache_root,
            stage=stage,
            rank=accelerator.process_index,
            kind=cache_kind,
        )
        tracker.load()
    pending_ids: List[str] = []
    if tracker is None:
        pending_ids = [str(sample_id) for sample_id in sample_ids]
    elif tracker.has_meta:
        for sample_id in sample_ids:
            sample_id = str(sample_id)
            if not tracker.is_done(sample_id):
                pending_ids.append(sample_id)
    else:
        for sample_id in sample_ids:
            sample_id = str(sample_id)
            if cache_kind == "hbins":
                has_cache = provider.has_hbins_cache(sample_id)
            else:
                has_cache = provider.has_cache(sample_id)
            if has_cache:
                tracker.mark_done(sample_id)
            else:
                pending_ids.append(sample_id)
    if not pending_ids:
        if tracker is not None:
            tracker.close()
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        return

    if accelerator.is_main_process:
        cache_note = f" -> {cache_dir}" if cache_dir else ""
        kind_label = "embeddings" if cache_kind == "tokens" else cache_kind
        msg = (
            f"[info] running {kind_label} inference for {stage} "
            f"(total {len(split)} samples, num_processes={accelerator.num_processes}){cache_note}"
        )
        if logger is None:
            print(msg, flush=True)
        else:
            logger.print(msg)

    progress_cm = build_progress(progress_mininterval_sec) if accelerator.is_main_process else nullcontext()
    start = time.time()
    with progress_cm as progress:
        task = None
        if progress is not None:
            task = progress.add_task(f"{stage} embeddings", total=len(pending_ids))
        for sample_id in pending_ids:
            if cache_kind == "hbins":
                provider.get_hbins(sample_id)
            else:
                provider.get(sample_id)
            if tracker is not None:
                tracker.mark_done(sample_id)
            if task is not None:
                progress.advance(task)
    if tracker is not None:
        tracker.close()
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        elapsed = time.time() - start
        msg = f"[info] {stage} embedding inference done in {elapsed:.1f}s"
        if logger is None:
            print(msg, flush=True)
        else:
            logger.print(msg)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target) & torch.isfinite(pred)
    if not torch.any(mask):
        return pred.sum() * 0.0
    diff = pred[mask] - target[mask]
    return (diff * diff).mean()


def mask_nonfinite_tokens(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    chunk_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if chunk_tokens <= 0 or hidden.numel() == 0:
        return hidden, mask
    squeeze = False
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze = True
    if mask.shape[0] != hidden.shape[0] or mask.shape[1] != hidden.shape[1]:
        raise ValueError("mask shape must match hidden tokens.")
    mask_out = mask.clone()
    nonfinite = False
    length = hidden.shape[1]
    for start in range(0, length, chunk_tokens):
        end = min(start + chunk_tokens, length)
        finite_chunk = torch.isfinite(hidden[:, start:end]).all(dim=-1)
        if not bool(finite_chunk.all()):
            nonfinite = True
        mask_out[:, start:end] &= finite_chunk
    if nonfinite:
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)
    if squeeze:
        hidden = hidden.squeeze(0)
        mask_out = mask_out.squeeze(0)
    return hidden, mask_out


def build_optimizer(cfg: DictConfig, params: List[nn.Parameter]) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.optimizer or {}
    optimizer_name = str(optimizer_cfg.get("name") or "adamw").lower()
    optimizer_lr = float(optimizer_cfg.get("lr") or cfg.lr)
    if optimizer_name == "sgd":
        momentum = float(optimizer_cfg.get("momentum") or 0.0)
        return torch.optim.SGD(params, lr=optimizer_lr, momentum=momentum)
    if optimizer_name == "adamw":
        weight_decay = float(optimizer_cfg.get("weight_decay") or 0.0)
        return torch.optim.AdamW(params, lr=optimizer_lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def resolve_target_format(
    target_format: Optional[str],
    sample_target: torch.Tensor,
    num_bins: int,
    upper_len: int,
) -> str:
    fmt = None
    if target_format is not None:
        fmt = str(target_format).lower()
        if fmt in ("auto", "none", ""):
            fmt = None
        elif fmt not in ("upper", "full"):
            raise ValueError(f"Unsupported target_format: {target_format}")

    if fmt is None:
        if sample_target.dim() == 1 and sample_target.numel() == upper_len:
            fmt = "upper"
        elif sample_target.dim() >= 2 and sample_target.shape[-2:] == (num_bins, num_bins):
            fmt = "full"
        else:
            raise ValueError("Unable to infer target_format from sample target.")
    return fmt


def compute_pred_and_loss(
    tokens: torch.Tensor,
    start_bp: torch.Tensor,
    end_bp: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    target_format: str,
    upper_idx: torch.Tensor,
    pooler: nn.Module,
    proj: nn.Module,
    head: nn.Module,
    nonfinite_chunk_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = tokens
    if nonfinite_chunk_tokens > 0:
        hidden, mask = mask_nonfinite_tokens(hidden, mask, nonfinite_chunk_tokens)
    pooled = pooler(hidden, start_bp, end_bp, mask=mask)
    if isinstance(proj, nn.Linear):
        pooled = pooled.to(proj.weight.dtype)
    pooled = proj(pooled)
    pred_full = head(pooled)
    pred_upper = pred_full[:, upper_idx[0], upper_idx[1]]
    if target_format == "full":
        target_upper = target[:, upper_idx[0], upper_idx[1]]
    else:
        target_upper = target
    loss = masked_mse(pred_upper, target_upper)
    return pred_upper, loss


def compute_pred_and_loss_hbins(
    h_bins: torch.Tensor,
    target: torch.Tensor,
    target_format: str,
    upper_idx: torch.Tensor,
    proj: nn.Module,
    head: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(proj, nn.Linear):
        h_bins = h_bins.to(proj.weight.dtype)
    if not torch.isfinite(h_bins).all():
        h_bins = torch.nan_to_num(h_bins, nan=0.0, posinf=0.0, neginf=0.0)
    h_bins = proj(h_bins)
    pred_full = head(h_bins)
    pred_upper = pred_full[:, upper_idx[0], upper_idx[1]]
    if target_format == "full":
        target_upper = target[:, upper_idx[0], upper_idx[1]]
    else:
        target_upper = target
    loss = masked_mse(pred_upper, target_upper)
    return pred_upper, loss


def run_eval(
    accelerator: Accelerator,
    loader: DataLoader,
    target_format: str,
    upper_idx: torch.Tensor,
    strata: List[np.ndarray],
    forward_batch,
) -> Dict[str, float]:
    loss_sum = 0.0
    loss_count = 0
    scc_sum = 0.0
    scc_count = 0
    pearson_sum = 0.0
    pearson_count = 0
    upper_idx_np = (upper_idx[0].cpu().numpy(), upper_idx[1].cpu().numpy())
    for batch in loader:
        with torch.inference_mode(), accelerator.autocast():
            pred_upper, loss = forward_batch(batch)
        loss_sum += float(loss.detach().cpu().item())
        loss_count += 1
        target_np = batch["target"].detach().cpu().numpy()
        if target_format == "full":
            target_np = target_np[:, upper_idx_np[0], upper_idx_np[1]]
        pred_np = pred_upper.detach().cpu().numpy()
        for idx in range(pred_np.shape[0]):
            pred_row = pred_np[idx].ravel()
            true_row = target_np[idx].ravel()
            scc_val = scc_from_upper(pred_row, true_row, strata)
            if math.isfinite(scc_val):
                scc_sum += float(scc_val)
                scc_count += 1
            pearson_val = pearson_from_upper(pred_row, true_row)
            if math.isfinite(pearson_val):
                pearson_sum += float(pearson_val)
                pearson_count += 1

    loss_sum_t = accelerator.reduce(torch.tensor(loss_sum, device=accelerator.device), reduction="sum")
    loss_count_t = accelerator.reduce(
        torch.tensor(loss_count, device=accelerator.device, dtype=torch.long), reduction="sum"
    )
    scc_sum_t = accelerator.reduce(torch.tensor(scc_sum, device=accelerator.device), reduction="sum")
    scc_count_t = accelerator.reduce(
        torch.tensor(scc_count, device=accelerator.device, dtype=torch.long), reduction="sum"
    )
    pearson_sum_t = accelerator.reduce(torch.tensor(pearson_sum, device=accelerator.device), reduction="sum")
    pearson_count_t = accelerator.reduce(
        torch.tensor(pearson_count, device=accelerator.device, dtype=torch.long), reduction="sum"
    )

    avg_loss = float((loss_sum_t / torch.clamp(loss_count_t, min=1)).item())
    avg_scc = float((scc_sum_t / torch.clamp(scc_count_t, min=1)).item())
    avg_pearson = float((pearson_sum_t / torch.clamp(pearson_count_t, min=1)).item())
    return {
        "eval/loss": avg_loss,
        "eval/scc": avg_scc,
        "eval/corr": avg_pearson,
    }


@timer()
@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg: RunConfig) -> None:
    cfg, logger = init_experiment(cfg)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_acc_steps,
        cpu=str(cfg.device).lower() == "cpu",
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    dtype = resolve_dtype(cfg.dtype)

    num_bins = int(cfg.num_bins)
    bin_size = int(cfg.bin_size)
    upper_offset = int(cfg.upper_offset)
    expected_upper = upper_triangular_length(num_bins, upper_offset)
    upper_idx = upper_triangular_indices(num_bins, upper_offset, device=device)
    strata = build_strata_indices(num_bins, upper_offset)

    data_cfg = cfg.dataset or {}
    if str(data_cfg.get("type") or data_cfg.get("source") or "").lower() != "cmp_seq":
        raise ValueError("This release script only supports dataset.type=cmp_seq.")

    if bool(cfg.precompute_embeddings) and int(data_cfg.get("num_workers", 0) or 0) > 0:
        raise ValueError(
            "precompute_embeddings requires dataset.num_workers=0 to avoid duplicating cached embeddings."
        )

    cmp_dataset = load_hf_dataset_from_cfg(
        data_cfg,
        hf_token=cfg.hf_token,
        path_resolver=to_absolute_path,
        cache_dir=resolve_hf_cache_dir(cfg),
        local_files_only=bool(cfg.get("offline_mode", False)),
    )
    if not hasattr(cmp_dataset, "keys"):
        raise ValueError("CMP dataset must be a DatasetDict with train/validation/test.")

    train_split = resolve_cmp_split(cmp_dataset, data_cfg.get("train_split", "train"))
    eval_split = resolve_cmp_split(cmp_dataset, data_cfg.get("eval_split", "validation"))
    test_split = resolve_cmp_split(cmp_dataset, data_cfg.get("test_split", "test"))

    train_split = subset_dataset(train_split, int(cfg.num_train_samples))
    eval_split = subset_dataset(eval_split, int(cfg.num_valid_samples))
    test_split = subset_dataset(test_split, int(cfg.num_test_samples))

    ckpt_file = resolve_checkpoint_file(cfg)
    ckpt_tag = os.path.basename(str(cfg.ckpt or ""))
    disable_routing = not (ckpt_tag.startswith("GeneZip") or ckpt_tag.startswith("HNet"))

    emb_layer, emb_stream = resolve_emb_position(cfg.emb_position)
    tokenizer_name = str(cfg.tokenizer).lower()

    use_cache = bool(cfg.use_cache)
    cache_dir = _normalize_optional_str(cfg.get("emb_cache")) if use_cache else None
    if cache_dir:
        cache_dir = to_absolute_path(str(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)

    modeling = str(cfg.get("modeling", "identity")).lower()
    use_hbins_cfg = data_cfg.get("use_hbins")
    use_hbins = bool(use_hbins_cfg) if use_hbins_cfg is not None else modeling == "identity"
    if use_hbins and modeling != "identity":
        raise ValueError("CMP h_bins requires modeling='identity'.")
    cache_hbins = bool(data_cfg.get("cache_hbins", True)) and use_cache
    cache_tokens = use_cache and not use_hbins

    label_key = str(data_cfg.get("label_key", "label_ut"))
    mask_key = str(data_cfg.get("mask_key", "mask_ut"))
    target_index = data_cfg.get("target_index")
    sequence_key = str(data_cfg.get("sequence_key", "sequence"))
    sequence_format = str(data_cfg.get("sequence_format", "string"))
    reference_id = data_cfg.get("reference_id") or "hg38"

    def build_provider(split):
        return SequenceProvider(
            split,
            sequence_key=sequence_key,
            sequence_format=sequence_format,
            reference_id=reference_id,
            tokenizer_name=tokenizer_name,
            embedding_layer=emb_layer,
            embedding_stream=emb_stream,
            trunk_len=int(cfg.trunk_len),
            routing_step=cfg.routing_step,
            encode_batch_size=int(cfg.encode_batch_size),
            validate_spans=bool(cfg.validate_spans),
            checkpoint_path=ckpt_file,
            device=device,
            dtype=dtype,
            strict=bool(cfg.strict),
            disable_routing=disable_routing,
            cache_dir=cache_dir,
            cache_tokens=cache_tokens,
            cache_embeddings=False,
            num_bins=num_bins,
            bin_size=bin_size,
            cache_hbins=cache_hbins,
        )

    train_provider = build_provider(train_split)
    eval_provider = build_provider(eval_split) if eval_split is not None else None
    test_provider = build_provider(test_split) if test_split is not None else None

    if bool(cfg.precompute_embeddings) and use_cache:
        if use_hbins:
            precompute_cmp_embeddings(
                "train",
                train_split,
                train_provider,
                accelerator,
                cfg.get("progress_mininterval_sec"),
                cache_dir=cache_dir,
                logger=logger,
                cache_kind="hbins",
            )
        else:
            precompute_cmp_embeddings(
                "train",
                train_split,
                train_provider,
                accelerator,
                cfg.get("progress_mininterval_sec"),
                cache_dir=cache_dir,
                logger=logger,
                cache_kind="tokens",
            )
        if eval_provider is not None and eval_split is not None:
            precompute_cmp_embeddings(
                "validation",
                eval_split,
                eval_provider,
                accelerator,
                cfg.get("progress_mininterval_sec"),
                cache_dir=cache_dir,
                logger=logger,
                cache_kind="hbins" if use_hbins else "tokens",
            )
        if test_provider is not None and test_split is not None:
            precompute_cmp_embeddings(
                "test",
                test_split,
                test_provider,
                accelerator,
                cfg.get("progress_mininterval_sec"),
                cache_dir=cache_dir,
                logger=logger,
                cache_kind="hbins" if use_hbins else "tokens",
            )

    if use_hbins:
        train_dataset = CMPHbinsDataset(
            train_split,
            train_provider,
            label_key=label_key,
            mask_key=mask_key,
            target_index=target_index,
        )
        eval_dataset = (
            CMPHbinsDataset(
                eval_split,
                eval_provider,
                label_key=label_key,
                mask_key=mask_key,
                target_index=target_index,
            )
            if eval_provider is not None
            else None
        )
        test_dataset = (
            CMPHbinsDataset(
                test_split,
                test_provider,
                label_key=label_key,
                mask_key=mask_key,
                target_index=target_index,
            )
            if test_provider is not None
            else None
        )
        collate_fn = collate_cmp_hbins
    else:
        train_dataset = CMPTrainingDataset(
            train_split,
            train_provider,
            label_key=label_key,
            mask_key=mask_key,
            target_index=target_index,
        )
        eval_dataset = (
            CMPTrainingDataset(
                eval_split,
                eval_provider,
                label_key=label_key,
                mask_key=mask_key,
                target_index=target_index,
            )
            if eval_provider is not None
            else None
        )
        test_dataset = (
            CMPTrainingDataset(
                test_split,
                test_provider,
                label_key=label_key,
                mask_key=mask_key,
                target_index=target_index,
            )
            if test_provider is not None
            else None
        )
        collate_fn = collate_cmp_features

    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=bool(data_cfg.get("shuffle", True)),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(cfg.eval_batch_size),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.eval_batch_size),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
        )

    sample = train_dataset[0]
    if use_hbins:
        token_dim = int(sample["h_bins"].shape[-1])
    else:
        feat = sample["feat"]
        tokens = feat.embedding if feat.embedding is not None else feat.tokens
        if tokens is None:
            raise ValueError("CMP feature record missing embedding/tokens.")
        token_dim = int(tokens.shape[-1])

    target_format = resolve_target_format("auto", sample["target"], num_bins=num_bins, upper_len=expected_upper)
    if target_format == "upper" and sample["target"].numel() != expected_upper:
        raise ValueError("Upper-tri target size mismatch; check num_bins/upper_offset")

    pooler = None
    if not use_hbins:
        pooling = str(cfg.pooling).lower()
        if pooling == "coverage_aware_pooling":
            pooler = CoveragePooler(num_bins=num_bins, bin_size=bin_size)
        elif pooling == "mean":
            pooler = MeanPooler(num_bins=num_bins, bin_size=bin_size)
        else:
            raise ValueError(f"Unsupported pooling={pooling}. Use 'mean' or 'coverage_aware_pooling'.")

    proj, head = build_head(cfg, input_dim=token_dim, num_bins=num_bins)
    params = [p for module in (proj, head) for p in module.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg, params)

    proj, head, optimizer = accelerator.prepare(proj, head, optimizer)
    train_loader = accelerator.prepare_data_loader(train_loader, device_placement=False)
    if eval_loader is not None:
        eval_loader = accelerator.prepare_data_loader(eval_loader, device_placement=False)
    if test_loader is not None:
        test_loader = accelerator.prepare_data_loader(test_loader, device_placement=False)

    proj.train()
    head.train()

    def forward_batch(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_hbins:
            h_bins = batch["h_bins"].to(device=device, dtype=dtype)
            target = batch["target"].to(device=device)
            return compute_pred_and_loss_hbins(
                h_bins,
                target,
                target_format,
                upper_idx,
                proj,
                head,
            )
        if pooler is None:
            raise RuntimeError("pooler must be initialized for token-level CMP inputs.")
        tokens = batch["tokens"].to(device=device, dtype=dtype)
        start_bp = batch["start_bp"].to(device=device)
        end_bp = batch["end_bp"].to(device=device)
        mask = batch["mask"].to(device=device)
        target = batch["target"].to(device=device)
        nonfinite_chunk_tokens = int(cfg.get("nonfinite_chunk_tokens", 4096) or 0)
        return compute_pred_and_loss(
            tokens,
            start_bp,
            end_bp,
            mask,
            target,
            target_format,
            upper_idx,
            pooler,
            proj,
            head,
            nonfinite_chunk_tokens=nonfinite_chunk_tokens,
        )

    train_steps = int(cfg.train_steps)
    if train_steps <= 0:
        raise ValueError("train_steps must be > 0")
    eval_steps = int(cfg.eval_steps)
    log_every = max(int(cfg.log_every), 1)
    max_grad_norm = float(cfg.max_grad_norm)

    step = 0
    accum_loss = 0.0
    accum_count = 0
    data_iter = iter(train_loader)
    while step < train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.accumulate(proj, head):
            with accelerator.autocast():
                pred_upper, loss = forward_batch(batch)
            loss_value = float(loss.detach().cpu().item())
            accum_loss += loss_value
            accum_count += 1
            accelerator.backward(loss)
            if not accelerator.sync_gradients:
                continue

            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            mean_loss = accum_loss / max(accum_count, 1)
            accum_loss = 0.0
            accum_count = 0

            if step % log_every == 0:
                log_payload = {"train/loss": mean_loss, "train/lr": optimizer.param_groups[0]["lr"]}
                if logger is not None:
                    logger.log_metrics(log_payload, step=step)

            if eval_steps > 0 and eval_loader is not None and (step + 1) % eval_steps == 0:
                proj.eval()
                head.eval()
                metrics = run_eval(
                    accelerator,
                    eval_loader,
                    target_format,
                    upper_idx,
                    strata,
                    forward_batch,
                )
                test_metrics = None
                if test_loader is not None:
                    test_metrics = run_eval(
                        accelerator,
                        test_loader,
                        target_format,
                        upper_idx,
                        strata,
                        forward_batch,
                    )
                if logger is not None:
                    payload = dict(metrics)
                    if test_metrics is not None:
                        payload.update({f"test/{k.split('/', 1)[1]}": v for k, v in test_metrics.items()})
                    logger.log_metrics(payload, step=step)
                proj.train()
                head.train()

            step += 1

    finish_experiment(cfg, logger)


if __name__ == "__main__":
    main()
