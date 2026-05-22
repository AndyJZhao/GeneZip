from __future__ import annotations

from typing import Optional, Tuple

import torch


def coverage_aware_pooling_num_den(
    embeddings: torch.Tensor,
    token_spans: torch.Tensor,
    bin_size: int,
    *,
    num_bins: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape (T, D) for coverage-aware pooling.")
    if token_spans.dim() != 2 or token_spans.shape[1] != 2:
        raise ValueError("token_spans must have shape (T, 2) for coverage-aware pooling.")
    if token_spans.shape[0] != embeddings.shape[0]:
        raise ValueError("token_spans length must match embeddings.")
    bin_size = int(bin_size)
    if bin_size <= 0:
        raise ValueError("bin_size must be positive for coverage-aware pooling.")

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
    w = (end - start)[same].to(acc_dtype)
    if idx.numel() > 0:
        num.index_add_(0, idx, emb[same] * w.unsqueeze(1))
        den.index_add_(0, idx, w)

    multi = ~same
    sb = start_bin[multi]
    eb = end_bin[multi]
    emb_multi = emb[multi]
    st = start[multi]
    en = end[multi]
    if sb.numel() > 0:
        w0 = ((sb + 1) * bin_size - st).to(acc_dtype)
        w1 = (en - eb * bin_size).to(acc_dtype)
        num.index_add_(0, sb, emb_multi * w0.unsqueeze(1))
        den.index_add_(0, sb, w0)
        num.index_add_(0, eb, emb_multi * w1.unsqueeze(1))
        den.index_add_(0, eb, w1)
        counts = (eb - sb - 1).clamp(min=0)
        if counts.numel() > 0:
            base = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
            if base.numel() > 0:
                offsets = torch.arange(base.numel(), device=embeddings.device) - base
                bins = sb.repeat_interleave(counts) + 1 + offsets
                emb_rep = emb_multi.repeat_interleave(counts, dim=0)
                w_mid = torch.full(
                    (base.numel(),), float(bin_size), device=embeddings.device, dtype=acc_dtype
                )
                num.index_add_(0, bins, emb_rep * w_mid.unsqueeze(1))
                den.index_add_(0, bins, w_mid)

    return num, den


def coverage_aware_pooling(
    embeddings: torch.Tensor,
    token_spans: torch.Tensor,
    bin_size: int,
    *,
    num_bins: Optional[int] = None,
    eps: float = 1e-6,
    return_den: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    num, den = coverage_aware_pooling_num_den(
        embeddings,
        token_spans,
        bin_size,
        num_bins=num_bins,
    )
    pooled = num / (den[:, None] + eps)
    pooled = pooled.to(embeddings.dtype)
    return (pooled, den) if return_den else pooled
