from __future__ import annotations

from typing import Tuple

import torch


def apply_routing_floor(
    boundary_mask: torch.Tensor,
    boundary_prob: torch.Tensor,
    *,
    k_min: int,
    mask: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, dict[str, int] | None]:
    k_min = max(0, int(k_min))
    if k_min <= 0:
        if return_stats:
            return boundary_mask, None
        return boundary_mask

    boundary_mask = boundary_mask.bool()
    p = boundary_prob[..., -1]
    min_val = torch.finfo(p.dtype).min

    if cu_seqlens is not None:
        updated = boundary_mask.clone()
        num_seg = int(cu_seqlens.numel() - 1)
        triggers = 0
        deficit_total = 0
        for i in range(num_seg):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            length = e - s
            if length <= 0:
                continue
            updated[s] = True  # must-keep: segment start token
            seg_k_min = min(k_min, length)
            if seg_k_min <= 0:
                continue
            seg_mask = updated[s:e]
            count = int(seg_mask.sum().item())
            if count >= seg_k_min:
                continue
            candidates = ~seg_mask
            cand_count = int(candidates.sum().item())
            if cand_count <= 0:
                continue
            deficit = min(seg_k_min - count, cand_count)
            scores = p[s:e].masked_fill(~candidates, min_val)
            idx_local = torch.topk(scores, k=deficit).indices
            updated[s + idx_local] = True
            triggers += 1
            deficit_total += deficit
        if return_stats:
            return updated, {
                "segments": num_seg,
                "triggers": triggers,
                "deficit": deficit_total,
            }
        return updated

    if boundary_mask.dim() == 1:
        boundary_mask = boundary_mask.unsqueeze(0)
        p = p.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    updated = boundary_mask.clone()
    valid_mask = mask if mask is not None else torch.ones_like(updated, dtype=torch.bool)
    triggers = 0
    deficit_total = 0
    for i in range(updated.shape[0]):
        valid = valid_mask[i].bool()
        length = int(valid.sum().item())
        if length <= 0:
            continue
        first = int(valid.float().argmax().item())
        updated[i, first] = True  # must-keep: first valid token
        seg_k_min = min(k_min, length)
        if seg_k_min <= 0:
            continue
        seg_mask = updated[i] & valid
        count = int(seg_mask.sum().item())
        if count >= seg_k_min:
            continue
        candidates = (~seg_mask) & valid
        cand_count = int(candidates.sum().item())
        if cand_count <= 0:
            continue
        deficit = min(seg_k_min - count, cand_count)
        scores = p[i].masked_fill(~candidates, min_val)
        idx = torch.topk(scores, k=deficit).indices
        updated[i, idx] = True
        triggers += 1
        deficit_total += deficit

    if return_stats:
        segments = int(updated.shape[0])
        stats = {"segments": segments, "triggers": triggers, "deficit": deficit_total}
        return (updated.squeeze(0), stats) if squeeze else (updated, stats)
    return updated.squeeze(0) if squeeze else updated


def apply_routing_ceiling(
    boundary_mask: torch.Tensor,
    boundary_prob: torch.Tensor,
    *,
    k_max: int,
    mask: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, dict[str, int] | None]:
    """
    Enforce an upper bound (ceiling) on how many tokens can be selected.

    SAFETY FIXES (for H-Net DeChunkLayer correctness):
      1) Never allow a sample/segment to have zero selected tokens.
      2) Never drop the FIRST VALID token per sample (non-packed), or the
         SEGMENT START token per segment (packed). This prevents DeChunkLayer's
         plug_back_idx = cumsum(boundary_mask)-1 from producing -1 indices.
      3) When dropping, cap surplus by the number of droppable candidates to
         avoid topk(k > cand_count).

    Behavior:
      - Compute kmax per sample/segment as min(k_max, length),
        ignoring constraints <= 0 (treated as "no constraint").
      - Ensure kmax >= 1 when length > 0.
      - If selected count > kmax: drop the LOWEST boundary_prob[..., -1] among
        selected & droppable positions.

    Args:
        boundary_mask: bool tensor (L) / (B, L) for non-packed, or (T,) for packed.
        boundary_prob: float tensor (..., C) with keep-score in last channel.
        k_max: upper bound absolute count, <=0 disables ceiling.
        mask: optional valid positions (B, L) for non-packed (e.g., attention_mask).
        cu_seqlens: optional packed segment boundaries (num_seg+1,).
        return_stats: whether to return stats dict.

    Returns:
        updated boundary_mask (same shape as input), and optionally stats:
            {"segments": int, "triggers": int, "surplus": int}
    """
    k_max = int(k_max)
    if k_max <= 0:
        if return_stats:
            return boundary_mask, None
        return boundary_mask

    boundary_mask = boundary_mask.bool()

    # score for "keep/selected" assumed to be last channel
    p = boundary_prob[..., -1]
    max_val = torch.finfo(p.dtype).max

    def _k_max(length: int) -> int:
        length = max(0, int(length))
        k = max(0, min(k_max, length))
        if length > 0:
            k = max(k, 1)
        return k

    # -------- packed mode: boundary_mask is 1D over total_tokens --------
    if cu_seqlens is not None:
        updated = boundary_mask.clone()
        num_seg = int(cu_seqlens.numel() - 1)
        triggers = 0
        surplus_total = 0

        for i in range(num_seg):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            length = e - s
            if length <= 0:
                continue

            # must-keep: segment start token
            updated[s] = True

            kmax = _k_max(length)

            seg_mask = updated[s:e]
            count = int(seg_mask.sum().item())
            if count <= kmax:
                continue

            surplus = count - kmax

            must_keep = torch.zeros_like(seg_mask, dtype=torch.bool)
            must_keep[0] = True

            # only drop among selected & NOT must_keep
            candidates = seg_mask & (~must_keep)
            cand_count = int(candidates.sum().item())
            if cand_count <= 0:
                # nothing droppable; keep as-is
                continue

            surplus = min(surplus, cand_count)
            if surplus <= 0:
                continue

            scores = p[s:e].masked_fill(~candidates, max_val)
            idx_local = torch.topk(scores, k=surplus, largest=False).indices  # lowest scores
            updated[s + idx_local] = False
            triggers += 1
            surplus_total += surplus

        if return_stats:
            return updated, {"segments": num_seg, "triggers": triggers, "surplus": surplus_total}
        return updated

    # -------- non-packed mode: boundary_mask is (L) or (B,L) --------
    squeeze = False
    if boundary_mask.dim() == 1:
        boundary_mask = boundary_mask.unsqueeze(0)
        p = p.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        squeeze = True

    updated = boundary_mask.clone()
    valid_mask = mask.bool() if mask is not None else torch.ones_like(updated, dtype=torch.bool)

    triggers = 0
    surplus_total = 0
    for i in range(updated.shape[0]):
        valid = valid_mask[i]
        length = int(valid.sum().item())
        if length <= 0:
            continue

        # must-keep: first valid token (handles left-padding)
        first = int(valid.float().argmax().item())
        updated[i, first] = True

        kmax = _k_max(length)

        seg_mask = updated[i] & valid
        count = int(seg_mask.sum().item())
        if count <= kmax:
            continue

        surplus = count - kmax

        must_keep = torch.zeros_like(seg_mask, dtype=torch.bool)
        must_keep[first] = True

        # only drop among selected & valid & NOT must_keep
        candidates = seg_mask & (~must_keep)
        cand_count = int(candidates.sum().item())
        if cand_count <= 0:
            continue

        surplus = min(surplus, cand_count)
        if surplus <= 0:
            continue

        scores = p[i].masked_fill(~candidates, max_val)
        idx = torch.topk(scores, k=surplus, largest=False).indices
        updated[i, idx] = False
        triggers += 1
        surplus_total += surplus

    if return_stats:
        stats = {"segments": int(updated.shape[0]), "triggers": triggers, "surplus": surplus_total}
        return (updated.squeeze(0), stats) if squeeze else (updated, stats)
    return updated.squeeze(0) if squeeze else updated
