import math
import re
from collections.abc import Sequence

import torch
import torch.nn as nn

from src.genezip.data import normalize_region_name, _normalize_region_label_map
from src.genezip.routing_floor import apply_routing_ceiling, apply_routing_floor

from src.hnet.models.config_hnet import HNetConfig
from src.hnet.models.mixer_seq import HNetForCausalLM
from src.hnet.modules.dc import RoutingModule
from src.utils.log import logger as base_logger

REGION_TOKEN_PATTERN = re.compile(r"^(?P<name>[A-Za-z]+)(?P<ratio>[0-9]+(?:\.[0-9]+)?)$")


def _resolve_stage_tokens(value, stage_idx: int | None) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return 0
        if stage_idx is None:
            return int(value[0] or 0)
        idx = min(max(int(stage_idx), 0), len(value) - 1)
        return int(value[idx] or 0)
    return int(value or 0)


def parse_region_info(region_info: str) -> dict[str, float]:
    """
    Parse a region info string like "promoter1_cds2_utr4_exon4_intron16_nig8_dig32"
    into a mapping {region_name: multiplier}.
    """
    parsed: dict[str, float] = {}
    for token in str(region_info).split("_"):
        if not token:
            continue
        match = REGION_TOKEN_PATTERN.match(token)
        if not match:
            raise ValueError(
                f"Invalid region token '{token}'. Expected format {{region}}{{ratio}}."
            )
        name = normalize_region_name(match.group("name"))
        ratio = float(match.group("ratio"))
        if name in parsed and parsed[name] != ratio:
            base_logger.warning(
                f"[region_info] duplicate ratio for '{name}' -> using min({parsed[name]}, {ratio})."
            )
            ratio = min(parsed[name], ratio)
        parsed[name] = ratio
    return parsed


def build_region_target_bp_per_token(
        region_info: str,
        bp_per_token: float,
        region_label_map: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Returns:
        region_targets: {region_name: target_bp_per_token}
        region_label_map: mapping region_name -> label id used in datasets
    """
    if region_label_map is None:
        raise ValueError("region_label_map is required for region target construction.")
    region_label_map = _normalize_region_label_map(region_label_map)

    region_multipliers = parse_region_info(region_info)
    collapsed: dict[str, float] = {}
    for name, mult in region_multipliers.items():
        canonical = normalize_region_name(name)
        if canonical in collapsed:
            prev = collapsed[canonical]
            if mult != prev:
                base_logger.warning(
                    f"[region_info] collapsing multiple ratios for '{canonical}' -> using min({prev}, {mult})."
                )
                collapsed[canonical] = min(prev, mult)
        else:
            collapsed[canonical] = mult

    unknown = [name for name in collapsed.keys() if name not in region_label_map]
    if unknown:
        raise ValueError(
            f"[region_info] unknown region names {unknown}; available labels={sorted(region_label_map.keys())}"
        )

    region_targets = {name: float(bp_per_token) * float(mult) for name, mult in collapsed.items()}
    return region_targets, region_label_map


def compute_hnet_lr_multipliers(model, model_config_dict) -> list[float]:
    """
    Compute stage-wise LR multipliers following H-Net Appendix C.
    """
    d_model = list(getattr(model.config, "d_model", []) or [])
    if not d_model:
        return [1.0]

    try:
        n_gpt = float(model_config_dict.get("n_gpt", 1.0) or 1.0)
    except (TypeError, ValueError):
        n_gpt = 1.0

    # Without explicit per-stage compression ratios, assume N_i=1 for all stages (N_S=1).
    n_list = [1.0 for _ in d_model]

    d_s = float(d_model[-1])
    multipliers = []
    prefix = 1.0
    for stage_idx, d_stage in enumerate(d_model):
        ratio = 1.0 / max(prefix, 1e-12)
        scale = n_gpt * ratio * (d_s / float(d_stage))
        multipliers.append(math.sqrt(max(scale, 1e-12)))
        prefix *= n_list[stage_idx]
    return multipliers


class RegionCompressionTracker:
    def __init__(self, region_names: list[str]):
        self.region_names = list(dict.fromkeys(region_names))
        self.reset()

    def reset(self):
        self.region_bp = {k: 0.0 for k in self.region_names}
        self.region_tokens = {k: 0.0 for k in self.region_names}
        self.region_prob = {k: 0.0 for k in self.region_names}
        self._prob_collected = False

    def update(self, stats: dict[str, tuple]):
        for name, values in stats.items():
            if name not in self.region_names:
                continue
            if not isinstance(values, (tuple, list)) or len(values) < 2:
                continue
            bp = values[0]
            tokens = values[1]
            self.region_bp[name] += self._to_float(bp)
            self.region_tokens[name] += self._to_float(tokens)
            if len(values) > 2:
                prob_sum = values[2]
                self.region_prob[name] += self._to_float(prob_sum)
                self._prob_collected = True

    def metrics(self, prefix: str = "eval_") -> dict[str, float]:
        metrics = {}
        for name in self.region_names:
            bp = self.region_bp.get(name, 0.0)
            tokens = self.region_tokens.get(name, 0.0)
            if bp <= 0:
                continue
            metrics[f"{prefix}bp_per_token/{name}"] = bp / (tokens if tokens > 0 else 1.0)
        return metrics

    def summary_metrics(self, prefix: str, name_mapper, region_names=None) -> dict[str, float]:
        if region_names is None:
            region_names = self.region_names
        metrics = {}
        total_bp = sum(self.region_bp.get(name, 0.0) for name in region_names)
        total_tokens = sum(self.region_tokens.get(name, 0.0) for name in region_names)
        total_prob = sum(self.region_prob.get(name, 0.0) for name in region_names)
        if total_bp > 0:
            metrics[f"{prefix}F"] = total_tokens / total_bp
            if self._prob_collected:
                metrics[f"{prefix}G"] = total_prob / total_bp
            if total_tokens > 0:
                metrics[f"{prefix}avg_bp_per_token"] = total_bp / total_tokens
        for name in region_names:
            bp = self.region_bp.get(name, 0.0)
            if bp <= 0:
                continue
            region_key = name_mapper(name)
            metrics[f"{prefix}F_{region_key}"] = self.region_tokens.get(name, 0.0) / bp
            if self._prob_collected:
                metrics[f"{prefix}G_{region_key}"] = self.region_prob.get(name, 0.0) / bp
        return metrics

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)


class RegionLossTracker:
    def __init__(self, region_names: list[str]):
        self.region_names = list(dict.fromkeys(region_names))
        self.reset()

    def reset(self):
        self.region_loss_sum = {k: 0.0 for k in self.region_names}
        self.region_loss_count = {k: 0.0 for k in self.region_names}

    def update(self, stats: dict[str, tuple]):
        for name, values in stats.items():
            if name not in self.region_names:
                continue
            if not isinstance(values, (tuple, list)) or len(values) < 2:
                continue
            loss_sum = values[0]
            count = values[1]
            self.region_loss_sum[name] += self._to_float(loss_sum)
            self.region_loss_count[name] += self._to_float(count)

    def metrics(self, prefix: str, name_mapper) -> dict[str, float]:
        metrics = {}
        for name in self.region_names:
            count = self.region_loss_count.get(name, 0.0)
            if count <= 0:
                continue
            region_key = name_mapper(name)
            avg_loss = self.region_loss_sum[name] / count
            metrics[f"{prefix}ppl_{region_key}"] = math.exp(avg_loss)
        return metrics

    @staticmethod
    def _to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)


class RoutingModuleWithFloor(RoutingModule):
    def __init__(
        self,
        d_model: int,
        *,
        routing_floor,
        stage_idx: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__(d_model, device=device, dtype=dtype)
        self._routing_floor = routing_floor
        self._stage_idx = None if stage_idx is None else int(stage_idx)

    def _routing_floor_tokens(self) -> int:
        return _resolve_stage_tokens(getattr(self._routing_floor, "k_min_list", []), self._stage_idx)

    @staticmethod
    def _count_selected_tokens(boundary_mask) -> float:
        if boundary_mask.dim() == 1:
            return float(boundary_mask.sum().item())
        return float(boundary_mask.sum(dim=-1).float().mean().item())

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        output = super().forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params,
        )
        output.selected_tokens_pre = self._count_selected_tokens(output.boundary_mask)
        k_min = self._routing_floor_tokens()
        if k_min <= 0:
            return output
        floored_mask, floor_stats = apply_routing_floor(
            output.boundary_mask,
            output.boundary_prob,
            k_min=k_min,
            mask=mask,
            cu_seqlens=cu_seqlens,
            return_stats=True,
        )
        output.boundary_mask = floored_mask
        if floor_stats is not None:
            output.floor_stats = floor_stats
        return output


def _wrap_routing_modules(module: nn.Module, routing_floor) -> None:
    stage_idx = getattr(module, "stage_idx", None)
    for name, child in module.named_children():
        if isinstance(child, RoutingModuleWithFloor):
            continue
        if isinstance(child, RoutingModule):
            device = child.q_proj_layer.weight.device
            dtype = child.q_proj_layer.weight.dtype
            wrapped = RoutingModuleWithFloor(
                child.d_model,
                routing_floor=routing_floor,
                stage_idx=stage_idx,
                device=device,
                dtype=dtype,
            )
            wrapped.load_state_dict(child.state_dict())
            setattr(module, name, wrapped)
        else:
            _wrap_routing_modules(child, routing_floor)

def _wrap_routing_modules_ceil(module: nn.Module, routing_ceiling) -> None:
    stage_idx = getattr(module, "stage_idx", None)
    for name, child in module.named_children():
        if isinstance(child, RoutingModuleWithCeiling):
            continue
        if isinstance(child, RoutingModule):
            device = child.q_proj_layer.weight.device
            dtype = child.q_proj_layer.weight.dtype
            wrapped = RoutingModuleWithCeiling(
                child.d_model,
                routing_ceiling=routing_ceiling,
                stage_idx=stage_idx,
                device=device,
                dtype=dtype,
            )
            wrapped.load_state_dict(child.state_dict())
            setattr(module, name, wrapped)
        else:
            _wrap_routing_modules_ceil(child, routing_ceiling)


class HNetForCausalLMWithRoutingFloor(HNetForCausalLM):
    def __init__(
        self,
        config: HNetConfig,
        routing_floor,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        _wrap_routing_modules(self.backbone, routing_floor)
        self._routing_floor = routing_floor

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        routing_step: int | None = None,
        **mixer_kwargs,
    ):
        del routing_step  # schedule removed; kept for callsite compatibility
        return super().forward(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            **mixer_kwargs,
        )

class RoutingModuleWithCeiling(RoutingModule):
    def __init__(
        self,
        d_model: int,
        *,
        routing_ceiling,
        stage_idx: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__(d_model, device=device, dtype=dtype)
        self._routing_ceiling = routing_ceiling
        self._stage_idx = None if stage_idx is None else int(stage_idx)

    def _set_out(self, out, key: str, value):
        # support dict-like or attribute-like
        if isinstance(out, dict):
            out[key] = value
        else:
            setattr(out, key, value)

    def _get_out(self, out, key: str, default=None):
        if isinstance(out, dict):
            return out.get(key, default)
        return getattr(out, key, default)

    def _routing_ceiling_tokens(self) -> int:
        return _resolve_stage_tokens(
            getattr(self._routing_ceiling, "k_max_list", []),
            self._stage_idx,
        )

    @staticmethod
    def _count_selected_tokens(boundary_mask) -> float:
        if boundary_mask.dim() == 1:
            return float(boundary_mask.sum().item())
        return float(boundary_mask.sum(dim=-1).float().mean().item())

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        output = super().forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params,
        )

        # -------- save PRE-ceiling copies (Fix C) --------
        bm_pre = self._get_out(output, "boundary_mask", None)
        bp_pre = self._get_out(output, "boundary_prob", None)

        if bm_pre is not None:
            self._set_out(output, "boundary_mask_pre", bm_pre)
            self._set_out(output, "selected_tokens_pre", self._count_selected_tokens(bm_pre))
        if bp_pre is not None:
            # keep full boundary_prob tensor; trainer will take last channel if needed
            self._set_out(output, "boundary_prob_pre", bp_pre)

        # ===== apply ceiling (max) =====
        k_max = self._routing_ceiling_tokens()
        if k_max > 0:
            capped_mask, ceil_stats = apply_routing_ceiling(
                bm_pre,
                bp_pre,
                k_max=k_max,
                mask=mask,
                cu_seqlens=cu_seqlens,
                return_stats=True,
            )
            self._set_out(output, "boundary_mask", capped_mask)
            if ceil_stats is not None:
                self._set_out(output, "ceiling_stats", ceil_stats)

        # post stats
        bm_post = self._get_out(output, "boundary_mask", None)
        if bm_post is not None:
            self._set_out(output, "selected_tokens_post", self._count_selected_tokens(bm_post))

        return output


class HNetForCausalLMWithRoutingCeiling(HNetForCausalLM):
    def __init__(
        self,
        config: HNetConfig,
        routing_ceiling,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        _wrap_routing_modules_ceil(self.backbone, routing_ceiling)
        self._routing_ceiling = routing_ceiling

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        routing_step: int | None = None,
        **mixer_kwargs,
    ):
        del routing_step  # schedule removed; kept for callsite compatibility
        return super().forward(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            **mixer_kwargs,
        )


class RoutingModuleWithFloorAndCeiling(RoutingModule):
    def __init__(
        self,
        d_model: int,
        *,
        routing_floor,
        routing_ceiling,
        stage_idx: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__(d_model, device=device, dtype=dtype)
        self._routing_floor = routing_floor
        self._routing_ceiling = routing_ceiling
        self._stage_idx = None if stage_idx is None else int(stage_idx)

    def _set_out(self, out, key: str, value):
        if isinstance(out, dict):
            out[key] = value
        else:
            setattr(out, key, value)

    def _get_out(self, out, key: str, default=None):
        if isinstance(out, dict):
            return out.get(key, default)
        return getattr(out, key, default)

    def _routing_floor_tokens(self) -> int:
        return _resolve_stage_tokens(
            getattr(self._routing_floor, "k_min_list", []),
            self._stage_idx,
        )

    def _routing_ceiling_tokens(self) -> int:
        return _resolve_stage_tokens(
            getattr(self._routing_ceiling, "k_max_list", []),
            self._stage_idx,
        )

    @staticmethod
    def _count_selected_tokens(boundary_mask) -> float:
        if boundary_mask.dim() == 1:
            return float(boundary_mask.sum().item())
        return float(boundary_mask.sum(dim=-1).float().mean().item())

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        output = super().forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params,
        )

        bm_pre = self._get_out(output, "boundary_mask", None)
        bp_pre = self._get_out(output, "boundary_prob", None)
        if bm_pre is None or bp_pre is None:
            return output

        self._set_out(output, "boundary_mask_pre", bm_pre)
        self._set_out(output, "boundary_prob_pre", bp_pre)
        self._set_out(output, "selected_tokens_pre", self._count_selected_tokens(bm_pre))

        k_min = self._routing_floor_tokens()
        floored_mask = bm_pre
        if k_min > 0:
            floored_mask, floor_stats = apply_routing_floor(
                bm_pre,
                bp_pre,
                k_min=k_min,
                mask=mask,
                cu_seqlens=cu_seqlens,
                return_stats=True,
            )
            self._set_out(output, "boundary_mask", floored_mask)
            if floor_stats is not None:
                self._set_out(output, "floor_stats", floor_stats)

        k_max = self._routing_ceiling_tokens()
        if k_max > 0:
            capped_mask, ceil_stats = apply_routing_ceiling(
                floored_mask,
                bp_pre,
                k_max=k_max,
                mask=mask,
                cu_seqlens=cu_seqlens,
                return_stats=True,
            )
            self._set_out(output, "boundary_mask", capped_mask)
            if ceil_stats is not None:
                self._set_out(output, "ceiling_stats", ceil_stats)

        bm_post = self._get_out(output, "boundary_mask", None)
        if bm_post is not None:
            self._set_out(output, "selected_tokens_post", self._count_selected_tokens(bm_post))

        return output


def _wrap_routing_modules_floor_ceiling(
    module: nn.Module,
    routing_floor,
    routing_ceiling,
) -> None:
    stage_idx = getattr(module, "stage_idx", None)
    for name, child in module.named_children():
        if isinstance(child, RoutingModuleWithFloorAndCeiling):
            continue
        if isinstance(child, RoutingModule):
            device = child.q_proj_layer.weight.device
            dtype = child.q_proj_layer.weight.dtype
            wrapped = RoutingModuleWithFloorAndCeiling(
                child.d_model,
                routing_floor=routing_floor,
                routing_ceiling=routing_ceiling,
                stage_idx=stage_idx,
                device=device,
                dtype=dtype,
            )
            wrapped.load_state_dict(child.state_dict())
            setattr(module, name, wrapped)
        else:
            _wrap_routing_modules_floor_ceiling(
                child,
                routing_floor,
                routing_ceiling,
            )


class HNetForCausalLMWithRoutingFloorAndCeiling(HNetForCausalLM):
    def __init__(
        self,
        config: HNetConfig,
        routing_floor,
        routing_ceiling,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config, device=device, dtype=dtype)
        _wrap_routing_modules_floor_ceiling(self.backbone, routing_floor, routing_ceiling)
        self._routing_floor = routing_floor
        self._routing_ceiling = routing_ceiling

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        routing_step: int | None = None,
        **mixer_kwargs,
    ):
        del routing_step  # schedule removed; kept for callsite compatibility
        return super().forward(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            **mixer_kwargs,
        )
