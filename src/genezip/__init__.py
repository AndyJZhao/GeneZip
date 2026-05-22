from __future__ import annotations

"""GeneZip package.

Keep imports lightweight so config helpers can be used without importing the full
model stack (and optional GPU kernel deps).
"""

__all__ = [
    "RoutingCeilingConfig",
    "RoutingFloorConfig",
    "parse_routing_ceiling_config",
    "parse_routing_floor_config",
    "HNetForCausalLMWithRoutingFloor",
    "coverage_aware_pooling",
    "coverage_aware_pooling_num_den",
]


def __getattr__(name: str):
    if name in {
        "RoutingCeilingConfig",
        "RoutingFloorConfig",
        "parse_routing_ceiling_config",
        "parse_routing_floor_config",
    }:
        from .config import (
            RoutingCeilingConfig,
            RoutingFloorConfig,
            parse_routing_ceiling_config,
            parse_routing_floor_config,
        )

        return {
            "RoutingCeilingConfig": RoutingCeilingConfig,
            "RoutingFloorConfig": RoutingFloorConfig,
            "parse_routing_ceiling_config": parse_routing_ceiling_config,
            "parse_routing_floor_config": parse_routing_floor_config,
        }[name]

    if name in {"coverage_aware_pooling", "coverage_aware_pooling_num_den"}:
        from .embedding import coverage_aware_pooling, coverage_aware_pooling_num_den

        return {
            "coverage_aware_pooling": coverage_aware_pooling,
            "coverage_aware_pooling_num_den": coverage_aware_pooling_num_den,
        }[name]

    if name == "HNetForCausalLMWithRoutingFloor":
        from .region_aware_hnet import HNetForCausalLMWithRoutingFloor

        return HNetForCausalLMWithRoutingFloor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
