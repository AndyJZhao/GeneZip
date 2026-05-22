from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class RoutingFloorConfig:
    k_min_list: list[int] = field(default_factory=list)

    def enabled(self) -> bool:
        return any(int(value or 0) > 0 for value in self.k_min_list)

@dataclass
class RoutingCeilingConfig:
    k_max_list: list[int] = field(default_factory=list)

    def enabled(self) -> bool:
        return any(int(value or 0) > 0 for value in self.k_max_list)


def parse_routing_floor_config(cfg_dict: Mapping[str, Any] | None) -> RoutingFloorConfig:
    if not cfg_dict:
        return RoutingFloorConfig()
    routing_floor = cfg_dict.get("routing_floor")
    source = routing_floor if isinstance(routing_floor, Mapping) else cfg_dict
    k_min = source.get("k_min_list", [])
    if isinstance(k_min, Sequence) and not isinstance(k_min, (str, bytes)):
        k_min_list = [max(0, int(value or 0)) for value in k_min]
    else:
        k_min_list = [max(0, int(k_min or 0))]
    return RoutingFloorConfig(
        k_min_list=k_min_list,
    )

def parse_routing_ceiling_config(cfg_dict: Mapping[str, Any] | None) -> RoutingCeilingConfig:
    if not cfg_dict:
        return RoutingCeilingConfig()
    routing_ceiling = cfg_dict.get("routing_ceiling")
    source = routing_ceiling if isinstance(routing_ceiling, Mapping) else cfg_dict
    k_max = source.get("k_max_list", [])
    if isinstance(k_max, Sequence) and not isinstance(k_max, (str, bytes)):
        k_max_list = [max(0, int(value or 0)) for value in k_max]
    else:
        k_max_list = [max(0, int(k_max or 0))]
    return RoutingCeilingConfig(
        k_max_list=k_max_list,
    )
