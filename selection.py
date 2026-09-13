from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence, Tuple


AGGRESSIVE_HARD_EXCLUDE_REGEX = r"^decoder\.upsamples\.14\.residual\.(2|6)$"
VALID_PRESETS = ("Conservative", "Balanced", "Aggressive")


@dataclass(frozen=True)
class LayerInfo:
    name: str
    in_channels: int
    out_channels: int

    @property
    def min_channels(self) -> int:
        return min(self.in_channels, self.out_channels)


def channel_tiers(layers: Sequence[LayerInfo]) -> List[int]:
    """Distinct eligible min(in,out) channel tiers, largest first."""
    return sorted({layer.min_channels for layer in layers}, reverse=True)


def normalize_preset(preset: str) -> str:
    normalized = str(preset).strip().lower()
    if normalized not in {"conservative", "balanced", "aggressive"}:
        raise ValueError(
            f"Unknown preset: {preset!r}. Expected one of: {', '.join(VALID_PRESETS)}"
        )
    return normalized


def resolve_min_channels(
    layers: Sequence[LayerInfo],
    preset: str,
) -> Tuple[int, List[int]]:
    """
    Resolve the architecture-adaptive channel threshold.

    Conservative: highest available channel tier only.
    Balanced:     highest two available channel tiers.
    Aggressive:   all eligible tiers except the validated U14 noise-sensitive pair.
    """
    tiers = channel_tiers(layers)
    if not tiers:
        raise ValueError("Cannot resolve a preset without eligible layers")

    preset_key = normalize_preset(preset)
    if preset_key == "conservative":
        return tiers[0], tiers
    if preset_key == "balanced":
        return tiers[1] if len(tiers) >= 2 else tiers[0], tiers
    return tiers[-1], tiers


def select_layers(
    layers: Sequence[LayerInfo],
    preset: str,
) -> List[LayerInfo]:
    preset_key = normalize_preset(preset)
    min_channels, _ = resolve_min_channels(layers, preset_key)
    aggressive_hard_exclude_re = (
        re.compile(AGGRESSIVE_HARD_EXCLUDE_REGEX) if preset_key == "aggressive" else None
    )

    selected: List[LayerInfo] = []
    for layer in layers:
        if layer.min_channels < min_channels:
            continue
        if aggressive_hard_exclude_re is not None and aggressive_hard_exclude_re.search(layer.name):
            continue
        selected.append(layer)
    return selected
