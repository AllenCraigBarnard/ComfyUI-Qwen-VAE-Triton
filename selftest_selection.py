#!/usr/bin/env python3
"""Standalone PRC2 selection-policy self-test. Does not require ComfyUI."""

from selection import AGGRESSIVE_HARD_EXCLUDE_REGEX, LayerInfo, VALID_PRESETS, select_layers


def main() -> None:
    layers = [
        LayerInfo("decoder.middle.0.residual.2", 384, 384),
        LayerInfo("decoder.middle.0.residual.6", 384, 384),
        LayerInfo("decoder.upsamples.10.residual.2", 192, 192),
        LayerInfo("decoder.upsamples.10.residual.6", 192, 192),
        LayerInfo("decoder.upsamples.12.residual.2", 96, 96),
        LayerInfo("decoder.upsamples.12.residual.6", 96, 96),
        LayerInfo("decoder.upsamples.13.residual.2", 96, 96),
        LayerInfo("decoder.upsamples.13.residual.6", 96, 96),
        LayerInfo("decoder.upsamples.14.residual.2", 96, 96),
        LayerInfo("decoder.upsamples.14.residual.6", 96, 96),
    ]

    assert VALID_PRESETS == ("Conservative", "Balanced", "Aggressive")

    conservative = {x.name for x in select_layers(layers, "Conservative")}
    assert conservative == {
        "decoder.middle.0.residual.2",
        "decoder.middle.0.residual.6",
    }

    balanced = {x.name for x in select_layers(layers, "Balanced")}
    assert "decoder.upsamples.10.residual.2" in balanced
    assert "decoder.upsamples.12.residual.2" not in balanced

    aggressive = {x.name for x in select_layers(layers, "Aggressive")}
    assert "decoder.upsamples.14.residual.2" not in aggressive
    assert "decoder.upsamples.14.residual.6" not in aggressive
    assert "decoder.upsamples.12.residual.2" in aggressive
    assert "decoder.upsamples.13.residual.6" in aggressive

    assert AGGRESSIVE_HARD_EXCLUDE_REGEX == r"^decoder\.upsamples\.14\.residual\.(2|6)$"
    print("v0.2.2 selection policy self-test: PASS")


if __name__ == "__main__":
    main()
