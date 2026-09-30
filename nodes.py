from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

import torch

import comfy.model_patcher
from comfy.ldm.wan.vae import CausalConv3d
from comfy.patcher_extension import CallbacksMP

from .selection import AGGRESSIVE_HARD_EXCLUDE_REGEX, LayerInfo, resolve_min_channels, select_layers
from .triton_w8a8 import W8A8CausalConv3d

LOGGER = logging.getLogger("ComfyUI-Qwen-VAE-Triton")
VERSION = "0.2.3"


def _eligible_qwen_decoder_layers(model) -> List[Tuple[str, CausalConv3d]]:
    layers: List[Tuple[str, CausalConv3d]] = []
    for name, mod in model.named_modules():
        if not name.startswith("decoder."):
            continue
        if not isinstance(mod, CausalConv3d):
            continue
        if tuple(mod.kernel_size) != (3, 3, 3):
            continue
        if tuple(mod.stride) != (1, 1, 1):
            continue
        if tuple(mod.dilation) != (1, 1, 1):
            continue
        if mod.groups != 1:
            continue
        # Keep latent-facing input and RGB-facing output automatically native.
        if mod.in_channels < 64 or mod.out_channels < 64:
            continue
        if mod.in_channels % 32 != 0:
            continue
        layers.append((name, mod))
    return layers


def _convert_conv_layout(model) -> None:
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv3d):
            mod.to(memory_format=torch.channels_last_3d)


def _clone_vae_for_object_patching(vae):
    vae = copy.copy(vae)
    model = vae.first_stage_model

    if vae.patcher.is_dynamic():
        model.to(vae.vae_dtype)
        new_patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=vae.patcher.load_device,
            offload_device=vae.patcher.offload_device,
        )
        new_patcher.parent = vae.patcher
        vae.patcher = new_patcher
    else:
        vae.patcher = vae.patcher.clone()

    return vae


class PatchQwenVAEW8A8:
    """Layer-selective Triton W8A8 patch for the Qwen-Image/Wan VAE decoder."""

    CATEGORY = "Krea2 Optimization/VAE"
    FUNCTION = "patch"
    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "Preset": (
                    ["Conservative", "Balanced", "Aggressive"],
                    {
                        "default": "Balanced",
                        "tooltip": (
                            "Conservative = highest decoder channel tier only; "
                            "Balanced = highest two tiers; Aggressive = all eligible tiers "
                            "except the validated U14 residual.2/residual.6 native-precision pair."
                        ),
                    },
                ),
                "disable_node": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, pass the input VAE through unchanged without applying W8A8 patches.",
                    },
                ),
            }
        }

    def patch(self, vae, Preset, disable_node):
        if disable_node:
            report = (
                f"ComfyUI-Qwen-VAE-Triton {VERSION}\n"
                "Node disabled: true\n"
                "VAE passed through unchanged."
            )
            LOGGER.info("\n%s", report)
            return (vae, report)

        vae = _clone_vae_for_object_patching(vae)
        model = vae.first_stage_model

        eligible = _eligible_qwen_decoder_layers(model)
        if not eligible:
            raise RuntimeError(
                "No eligible Qwen/Wan decoder CausalConv3d layers were found. "
                "This node currently targets the Qwen-Image/Wan VAE architecture only."
            )

        info = [LayerInfo(name=n, in_channels=m.in_channels, out_channels=m.out_channels) for n, m in eligible]
        resolved_min_channels, available_tiers = resolve_min_channels(info, Preset)
        selected_info = select_layers(info, Preset)
        selected_names = {x.name for x in selected_info}

        patches: Dict[str, W8A8CausalConv3d] = {}
        for name, mod in eligible:
            if name not in selected_names:
                continue
            patches[name] = W8A8CausalConv3d(
                mod,
                layer_name=name,
                activation_clip_ratio=1.0,
                weight_clip_ratio=1.0,
                fallback_on_error=True,
                profile=False,
            )

        if not patches:
            raise RuntimeError(
                f"Preset {Preset!r} selected zero W8A8 layers. "
                f"Resolved minimum channel tier: {resolved_min_channels}; "
                f"available tiers: {available_tiers}."
            )

        for name, obj in patches.items():
            vae.patcher.add_object_patch(name, obj)

        # Keep the VAE resident while runtime INT8 buffers are active.
        vae.disable_offload = True

        # Reapply channels-last layout after ComfyUI loads/rematerializes the VAE.
        def _reapply_channels_last(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            _convert_conv_layout(patcher.model)

        vae.patcher.add_callback_with_key(
            CallbacksMP.ON_LOAD,
            "qwen_vae_w8a8_channels_last",
            _reapply_channels_last,
        )
        _convert_conv_layout(model)

        selected_sorted = sorted(patches)
        native_sorted = sorted(name for name, _ in eligible if name not in selected_names)
        preset_key = str(Preset).strip().lower()
        report_lines = [
            f"ComfyUI-Qwen-VAE-Triton {VERSION}",
            f"Preset: {Preset}",
            "Node disabled: false",
            f"Eligible decoder Conv3D layers: {len(eligible)}",
            f"Available channel tiers (min(in,out)): {available_tiers}",
            f"Resolved minimum channel tier: {resolved_min_channels}",
            f"Built-in aggressive exclusion: {AGGRESSIVE_HARD_EXCLUDE_REGEX if preset_key == 'aggressive' else 'none'}",
            f"W8A8 patched layers: {len(selected_sorted)}",
            f"Native precision eligible layers: {len(native_sorted)}",
            "Activation clip ratio: 1.0000 (fixed)",
            "Weight clip ratio: 1.0000 (fixed)",
            "channels_last: true (fixed)",
            "fallback_on_error: true (fixed)",
            "W8A8 layers:",
            *[f"  {name}" for name in selected_sorted],
        ]
        if native_sorted:
            report_lines.extend(["Native eligible layers:", *[f"  {name}" for name in native_sorted]])

        report = "\n".join(report_lines)
        LOGGER.info("\n%s", report)
        return (vae, report)


NODE_CLASS_MAPPINGS = {
    "PatchQwenVAEW8A8": PatchQwenVAEW8A8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchQwenVAEW8A8": "Patch Qwen VAE Triton W8A8",
}
