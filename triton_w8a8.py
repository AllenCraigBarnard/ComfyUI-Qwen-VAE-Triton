from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - depends on Comfy runtime
    raise ImportError(
        "ComfyUI-Qwen-VAE-Triton requires Triton. The user's ComfyUI ROCm/Triton environment "
        "must provide a working 'triton' Python package."
    ) from exc

from comfy.ldm.wan.vae import CausalConv3d

CL3D = torch.channels_last_3d
LOGGER = logging.getLogger("ComfyUI-Qwen-VAE-Triton")


@triton.jit
def _quantize_padded_s8(
    x_ptr,
    cache_ptr,
    out_ptr,
    amax_ptr,
    C,
    C_PAD,
    S_PAD,
    S_CACHE,
    N_TOTAL,
    CLIP_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Quantize NDHWC payload/cache into one padded INT8 activation buffer."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_TOTAL

    amax = tl.maximum(tl.load(amax_ptr).to(tl.float32) * CLIP_RATIO, 1e-8)
    inv = 127.0 / amax

    s_idx = offs // C_PAD
    c_idx = offs % C_PAD
    valid_c = mask & (c_idx < C)
    in_cache = valid_c & (s_idx >= S_PAD) & (s_idx < S_PAD + S_CACHE)
    in_payload = valid_c & (s_idx >= S_PAD + S_CACHE)

    payload_v = tl.load(
        x_ptr + (s_idx - S_PAD - S_CACHE) * C + c_idx,
        mask=in_payload,
        other=0.0,
    ).to(tl.float32)
    cache_v = tl.load(
        cache_ptr + (s_idx - S_PAD) * C + c_idx,
        mask=in_cache,
        other=0.0,
    ).to(tl.float32)
    v = payload_v + cache_v
    v = v * inv
    v = tl.minimum(tl.maximum(v + tl.where(v >= 0, 0.5, -0.5), -127.0), 127.0)
    tl.store(out_ptr + offs, v.to(tl.int8), mask=mask)


@triton.jit
def _conv3d_s8_implicit_gemm(
    x_ptr,
    w_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    H,
    W,
    C,
    C_OUT,
    C3,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """3x3x3 stride-1/pad-1 spatial causal conv as implicit INT8 GEMM."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < C_OUT

    w_o = offs_m % W
    h_o = (offs_m // W) % H
    t_o = offs_m // (W * H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # 3 temporal taps x 3 vertical taps. The 3 horizontal taps are packed
    # contiguously into C3 = 3*C and traversed by BLOCK_K chunks.
    for g in tl.static_range(9):
        dt = g // 3
        dh = g % 3
        t_i = t_o + dt
        h_i = h_o + dh - 1
        h_ok = mask_m & (h_i >= 0) & (h_i < H)
        base = ((t_i * H + h_i) * W + (w_o - 1)) * C

        for k0 in range(0, C3, BLOCK_K):
            dw = k0 // C
            spatial_ok = h_ok & (w_o + dw - 1 >= 0) & (w_o + dw - 1 < W)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                x_ptr + base[:, None] + offs_k[None, :],
                mask=spatial_ok[:, None],
                other=0,
            )
            b = tl.load(
                w_ptr + (g * C3 + offs_k)[:, None] * C_OUT + offs_n[None, :],
                mask=mask_n[None, :],
                other=0,
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    y = acc.to(tl.float32) * tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    if HAS_BIAS:
        y += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * C_OUT + offs_n[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _pack_weight_per_output_channel(
    weight: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return kernel-packed int8 weight, FP32 per-output scale, padded C-in."""
    if weight.ndim != 5:
        raise ValueError(f"Expected Conv3D weight rank 5, got {weight.ndim}")
    if not (0.5 <= clip_ratio <= 1.0):
        raise ValueError("weight_clip_ratio must be in [0.5, 1.0]")

    with torch.no_grad():
        absmax = weight.detach().float().abs().amax(dim=(1, 2, 3, 4)).clamp(min=1e-8)
        scale = (absmax * float(clip_ratio)) / 127.0
        q = torch.round(weight.detach().float() / scale[:, None, None, None, None])
        q = q.clamp(-127, 127).to(torch.int8)

        c_in = int(weight.shape[1])
        c_pad = ((c_in + 63) // 64) * 64
        if c_pad != c_in:
            q = F.pad(q, (0, 0, 0, 0, 0, 0, 0, c_pad - c_in))

        # [Cout,Cin,Kt,Kh,Kw] -> [Kt,Kh,Kw,Cin,Cout] -> [K,Cout]
        packed = q.permute(2, 3, 4, 1, 0).reshape(-1, q.shape[0]).contiguous()
        return packed, scale.contiguous(), c_pad


class W8A8CausalConv3d(CausalConv3d):
    """
    Qwen/Wan CausalConv3d replacement.

    - W8: symmetric INT8, per-output-channel weight scale.
    - A8: dynamic symmetric INT8 activation scale.
    - Accumulation: INT32 in Triton; dequantization in FP32; output matches input dtype.
    - Unsupported execution falls back to the original BF16/FP16 convolution semantics.
    """

    def __init__(
        self,
        orig: CausalConv3d,
        layer_name: str,
        activation_clip_ratio: float = 1.0,
        weight_clip_ratio: float = 1.0,
        fallback_on_error: bool = True,
        profile: bool = False,
    ):
        super().__init__(
            orig.in_channels,
            orig.out_channels,
            orig.kernel_size,
            stride=orig.stride,
            padding=1,
            dilation=orig.dilation,
            groups=orig.groups,
            bias=orig.bias is not None,
        )
        self.weight = orig.weight
        self.bias = orig.bias
        self.layer_name = layer_name
        self.activation_clip_ratio = float(activation_clip_ratio)
        self.weight_clip_ratio = float(weight_clip_ratio)
        self.fallback_on_error = bool(fallback_on_error)
        self.profile = bool(profile)
        self._last_amax: Optional[torch.Tensor] = None
        self._logged_active = False
        self._logged_fallback = set()

        packed_w, weight_scale, c_pad = _pack_weight_per_output_channel(
            orig.weight,
            self.weight_clip_ratio,
        )
        self.int8_cpad = c_pad
        self.register_buffer("int8_weight", packed_w, persistent=False)
        # Activation quantization is qx ~= x / a_scale, where a_scale = amax*clip/127.
        # Kernel receives combined dequant scale a_scale * w_scale.
        self.register_buffer("weight_scale", weight_scale, persistent=False)

    def _fallback(self, x, cache_x, cache_list, cache_idx, reason: str):
        if reason not in self._logged_fallback:
            LOGGER.info("W8A8 fallback [%s]: %s", self.layer_name, reason)
            self._logged_fallback.add(reason)
        return super().forward(x, cache_x=cache_x, cache_list=cache_list, cache_idx=cache_idx)

    def forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
        if not x.is_cuda:
            return self._fallback(x, cache_x, cache_list, cache_idx, "non-accelerator tensor")
        if x.ndim != 5:
            return self._fallback(x, cache_x, cache_list, cache_idx, f"rank {x.ndim} input")
        if x.shape[0] != 1:
            return self._fallback(x, cache_x, cache_list, cache_idx, f"batch size {x.shape[0]}")
        if self.groups != 1:
            return self._fallback(x, cache_x, cache_list, cache_idx, f"groups={self.groups}")
        if tuple(self.kernel_size) != (3, 3, 3) or tuple(self.stride) != (1, 1, 1):
            return self._fallback(x, cache_x, cache_list, cache_idx, "unsupported kernel/stride")

        try:
            out = self._w8a8_forward(x, cache_x, cache_list, cache_idx)
            if not self._logged_active:
                LOGGER.info(
                    "W8A8 Triton active [%s]: %d -> %d channels, activation_clip=%.4f, weight_clip=%.4f",
                    self.layer_name,
                    self.in_channels,
                    self.out_channels,
                    self.activation_clip_ratio,
                    self.weight_clip_ratio,
                )
                self._logged_active = True
            return out
        except Exception as exc:
            if not self.fallback_on_error:
                raise
            LOGGER.exception("W8A8 kernel failed for %s; falling back to native convolution", self.layer_name)
            return self._fallback(x, cache_x, cache_list, cache_idx, f"kernel exception: {type(exc).__name__}")

    def _w8a8_forward(self, x, cache_x, cache_list, cache_idx):
        if cache_list is not None:
            cache_x = cache_list[cache_idx]
            cache_list[cache_idx] = None

        if not x.is_contiguous(memory_format=CL3D) or x.is_contiguous():
            x = x.contiguous(memory_format=CL3D)

        _, C, T, H, W = x.shape
        cache_t = 0 if cache_x is None else int(cache_x.shape[2])
        pad_t = max(0, 2 - cache_t)
        t_in = T + cache_t + pad_t

        # Global dynamic A8 scale for this layer invocation. Layer-selective mixed
        # precision is the primary quality control in v0.1.x; groupwise activation
        # scaling is intentionally deferred until benchmarked on gfx1151.
        amax = x.detach().abs().amax().float()
        if cache_t and self._last_amax is not None:
            amax = torch.maximum(amax, self._last_amax)
        self._last_amax = amax.detach()

        x_rows = x.permute(0, 2, 3, 4, 1).reshape(-1)
        if cache_t:
            if not cache_x.is_contiguous(memory_format=CL3D) or cache_x.is_contiguous():
                cache_x = cache_x.contiguous(memory_format=CL3D)
            cache_rows = cache_x.permute(0, 2, 3, 4, 1).reshape(-1)
        else:
            cache_rows = x_rows

        c_pad = self.int8_cpad
        hw = H * W
        n_total = t_in * hw * c_pad
        qx = torch.empty(n_total, device=x.device, dtype=torch.int8)

        _quantize_padded_s8[(triton.cdiv(n_total, 4096),)](
            x_rows,
            cache_rows,
            qx,
            amax,
            C,
            c_pad,
            pad_t * hw,
            cache_t * hw,
            n_total,
            CLIP_RATIO=self.activation_clip_ratio,
            BLOCK=4096,
            num_warps=8,
        )

        c_out = self.out_channels
        activation_scale = (amax * self.activation_clip_ratio).clamp(min=1e-8) / 127.0
        scale_vec = self.weight_scale.float() * activation_scale
        bias = self.bias
        t_out = t_in - 2
        m = t_out * hw
        out = torch.empty(m, c_out, device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(m, 128), triton.cdiv(c_out, 64))

        if self.profile:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        _conv3d_s8_implicit_gemm[grid](
            qx,
            self.int8_weight,
            out,
            scale_vec,
            bias.float() if bias is not None else scale_vec,
            H,
            W,
            c_pad,
            c_out,
            3 * c_pad,
            m,
            BLOCK_M=128,
            BLOCK_N=64,
            BLOCK_K=64,
            HAS_BIAS=bias is not None,
            num_warps=4,
            num_stages=4,
        )

        if self.profile:
            ev1.record()
            torch.cuda.synchronize()
            LOGGER.info("W8A8 profile [%s]: %.3f ms", self.layer_name, ev0.elapsed_time(ev1))

        return out.view(1, t_out, H, W, c_out).permute(0, 4, 1, 2, 3)
