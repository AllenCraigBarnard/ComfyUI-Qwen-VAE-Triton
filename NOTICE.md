# Notice

This project contains a standalone quality-focused implementation of a Triton W8A8 implicit-GEMM convolution path for ComfyUI's Qwen-Image/Wan VAE.

The design and portions of the implementation were informed by the GPL-3.0 licensed `Patch Triton VAE` implementation in `kijai/ComfyUI-KJNodes`, particularly its INT8 implicit-GEMM Conv3D approach, Qwen/Wan causal-convolution handling, and ComfyUI object-patcher integration.

Upstream project:

- ComfyUI-KJNodes: https://github.com/kijai/ComfyUI-KJNodes
- Upstream license: GNU General Public License v3.0

To remain compatible with the upstream licensing basis, this package is distributed under **GPL-3.0-only**. See `LICENSE`.

This package is unofficial and is not endorsed by Krea, Qwen/Alibaba, ComfyUI, AMD, or the KJNodes author.
