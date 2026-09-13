# Changelog

## 0.2.0 — Initial public release

- Promoted the validated `0.2.0-rc2` runtime into the first public release.
- Node UI exposes only `Preset` and `disable_node`.
- `Preset` exposes exactly `Conservative`, `Balanced`, and `Aggressive`.
- Preserved the validated Aggressive quality policy that keeps both U14 residual layers in native precision:
  - `decoder.upsamples.14.residual.2`
  - `decoder.upsamples.14.residual.6`
- Internal production defaults remain fixed: activation clip ratio 1.0, weight clip ratio 1.0, channels-last enabled, fallback-on-error enabled, profiling disabled.
- Includes benchmark data in `benchmarks/benchmarks.csv`.
- Includes example output image in `assets/examples.png`.
- Includes importable ComfyUI example workflow in `workflows/example_workflow.json`.
- Distributed under GPL-3.0-only with upstream attribution in `NOTICE.md`.
- Finalized Comfy Registry metadata as `qwen-vae-triton` under publisher `puppet-vision`.
- Added Registry icon at `assets/logo.png` and GitHub repository/issue URLs.

## 0.2.0-rc2 — Pre-Release Candidate 2

- Simplified the node UI for release preparation.
- Removed all tuning widgets except `Preset`.
- `Preset` now exposes exactly three values: `Conservative`, `Balanced`, and `Aggressive`.
- Added `disable_node` boolean passthrough toggle.
- Internal production defaults are fixed: activation clip ratio 1.0, weight clip ratio 1.0, channels-last enabled, fallback-on-error enabled, profiling disabled.
- Preserved the validated Aggressive quality policy that keeps both U14 residual layers in native precision:
  - `decoder.upsamples.14.residual.2`
  - `decoder.upsamples.14.residual.6`
- Added benchmark CSV to `benchmarks/benchmarks.csv`.
- Rebuilt README for GitHub release preparation.
- Aligned package metadata with the upstream GPL-3.0 license as `GPL-3.0-only`.

## 0.2.0-rc1 — Pre-Release Candidate 1

- Promoted the validated two-layer U14 exclusion policy into the first benchmark candidate.
- No Triton kernel math or quantization formula changes relative to the validated v0.1.2 policy.

## 0.1.2

- Hard-coded the validated aggressive-preset U14 residual pair exclusion.

## 0.1.1

- Replaced hard-coded preset thresholds with architecture-adaptive channel tiers.

## 0.1.0

- Initial experimental decoder-only Triton W8A8 implicit-GEMM Conv3D package.
