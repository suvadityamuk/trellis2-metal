---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}tags:
- metal
- mps
- trellis
- trellis2
- image-to-3d
- apple-silicon
---

# {{ repo_id }}

Metal compute kernels from a Metal port of the [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D pipeline,
packaged for the [`kernels` library](https://github.com/huggingface/kernels). They are the fused building blocks of the
flow transformer's fast path, plus two mesh/texture kernels from the port's GLB export (nearest triangle on a linear
BVH, and wavefront Telea inpainting), and operate directly on PyTorch MPS tensors.

**Requirements:** Apple silicon, macOS 26 or later (the GEMM uses Metal 4's MetalPerformancePrimitives), PyTorch with MPS.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
import torch
from kernels import get_kernel

# This repository is not from a trusted publisher, so `trust_remote_code=True` is required.
t2 = get_kernel("{{ repo_id }}", version={{ version }}, trust_remote_code=True)

x = torch.randn(16384, 1536, device="mps").to(torch.bfloat16)
w = (torch.randn(6144, 1536, device="mps") / 39).to(torch.bfloat16)
b = torch.randn(6144, device="mps")

y = t2.gemm_bias_gelu(x, w, b)  # gelu(x @ w.T + b, approximate="tanh") in one pass
```

The transformer functions take **contiguous MPS tensors**. Activations are `bfloat16`; the residual stream and the parameters
applied to it are `float32`. Nothing here is differentiable: the kernels are for inference.

## Available functions

| Function | What it computes |
|---|---|
| `gemm(x, weight, out=None)` | `x @ weight.T` for bf16 `x` [M, K] and `weight` [N, K]. |
| `gemm_bias_gelu(x, weight, bias, out=None)` | `gelu(x @ weight.T + bias, approximate="tanh")`; the bias and activation are applied while each output tile is still in registers. `bias` is f32 [N]. |
| `bias_gelu_(x, bias)` | In place `gelu(x + bias, approximate="tanh")` on bf16 `x` [N, C]. |
| `resid_ln(x, vec, branch=None, out=None, *, residual=True, layer_norm=True, fused_affine=False, eps=1e-6)` | Residual update of the f32 stream `x` [N, C] (in place) with the bf16 `branch` output, fused with the layer norm (+ affine) that feeds the next branch, returned as bf16. `vec` is f32 [4, C]: `(branch_bias, branch_scale, ln_weight, ln_bias)`. C must be a multiple of 32. |
| `qprep(src, bias, heads, col_offset=0, gamma=None, phases=None, out=None)` | Splits q, k or v out of a fused bf16 projection [N, R] into attention layout [heads, N, 128], adding the bias and optionally applying QK RMS-norm (`gamma`) and rotary embeddings (`phases` [N, 64, 2] of cos/sin). The head dimension is fixed at 128. |

Mesh and texture functions (f32 / int32 / uint8 tensors; the query and fill loops run on the GPU, the BVH build and
the level ordering on the CPU):

| Function | What it computes |
|---|---|
| `lbvh_build(vertices, faces)` | Linear BVH (30-bit Morton codes, Karras 2012) over a triangle mesh; returns an opaque `(nodes, tris)` pair of int32 MPS tensors. |
| `mesh_nearest(bvh, points, max_distance=None)` | For f32 `points` [N, 3]: unsigned distance to the mesh, index of the nearest triangle and the barycentric coordinates of the closest point. With `max_distance`, points farther away get `inf` / `-1`. |
| `mesh_udf(vertices, faces, points)` | Convenience: build + `mesh_nearest`, distance only. |
| `telea_inpaint(image, mask, radius=3)` | Telea (2004) fast-marching-style inpainting of a uint8 image [H, W, C] (C ≤ 8) where `mask` is non-zero. Texels are filled in order of their 4-connected distance to the known region, one GPU dispatch per level (a wavefront), instead of OpenCV's serial heap order. |

See the docstrings (`help(t2.resid_ln)`) for the exact rounding of each step.

## Numerics

Results are rounded to bfloat16 with round-to-nearest-even. Elementwise kernels follow the documented sequence of
operations; reductions (the GEMM accumulation, the layer-norm statistics) use a different summation order than PyTorch,
so compare with a tolerance of about one bfloat16 step rather than bit for bit.

`mesh_nearest` evaluates exact point-triangle distance in f32 (Ericson's closest-point regions, degenerate triangles
handled as segments/points); the tests compare it with a float64 brute force. Ties between triangles at equal
distance may resolve to a different face index than a brute-force argmin. `telea_inpaint` is deterministic. Against
a NumPy implementation of the same level order it is exact or one grey level off on isolated rounding ties (measured:
at most 0.02% of values) when the shaders are compiled without fast math, and up to two grey levels off on under 1% of
values with the fast-math shaders that kernel-builder compiles by default; it is **not** bit-compatible with
`cv2.inpaint(..., cv2.INPAINT_TELEA)`, whose fill order is a serial priority queue (see the tests for the measured
difference).

## Benchmarks
{% if has_benchmark %}

A benchmarking script is available for this kernel. Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% endif %}

Measured on an Apple M5 Pro (macOS 26, PyTorch 2.15) with 16384 tokens and 1536 channels, against the same computation
written with PyTorch MPS ops, run eagerly and through `torch.compile` (default mode; `max-autotune` gave the same
times). Best of 15 calls after warm-up, all three measured in the same session, compilation time excluded:

| Workload | This kernel | PyTorch MPS, eager | PyTorch MPS, `torch.compile` |
|---|---|---|---|
| `gemm`, 1536 → 4608 | 7.3 ms | 7.6 ms | 7.6 ms |
| `gemm_bias_gelu`, 1536 → 6144 | 10.4 ms | 25.1 ms | 11.8 ms |
| `resid_ln` (layer norm only) | 0.87 ms | 2.20 ms | 0.94 ms |

`torch.compile` fuses the elementwise work after the matrix product into one generated Metal shader, which closes most
of the gap to eager mode: against compiled PyTorch these kernels are 4% to 14% faster rather than 2.4x.
{% if source %}

## Source

The kernel-builder formatted source for this kernel is in the `kernel/` directory of {{ source }}.
{% endif %}

## Related

Part of [TRELLIS.2 everywhere](https://huggingface.co/collections/hugging-3d/trellis2-everywhere-6b062cf7d5ae3eca49def35b), built on Microsoft's [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B) ([paper](https://huggingface.co/papers/2512.14692)).

| | Hugging Face | GitHub |
|---|---|---|
| Guide | [The Definitive Fine-Tuning Guide for 3D Generation Models](https://huggingface.co/blog/3d-lora-finetuning) | |
| Fine-tuning (LoRA) | [footwear LoRA](https://huggingface.co/suvadityamuk/trellis-2-finetune-gso) · [trainer Space](https://huggingface.co/spaces/hugging-3d/3d-model-lora-trainer) · [subset builder](https://huggingface.co/spaces/hugging-3d/data-finetuning-subset-builder) | [trellis2-finetuning](https://github.com/suvadityamuk/trellis2-finetuning) |
| Data | [GSO footwear](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents) · [GSO](https://huggingface.co/datasets/hugging-3d/gso-trellis2-latents) · [Smithsonian](https://huggingface.co/datasets/hugging-3d/smithsonian-trellis2-latents) · [NASA](https://huggingface.co/datasets/hugging-3d/nasa-trellis2-latents) latents | |
| Diffusers port | [TRELLIS.2-4B-diffusers-3d](https://huggingface.co/suvadityamuk/TRELLIS.2-4B-diffusers-3d) | [diffusers-3d](https://github.com/suvadityamuk/diffusers-3d) |
| NVIDIA (CUDA C++) | [kernels](https://huggingface.co/kernels/suvadityamuk/trellis2-cuda) · [Space](https://huggingface.co/spaces/suvadityamuk/trellis2-cuda) | [trellis2-cuda](https://github.com/suvadityamuk/trellis2-cuda) |
| Apple Silicon (Metal) | [kernels](https://huggingface.co/kernels/suvadityamuk/trellis2-metal) | [trellis2-metal](https://github.com/suvadityamuk/trellis2-metal) |
| AMD (ROCm/HIP) | [kernels](https://huggingface.co/kernels/suvadityamuk/trellis2-rocm) | [trellis2-rocm](https://github.com/suvadityamuk/trellis2-rocm) |
