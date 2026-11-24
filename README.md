# TRELLIS.2 on Metal

Image → 3D mesh with per-vertex PBR attributes for [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2)
(`Trellis2ImageTo3DPipeline.run`, pipeline types `512` and `1024_cascade`), and on to a textured GLB
(`o_voxel.postprocess.to_glb`, see *Image → GLB*), as a native Objective-C++/Metal program for
Apple-silicon GPUs with no Python at runtime. Two builds from one binary:

- **parity** (`--mode parity`, default): bitwise identical to the official PyTorch-MPS pipeline, from the input image to
  the final mesh vertices, faces and voxel attributes. It loads the precompiled Metal kernel library that ships inside
  torch 2.15 and issues the same kernel dispatches and MPS/MPSGraph calls as torch does, in the same order.
- **fast** (`--mode fast`): tolerance-bounded; the flow transformers run on fused Metal kernels written for this port.
  See below.

## Results (M5 Pro, 20-core GPU, 64 GB, 5 example images, warm process, 1 warmup, seconds)

`512`:

| Image | tokens | Official PyTorch-MPS | Official `torch.compile` | Metal parity | Metal fast |
|---|---|---|---|---|---|
| 0a34 | 2867 | 51.3 | 47.6 | 37.3 | **26.8** |
| 0e49 | 5425 | 84.3 | 78.6 | 58.1 | **43.3** |
| 0f16 | 3819 | 59.8 | 56.8 | 44.5 | **33.0** |
| 130c | 1848 | 36.0 | 34.2 | 30.1 | **21.1** |
| T.png | 3530 | 58.0 | 54.8 | 42.6 | **30.9** |
| **Sum** | | 289.4 | 271.9 (1.06×) | 212.5 (1.36×) | **155.1 (1.87×)** |

`1024_cascade`:

| Image | tokens | Official PyTorch-MPS | Official `torch.compile` | Metal parity | Metal fast |
|---|---|---|---|---|---|
| 0a34 | 13360 | 226.5 | 235.1 | 176.2 | **140.8** |
| 0e49 | 27054 | 562.8 | 596.9 | 448.0 | **377.5** |
| 0f16 | 17482 | 321.4 | 348.6 | 248.3 | **209.0** |
| 130c | 7011 | 111.6 | 119.0 | 89.1 | **65.5** |
| T.png | 16362 | 295.4 | 311.4 | 226.9 | **183.2** |
| **Sum** | | 1517.7 | 1611.0 (0.94×) | 1188.5 (1.28×) | **976.1 (1.55×)** |

The fast column predates the Metal 4 GEMM described under *Metal-specific experiments*, which makes the fast build a
further 4.4% (`512`) and 3.4% (`1024_cascade`) faster in a same-session A/B.

The `torch.compile` column is the official pipeline with the flow transformers and the DINOv3 encoder wrapped in
`torch.compile(dynamic=True)` (`tools/bench_ref.py --compile`, the same wrapping as the CUDA port's baseline), with
compilation absorbed by the warmup run (117 s at `512`, 298 s at `1024_cascade`, against 51 s and 227 s eager). It is
6% faster than eager at `512`, almost all of it in the dense sparse-structure sampler (87.9 → 75.2 s), and 6% slower
at `1024_cascade` (shape SLat 747 → 808 s, texture SLat 371 → 414 s). The sparse samplers gain nothing because their
time is in the bf16 matmul and attention kernels, which compilation does not change, and the upstream sparse attention
code breaks the graph on every call (a function-local `import`). It was measured in a later session than the other
columns, so differences of a few percent may be session-to-session drift. Its meshes are not bit-identical to the
eager ones (vertex counts differ by a few percent, up to 26% on one image); their distance to eager was not measured.

Times are the whole pipeline (preprocess, DINOv3, three samplers, two decoders, mesh extraction), with model loading
excluded. Raw numbers are in `bench/*.json`. All ten parity meshes are bit-identical to the official ones.

![total time](docs/speedup.png)

![per-image stage breakdown, 512](docs/timing_512.png)

![per-image stage breakdown, 1024_cascade](docs/timing_1024c.png)

Per stage, summed over the five images:

| Stage | Official 512 | parity 512 | fast 512 | Official 1024c | parity 1024c | fast 1024c |
|---|---|---|---|---|---|---|
| DINOv3 cond | 0.8 | 0.8 | 0.8 | 7.7 | 4.5 | 4.5 |
| sparse structure | 87.9 | 90.6 | 61.6 | 87.9 | 92.1 | 61.7 |
| shape SLat | 84.1 | 70.0 | 50.6 | 747.3 | 689.1 | 552.7 |
| texture SLat | 48.0 | 40.2 | 30.9 | 370.5 | 352.2 | 304.8 |
| shape decoder | 34.9 | 5.6 | 5.8 | 152.5 | 26.1 | 27.0 |
| texture decoder | 15.4 | 5.3 | 5.4 | 79.8 | 24.4 | 25.2 |

Where the time goes: the samplers are bf16 matmul and attention, and both sides end up in the same Apple kernels
(`MPSNDArrayMatrixMultiplication`, torch's prefill-attention kernel). On this GPU those run at about 31 and 25 TFLOPS,
and the fast build sits close to that ceiling, so the gain at 1024 (where attention is quadratic in 13k–27k tokens) is
smaller than at 512. The decoders are 3–6× faster in both builds because the sparse convolutions run as chunked
gather + matmul on the GPU with pooled buffers, while the official run goes through the pure-PyTorch fallback of
FlexGEMM/spconv on MPS.

## Outputs

Rendered with `build/render` (a small Metal rasterizer: vertex base color with simple shading, same camera):

![renders 512](docs/renders_512.png)

![renders 1024_cascade](docs/renders_1024c.png)

Turntable of the Metal fast output for `T.png`:

![turntable](docs/T_fast.gif)

## Image → GLB (`--glb`)

`build/t2 --glb out.glb` continues from the mesh to a textured GLB, reimplementing `o_voxel.postprocess.to_glb` with
the settings of the official `example.py` (`decimation_target=1000000`, `texture_size=4096`, `remesh=True`,
`remesh_band=1`, `remesh_project=0`; `--faces` and `--texture-size` change the first two). It runs after either build's
mesh and never touches the parity path. There is no bitwise reference for this stage (the official one needs CUDA-only
`cumesh`/`nvdiffrast`, so there is **no official `to_glb` baseline on a Mac**); it is gated by tolerance and unit tests.

| Stage | How | Where |
|---|---|---|
| fill holes | boundary loops under the official perimeter threshold, fan-filled | CPU |
| remesh | LBVH over the input triangles (63-bit Morton codes, Karras tree; built on CPU threads in shared buffers), unsigned distance to it for the grid vertices of a one-voxel band (the cost of the remesh), dual contouring of the band | Metal compute for the queries, CPU threads for the rest |
| simplify | parallel quadric edge collapse on CuMesh's schedule: plane-distance quadrics, repeated independent-set selection per pricing pass, last pass lands on the face target | CPU, all performance cores |
| UV unwrap | chart clustering after CuMesh's `compute_charts`, charts split into 64 groups, one xatlas instance per group in parallel, group atlases tiled into one | CPU, all performance cores |
| bake | UV-space rasterization, then one fused kernel per texel: nearest point on the *input* mesh (LBVH) → trilinear sample of the sparse voxel attributes | Metal render + compute |
| inpaint | wavefront TELEA: 4-connected distance levels, one dispatch per level; base colour, alpha, metallic, roughness together | Metal compute |
| pack | uint8 conversion, RGBA base colour and (0, roughness, metallic) | Metal compute |
| export | direct binary glTF writer; WebP (`EXT_texture_webp`, quality 80, method 0, as trimesh writes) or PNG (`--texture-format png`), the two textures encoded on separate threads | CPU |

Remesh, simplification and chart clustering follow the CuMesh algorithms (MIT) that the official `to_glb` calls;
the differences in the simplifier (direct plane-distance quadric, several independent sets per pricing pass, landing
on the face target) are listed at the top of `src/glb/simplify.h`. MSL is compiled at runtime with fast math off and `fp contract(off)`. Vendored, unmodified, under `third_party/`:
[xatlas](https://github.com/jpcy/xatlas) (MIT) and the encoder subset of libwebp 1.5.0 with sharpyuv (BSD-3-Clause,
with `COPYING`, `PATENTS`, `AUTHORS`). Still no CMake and no other dependency.

### Timing (same protocol as above; seconds)

Three configurations, each run twice: old fast (the binary before this change, image → mesh), new fast without `--glb`
(image → mesh), and new fast with `--glb` (image → GLB). The runs were strictly sequential with nothing else running on
the machine and a 5-minute idle cooldown before each, in the order old, new, glb, glb, new, old; macOS reported no
thermal warning before or after any run. Per-image values are the mean of the two runs. Official PyTorch-MPS numbers
are from the earlier session of the table at the top. Raw numbers: `bench/e2e_cool/`.

`512`:

| Image | Official PyTorch-MPS image→mesh | Old fast image→mesh | New fast image→mesh (no `--glb`) | New fast image→mesh (with `--glb`) | `to_glb` | export | New image→GLB | faces | GLB MB |
|---|---|---|---|---|---|---|---|---|---|
| 0a34 | 51.3 | 24.1 | 24.3 | 24.6 | 3.17 | 0.48 | 28.3 | 999,452 | 44.0 |
| 0e49 | 84.3 | 40.7 | 41.0 | 41.3 | 3.85 | 0.64 | 45.8 | 997,456 | 47.4 |
| 0f16 | 59.8 | 31.0 | 31.2 | 31.2 | 3.06 | 0.43 | 34.7 | 999,934 | 37.4 |
| 130c | 36.0 | 19.8 | 19.9 | 19.9 | 3.01 | 0.34 | 23.3 | 999,844 | 39.9 |
| T.png | 58.0 | 29.0 | 29.1 | 29.2 | 3.33 | 0.60 | 33.1 | 999,678 | 40.8 |
| **Sum** | 289.4 | 144.6 | 145.5 | 146.2 | 16.4 | 2.5 | **165.1** | | |

Image→mesh sums of the two runs: old 143.6 / 145.5, new without `--glb` 145.4 / 145.5, new with `--glb` 145.5 / 146.9; image→GLB 164.3 / 165.9.

`1024_cascade`:

| Image | Official PyTorch-MPS image→mesh | Old fast image→mesh | New fast image→mesh (no `--glb`) | New fast image→mesh (with `--glb`) | `to_glb` | export | New image→GLB | faces | GLB MB |
|---|---|---|---|---|---|---|---|---|---|
| 0a34 | 226.5 | 137.4 | 138.5 | 142.1 | 7.55 | 0.69 | 150.3 | 996,270 | 46.7 |
| 0e49 | 562.8 | 393.8 | 411.9 | 422.1 | 13.12 | 0.74 | 436.0 | 993,496 | 51.6 |
| 0f16 | 321.4 | 212.9 | 229.6 | 226.1 | 7.53 | 0.65 | 234.3 | 999,416 | 41.1 |
| 130c | 111.6 | 67.8 | 71.4 | 69.8 | 4.29 | 0.59 | 74.7 | 999,858 | 46.7 |
| T.png | 295.4 | 189.5 | 204.1 | 200.4 | 7.67 | 0.72 | 208.8 | 998,510 | 43.9 |
| **Sum** | 1517.7 | 1001.3 | 1055.6 | 1060.6 | 40.2 | 3.4 | **1104.1** | | |

Image→mesh sums of the two runs: old 1030.6 / 972.0, new without `--glb` 1057.9 / 1053.3, new with `--glb` 1060.5 / 1060.7; image→GLB 1104.0 / 1104.2.

`to_glb` here excludes export. Per stage, sum of five images:

| | remesh | simplify | UV unwrap | bake | inpaint + pack | export | peak footprint old → new |
|---|---|---|---|---|---|---|---|
| `512` | 0.87 | 4.69 | 9.85 | 0.44 | 0.58 | 2.50 | 19.26 → 19.45 GB |
| `1024_cascade` | 3.62 | 25.98 | 9.16 | 0.71 | 0.69 | 3.39 | 42.64 → 43.05 GB |

Reading the image→mesh columns: the model source is identical in the old and new binaries, and running `to_glb`
between images does not slow the samplers (with and without `--glb` are within 1% at both sizes). At `512` all six
runs are within 143.6–146.9 s. At `1024_cascade` the old binary's two five-image runs (1030.6 and 972.0 s) differ from
each other by 6% and are both below the four new-binary runs (1053–1061 s). A single-image A/B at `1024_cascade`
(0f16, three interleaved runs each, same cooldown protocol) shows no difference: old 210.4 / 216.7 / 210.4 s, new
212.7 / 212.1 / 217.6 s, new rebuilt from clean 214.7 / 214.6 / 208.8 s. The gap in the five-image sums is therefore
unexplained and may be run-to-run variance of long runs; it was not resolved. An earlier, uncooled session with other
applications open was 10–15% slower throughout (`bench/e2e/`).

For context only, on **different hardware**: the other ports report for the official `to_glb` 608.6 s (ROCm) and
363.5 s (CUDA), and for their fast `to_glb` 14.0 s (ROCm) and 15.2 s (CUDA), each a sum over the same five images.

### A/B on this Mac (`build/glb`, `T.png` `1024_cascade` mesh, 12.7 M input faces)

| Variant | Result |
|---|---|
| default, best of 3 | 7.17 s: remesh 0.47, simplify 3.93, UV unwrap 1.88, bake 0.11, inpaint + pack 0.12, export 0.67 |
| naive CPU on this Mac: one serial xatlas over the whole mesh | UV unwrap 9.73 s (5.2× slower), total 15.18 s |
| naive CPU on this Mac: `cv2.inpaint` TELEA on the four 4096² maps | 7.96 s, against about 0.1 s for the Metal wavefront |
| PNG textures | export 0.54 s, 82.6 MB instead of 43.9 MB |
| WebP method 4 (dropped) | export 4.07 s for 42.7 MB: 6× the time for 3% of the size |
| WebP alpha quality 0 (option, not default) | export 0.30 s, 40.9 MB; lossy alpha differs from what trimesh writes |

Raw numbers: `bench/to_glb_ab/`. Inside the pipeline process `to_glb` is slightly slower than standalone
(`T.png`: 7.7 s against 7.2 s); it runs right after the samplers with their buffers resident.
The GLB is deterministic: `T.png` `1024_cascade` gave the same bytes in the standalone run and both pipeline runs.

### Tolerance

End to end against the ROCm port's official-eager GLBs at `1024_cascade` (`tools/tolerance.py`,
`bench/tolerance_glb_1024c.json`): symmetric Chamfer distance / bounding-box diagonal, and base-colour L1 (0–255) at
matched surface samples. The ROCm rows are the official pipeline in other configurations against the same reference,
i.e. the band users already accept. The Metal row also contains the sampling difference of a different backend, not
only `to_glb`.

| vs ROCm official eager | 0a34 | 0e49 | 0f16 | 130c | T.png |
|---|---|---|---|---|---|
| ROCm official `torch.compile` | 2.20e-3 / 22.2 | 3.83e-3 / 22.8 | 2.74e-3 / 21.4 | 4.32e-3 / 18.9 | 2.53e-3 / 25.4 |
| ROCm parity | 2.21e-3 / 21.7 | 3.68e-3 / 22.6 | 2.75e-3 / 26.3 | 4.34e-3 / 19.3 | 2.50e-3 / 28.9 |
| ROCm fast | 2.14e-3 / 17.0 | 3.54e-3 / 21.4 | 2.66e-3 / 28.7 | 4.30e-3 / 23.7 | 2.70e-3 / 26.2 |
| **Metal fast + `--glb`** | 2.33e-3 / 21.5 | 3.73e-3 / 21.1 | 3.17e-3 / 25.0 | 4.69e-3 / 22.2 | 2.60e-3 / 29.9 |

Metal is inside the band on 0e49 and on Chamfer for T.png, and just above it elsewhere: Chamfer by 0.1e-3 (0a34),
0.4e-3 (0f16) and 0.35e-3 (130c) of the diagonal, colour by 1.0 on T.png. The band of the CUDA port's configurations
was not recomputed here.

`to_glb` in isolation, each GLB against the mesh and voxels it was made from (`tools/check_glb.py`,
`bench/to_glb_gates/`), range over the five images. The remesh sits one voxel off the input surface by construction
(`remesh_band=1`), which is the distance seen:

| | voxel | GLB→mesh mean (p99) | mesh→GLB mean | base colour L1 | roughness L1 | metallic L1 |
|---|---|---|---|---|---|---|
| `512` | 1.95e-3 | 2.1–2.65e-3 (3.0–4.4e-3) | 2.3–3.3e-3 | 2.33–3.56 | 0.01–3.0 | 0.06–5.07 |
| `1024_cascade` | 9.77e-4 | 1.15–1.95e-3 (2.0–3.95e-3) | 1.28–2.58e-3 | 2.56–4.36 | 0.45–2.54 | 0.05–3.22 |

Simplification quality against the reference GLBs: all ten Metal GLBs load in trimesh with unit normals, UVs in
[0, 1], 0 degenerate faces and 0 boundary edges; at `1024_cascade` they have 2,503–21,973 non-manifold edges at
993k–1,000k faces. The ROCm official-eager GLBs have 2,882–23,313 non-manifold edges, 8–70 boundary edges, 4–30
degenerate faces and a few zero-length normals at about 950k–970k faces. So the topology is comparable, and the
geometric distance is the Chamfer row above.

Unit tests, `build/glb_test` (24 checks, all pass; log in `bench/to_glb_gates/glb_test_T1024c.log`):

- LBVH nearest-triangle against brute force: random soups, capped queries, 1 and 2 triangles, and the 12.7 M-face real
  mesh; max distance error 1.3e-5 on the soups and 5.9e-8 on the real mesh, same face except for exact ties.
- Rasterizer against a CPU rasterizer: fill-rule grid exact; random triangles differ on 0.012% of texels, all within
  0.002 texel of an edge (Metal snaps vertices to 1/256 texel).
- TELEA on the GPU against the same algorithm on the CPU: 0 of 55,698,396 values differ on the real 4096² textures.
  It is not `cv2.inpaint`: OpenCV fills by a serial priority queue. Measured against OpenCV on the same textures: mean
  |d| 1.11 (p99 10) on the first ring of filled texels, 3.50 over all filled texels (base colour, 0–255).
- Bake against a brute-force nearest point + trilinear sample: 100% of 3,000 texels equal on the real mesh; 99.5%
  equal and the rest off by one on the synthetic one.
- Atlas: every simplified face mapped exactly once; 59 of 7,494,233 covered texels belong to two faces; texel density
  p5/median 0.96, p95/median 1.13.
- GLB validity in a standard parser (trimesh, offline): `tools/check_glb.py`.

### Model side

Nothing was changed in the samplers or decoders for the GLB work. The ROCm fast build's ideas are already in the Metal
fast build (fused kernels, K/V caching, closed-form unconditional cross-attention, the Metal 4 GEMM). Batching the two
CFG branches was not pursued: matmul and attention already run at the kernels' TFLOPS ceiling, so a batch of two costs
twice one. Mesh extraction takes about 0.14 s, so there is nothing to win there.

### Renders

`build/render --glb` draws a GLB with its textures. Columns: input, Metal fast mesh (vertex colour), Metal GLB
(textured, then geometry only), ROCm official GLB (textured, then geometry only).

![GLB renders 1024_cascade](docs/renders_glb_1024c.png)

![GLB renders 512](docs/renders_glb_512.png)

## Fast build

| Change | Effect (T.png 512, per model call) |
|---|---|
| Residual stream in f32, one fused kernel between matmuls (residual + gate + bias + LayerNorm + modulation; bias + RMS norm + RoPE; bias + GELU) instead of 10–20 torch dispatches | dense DiT call 0.78 s → 0.52 s |
| Bias-less bf16 matmuls (bias folded into the next fused kernel); attention writes straight into the layout the output projection reads | sparse DiT call (3530 tokens) 0.64 s → 0.46 s |
| Cross-attention K/V of the image condition computed once per sampler run | part of the above |
| Unconditional branch uses the closed form of its cross attention (all keys identical → one constant row) | uncond call 0.45 s dense / 0.40 s sparse |

The fused kernels round to bf16 at the same points as the reference, so a single model call differs from parity by a
relative L2 of 4–7e-3, which is the accumulated bf16 rounding noise of the reference itself (the cond-vs-uncond
difference that guidance amplifies is 0.2–0.3). The decoders and mesh extraction are the same code in both builds.

**Tolerance gates.**
1. Sampler trajectory vs the official pipeline, compared with what the official pipeline does to itself when only its
   attention backend is switched (`PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1`, MPSGraph instead of the prefill kernel).
   Sparse-structure stage, T.png: relative L2 goes 2.1e-4 → 4.7e-2 over the 12 steps for the backend switch (3542
   tokens vs 3530) and 3.1e-4 → 7.3e-2 for the fast build (3549 tokens).
2. End-to-end mesh vs official (`tools/gates.py`: symmetric Chamfer distance over the bbox diagonal / base-color L1 on
   a 0–255 scale at matched surface samples). The parity column is an identical mesh, so it shows the sampling floor
   of the metric.

`512`:

| Image | official (alt attention) vs official | Metal parity vs official | Metal fast vs official |
|---|---|---|---|
| 0a34 | 2.7e-3 / 31.3 | 1.3e-3 / 6.6 | 2.9e-3 / 28.4 |
| 0e49 | 4.1e-3 / 16.1 | 1.8e-3 / 4.1 | 4.1e-3 / 19.7 |
| 0f16 | 3.1e-3 / 12.7 | 1.7e-3 / 3.1 | 3.8e-3 / 17.5 |
| 130c | 4.8e-3 / 24.6 | 0.9e-3 / 1.2 | 5.0e-3 / 23.9 |
| T.png | 3.1e-3 / 25.4 | 1.5e-3 / 4.1 | 3.1e-3 / 22.5 |

`1024_cascade`:

| Image | official (alt attention) vs official | Metal parity vs official | Metal fast vs official |
|---|---|---|---|
| 0a34 | 2.8e-3 / 15.0 | 1.2e-3 / 4.2 | 2.9e-3 / 27.0 |
| 0e49 | 4.2e-3 / 18.8 | 1.9e-3 / 7.3 | 4.2e-3 / 17.9 |
| 0f16 | 3.1e-3 / 28.2 | 1.7e-3 / 7.5 | 3.6e-3 / 28.1 |
| 130c | 4.9e-3 / 22.1 | 0.7e-3 / 1.5 | 5.3e-3 / 24.9 |
| T.png | 3.2e-3 / 26.9 | 1.5e-3 / 6.3 | 3.0e-3 / 28.0 |

The fast build lands at the level of the backend switch: slightly above it on some rows (geometry up to 0.5e-3 on
0f16, color 27.0 vs 15.0 on 0a34 at 1024) and below on others. The sampler is chaotic, so any rounding-level change
gives a different but equally valid sample; this is visible in the renders as small colour and detail differences
(for example the gem and tube colours).

## Metal-specific experiments (after the benchmark above)

Each idea was tried in isolation on the M5 Pro (bf16, 12 heads × 128, 4096 and 16384 tokens). One was kept.

| Idea | Result | Verdict |
|---|---|---|
| Own GEMM on Metal 4 `MetalPerformancePrimitives` `matmul2d` (64×128 output tiles, 4 simdgroups), with the MLP bias + GELU applied while the tile is still in registers | bit-identical to `MPSNDArrayMatrixMultiplication`; 31–32 vs 29–31 TFLOPS; GELU pass 2.4 ms → 1.1 ms at 16384 tokens | **kept** (fast build; `T2_GEMM=mps` or no Metal 4 falls back) |
| Tile shape of that GEMM | 64×32: 12.8 TFLOPS, 64×64: 23.7, 64×128: 32.3, 128-row tiles and 8 simdgroups slower | 64×128×4 |
| Retune torch's flash-attention kernel (query/key block, simdgroups per threadgroup, barriers) | stock {64, 32, 4} is the best at 23–25 TFLOPS; larger key blocks 4–11 | no gain |
| Flash attention on threadgroup-scope `matmul2d` (scores in threadgroup memory) | 12–13 TFLOPS vs 25 | loses |
| Attention as two MPS matmuls per head | 69 ms without the softmax vs 68 ms for the whole flash kernel, and needs N² score memory | loses |
| fp16 instead of bf16 | same throughput | no gain |

Effect of the kept change, interleaved A/B in one process (`build/gemm_ab`, sparse flow model, per call, two runs):
−4.6% / −6.3% at 16362 tokens, within noise (+3.3% / 0.0%) at 3530 tokens; outputs are identical, so the tolerance
gates above are unchanged. At 16384 tokens a block is 55% self attention, 35% matmul and 7% elementwise passes
(`build/epi_bench`), so the remaining matmul headroom is small and attention is at the ceiling of the kernel.

Whole pipeline, same session, one process per (image, backend) with the backend order flipped per image
(`tools/ab_bench.sh`, raw lines in `bench/ab/`), sum over the five images:

| Pipeline | fast, MPS matmul | fast, Metal 4 GEMM | Change |
|---|---|---|---|
| `512` | 153.3 s | 146.6 s | −4.4% (per image −2.8% to −8.1%) |
| `1024_cascade` | 1083.6 s | 1046.2 s | −3.4% (per image −2.0% to −5.1%) |

The tables at the top were measured before this change and are left as they are: absolute times drift between sessions
(the unchanged MPS path took 976.1 s for `1024_cascade` then and 1083.6 s here, and the untouched decoders moved with
it), so only same-session comparisons are meaningful at the few-percent level.

## Memory and smaller GPUs

Peak process footprint in the default mode (all models resident, buffers pooled):

| Pipeline | Official (MPS peak) | Metal parity / fast |
|---|---|---|
| 512 | 25.6 GB | 16–19 GB |
| 1024_cascade | 48.2 GB | 35–49 GB (largest image: 10.9M voxels) |

`--lowmem` loads each model only for its stage and caps the free-buffer pool (`--pool-gb`, default 1 GB in this mode).
Fast build, T.png, `512`, on the same machine:

| Flags | GPU buffer pool peak | Total |
|---|---|---|
| default | ~17 GB | 30 s |
| `--lowmem --pool-gb 4` | 6.3 GB | 29 s |
| `--lowmem --pool-gb 3` | 5.7 GB | 31 s |
| `--lowmem` (1 GB) | 3.8 GB | 35 s |
| `--lowmem --pool-gb 0.25` | 3.0 GB | 40 s |

The pool figure excludes MPS-internal scratch, so leave a few GB of headroom. That puts `512` within reach of 16 GB
machines (M1/M2/M3 base and Pro) with `--lowmem`, and plausibly 8 GB machines with `--lowmem --pool-gb 0.25` and
swap, though neither has been run on such hardware here: only the M5 Pro was available. `1024_cascade` needs a 48–64 GB
machine for large objects; `--max-tokens` lowers the cascade's token budget the same way the official pipeline does.
Nothing in the kernels is specific to a GPU family: they are plain MSL compute kernels plus MPS matmul (the Metal 4 GEMM is optional and falls back to it), and
elementwise ops over more than 2³¹ bytes are split as torch does.

## Running inference

Requires an Apple-silicon Mac, macOS 26 and the Command Line Tools (`xcode-select --install`; no Xcode or CMake, own
shaders are compiled at runtime). Python is needed once for setup, not at runtime.

**1. Checkpoints.** They are read as safetensors from `models/` (not in this repo). DINOv3 is a gated model, so accept
its licence on Hugging Face and log in (`hf auth login`) first.

```bash
pip install -U huggingface_hub
mkdir -p models
ln -s "$(hf download microsoft/TRELLIS.2-4B)" models/trellis2
ln -s "$(hf download microsoft/TRELLIS-image-large --include 'ckpts/ss_dec_conv3d_16l8_fp16.*')/ckpts" models/ss_dec
ln -s "$(hf download facebook/dinov3-vitl16-pretrain-lvd1689m config.json model.safetensors)" models/dino
```

**2. torch's Metal kernel library.** Both builds load the precompiled kernel library that ships inside the
torch 2.15 wheel (the parity build for everything, the fast build for attention and the non-transformer stages).
torch itself is not used at runtime.

```bash
python3 -m venv ref/venv && ref/venv/bin/pip install torch==2.15.0
ref/venv/bin/python tools/extract_torch_metallibs.py ref/venv/lib/python3*/site-packages/torch/lib/libtorch_cpu.dylib build/torchlibs
```

**3. Build and run.** The input image needs an alpha channel (PNG or WebP with the background already removed).

```bash
make build/t2 build/render
./build/t2 --image in.png --type 512 --mode fast --glb out.glb               # image -> textured GLB, about 33 s on an M5 Pro
./build/t2 --image in.png --type 512 --mode fast --out mesh.safetensors      # mesh + voxel attributes only, about 30 s
./build/t2 --image in.png --type 1024_cascade --mode fast --out mesh.safetensors   # higher resolution, 48-64 GB machines
./build/t2 --image in.png --type 512 --mode parity --out mesh.safetensors    # bit-identical to official PyTorch-MPS
./build/t2 --image in.png --mode fast --lowmem --out mesh.safetensors        # small-memory machines, see above
./build/render --mesh mesh.safetensors --out view --views 8                  # PNG renders of the result
./build/render --glb out.glb --out view --views 8                            # the same for a GLB, with its textures
```

The output file holds `vertices` [V,3], `faces` [F,3], and the voxel `coords` and `attrs` (base color, metallic,
roughness, alpha) of the official `MeshWithVoxel`. Other options: `--seed N` (default 42), `--max-tokens N`,
`--pool-gb G`, `--quiet`. `T2_GEMM=mps` switches the fast build's matmul back to MPS.

GLB options (defaults are the official `example.py` settings): `--faces N` (decimation target, 1000000),
`--texture-size N` (4096), `--texture-format webp|png` (WebP quality 80 with `EXT_texture_webp`, as the official
export writes; PNG for viewers without WebP support, about twice the file size), `--webp-alpha-quality 0..100`
(100 = lossless alpha plane, the default). `--glb` works with either `--mode`; `--out` and `--glb` can be combined, and
with several `--image` arguments both name a directory. `make build/glb` builds the standalone converter:
`./build/glb --mesh mesh.safetensors --out out.glb [-v]`.

**Benchmarks and checks.**

```bash
./build/t2 --image a.png --image b.png --warmup 1 --bench out.json --out dir   # timing as in tools/bench_ref.py
./build/t2 --image T.png --type 512 --ref refdump/T_512.safetensors            # prints PARITY OK / FAILED
tools/run_bench.sh 512 1024_cascade                                            # official vs parity vs fast, sequential
ref/venv/bin/python tools/bench_ref.py --compile --type 512 --images a.png --out out.json   # official under torch.compile
tools/ab_bench.sh 512 1024_cascade                                             # fast build: MPS matmul vs Metal 4 GEMM
make build/glb_test && ./build/glb_test --mesh mesh.safetensors                # to_glb kernels against brute force
ref/venv/bin/python tools/check_glb.py out.glb mesh.safetensors                # GLB validity + to_glb in isolation
ref/venv/bin/python tools/tolerance.py --out t.json REF=dir METAL=dir          # Chamfer / colour L1 between GLB sets
```

The official reference (parity checks and baselines only) lives in `ref/` (not in this repo: the official TRELLIS.2
repo in `ref/TRELLIS.2`, a torch 2.15 venv, and the CUDA-only dependencies replaced by the shims in `tools/shims`). `tools/extract_torch_metallibs.py` pulls torch's
kernel library out of `libtorch_cpu.dylib`, and `tools/ref_dump.py` writes the stage-boundary dumps that
`build/parity`, `build/parity_dec` and `build/t2 --ref` compare against.

**Scope and assumptions.**
- The bitwise baseline is image → `MeshWithVoxel`. The official `to_glb` (CuMesh, nvdiffrast, FlexGEMM) is CUDA-only
  and does not run on a Mac, so there is no official Mac baseline for GLB export, in time or in bits: `--glb` is a
  reimplementation checked by tolerance (see *Image → GLB*), and it leaves the parity guarantee on the mesh untouched.
- Input images need an alpha channel; background removal (rembg) is not ported.
- Bitwise parity is against torch 2.15 on this macOS version. It depends on Apple's MPS kernels, so a different OS or
  chip family may need the reference dumps regenerated before `PARITY OK` means anything there.
- Never time two GPU jobs at once on a unified-memory machine.

## Hugging Face kernels (`kernel/`)

`kernel/` packages the fast path's standalone Metal kernels for the [`kernels`](https://github.com/huggingface/kernels)
library, as a PyTorch extension that works on MPS tensors: `gemm`, `gemm_bias_gelu` (the Metal 4 MPP GEMM with fused
bias+GELU), `bias_gelu_`, `resid_ln` (residual update + layer norm) and `qprep` (q/k/v split with QK-norm and RoPE).
Hub repository: `suvadityamuk/trellis2-metal` (version 0).

Not yet published (in the source tree only, built and tested with the development build below, not with
`kernel-builder`): two kernels from `to_glb`, `lbvh_build` / `mesh_nearest` / `mesh_udf` (nearest triangle, distance and
barycentrics on a linear BVH, tested against a float64 brute force) and `telea_inpaint` (wavefront Telea inpainting of
uint8 images, tested against a NumPy implementation of the same fill order and, loosely, against OpenCV).

```python
from kernels import get_kernel
t2 = get_kernel("suvadityamuk/trellis2-metal", version=0, trust_remote_code=True)
y = t2.gemm_bias_gelu(x, weight, bias)   # x [M,K] bf16, weight [N,K] bf16, bias [N] f32, all on MPS
```

Building and publishing needs Nix, `kernel-builder`, Xcode 26 and the Metal Toolchain
(see the [builder docs](https://github.com/huggingface/kernels/blob/main/docs/source/builder/metal.md)):

```sh
cd kernel
kernel-builder check-config
kernel-builder build-and-copy -L         # -> kernel/build/<variant>/
pytest tests                             # with LOCAL_KERNELS="suvadityamuk/trellis2-metal=$PWD"
kernel-builder build-and-upload          # publishes to the Hub
```

For quick iteration without that toolchain, `PYTHON=../ref/venv/bin/python dev/build_dev.sh ../build/kernel-dev`
builds a development-only extension (shaders compiled at runtime; never publish it), tested with
`T2_KERNEL_DEV_PATH=$PWD/../build/kernel-dev python -m pytest tests`.

## Layout

- `src/torchk/`: replay of torch-MPS kernel dispatches (elementwise, reductions, matmul, attention, conv, norms).
- `src/rt/`: tensors, buffer pool, safetensors I/O, torch-compatible RNG.
- `src/models/`: DINOv3, dense and sparse flow DiTs, sampler, sparse-structure / shape / texture decoders, mesh extraction.
- `src/fast/flow_fast.h`: the fast build's fused kernels and execution plan. `src/pipeline.h`: the driver for both builds.
- `src/glb/`: `to_glb` (LBVH, remesh, simplify, UV atlas, bake + TELEA kernels, GLB writer). `third_party/`: vendored
  xatlas and libwebp encoder sources.
- `apps/`: `t2` (pipeline), `glb` (standalone `to_glb`), `glb_test` (its kernel tests), `parity*` (per-stage bitwise
  checks), `fast_check` (single-call error and timing), `render` (Metal rasterizer for meshes and GLBs), `mmbench` (matmul/attention throughput); experiments: `mm_mpp`, `gemm_ab`, `epi_bench`,
  `attn_tune`, `attn_tg`, `attn_mm`.
- `tools/`: offline-only Python (reference dumps, official timing, gates, plots, visuals). `bench/`: raw timings and
  outputs. `docs/`: plots and renders.
