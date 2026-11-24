"""Metal kernels from the TRELLIS.2 Metal port's flow-transformer fast path.

All functions take contiguous MPS tensors. Activations are bfloat16, the residual
stream and the parameters applied to it are float32. Requires macOS 26 (Metal 4).
"""

from typing import Optional, Tuple

import torch

from ._ops import ops

_HEAD_DIM = 128


def gemm(
    x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ weight.T`` for bfloat16 ``x`` [M, K] and ``weight`` [N, K] (the layout of ``nn.Linear.weight``)."""
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    ops.gemm(out, x, weight)
    return out


def gemm_bias_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``gelu(x @ weight.T + bias, approximate="tanh")`` in one pass. ``bias`` is float32 [N]."""
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    ops.gemm_bias_gelu(out, x, weight, bias)
    return out


def bias_gelu_(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """In place ``gelu(x + bias, approximate="tanh")`` on bfloat16 ``x`` [N, C] with a float32 ``bias`` [C]."""
    ops.bias_gelu(x, bias)
    return x


def resid_ln(
    x: torch.Tensor,
    vec: torch.Tensor,
    branch: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    *,
    residual: bool = True,
    layer_norm: bool = True,
    fused_affine: bool = False,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Residual update of a float32 stream fused with the layer norm that feeds the next branch.

    ``x`` is float32 [N, C] and is updated in place, ``vec`` is float32 [4, C] holding
    ``(branch_bias, branch_scale, ln_weight, ln_bias)``, ``branch`` is bfloat16 [N, C].

    - ``residual``: ``x += (branch + branch_bias) * branch_scale`` (``branch=None`` stands for zeros),
      every step rounded to bfloat16.
    - ``layer_norm``: returns ``layer_norm(x) * ln_weight + ln_bias`` as bfloat16 [N, C], computed
      from the updated ``x``. ``fused_affine`` rounds the affine once instead of after every step.

    C must be a multiple of 32. Returns ``out`` when ``layer_norm`` is set, otherwise ``None``.
    """
    flags = (1 if residual else 0) | (2 if layer_norm else 0) | (8 if fused_affine else 0)
    if residual and branch is None:
        flags |= 4
    if layer_norm and out is None:
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    ops.resid_ln(
        x,
        branch if branch is not None else x,
        vec,
        out if layer_norm else x,
        flags,
        eps,
    )
    return out if layer_norm else None


def qprep(
    src: torch.Tensor,
    bias: torch.Tensor,
    heads: int,
    col_offset: int = 0,
    gamma: Optional[torch.Tensor] = None,
    phases: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Splits one of q/k/v out of a fused projection into attention layout, with optional QK norm and RoPE.

    ``src`` is bfloat16 [N, R] (e.g. the output of a qkv projection) and ``bias`` float32 [R].
    Columns ``col_offset .. col_offset + heads * 128`` become ``out`` [heads, N, 128] (bfloat16):

    - ``v = src + bias``
    - ``gamma`` (float32 [heads * 128]): ``v = normalize(v, dim=-1) * gamma * sqrt(128)`` per head
    - ``phases`` (float32 [N, 64, 2] of ``(cos, sin)``): rotates each consecutive pair of ``v``
    """
    if out is None:
        out = torch.empty(
            (heads, src.shape[0], _HEAD_DIM), dtype=src.dtype, device=src.device
        )
    elif out.shape[0] != heads:
        raise ValueError(f"out must have {heads} heads, got {out.shape[0]}")
    flags = (1 if gamma is not None else 0) | (2 if phases is not None else 0)
    ops.qprep(
        out,
        src,
        bias,
        gamma if gamma is not None else bias,
        phases if phases is not None else bias,
        col_offset,
        flags,
    )
    return out


def lbvh_build(
    vertices: torch.Tensor, faces: torch.Tensor, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds a linear BVH over a triangle mesh on the CPU and returns it as two MPS tensors.

    ``vertices`` is [V, 3] (converted to float32), ``faces`` [F, 3] (converted to int32). The
    result ``(nodes, tris)`` is opaque (raw node and triangle records as int32) and is what
    :func:`mesh_nearest` takes. Degenerate triangles are allowed.
    """
    if device is None:
        device = vertices.device if vertices.device.type == "mps" else torch.device("mps")
    nodes, tris = ops.lbvh_build(
        vertices.detach().to("cpu", torch.float32).contiguous(),
        faces.detach().to("cpu", torch.int32).contiguous(),
    )
    return nodes.to(device), tris.to(device)


def mesh_nearest(
    bvh: Tuple[torch.Tensor, torch.Tensor],
    points: torch.Tensor,
    max_distance: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest triangle to every point of float32 ``points`` [M, 3] (MPS), exact (not approximate).

    Returns ``(distance, face, bary)``: float32 [M] unsigned distances, int32 [M] indices into
    the ``faces`` given to :func:`lbvh_build` (lowest index on exact ties), and float32 [M, 3]
    barycentric coordinates of the closest point. With ``max_distance`` set, points farther
    than that from every triangle get ``distance = inf`` and ``face = -1``.
    """
    nodes, tris = bvh
    m = points.shape[0]
    dist = torch.empty((m,), dtype=torch.float32, device=points.device)
    face = torch.empty((m,), dtype=torch.int32, device=points.device)
    bary = torch.empty((m, 3), dtype=torch.float32, device=points.device)
    ops.mesh_nearest(
        dist,
        face,
        bary,
        nodes,
        tris,
        points,
        float("inf") if max_distance is None else float(max_distance),
    )
    return dist, face, bary


def mesh_udf(
    vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Unsigned distance from float32 ``points`` [M, 3] (MPS) to the mesh. Builds the BVH on every call."""
    return mesh_nearest(lbvh_build(vertices, faces, points.device), points)[0]


def telea_inpaint(
    image: torch.Tensor, mask: torch.Tensor, radius: int = 3
) -> torch.Tensor:
    """Telea inpainting of a uint8 ``image`` [H, W, C] (MPS, C <= 8) where ``mask`` [H, W] is non-zero.

    The counterpart of ``cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)``: the same
    per-pixel estimate, but pixels are filled in order of their 4-connected distance to the
    known region (all pixels of one level in parallel) instead of OpenCV's fast-marching
    order, so results agree closely next to known pixels and differ more deep inside large
    holes. Known pixels are returned unchanged; with no known pixel nothing is filled.
    """
    if image.dim() != 3:
        raise ValueError(f"image must be [H, W, C], got {tuple(image.shape)}")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask must be {tuple(image.shape[:2])}, got {tuple(mask.shape)}"
        )
    lv, order, offsets = ops.telea_levels(
        (mask.detach() != 0).to("cpu", torch.uint8).contiguous()
    )
    planar = image.permute(2, 0, 1).clone(memory_format=torch.contiguous_format)
    if order.numel() > 0:
        ops.telea_fill(
            planar,
            lv.to(image.device),
            order.to(image.device),
            offsets.tolist(),
            radius,
        )
    return planar.permute(1, 2, 0).contiguous()


__all__ = [
    "gemm",
    "gemm_bias_gelu",
    "bias_gelu_",
    "resid_ln",
    "qprep",
    "lbvh_build",
    "mesh_nearest",
    "mesh_udf",
    "telea_inpaint",
]
