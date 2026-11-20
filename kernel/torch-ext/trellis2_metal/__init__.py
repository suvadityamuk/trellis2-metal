"""Metal kernels from the TRELLIS.2 Metal port's flow-transformer fast path.

All functions take contiguous MPS tensors. Activations are bfloat16, the residual
stream and the parameters applied to it are float32. Requires macOS 26 (Metal 4).
"""

from typing import Optional

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


__all__ = ["gemm", "gemm_bias_gelu", "bias_gelu_", "resid_ln", "qprep"]
