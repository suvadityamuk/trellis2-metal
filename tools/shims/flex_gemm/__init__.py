"""Minimal pure-PyTorch stand-in for FlexGEMM (CUDA/Triton only upstream) so the official TRELLIS.2 code runs on MPS.
Semantics follow FlexGEMM's submanifold conv: out[i] = bias + sum_v W[:, v, :] @ x[nbr(i, v)], v = kd*9 + kh*3 + kw,
neighbor offset (x, y, z) = (kd-1, kh-1, kw-1)."""
from . import ops
