import math

import pytest
import torch
import torch.nn.functional as F

BF = torch.bfloat16


def rb(t: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to bfloat16 precision."""
    return t.to(BF).float()


def assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor, atol: float):
    """`actual` (bf16) matches `expected` (f32) to about one bfloat16 step (2^-8 relative), plus `atol` where
    the result is a difference of larger values or sits next to zero."""
    assert actual.dtype == BF
    torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), rtol=2.0**-7, atol=atol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(64, 128, 128), (1, 7, 3), (65, 129, 257), (1000, 1536, 300), (130, 96, 8192)])
def test_gemm(kernel, device, shape):
    M, K, N = shape
    g = torch.Generator().manual_seed(0)
    x = rb(torch.randn(M, K, generator=g))
    w = rb(torch.randn(N, K, generator=g) / math.sqrt(K))
    y = kernel.gemm(x.to(device, BF), w.to(device, BF))
    assert y.shape == (M, N) and y.dtype == BF
    ref = x.double() @ w.double().T
    torch.testing.assert_close(y.cpu().double(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.kernels_ci
def test_gemm_matches_torch_mps(kernel, device):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(777, 1536, generator=g).to(device, BF)
    w = (torch.randn(1536, 1536, generator=g) / 39.0).to(device, BF)
    torch.testing.assert_close(kernel.gemm(x, w).float(), (x @ w.T).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.kernels_ci
def test_gemm_views_and_out(kernel, device):
    g = torch.Generator().manual_seed(2)
    xs = torch.randn(4, 100, 96, generator=g).to(device, BF)
    w = torch.randn(200, 96, generator=g).to(device, BF)
    out = torch.full((3, 100, 200), 42.0, dtype=BF, device=device)
    kernel.gemm(xs[2], w, out[1])
    expect = torch.full((3, 100, 200), 42.0, dtype=BF, device=device)
    expect[1] = kernel.gemm(xs[2].clone(), w)
    assert torch.equal(out, expect)
    torch.testing.assert_close(out[1].float(), (xs[2] @ w.T).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(64, 128, 128), (65, 129, 257), (300, 1536, 6144)])
def test_gemm_bias_gelu(kernel, device, shape):
    M, K, N = shape
    g = torch.Generator().manual_seed(3)
    x = torch.randn(M, K, generator=g).to(device, BF)
    w = (torch.randn(N, K, generator=g) / math.sqrt(K)).to(device, BF)
    b = torch.randn(N, generator=g).to(device)
    y = kernel.gemm_bias_gelu(x, w, b)
    # exactly the unfused sequence
    z = kernel.bias_gelu_(kernel.gemm(x, w), b)
    assert torch.equal(y, z)
    ref = F.gelu(x.cpu().double() @ w.cpu().double().T + b.cpu().double(), approximate="tanh")
    torch.testing.assert_close(y.cpu().double(), ref, rtol=3e-2, atol=3e-2)


@pytest.mark.kernels_ci
def test_bias_gelu(kernel, device):
    g = torch.Generator().manual_seed(4)
    x = rb(torch.randn(513, 96, generator=g) * 3)
    b = torch.randn(96, generator=g)
    y = kernel.bias_gelu_(x.to(device, BF), b.to(device))
    ref = F.gelu(rb(x + b).double(), approximate="tanh").float()
    assert_bf16_close(y, ref, atol=1e-4)


def resid_ln_ref(x, vec, branch, residual, layer_norm, fused_affine, eps):
    ob, gs, a, b = vec
    if residual:
        o = branch.float() if branch is not None else torch.zeros_like(x)
        x = rb(x + rb(rb(o + ob) * gs))
    h = None
    if layer_norm:
        nv = F.layer_norm(x.double(), x.shape[-1:], eps=eps).float()
        h = nv * a + b if fused_affine else rb(rb(nv) * a) + b
    return x, h


@pytest.mark.kernels_ci
@pytest.mark.parametrize("residual,layer_norm,fused_affine,with_branch", [
    (True, True, False, True), (True, True, True, True), (True, False, False, True),
    (False, True, False, False), (True, True, False, False),
])
def test_resid_ln(kernel, device, residual, layer_norm, fused_affine, with_branch):
    N, C = 301, 1536
    g = torch.Generator().manual_seed(5)
    x = torch.randn(N, C, generator=g)
    vec = torch.randn(4, C, generator=g)
    branch = rb(torch.randn(N, C, generator=g)) if with_branch else None
    xd = x.to(device)
    h = kernel.resid_ln(xd, vec.to(device), branch.to(device, BF) if with_branch else None,
                        residual=residual, layer_norm=layer_norm, fused_affine=fused_affine, eps=1e-6)
    xr, hr = resid_ln_ref(x, vec, branch, residual, layer_norm, fused_affine, 1e-6)
    # the residual update is elementwise and exactly specified
    assert torch.equal(xd.cpu(), xr)
    if layer_norm:
        assert h.dtype == BF and h.shape == (N, C)
        assert_bf16_close(h, hr, atol=2e-2)
    else:
        assert h is None


@pytest.mark.kernels_ci
@pytest.mark.parametrize("use_gamma,use_phases", [(False, False), (True, False), (True, True), (False, True)])
def test_qprep(kernel, device, use_gamma, use_phases):
    N, H = 203, 12
    R = 3 * H * 128
    g = torch.Generator().manual_seed(6)
    src = rb(torch.randn(N, R, generator=g))
    bias = torch.randn(R, generator=g) * 0.1
    gamma = torch.randn(H * 128, generator=g) if use_gamma else None
    ang = torch.rand(N, 64, generator=g) * 6.283
    phases = torch.stack([ang.cos(), ang.sin()], -1) if use_phases else None
    for part in range(3):
        off = part * H * 128
        out = kernel.qprep(src.to(device, BF), bias.to(device), H, off,
                           gamma.to(device) if use_gamma else None, phases.to(device) if use_phases else None)
        assert out.shape == (H, N, 128) and out.dtype == BF
        v = rb(src[:, off:off + H * 128] + bias[off:off + H * 128]).view(N, H, 128)
        if use_gamma:
            v = F.normalize(v.double(), dim=-1).float() * gamma.view(1, H, 128) * math.sqrt(128.0)
        if use_phases:
            p = v.view(N, H, 64, 2)
            c, s = phases[:, None, :, 0], phases[:, None, :, 1]
            v = torch.stack([p[..., 0] * c - p[..., 1] * s, p[..., 0] * s + p[..., 1] * c], -1).view(N, H, 128)
        expected = v.permute(1, 0, 2)
        if use_gamma or use_phases:
            assert_bf16_close(out, expected, atol=2e-2)
        else:
            assert torch.equal(out.cpu(), expected.to(BF))


@pytest.mark.kernels_ci
def test_argument_checks(kernel, device):
    x = torch.zeros(4, 8, dtype=BF, device=device)
    w = torch.zeros(5, 8, dtype=BF, device=device)
    with pytest.raises(RuntimeError):
        kernel.gemm(x.float(), w)
    with pytest.raises(RuntimeError):
        kernel.gemm(x, torch.zeros(5, 9, dtype=BF, device=device))
    with pytest.raises(RuntimeError):
        kernel.gemm(x.T, torch.zeros(5, 4, dtype=BF, device=device))  # not contiguous
    with pytest.raises(RuntimeError):
        kernel.resid_ln(torch.zeros(4, 33, device=device), torch.zeros(4, 33, device=device), residual=False)
