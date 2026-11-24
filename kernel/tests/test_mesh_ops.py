"""Mesh post-processing ops: lbvh_build / mesh_nearest / mesh_udf against a float64 brute force, telea_inpaint against
a NumPy implementation of the same wavefront algorithm and (loosely, the fill order differs) against OpenCV."""

import os

import numpy as np
import pytest
import torch


def closest_on_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Squared distance from every point [M, 3] to every triangle [F, 3, 3] in float64 -> [M, F]."""

    def seg(p, a, b):
        ab = b - a
        l2 = (ab * ab).sum(-1)
        t = np.where(l2 > 0, ((p - a) * ab).sum(-1) / np.where(l2 > 0, l2, 1.0), 0.0).clip(0, 1)
        q = a + ab * t[..., None]
        return ((p - q) ** 2).sum(-1)

    p = p[:, None, :].astype(np.float64)
    a, b, c = (tri[None, :, k, :].astype(np.float64) for k in range(3))
    d2 = np.minimum(np.minimum(seg(p, a, b), seg(p, b, c)), seg(p, c, a))
    n = np.cross(b - a, c - a)
    nn = (n * n).sum(-1)
    ok = nn > 1e-300
    nn1 = np.where(ok, nn, 1.0)
    t = ((p - a) * n).sum(-1) / nn1
    q = p - n * t[..., None]  # projection on the plane
    inside = ok
    for u, v in ((a, b), (b, c), (c, a)):
        inside = inside & ((np.cross(v - u, q - u) * n).sum(-1) >= 0)
    return np.where(inside, np.minimum(d2, t * t * nn1), d2)


def random_mesh(nf: int, seed: int, degenerate: bool = False):
    g = np.random.default_rng(seed)
    nv = max(3, nf // 2 + 3)
    v = g.uniform(-0.5, 0.5, (nv, 3)).astype(np.float32)
    f = g.integers(0, nv, (nf, 3)).astype(np.int32)  # random soup: repeated vertices give zero-area triangles
    if not degenerate:
        f[:, 1] = (f[:, 0] + 1 + g.integers(0, nv - 2, nf)) % nv
        f[:, 2] = np.where((f[:, 2] == f[:, 0]) | (f[:, 2] == f[:, 1]), (np.maximum(f[:, 0], f[:, 1]) + 1) % nv, f[:, 2])
    return v, f


def check_nearest(kernel, device, v, f, pts, atol=3e-5):
    bvh = kernel.lbvh_build(torch.from_numpy(v), torch.from_numpy(f))
    dist, face, bary = kernel.mesh_nearest(bvh, torch.from_numpy(pts).to(device))
    dist, face, bary = dist.cpu().numpy(), face.cpu().numpy(), bary.cpu().numpy().astype(np.float64)
    ref = np.sqrt(closest_on_triangles(pts, v[f]))
    best = ref.min(1)
    assert np.abs(dist - best).max() <= atol
    assert face.min() >= 0 and face.max() < len(f)
    # the reported face attains the minimum, and bary reproduces the closest point on it
    assert np.abs(ref[np.arange(len(pts)), face] - best).max() <= atol
    q = (v[f[face]].astype(np.float64) * bary[:, :, None]).sum(1)
    assert np.abs(np.linalg.norm(pts - q, axis=1) - dist).max() <= atol
    assert bary.min() >= 0 and np.abs(bary.sum(1) - 1).max() <= 1e-5
    return dist


@pytest.mark.kernels_ci
@pytest.mark.parametrize("nf", [1, 2, 3, 37, 4000])
def test_mesh_nearest(kernel, device, nf):
    v, f = random_mesh(nf, nf)
    g = np.random.default_rng(100 + nf)
    pts = g.uniform(-0.7, 0.7, (1500, 3)).astype(np.float32)
    # some queries exactly on vertices and on the surface
    pts[:50] = v[g.integers(0, len(v), 50)]
    w = g.dirichlet(np.ones(3), 50).astype(np.float32)
    pts[50:100] = (v[f[g.integers(0, nf, 50)]] * w[:, :, None]).sum(1)
    d = check_nearest(kernel, device, v, f, pts)
    assert d[:50].max() <= 3e-5 or nf < 37  # vertices of tiny meshes may be unreferenced


@pytest.mark.kernels_ci
def test_mesh_nearest_degenerate_triangles(kernel, device):
    v, f = random_mesh(600, 7, degenerate=True)
    f[:100, 2] = f[:100, 1]  # segments
    f[100:150, 1] = f[100:150, 0]
    f[100:150, 2] = f[100:150, 0]  # points
    v = np.concatenate([v, np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], np.float32)])
    f = np.concatenate([f, np.array([[len(v) - 3, len(v) - 2, len(v) - 1]], np.int32)])  # collinear
    pts = np.random.default_rng(8).uniform(-0.7, 0.7, (2000, 3)).astype(np.float32)
    check_nearest(kernel, device, v, f, pts)


@pytest.mark.kernels_ci
def test_mesh_nearest_max_distance_and_views(kernel, device):
    v, f = random_mesh(500, 3)
    pts = np.random.default_rng(4).uniform(-1.5, 1.5, (3001, 3)).astype(np.float32)
    best = np.sqrt(closest_on_triangles(pts, v[f]).min(1))
    cap = float(np.median(best))
    bvh = kernel.lbvh_build(torch.from_numpy(v), torch.from_numpy(f))
    base = torch.from_numpy(np.concatenate([np.zeros((5, 3), np.float32), pts])).to(device)
    view = base[5:]  # contiguous view with a storage offset
    dist, face, _ = kernel.mesh_nearest(bvh, view, max_distance=cap)
    dist, face = dist.cpu().numpy(), face.cpu().numpy()
    far, near = best > cap + 1e-4, best < cap - 1e-4
    assert (face[far] == -1).all() and np.isinf(dist[far]).all()
    assert (face[near] >= 0).all() and np.abs(dist[near] - best[near]).max() <= 3e-5
    udf = kernel.mesh_udf(torch.from_numpy(v), torch.from_numpy(f), view).cpu().numpy()
    assert np.abs(udf - best).max() <= 3e-5


def telea_reference(img: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """The wavefront Telea fill in float32 NumPy: img uint8 [H, W, C], mask [H, W] bool."""
    H, W, C = img.shape
    out = img.copy()
    lv = np.where(mask, -1, 0).astype(np.int32)
    L = 0
    while True:  # 4-connected levels
        L += 1
        k = lv == L - 1
        nb = np.zeros_like(k)
        nb[1:] |= k[:-1]
        nb[:-1] |= k[1:]
        nb[:, 1:] |= k[:, :-1]
        nb[:, :-1] |= k[:, 1:]
        new = nb & (lv == -1)
        if not new.any():
            break
        lv[new] = L
    f32 = np.float32

    for L in range(1, L):
        def kn(r, c):
            return 0 <= r < H and 0 <= c < W and 0 <= lv[r, c] < L

        src = out.copy()  # pixels of one level only read lower levels, which are final
        for y, x in zip(*np.nonzero(lv == L)):
            fL = f32(L)
            xr, xl, yd, yu = kn(y, x + 1), kn(y, x - 1), kn(y + 1, x), kn(y - 1, x)
            gx = ((f32(lv[y, x + 1]) - f32(lv[y, x - 1])) * f32(0.5) if xl else f32(lv[y, x + 1]) - fL) if xr else (fL - f32(lv[y, x - 1]) if xl else f32(0))
            gy = ((f32(lv[y + 1, x]) - f32(lv[y - 1, x])) * f32(0.5) if yu else f32(lv[y + 1, x]) - fL) if yd else (fL - f32(lv[y - 1, x]) if yu else f32(0))
            Ia, Jx, Jy = np.zeros(C, f32), np.zeros(C, f32), np.zeros(C, f32)
            s = f32(1e-20)
            for k in range(y - radius, y + radius + 1):
                for l in range(x - radius, x + radius + 1):
                    if (k - y) ** 2 + (l - x) ** 2 > radius * radius or not kn(k, l):
                        continue
                    ry, rx = f32(y - k), f32(x - l)
                    vl = f32(rx * rx + ry * ry)
                    dr = f32(f32(rx * gx) + f32(ry * gy))
                    if abs(dr) <= f32(0.01):
                        dr = f32(1e-6)
                    w = abs(f32(dr / f32(f32(vl * np.sqrt(vl)) * f32(f32(1) + abs(f32(lv[k, l]) - fL)))))
                    r1, l1, d1, u1 = kn(k, l + 1), kn(k, l - 1), kn(k + 1, l), kn(k - 1, l)
                    for c in range(C):
                        v = f32(src[k, l, c])
                        ix = ((f32(src[k, l + 1, c]) - f32(src[k, l - 1, c])) * f32(2) if l1 else f32(src[k, l + 1, c]) - v) if r1 else (v - f32(src[k, l - 1, c]) if l1 else f32(0))
                        iy = ((f32(src[k + 1, l, c]) - f32(src[k - 1, l, c])) * f32(2) if u1 else f32(src[k + 1, l, c]) - v) if d1 else (v - f32(src[k - 1, l, c]) if u1 else f32(0))
                        Ia[c] += f32(w * v)
                        Jx[c] -= f32(f32(w * ix) * rx)
                        Jy[c] -= f32(f32(w * iy) * ry)
                    s = f32(s + w)
            sat = Ia / s + (Jx + Jy) / (np.sqrt(Jx * Jx + Jy * Jy) + f32(1e-20))
            out[y, x] = np.clip(np.rint(sat + f32(0.5)), 0, 255).astype(np.uint8)
    return out


def blob_mask(H, W, seed, frac=0.35):
    g = np.random.default_rng(seed)
    m = g.random((H, W)) < 0.04
    for _ in range(3):  # grow the seeds into blobs a few pixels wide
        n = m.copy()
        n[1:] |= m[:-1]
        n[:-1] |= m[1:]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n & (g.random((H, W)) < 0.9) | m
    return m


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape,radius", [((33, 47, 3), 3), ((40, 40, 1), 1), ((24, 31, 4), 2), ((30, 30, 6), 3), ((1, 50, 3), 3)])
def test_telea_matches_reference(kernel, device, shape, radius):
    H, W, C = shape
    g = np.random.default_rng(H * W + C)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([(127 + 100 * np.sin(xx / (3.0 + c) + yy / 5.0) + g.normal(0, 6, (H, W))).clip(0, 255) for c in range(C)], -1).astype(np.uint8)
    mask = blob_mask(H, W, 11 + C)
    assert mask.any() and not mask.all()
    out = kernel.telea_inpaint(torch.from_numpy(img).to(device), torch.from_numpy(mask).to(device), radius)
    assert out.shape == img.shape and out.dtype == torch.uint8
    out = out.cpu().numpy()
    assert (out[~mask] == img[~mask]).all()
    ref = telea_reference(img, mask, radius)
    # Exact when the shaders are compiled without fast math (the development build with T2_FAST_MATH=0). The
    # builder's fast-math metallib may reassociate the sums, which can move a value across a rounding boundary, and
    # later levels read that texel (and twice its finite difference), so a few grey levels on a few texels are allowed.
    diff = np.abs(out.astype(int) - ref.astype(int))
    print(f"telea vs reference {shape} r={radius}: max {diff.max()}, differing {100 * (diff > 0).mean():.3f}%")
    if os.environ.get("T2_FAST_MATH") == "0":  # safe-math development build: at most a rounding tie apart from NumPy
        assert diff.max() <= 1 and (diff > 0).mean() < 1e-3
    assert diff.max() <= 4 and (diff > 0).mean() < 0.02


@pytest.mark.kernels_ci
def test_telea_trivial_masks(kernel, device):
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (20, 23, 3), generator=g, dtype=torch.uint8).to(device)
    none = torch.zeros(20, 23, dtype=torch.bool)
    assert torch.equal(kernel.telea_inpaint(img, none), img)
    assert torch.equal(kernel.telea_inpaint(img, ~none), img)  # no known pixel: nothing to fill from
    one = none.clone()
    one[7, 9] = True
    out = kernel.telea_inpaint(img, one.to(torch.uint8) * 255, radius=1)
    assert torch.equal(out.cpu()[~one], img.cpu()[~one])
    flat = torch.full((16, 16, 2), 93, dtype=torch.uint8, device=device)
    m = torch.zeros(16, 16, dtype=torch.bool)
    m[4:11, 5:12] = True
    # like OpenCV, the fill rounds (value + 0.5) and adds a unit-length gradient term, so a constant image is
    # reproduced only to within a few grey levels
    d = kernel.telea_inpaint(flat, m).cpu().int() - 93
    try:
        import cv2
        dc = cv2.inpaint(flat.cpu().numpy()[..., :1].copy(), m.numpy().astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(int) - 93
        print(f"constant image: kernel max |d| {int(d.abs().max())}, cv2 max |d| {int(np.abs(dc).max())}")
    except ImportError:
        pass
    assert d.abs().max() <= 3 and torch.equal(d[~m], torch.zeros_like(d[~m]))


def test_telea_close_to_opencv(kernel, device):
    cv2 = pytest.importorskip("cv2")
    H = W = 96
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([127 + 90 * np.sin(xx / 9.0), 127 + 90 * np.cos(yy / 11.0), 2.0 * xx + 0.5 * yy], -1).clip(0, 255).astype(np.uint8)
    mask = np.zeros((H, W), bool)
    mask[10:86, 30:33] = True  # scratches 2-3 pixels wide, the texture-seam case
    mask[50:52, 5:90] = True
    out = kernel.telea_inpaint(torch.from_numpy(img).to(device), torch.from_numpy(mask).to(device), 3).cpu().numpy()
    ref = cv2.inpaint(img, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    d = np.abs(out.astype(int) - ref.astype(int))[mask]
    print(f"telea vs cv2: mean {d.mean():.3f}, p99 {np.percentile(d, 99):.1f}, max {d.max()}")
    assert d.mean() < 5.0 and np.percentile(d, 99) < 24


def test_mesh_ops_argument_checks(kernel, device):
    v, f = random_mesh(10, 0)
    tv, tf = torch.from_numpy(v), torch.from_numpy(f)
    bvh = kernel.lbvh_build(tv, tf)
    pts = torch.zeros(8, 3, device=device)
    bad = tf.clone()
    bad[0, 0] = 10_000
    with pytest.raises(RuntimeError):
        kernel.lbvh_build(tv, bad)
    with pytest.raises(RuntimeError):
        kernel.lbvh_build(tv[:, :2], tf)
    with pytest.raises(RuntimeError):
        kernel.mesh_nearest(bvh, pts.to(torch.int32))
    with pytest.raises(RuntimeError):
        kernel.mesh_nearest(bvh, torch.zeros(8, 6, device=device)[:, ::2])  # not contiguous
    with pytest.raises(RuntimeError):
        kernel.mesh_nearest((bvh[0], bvh[1][:-1]), pts)  # nodes / tris that do not belong together
    with pytest.raises(RuntimeError):
        kernel.mesh_nearest(bvh, pts, max_distance=-1.0)
    img = torch.zeros(8, 8, 3, dtype=torch.uint8, device=device)
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[2, 2] = True
    with pytest.raises(RuntimeError):
        kernel.telea_inpaint(img.float(), m)
    with pytest.raises(RuntimeError):
        kernel.telea_inpaint(torch.zeros(8, 8, 9, dtype=torch.uint8, device=device), m)
    with pytest.raises(RuntimeError):
        kernel.telea_inpaint(img, m, radius=0)
    with pytest.raises(ValueError):
        kernel.telea_inpaint(img, m[:4])
    assert kernel.mesh_nearest(bvh, pts[:0])[0].shape == (0,)
