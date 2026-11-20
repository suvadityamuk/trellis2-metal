"""Compare two image->3D outputs (MeshWithVoxel: vertices, faces, per-vertex voxel attrs).
  ref/venv/bin/python tools/compare_mesh.py A.{npz,safetensors} B.{npz,safetensors} [--samples 200000]
Prints: symmetric Chamfer distance / bbox diagonal, base-color L1 (0-255) at matched surface samples."""
import argparse, json, sys
import numpy as np
from scipy.spatial import cKDTree


def load(path):
    if path.endswith(".npz"):
        d = np.load(path); return {k: d[k] for k in ("vertices", "faces", "attrs")}
    from safetensors.numpy import load_file
    d = load_file(path); pre = "" if "vertices" in d else "out."  # reference dumps store the result as out.*
    return {k: d[pre + k] for k in ("vertices", "faces", "attrs")}


def sample(m, n, seed):
    v, f, a = m["vertices"].astype(np.float64), m["faces"].astype(np.int64), m["attrs"].astype(np.float64)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    s = np.sqrt(u); b = np.stack([1 - s, s * (1 - w), s * w], 1)
    pts = (tri[idx] * b[:, :, None]).sum(1)
    col = (a[f[idx]][:, :, :3] * b[:, :, None]).sum(1)
    return pts, np.clip(col, 0, 1) * 255.0


def compare(pa, pb, n=200000):
    A, B = load(pa), load(pb)
    xa, ca = sample(A, n, 0); xb, cb = sample(B, n, 1)
    diag = np.linalg.norm(B["vertices"].max(0) - B["vertices"].min(0))
    dab, iab = cKDTree(xb).query(xa); dba, iba = cKDTree(xa).query(xb)
    chamfer = 0.5 * (dab.mean() + dba.mean()) / diag
    color = 0.5 * (np.abs(ca - cb[iab]).mean() + np.abs(cb - ca[iba]).mean())
    return {"chamfer": float(chamfer), "color_l1": float(color), "verts_a": int(len(A["vertices"])), "verts_b": int(len(B["vertices"]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--samples", type=int, default=200000)
    a = ap.parse_args()
    print(json.dumps(compare(a.a, a.b, a.samples)))
