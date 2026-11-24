"""Offline checks of a GLB written by build/t2 --glb / build/glb (run with ref/venv, nothing here is used at runtime).

  python tools/check_glb.py out.glb [mesh.safetensors]

1. Validity: the file loads in a standard glTF parser (trimesh), has normals, UVs in [0, 1], both textures decode.
2. to_glb in isolation (when the MeshWithVoxel safetensors it was made from is given): distance between the GLB surface
   and the input mesh (the remesh sits remesh_band voxels off the input surface by construction), and base colour at
   GLB surface samples against the voxel colours at the same place (nearest voxel of the input attribute grid).
"""
import json, os, struct, sys
import numpy as np, trimesh
from scipy.spatial import cKDTree


def main():
    path = sys.argv[1]
    raw = open(path, "rb").read()
    magic, ver, total = struct.unpack("<III", raw[:12]); jl, jt = struct.unpack("<II", raw[12:20])
    assert magic == 0x46546C67 and ver == 2 and total == len(raw) and jt == 0x4E4F534A, "bad GLB container"
    js = json.loads(raw[20:20 + jl])
    m = trimesh.load(path, force="mesh")
    mat = m.visual.material
    base = np.asarray(mat.baseColorTexture.convert("RGB"), dtype=np.float32)
    mr = np.asarray(mat.metallicRoughnessTexture.convert("RGB"), dtype=np.float32)
    uv = np.asarray(m.visual.uv)
    assert uv.min() >= 0 and uv.max() <= 1, "UV outside [0, 1]"
    bin0 = 20 + jl + 8
    def acc(i, dt, k):
        a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
        return np.frombuffer(raw, dt, a["count"] * k, bin0 + bv.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(a["count"], k)
    pr = js["meshes"][0]["primitives"][0]
    nrm = acc(pr["attributes"]["NORMAL"], np.float32, 3); pos = acc(pr["attributes"]["POSITION"], np.float32, 3); idx = acc(pr["indices"], np.uint32, 1).reshape(-1, 3)
    assert len(nrm) == len(pos) and idx.max() < len(pos) and np.isfinite(pos).all() and np.isfinite(nrm).all(), "bad accessors"
    nl = np.linalg.norm(nrm, axis=1)
    if os.environ.get("CHECK_GLB_LENIENT"):  # for reference GLBs from other exporters: report instead of failing
        print(f"NORMALS: {int((abs(nl - 1) > 1e-3).sum())} of {len(nl)} not unit length")
    else:
        assert abs(nl - 1).max() < 1e-3, f"NORMAL not unit length: [{nl.min()}, {nl.max()}]"
    a_ = js["accessors"][pr["attributes"]["POSITION"]]
    assert np.allclose(pos.min(0), a_["min"], atol=1e-6) and np.allclose(pos.max(0), a_["max"], atol=1e-6), "POSITION min/max wrong"
    # geometric topology (vertices welded by position, since UV seams duplicate them)
    _, w = np.unique(pos, axis=0, return_inverse=True); wf = w.reshape(-1)[idx]
    deg = (wf[:, 0] == wf[:, 1]) | (wf[:, 1] == wf[:, 2]) | (wf[:, 0] == wf[:, 2])
    e = np.sort(np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]]), axis=1)
    _, ec = np.unique(e[:, 0].astype(np.int64) * (wf.max() + 1) + e[:, 1], return_counts=True)
    print(f"TOPOLOGY: welded verts {wf.max() + 1}, degenerate faces {int(deg.sum())}, edges {len(ec)}: boundary {int((ec == 1).sum())}, manifold {int((ec == 2).sum())}, non-manifold {int((ec > 2).sum())}")
    print(f"VALID GLB: {len(m.vertices)} verts, {len(m.faces)} faces, textures base {base.shape} mr {mr.shape}, images "
          f"{[i.get('mimeType') for i in js['images']]}, extensions {js.get('extensionsUsed', [])}, normals |n| in [{nl.min():.3f}, {nl.max():.3f}], "
          f"uv in [{uv.min():.4f}, {uv.max():.4f}], bounds {np.round(m.bounds, 3).tolist()}")
    if len(sys.argv) < 3:
        return
    from safetensors.numpy import load_file
    d = load_file(sys.argv[2])
    n = 400000
    pts, fid = trimesh.sample.sample_surface(m, n, seed=0)
    bary = trimesh.triangles.points_to_barycentric(m.triangles[fid], pts)
    suv = (uv[m.faces[fid]] * bary[:, :, None]).sum(1)
    h, w = base.shape[:2]
    x = np.clip((suv[:, 0] * w).astype(int), 0, w - 1); y = np.clip(((1 - suv[:, 1]) * h).astype(int), 0, h - 1)
    col = base[y, x]
    p = np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], 1)  # glTF y-up -> pipeline z-up
    src = trimesh.Trimesh(d["vertices"], d["faces"], process=False)
    sp, _ = trimesh.sample.sample_surface(src, 2 * n, seed=1)
    d_gs = cKDTree(sp).query(p)[0]; d_sg = cKDTree(p).query(sp)[0]
    res = 64
    while res <= d["coords"].max(): res *= 2
    centers = (d["coords"].astype(np.float32) + 0.5) / res - 0.5
    dist, idx = cKDTree(centers).query(p)
    ref = np.clip(d["attrs"][idx, :3] * 255, 0, 255)
    l1 = np.abs(col - ref).mean()
    mrs = mr[y, x]
    l1_r = np.abs(mrs[:, 1] - np.clip(d["attrs"][idx, 4] * 255, 0, 255)).mean(); l1_m = np.abs(mrs[:, 2] - np.clip(d["attrs"][idx, 3] * 255, 0, 255)).mean()
    print(f"ISOLATION: voxel {1 / res:.2e}; GLB->input mesh mean {d_gs.mean():.2e} p99 {np.percentile(d_gs, 99):.2e}; input->GLB mean {d_sg.mean():.2e} p99 {np.percentile(d_sg, 99):.2e}; "
          f"base colour L1 vs nearest voxel {l1:.2f}/255 (nearest voxel center at mean {dist.mean():.2e}); roughness L1 {l1_r:.2f}, metallic L1 {l1_m:.2f}")


if __name__ == "__main__":
    main()
