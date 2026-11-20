"""Official TRELLIS.2 pipeline timing on PyTorch-MPS (image -> MeshWithVoxel), warm process, per-stage breakdown.
  ref/venv/bin/python tools/bench_ref.py --type 512 --images a.png b.webp ... --out bench/official_512.json [--save DIR]
Model loading is excluded; one warmup run of the first image precedes the timed runs."""
import argparse, os, sys, time, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mps_ref  # must precede torch: sets PYTORCH_ENABLE_MPS_FALLBACK etc.
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--type", default="512"); ap.add_argument("--images", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--save", default="")
a = ap.parse_args()

pipe = mps_ref.load_pipeline()
T = {}
def now(): mps_ref.sync(); return time.perf_counter()
def timed(name, key):
    f = getattr(pipe, name)
    def g(*args, **kw):
        s = now(); r = f(*args, **kw); T[key] = T.get(key, 0.0) + now() - s; return r
    setattr(pipe, name, g)
for n, k in [("preprocess_image", "preprocess"), ("get_cond", "cond"), ("sample_sparse_structure", "sparse_structure"),
             ("sample_shape_slat", "shape_slat"), ("sample_shape_slat_cascade", "shape_slat"), ("sample_tex_slat", "tex_slat"),
             ("decode_shape_slat", "decode_shape"), ("decode_tex_slat", "decode_tex")]:
    timed(n, k)

def run(path, save=False):
    T.clear(); s = now()
    m = pipe.run(Image.open(path), seed=a.seed, pipeline_type=a.type)[0]
    t = dict(T); t["total"] = now() - s
    t["vertices"] = int(m.vertices.shape[0]); t["faces"] = int(m.faces.shape[0]); t["voxels"] = int(m.coords.shape[0])
    t["peak_mps_gb"] = torch.mps.driver_allocated_memory() / 2**30
    if save and a.save:
        os.makedirs(a.save, exist_ok=True)
        np.savez(os.path.join(a.save, os.path.splitext(os.path.basename(path))[0] + f"_{a.type}.npz"),
                 vertices=m.vertices.cpu().numpy(), faces=m.faces.cpu().numpy(), coords=m.coords.cpu().numpy(),
                 attrs=m.attrs.float().cpu().numpy())
    return t

for i in range(a.warmup): print("warmup", run(a.images[0]), flush=True)
R = {}
for p in a.images:
    k = os.path.basename(p); R[k] = run(p, True); print(a.type, k, R[k], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
