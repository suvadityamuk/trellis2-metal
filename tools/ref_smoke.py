import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mps_ref, torch
from PIL import Image
t0 = time.time(); pipe = mps_ref.load_pipeline(); print("load", time.time() - t0, flush=True)
print({k: str(next(m.parameters()).dtype) for k, m in pipe.models.items()})
P = type(pipe)
for fn in ["get_cond", "sample_sparse_structure", "sample_shape_slat", "sample_shape_slat_cascade", "sample_tex_slat", "decode_shape_slat", "decode_tex_slat"]:
    def mk(orig, fn):
        def w(self, *a, **k):
            mps_ref.sync(); t = time.time(); o = orig(self, *a, **k); mps_ref.sync(); print(f"[time] {fn}: {time.time()-t:.2f}s", flush=True); return o
        return w
    setattr(P, fn, mk(getattr(P, fn), fn))
img = Image.open(sys.argv[1]); t0 = time.time()
mesh = pipe.run(img, seed=42, pipeline_type=sys.argv[2])[0]
mps_ref.sync(); print("run", time.time() - t0)
print(mesh.vertices.shape, mesh.faces.shape, mesh.attrs.shape, mesh.coords.shape)
print("peak mem GB", torch.mps.driver_allocated_memory() / 2**30)
