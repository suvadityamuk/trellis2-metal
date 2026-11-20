"""Lists every aten op the official pipeline executes on MPS (name, input dtypes/ranks, strides flags), per stage.
Used to scope which torch-MPS kernels the parity build has to mirror."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mtltrace'))
import torch, trace
import mps_ref
from torch.utils._python_dispatch import TorchDispatchMode
from PIL import Image

stage = ["init"]; seen = collections.OrderedDict()
def desc(a):
    if isinstance(a, torch.Tensor):
        return f"{str(a.dtype).replace('torch.','')}{list(a.shape)}{'' if a.is_contiguous() else '!nc'}@{a.device.type}"
    if isinstance(a, (list, tuple)): return "[" + ",".join(desc(x) for x in a[:4]) + "]"
    return repr(a)[:24]
def shapeless(a):
    if isinstance(a, torch.Tensor):
        return f"{str(a.dtype).replace('torch.','')}/{a.dim()}d{'' if a.is_contiguous() else '!nc'}@{a.device.type}"
    if isinstance(a, (list, tuple)): return "[" + ",".join(shapeless(x) for x in a[:4]) + "]"
    return type(a).__name__
class Census(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        k = (stage[0], func._schema.name + ("." + func._schema.overload_name if func._schema.overload_name else ""), tuple(shapeless(a) for a in args))
        trace.mark(f"{k[0]}|{k[1]}|{' '.join(k[2])}")
        out = func(*args, **(kwargs or {}))
        if k not in seen: seen[k] = [0, " ".join(desc(a) for a in args) + (" " + repr({a: desc(b) for a, b in kwargs.items()}) if kwargs else "") + " -> " + desc(out)]
        seen[k][0] += 1
        return out

pipe = mps_ref.load_pipeline()
P = type(pipe)
for fn in ["get_cond", "sample_sparse_structure", "sample_shape_slat", "sample_shape_slat_cascade", "sample_tex_slat", "decode_shape_slat", "decode_tex_slat", "decode_latent"]:
    def mk(orig, fn):
        def w(self, *a, **k):
            prev = stage[0]; stage[0] = fn
            try: return orig(self, *a, **k)
            finally: stage[0] = prev
        return w
    setattr(P, fn, mk(getattr(P, fn), fn))
img = Image.open(sys.argv[1])
steps = {"steps": 2}
trace.start(sys.argv[3] + '.trace')
with Census():
    pipe.run(img, seed=42, pipeline_type=sys.argv[2], sparse_structure_sampler_params=steps, shape_slat_sampler_params=steps, tex_slat_sampler_params=steps)
trace.stop()
with open(sys.argv[3], "w") as f:
    for (st, op, sig), (n, ex) in seen.items(): f.write(f"{st}\t{op}\t{n}\t{ex}\n")
ops = collections.Counter()
for (st, op, sig), (n, ex) in seen.items(): ops[op] += n
print(len(seen), "signatures;", len(ops), "ops"); print(sorted(ops.items(), key=lambda x: -x[1]))
