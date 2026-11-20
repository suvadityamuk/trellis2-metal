"""Reference dump: runs the official pipeline on PyTorch-MPS and saves every stage boundary (and optional deep taps)
into one safetensors file that the Metal parity tests compare against bit for bit.

  ref/venv/bin/python tools/ref_dump.py <image> <512|1024_cascade> <out.safetensors> [--seed 42] [--steps N]
        [--taps REGEX] [--tapcalls K]

--taps: regex over "<model>.<module path>"; the output of each matching module is saved for its first K calls as
        tap.<model>.<module path>.<call>."""
import argparse, os, re, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
import mps_ref
from PIL import Image
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("ptype"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--taps", default="")
ap.add_argument("--tapcalls", type=int, default=2)
ap.add_argument("--stop", default="", help="stop after this stage (cond|ss|shape|tex)")
a = ap.parse_args()

D = {}
def put(name, t):
    if hasattr(t, "feats") and hasattr(t, "coords"):
        put(name + ".feats", t.feats); put(name + ".coords", t.coords); return
    if isinstance(t, (list, tuple)):
        for i, x in enumerate(t): put(f"{name}.{i}", x)
        return
    if isinstance(t, dict):
        for k, x in t.items(): put(f"{name}.{k}", x)
        return
    if isinstance(t, np.ndarray): t = torch.from_numpy(np.ascontiguousarray(t))
    if not torch.is_tensor(t): return
    t = t.detach()
    if t.is_complex(): t = torch.view_as_real(t)
    D[name] = t.cpu().contiguous()

pipe = mps_ref.load_pipeline()
if a.steps:
    for s in ("sparse_structure_sampler_params", "shape_slat_sampler_params", "tex_slat_sampler_params"):
        getattr(pipe, s)["steps"] = a.steps

class Stop(Exception): pass

# ---- stage boundaries -------------------------------------------------------------------------------------------
cnt = {}
def wrap(obj, meth, tag, post=None):
    f = getattr(obj, meth)
    def g(*args, **kw):
        r = f(*args, **kw)
        i = cnt.get(tag, 0); cnt[tag] = i + 1
        put(f"{tag}.{i}" if tag in ("cond",) else tag, r)
        if post: post(r)
        if a.stop and a.stop == tag: raise Stop()
        return r
    setattr(obj, meth, g)

_pp = pipe.preprocess_image
def pp(im):
    r = _pp(im); put("image", np.asarray(r)); return r
pipe.preprocess_image = pp
wrap(pipe, "get_cond", "cond")
wrap(pipe, "sample_sparse_structure", "ss")
wrap(pipe, "sample_shape_slat", "shape")
wrap(pipe, "sample_shape_slat_cascade", "shape")
wrap(pipe, "sample_tex_slat", "tex")

_dss = pipe.decode_shape_slat
def dss(slat, res):
    meshes, subs = _dss(slat, res)
    put("mesh.vertices", meshes[0].vertices); put("mesh.faces", meshes[0].faces)
    for i, s in enumerate(subs): put(f"subs.{i}", s)
    return meshes, subs
pipe.decode_shape_slat = dss
wrap(pipe, "decode_tex_slat", "texvox")

# raw decoder outputs (before sigmoid / mesh extraction) and the SS decoder logits
def hook_out(mod, tag):
    def h(m, inp, out):
        i = cnt.get(tag, 0); cnt[tag] = i + 1
        put(f"{tag}.{i}", out[0] if isinstance(out, tuple) else out)
    mod.register_forward_hook(h)
hook_out(pipe.models["sparse_structure_decoder"], "ssdec")
hook_out(pipe.models["shape_slat_decoder"].output_layer, "shapedec_h")

# sampler internals: noise, every model prediction, every step output
def wrap_sampler(s, tag):
    f_inf = s._inference_model; f_once = s.sample_once; f_sample = s.sample
    st = {"run": -1, "call": 0, "step": 0}
    def inf(model, x_t, t, cond=None, **kw):
        r = f_inf(model, x_t, t, cond, **kw)
        put(f"{tag}.{st['run']}.pred.{st['call']}", r); st["call"] += 1
        return r
    def once(model, x_t, t, t_prev, cond=None, **kw):
        r = f_once(model, x_t, t, t_prev, cond, **kw)
        put(f"{tag}.{st['run']}.x.{st['step']}", r.pred_x_prev); st["step"] += 1
        return r
    def sample(model, noise, *args, **kw):
        st["run"] += 1; st["call"] = 0; st["step"] = 0
        put(f"{tag}.{st['run']}.noise", noise)
        return f_sample(model, noise, *args, **kw)
    s._inference_model = inf; s.sample_once = once; s.sample = sample
wrap_sampler(pipe.sparse_structure_sampler, "ss_s")
wrap_sampler(pipe.shape_slat_sampler, "shape_s")
wrap_sampler(pipe.tex_slat_sampler, "tex_s")

# ---- deep taps --------------------------------------------------------------------------------------------------
if a.taps:
    rx = re.compile(a.taps)
    mods = dict(pipe.models); mods["dino"] = pipe.image_cond_model.model
    for mn, model in mods.items():
        for name, mod in model.named_modules():
            full = f"{mn}.{name}"
            if not rx.search(full): continue
            def mk(full):
                def h(m, inp, out):
                    i = cnt.get(full, 0); cnt[full] = i + 1
                    if i < a.tapcalls:
                        put(f"tap.{full}.{i}", out)
                        put(f"tapin.{full}.{i}", list(inp))
                return h
            mod.register_forward_hook(mk(full))

im = Image.open(a.image)
t0 = time.time()
try:
    out = pipe.run(im, seed=a.seed, pipeline_type=a.ptype, preprocess_image=True)
    m = out[0]
    put("out.vertices", m.vertices); put("out.faces", m.faces); put("out.attrs", m.attrs); put("out.coords", m.coords)
except Stop:
    pass
mps_ref.sync()
print("run", time.time() - t0)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
save_file(D, a.out)
with open(a.out + ".txt", "w") as f:
    for k, v in D.items(): f.write(f"{k}\t{v.dtype}\t{tuple(v.shape)}\n")
print("saved", len(D), "tensors ->", a.out)
