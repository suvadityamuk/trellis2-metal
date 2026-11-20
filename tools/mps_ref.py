"""Loads the official TRELLIS.2 pipeline (pinned checkout in ref/TRELLIS.2) on the Apple GPU through PyTorch-MPS.
Upstream hard-depends on CUDA-only packages (flex_gemm, flash_attn, cumesh, nvdiffrast, o_voxel._C); tools/shims
replaces exactly those with small pure-PyTorch equivalents. Nothing in trellis2/ itself is edited.
Offline reference / baseline only; never used by the Metal runtime."""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path[:0] = [os.path.join(ROOT, "tools", "shims"), os.path.join(ROOT, "ref", "TRELLIS.2"), os.path.join(ROOT, "ref", "TRELLIS.2", "o-voxel")]
import numpy as np, torch

DEVICE = torch.device(os.environ.get("T2_DEVICE", "mps"))
torch.Tensor.cuda = lambda self, *a, **k: self.to(DEVICE)
torch.nn.Module.cuda = lambda self, *a, **k: self.to(DEVICE)


def sync():
    if DEVICE.type == "mps": torch.mps.synchronize()


# ---- o_voxel._C: only the two hashmap entry points used by flexible_dual_grid_to_mesh ---------------------------
_C = types.ModuleType("o_voxel._C")
_HM = {}
def _key(c, W, H, D): return ((c[:, 0].long() * W + c[:, 1].long()) * H + c[:, 2].long()) * D + c[:, 3].long()
def hashmap_insert_3d_idx_as_val_cuda(keys, vals, coords, W, H, D):
    k = _key(coords, W, H, D); sk, order = torch.sort(k); _HM[id(keys)] = (sk, order)
def hashmap_lookup_3d_cuda(keys, vals, coords, W, H, D):
    sk, order = _HM.pop(id(keys)); c = coords.long()
    ok = ((c[:, 1:] >= 0) & (c[:, 1:] < torch.tensor([W, H, D]))).all(1)
    q = _key(coords, W, H, D); pos = torch.searchsorted(sk, q).clamp_(max=sk.numel() - 1)
    hit = ok & (sk[pos] == q)
    return torch.where(hit, order[pos], torch.full_like(q, 0xffffffff))
_C.hashmap_insert_3d_idx_as_val_cuda = hashmap_insert_3d_idx_as_val_cuda
_C.hashmap_lookup_3d_cuda = hashmap_lookup_3d_cuda
sys.modules["o_voxel._C"] = _C


def load_pipeline(low_vram=False):
    from trellis2.pipelines import rembg
    class _NoRembg:                      # example images carry alpha; background removal is never invoked
        def __init__(self, *a, **k): pass
        def to(self, *a): pass
        def __call__(self, im): raise RuntimeError("image has no alpha channel; rembg is not part of the Mac reference")
    rembg.BiRefNet = _NoRembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import trellis2.models.sc_vaes.fdg_vae as fdg
    from trellis2.representations import Mesh
    _fdg = fdg.flexible_dual_grid_to_mesh
    def fdg_cpu(coords, dual_vertices, intersected_flag, split_weight, **kw):
        # hashmap ops are CUDA-only upstream; run mesh extraction on the CPU (exact IEEE fp32) and move the result back
        import o_voxel.convert.flexible_dual_grid as m
        dev = coords.device
        for a in ("edge_neighbor_voxel_offset", "quad_split_1", "quad_split_2", "quad_split_train"):
            if hasattr(m.flexible_dual_grid_to_mesh, a) and getattr(m.flexible_dual_grid_to_mesh, a).device.type != "cpu":
                delattr(m.flexible_dual_grid_to_mesh, a)
        _init = m._init_hashmap
        m._init_hashmap = lambda grid_size, capacity, device: (torch.empty(1), torch.empty(1))
        try:
            v, f = _fdg(coords.cpu(), dual_vertices.float().cpu(), intersected_flag.cpu(),
                        None if split_weight is None else split_weight.float().cpu(), **kw)
        finally:
            m._init_hashmap = _init
        return v.to(dev), f.to(dev)
    fdg.flexible_dual_grid_to_mesh = fdg_cpu
    Mesh.fill_holes = lambda self, *a, **k: None      # cumesh (CUDA) only
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.low_vram = low_vram
    pipe.to(DEVICE)
    return pipe
