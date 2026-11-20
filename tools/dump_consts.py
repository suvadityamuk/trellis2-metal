"""Dump init-time CPU constants of the reference (timestep-embedding freqs, dense RoPE phases) for bitwise checks."""
import sys, numpy as np, torch
from safetensors.torch import save_file
out = {}
out["temb_freqs"] = torch.exp(-np.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128)
def rope(res, head_dim=128, dim=3):
    fd = head_dim // 2 // dim
    freqs = torch.arange(fd, dtype=torch.float32) / fd
    freqs = 1.0 / (10000.0 ** (freqs))
    coords = torch.meshgrid(*[torch.arange(r) for r in [res] * 3], indexing='ij')
    coords = torch.stack(coords, dim=-1).reshape(-1, 3)
    ph = torch.outer(coords.reshape(-1), freqs)
    ph = torch.polar(torch.ones_like(ph), ph).reshape(coords.shape[0], -1)
    padn = head_dim // 2 - ph.shape[-1]
    ph = torch.cat([ph, torch.polar(torch.ones(ph.shape[0], padn), torch.zeros(ph.shape[0], padn))], dim=-1)
    return freqs, torch.view_as_real(ph).contiguous()
f, p = rope(16)
out["rope_freqs"] = f; out["rope_dense16"] = p
save_file(out, sys.argv[1])
print({k: (v.dtype, tuple(v.shape)) for k, v in out.items()})
