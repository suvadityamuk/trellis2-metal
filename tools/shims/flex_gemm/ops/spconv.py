import numpy as np
import torch

CHUNK_BYTES = 1 << 30   # im2col budget per chunk


def set_algorithm(*a, **k): pass
def set_hashmap_ratio(*a, **k): pass


def build_neighbor_map(coords: torch.Tensor, dilation=(1, 1, 1)) -> torch.Tensor:
    """[N, 27] int32 index of the neighbor at offset v (N = missing)."""
    c = coords.detach().cpu().numpy().astype(np.int64)
    n = c.shape[0]
    S = int(c[:, 1:].max()) + 3
    key = ((c[:, 0] * S + c[:, 1] + 1) * S + c[:, 2] + 1) * S + c[:, 3] + 1
    order = np.argsort(key, kind='stable')
    skey = key[order]
    nbr = np.empty((n, 27), dtype=np.int32)
    for v in range(27):
        dx, dy, dz = (v // 9 - 1) * dilation[0], (v // 3 % 3 - 1) * dilation[1], (v % 3 - 1) * dilation[2]
        q = key + (dx * S + dy) * S + dz
        pos = np.searchsorted(skey, q)
        pos[pos >= n] = n - 1
        hit = skey[pos] == q
        nbr[:, v] = np.where(hit, order[pos], n)
    return torch.from_numpy(nbr).to(coords.device)


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias=None, neighbor_cache=None, dilation=(1, 1, 1)):
    N, Ci = feats.shape
    Co = weight.shape[0]
    assert weight.shape[1:4] == (3, 3, 3)
    if neighbor_cache is None:
        neighbor_cache = build_neighbor_map(coords, dilation)
    nbr = neighbor_cache
    w = weight.reshape(Co, 27 * Ci)
    xp = torch.cat([feats, feats.new_zeros(1, Ci)], dim=0)
    rows = max(1, CHUNK_BYTES // (27 * Ci * feats.element_size()))
    outs = []
    for s in range(0, N, rows):
        col = xp[nbr[s:s + rows].long()].reshape(-1, 27 * Ci)
        outs.append(torch.nn.functional.linear(col, w, bias))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return out, neighbor_cache
