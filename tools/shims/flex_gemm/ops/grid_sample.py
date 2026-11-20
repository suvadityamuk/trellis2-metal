import numpy as np
import torch


def grid_sample_3d(feats, coords, shape, grid, mode='trilinear'):
    """Sparse trilinear sampling (voxel centers at integer + 0.5; missing corners are skipped and the weights of the
    present corners renormalised, as FlexGEMM does). feats [N, C], coords [N, 4] (b, x, y, z), grid [B, M, 3]."""
    assert mode == 'trilinear'
    dev = feats.device
    c = coords.detach().cpu().numpy().astype(np.int64)
    W, H, D = [int(s) for s in shape[-3:]]
    key = (c[:, 1] * H + c[:, 2]) * D + c[:, 3]
    order = np.argsort(key, kind='stable'); skey = key[order]; n = len(key)
    g = grid.detach().float().cpu().numpy().reshape(-1, 3) - 0.5
    base = np.floor(g).astype(np.int64); frac = (g - base).astype(np.float32)
    f = feats.detach().float().cpu().numpy()
    out = np.zeros((g.shape[0], f.shape[1]), dtype=np.float32); wsum = np.zeros((g.shape[0], 1), dtype=np.float32)
    for k in range(8):
        o = np.array([(k >> 2) & 1, (k >> 1) & 1, k & 1])
        p = base + o
        w = np.prod(np.where(o == 1, frac, 1 - frac), axis=1, keepdims=True)
        ok = ((p >= 0) & (p < np.array([W, H, D]))).all(1)
        q = (p[:, 0] * H + p[:, 1]) * D + p[:, 2]
        pos = np.searchsorted(skey, q); pos[pos >= n] = n - 1
        ok &= skey[pos] == q
        idx = order[pos]
        out += np.where(ok[:, None], w * f[idx], 0); wsum += np.where(ok[:, None], w, 0)
    out = out / np.maximum(wsum, 1e-12)
    return torch.from_numpy(out).reshape(grid.shape[0], -1, f.shape[1]).to(dev)
