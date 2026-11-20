"""SDPA-backed stand-in for flash_attn's varlen API (CUDA only upstream). Queries are processed in chunks so the
N x N attention matrix is never materialised whole (MPS would need tens of GB at N ~ 40k)."""
import torch
import torch.nn.functional as F

CHUNK_BYTES = 2 << 30


def _attn(q, k, v):
    # q [Nq, H, D], k/v [Nk, H, D] -> [Nq, H, D]
    q = q.permute(1, 0, 2).unsqueeze(0); k = k.permute(1, 0, 2).unsqueeze(0); v = v.permute(1, 0, 2).unsqueeze(0)
    H, Nq, Nk = q.shape[1], q.shape[2], k.shape[2]
    rows = max(1, CHUNK_BYTES // (H * Nk * 4))
    if rows >= Nq:
        o = F.scaled_dot_product_attention(q, k, v)
    else:
        o = torch.cat([F.scaled_dot_product_attention(q[:, :, s:s + rows], k, v) for s in range(0, Nq, rows)], dim=2)
    return o[0].permute(1, 0, 2)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kw):
    cq = [int(x) for x in cu_seqlens_q]; ck = [int(x) for x in cu_seqlens_k]
    outs = [_attn(q[cq[i]:cq[i + 1]], k[ck[i]:ck[i + 1]], v[ck[i]:ck[i + 1]]) for i in range(len(cq) - 1)]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, **kw):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kw):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
