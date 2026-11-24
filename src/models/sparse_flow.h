// SLatFlowModel (sparse DiT over active voxels, shape and texture variants) — parity build. Batch size 1.
#pragma once
#include "common.h"

namespace t2 {

// other[batch_boardcast_map] of SparseTensor.__elemwise__: a [1, C] tensor materialised as [N, C] (data movement).
inline Tensor sp_bcast(const Tensor& row, const Tensor& zidx) {
  Tensor s = row; s.shape = {1, row.numel()}; s.strides = Tensor::cstrides(s.shape);
  return gather_rows(s, zidx);
}
inline Tensor sp_zero_index(int64_t N) { return zeros(tk::I32, {N}); }

// VarLenTensor.std(dim=[1], keepdim=True) for one batch element -> f32 [1,1].
// feats.mean(1) on the GPU, torch.segment_reduce('mean') on the CPU (sequential float32 sum / N).
inline Tensor sp_std(const Tensor& x) {
  auto mean1 = [](const Tensor& f) {
    Tensor red = mean_dims(f, {1}, true);
    tk::sync();
    const float* p = red.ptr<float>(); int64_t n = red.numel();
    float s = 0; for (int64_t i = 0; i < n; i++) s = s + p[i];
    if (!std::isnan(s)) s = s / (float)n;
    return from_cpu(tk::F32, {1, 1}, &s);
  };
  Tensor mean = mean1(x), mean2 = mean1(un("sqr", x));
  return un("sqrt", sub(mean2, un("sqr", mean)));
}

struct SparseFlow {
  int res = 32, C_in = 32, C_out = 32, ch = 1536, heads = 12, nblocks = 30;
  DT dt = tk::BF16;
  TimestepEmbedder temb; LinearP adaln, in_layer, out_layer;
  std::vector<BlockP> blocks;
  // per-coords state (the reference's spatial cache)
  Tensor phases, zidx, concat;  // C64 [N,1,64]; I32 [N] zeros; optional concat_cond feats [N, Cc]
  int64_t N = 0;
  static constexpr int64_t kChunkBytes = (int64_t)2 << 30;  // tools/shims/flash_attn.py CHUNK_BYTES

  void load(const std::string& json, const std::string& st) {
    Json cfg = loadJson(json); const Json& a = cfg["args"];
    res = (int)a["resolution"].i(); C_in = (int)a["in_channels"].i(); C_out = (int)a["out_channels"].i();
    ch = (int)a["model_channels"].i(); heads = (int)a["num_heads"].i(); nblocks = (int)a["num_blocks"].i();
    STFile f(st);
    temb.load(f, "t_embedder"); adaln.load(f, "adaLN_modulation.1", tk::F32);
    in_layer.load(f, "input_layer", tk::F32); out_layer.load(f, "out_layer", tk::F32);
    blocks.resize(nblocks);
    for (int i = 0; i < nblocks; i++) { blocks[i].load(f, "blocks." + std::to_string(i), dt); blocks[i].sa.heads = blocks[i].ca.heads = heads; }
  }

  // SparseRotaryPositionEmbedder phases, computed on the device like the reference. coords I32 [N,4].
  void setCoords(const Tensor& coords, const Tensor& concat_cond = Tensor()) {
    N = coords.shape[0]; concat = concat_cond;
    int hd = ch / heads, fd = hd / 2 / 3;
    std::vector<float> fr(fd);
    for (int i = 0; i < fd; i++) { float e = (float)i / (float)fd; fr[i] = 1.0f / std::pow(10000.0f, e); }
    Tensor freqs = from_cpu(tk::F32, {(int64_t)fd}, fr.data());
    Tensor idx = contiguous(coords.slice(1, 1, 4)).view({N * 3, 1});
    Tensor ph = mul(idx, freqs);                                  // torch.outer(indices, freqs)
    ph = bin("polar", full(tk::F32, ph.shape, tk::Scal(1.0)), ph).view({N, (int64_t)3 * fd});
    int64_t padn = hd / 2 - 3 * fd;
    if (padn > 0) ph = cat({ph, bin("polar", full(tk::F32, {N, padn}, tk::Scal(1.0)), full(tk::F32, {N, padn}, tk::Scal(0.0)))}, -1);
    phases = ph.unsqueeze(-2);
    zidx = sp_zero_index(N);
  }

  // tools/shims/flash_attn.py::_attn — q [Nq,H,D], k/v [Nk,H,D] views -> [Nq,H,D] view
  static Tensor attn(const Tensor& q3, const Tensor& k3, const Tensor& v3) {
    Tensor q = q3.permute({1, 0, 2}).unsqueeze(0), k = k3.permute({1, 0, 2}).unsqueeze(0), v = v3.permute({1, 0, 2}).unsqueeze(0);
    int64_t H = q.shape[1], Nq = q.shape[2], Nk = k.shape[2];
    int64_t rows = std::max<int64_t>(1, kChunkBytes / (H * Nk * 4));
    Tensor o;
    if (rows >= Nq) o = sdpa(q, k, v);
    else {
      std::vector<Tensor> parts;
      for (int64_t s = 0; s < Nq; s += rows) parts.push_back(sdpa(q.slice(2, s, std::min(Nq, s + rows)), k, v));
      o = cat(parts, 2);
    }
    return o.select(0, 0).permute({1, 0, 2});
  }

  Tensor selfAttn(const AttnP& p, const Tensor& x) const {
    int64_t H = p.heads;
    Tensor qkv = p.qkv(x).view({N, 3, H, -1});
    Tensor q = rms_norm(qkv.select(1, 0), p.qg), k = rms_norm(qkv.select(1, 1), p.kg), v = qkv.select(1, 2);
    q = apply_rope(q, phases); k = apply_rope(k, phases);
    Tensor s = stack({q, k, v}, 1);
    Tensor h = attn(s.select(1, 0), s.select(1, 1), s.select(1, 2));
    return p.out(contiguous(h).view({N, -1}));
  }
  Tensor crossAttn(const AttnP& p, const Tensor& x, const Tensor& ctx) const {
    int64_t L = ctx.shape[1], H = p.heads;
    Tensor q = p.q(x).view({N, H, -1});
    Tensor kv = p.kv(ctx).view({1, L, 2, H, -1});
    q = rms_norm(q, p.qg);
    Tensor k = rms_norm(kv.select(2, 0), p.kg), v = kv.select(2, 1);
    Tensor h = attn(q, k.select(0, 0), v.select(0, 0));
    return p.out(contiguous(h).view({N, -1}));
  }
  Tensor block(const BlockP& p, Tensor x, const Tensor& mod, const Tensor& ctx) const {
    auto m = to(add(p.modulation, mod), mod.dt).chunk(6, 1);
    auto bc = [&](const Tensor& r) { return sp_bcast(r, zidx); };
    Tensor h = layer_norm32(x, 1e-6f);
    h = add(mul(h, bc(add(1, m[1]))), bc(m[0]));
    h = selfAttn(p.sa, h);
    h = mul(h, bc(m[2]));
    x = add(x, h);
    h = layer_norm32(x, 1e-6f, &p.n2w, &p.n2b);
    h = crossAttn(p.ca, h, ctx);
    x = add(x, h);
    h = layer_norm32(x, 1e-6f);
    h = add(mul(h, bc(add(1, m[4]))), bc(m[3]));
    h = p.mlp(h);
    h = mul(h, bc(m[5]));
    x = add(x, h);
    return x;
  }
  // x f32 feats [N,C]; t f32 [1]; cond f32 [1,Lc,1024]
  Tensor operator()(const Tensor& x, const Tensor& t, const Tensor& cond) const {
    Tensor h = concat.defined() ? cat({x, concat}, -1) : x;
    h = in_layer(h);
    h = to(h, dt);
    Tensor te = adaln(silu(temb(t)));
    te = to(te, dt);
    Tensor c = to(cond, dt);
    for (auto& b : blocks) h = block(b, h, te, c);
    h = to(h, tk::F32);
    h = layer_norm(h, 1e-5f);
    return out_layer(h);
  }
};

// slat * std + mean  /  (slat - mean) / std  with per-channel lists from pipeline.json (SparseTensor elemwise)
inline Tensor jsonRow(const Json& j) {
  std::vector<float> v; for (auto& e : j.arr) v.push_back((float)e.d());
  return from_cpu(tk::F32, {1, (int64_t)v.size()}, v.data());
}
inline Tensor sp_denorm(const Tensor& x, const Json& norm) {
  Tensor z = sp_zero_index(x.shape[0]);
  return add(mul(x, sp_bcast(jsonRow(norm["std"]), z)), sp_bcast(jsonRow(norm["mean"]), z));
}
inline Tensor sp_norm(const Tensor& x, const Json& norm) {
  Tensor z = sp_zero_index(x.shape[0]);
  Tensor sd = sp_bcast(jsonRow(norm["std"]), z), mn = sp_bcast(jsonRow(norm["mean"]), z);
  return div(sub(x, mn), sd);
}

}  // namespace t2
