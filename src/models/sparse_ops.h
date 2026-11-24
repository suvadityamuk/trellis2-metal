// Sparse building blocks of the SLat VAE decoders — parity build.
// Mirrors ref/TRELLIS.2/trellis2/modules/sparse/** + models/sc_vaes/sparse_unet_vae.py as executed by the MPS
// reference with tools/shims/flex_gemm (neighbor-map gather + chunked F.linear).
// Float arithmetic goes through the torch kernel replay (t2 ops); coordinate hashing, neighbor maps, subdivision
// bookkeeping and gathers are integer work / data movement (CPU after tk::sync(), or xk gather kernels).
#pragma once
#include "common.h"

namespace t2 {

// ---- spatial cache ------------------------------------------------------------------------------------------------
// The reference caches one neighbor map per scale in SparseTensor._spatial_cache. Here a level is identified by its
// coords (content compare), so a cache can also be shared between the shape and the texture decoder.
struct SpLevel {
  Tensor coords;  // I32 [N,4] (batch, x, y, z)
  Tensor nbr;     // I32 [N,27], built lazily
};
typedef std::shared_ptr<SpLevel> SpLevelP;

// flex_gemm shim build_neighbor_map: nbr[i, v] = row of the voxel at coords[i] + (v//9-1, v//3%3-1, v%3-1), or N.
// (The shim uses a stable argsort + searchsorted over the same keys; with unique coords the result is identical.)
inline Tensor build_neighbor_map(const Tensor& coords) {
  tk::sync();
  const int64_t n = coords.shape[0];
  Tensor nbr(tk::I32, {n, 27});
  if (n == 0) return nbr;
  const int32_t* c = coords.ptr<int32_t>();
  int32_t mx = c[1];
  for (int64_t i = 0; i < n; i++) for (int k = 1; k < 4; k++) mx = std::max(mx, c[i * 4 + k]);
  const int64_t S = (int64_t)mx + 3;
  std::vector<int64_t> key(n);
  parallel_for(n, 1 << 16, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++)
      key[i] = (((int64_t)c[i * 4] * S + c[i * 4 + 1] + 1) * S + c[i * 4 + 2] + 1) * S + c[i * 4 + 3] + 1;
  });
  int lg = 4; while (((int64_t)1 << lg) < 2 * n) lg++;
  const uint64_t cap = (uint64_t)1 << lg, mask = cap - 1; const int sh = 64 - lg;
  std::vector<int64_t> hk(cap, -1); std::vector<int32_t> hv(cap);
  for (int64_t i = 0; i < n; i++) {
    uint64_t h = ((uint64_t)key[i] * 0x9E3779B97F4A7C15ull) >> sh;
    while (hk[h] >= 0) h = (h + 1) & mask;
    hk[h] = key[i]; hv[h] = (int32_t)i;
  }
  int32_t* o = nbr.ptr<int32_t>();
  parallel_for(n, 2048, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      for (int v = 0; v < 27; v++) {
        if (v == 13) { o[i * 27 + v] = (int32_t)i; continue; }
        const int64_t q = key[i] + ((int64_t)(v / 9 - 1) * S + (v / 3 % 3 - 1)) * S + (v % 3 - 1);
        uint64_t h = ((uint64_t)q * 0x9E3779B97F4A7C15ull) >> sh;
        int32_t r = (int32_t)n;
        while (hk[h] >= 0) { if (hk[h] == q) { r = hv[h]; break; } h = (h + 1) & mask; }
        o[i * 27 + v] = r;
      }
    }
  });
  return nbr;
}

struct SparseCache {
  std::vector<SpLevelP> levels;
  // returns the level with identical coords (creating it if needed)
  SpLevelP level(const Tensor& coords) {
    tk::sync();
    for (auto& l : levels)
      if (l->coords.shape == coords.shape &&
          (l->coords.st == coords.st || !memcmp(l->coords.raw(), coords.raw(), coords.nbytes()))) return l;
    auto l = std::make_shared<SpLevel>(); l->coords = coords; levels.push_back(l); return l;
  }
  void clear() { levels.clear(); }
};
inline const Tensor& neighbor_map(const SpLevelP& l) {
  if (!l->nbr.defined()) l->nbr = build_neighbor_map(l->coords);
  return l->nbr;
}

struct SparseT {
  Tensor feats, coords;
};

// ---- SparseConv3d (flex_gemm shim sparse_submanifold_conv3d) ----------------------------------------------------------
// w = weight.reshape(Co, 27*Ci) (checkpoint layout is already (Co,Kd,Kh,Kw,Ci)); xp = cat([feats, zeros(1,Ci)]);
// per chunk of rows = max(1, 2^30 // (27*Ci*elsize)): col = xp[nbr[s:s+rows]].reshape(-1, 27*Ci); F.linear(col, w, bias);
// chunks are torch.cat'ed. The gather is data movement (xk kernel; missing neighbor -> zero row).
inline Tensor sparse_conv3d(const Tensor& feats_, const Tensor& nbr, const Tensor& w, const Tensor* bias) {
  Tensor feats = contiguous(feats_);
  const int64_t N = feats.shape[0], Ci = feats.shape[1], Co = w.shape[0];
  const uint32_t es = feats.esize();
  if (w.shape[1] != 27 * Ci || nbr.shape[0] != N || (es != 2 && es != 4)) { fprintf(stderr, "sparse_conv3d: bad operands\n"); abort(); }
  const int64_t rows = std::max<int64_t>(1, ((int64_t)1 << 30) / (27 * Ci * es));
  auto chunk = [&](int64_t s, int64_t e) {
    Tensor col(feats.dt, {e - s, 27 * Ci});
    auto p = xk::pso(es == 2 ? "gather27_u16" : "gather27_u32");
    auto en = tk::enc();
    [en setComputePipelineState:p];
    tk::setBuf(en, feats.buf(), 0); tk::setBuf(en, nbr.buf(), 1); tk::setBuf(en, col.buf(), 2);
    struct { uint32_t C, N, rowOff, pad; } gp{(uint32_t)Ci, (uint32_t)N, (uint32_t)s, 0};
    tk::setVal(en, gp, 3);
    xk::dispatch2D(en, p, (uint64_t)(27 * Ci), (uint64_t)(e - s));
    tk::ctx().ndispatch++;
    return linear(col, w, bias);
  };
  if (N <= rows) return chunk(0, N);
  Tensor out(feats.dt, {N, Co});
  for (int64_t s = 0; s < N; s += rows) @autoreleasepool {
    const int64_t e = std::min(N, s + rows);
    Tensor o = chunk(s, e);
    copy_(out.slice(0, s, e), o);
    tk::flush();  // keep the 1 GB im2col buffers cycling through the pool instead of piling up unexecuted
  }
  return out;
}

struct SparseConvP {
  Tensor w, b;  // w [Co, 27*Ci]
  void load(const STFile& f, const std::string& p, DT dt) {
    Tensor W = f.get(p + ".weight", dt);
    if (W.ndim() != 5 || W.shape[1] != 3 || W.shape[2] != 3 || W.shape[3] != 3) { fprintf(stderr, "SparseConvP: %s is not (Co,3,3,3,Ci)\n", p.c_str()); abort(); }
    w = W.view({W.shape[0], 27 * W.shape[4]});
    b = f.get(p + ".bias", dt);
  }
  Tensor operator()(const Tensor& feats, const Tensor& nbr) const { return sparse_conv3d(feats, nbr, w, &b); }
};

// ---- subdivision (SparseChannel2Spatial / SparseUpsample without cache, i.e. the pred_subdiv / guide_subs path) ----------
// sub = subdivision.feats > 0 [N, 8]; subidx = sub.nonzero()[:, -1]; new_coords = repeat_interleave(coords * 2, N_leaf)
// + (subidx // 2**i % 2) on axis i; idx = repeat_interleave(arange(N), N_leaf). Row order = row-major over (n, subidx).
struct Subdiv {
  Tensor coords;  // I32 [M,4]
  Tensor idx;     // I32 [M]  parent row            (SparseUpsample: feats[idx])
  Tensor gidx;    // I32 [M]  idx * 8 + subidx      (SparseChannel2Spatial: feats.reshape(N*8,-1)[gidx])
};
inline Subdiv subdivide(const Tensor& coords, const Tensor& sub_) {
  tk::sync();
  Tensor sub = contiguous_cpu(sub_);
  const int64_t N = coords.shape[0];
  if (sub.shape[0] != N || sub.shape[1] != 8) { fprintf(stderr, "subdivide: bad subdivision tensor\n"); abort(); }
  std::vector<uint8_t> m(N); std::vector<int64_t> off(N + 1);
  parallel_for(N, 1 << 14, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      uint8_t bits = 0;
      for (int k = 0; k < 8; k++) {
        bool on;
        if (sub.dt == tk::F16) on = (float)sub.ptr<__fp16>()[i * 8 + k] > 0.0f;
        else if (sub.dt == tk::F32) on = sub.ptr<float>()[i * 8 + k] > 0.0f;
        else if (sub.dt == tk::BOOL || sub.dt == tk::U8) on = sub.ptr<uint8_t>()[i * 8 + k] != 0;
        else { fprintf(stderr, "subdivide: unsupported dtype\n"); abort(); }
        bits |= (uint8_t)on << k;
      }
      m[i] = bits;
    }
  });
  off[0] = 0;
  for (int64_t i = 0; i < N; i++) off[i + 1] = off[i] + __builtin_popcount(m[i]);
  const int64_t M = off[N];
  Subdiv r; r.coords = Tensor(tk::I32, {M, 4}); r.idx = Tensor(tk::I32, {M}); r.gidx = Tensor(tk::I32, {M});
  const int32_t* c = coords.ptr<int32_t>();
  int32_t* nc = r.coords.ptr<int32_t>(); int32_t* pi = r.idx.ptr<int32_t>(); int32_t* gi = r.gidx.ptr<int32_t>();
  parallel_for(N, 1 << 14, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      int64_t o = off[i];
      for (int k = 0; k < 8; k++) {
        if (!((m[i] >> k) & 1)) continue;
        nc[o * 4] = c[i * 4];
        nc[o * 4 + 1] = c[i * 4 + 1] * 2 + (k & 1);
        nc[o * 4 + 2] = c[i * 4 + 2] * 2 + ((k >> 1) & 1);
        nc[o * 4 + 3] = c[i * 4 + 3] * 2 + ((k >> 2) & 1);
        pi[o] = (int32_t)i; gi[o] = (int32_t)(i * 8 + k);
        o++;
      }
    }
  });
  return r;
}
// SparseChannel2Spatial feats: x.feats.reshape(N*8, -1)[idx*8 + subidx]
inline Tensor channel2spatial(const Tensor& feats, const Subdiv& sd) {
  Tensor f = contiguous(feats);
  return gather_rows(f.view({f.shape[0] * 8, -1}), sd.gidx);
}
// SparseUpsample feats: x.feats[idx]
inline Tensor upsample_nearest(const Tensor& feats, const Subdiv& sd) { return gather_rows(contiguous(feats), sd.idx); }

// x.repeat_interleave(r, dim=1) for [N,C]: x.unsqueeze(2).expand(N,C,r).clone().view(N, C*r)
inline Tensor repeat_interleave_dim1(const Tensor& x, int64_t r) {
  if (r == 1) return x;
  Tensor e = contiguous(x).unsqueeze(2); e.shape[2] = r; e.strides[2] = 0;
  Tensor o(x.dt, {x.shape[0], x.shape[1], r});
  copy_(o, e);
  return o.view({x.shape[0], x.shape[1] * r});
}

// ---- blocks ----------------------------------------------------------------------------------------------------------
// SparseConvNeXtBlock3d: h = conv(x); h = LayerNorm32(h) (affine, eps 1e-6); h = mlp(h); return h + x
struct SparseConvNeXtBlock {
  SparseConvP conv; Tensor nw, nb; LinearP l0, l2;
  void load(const STFile& f, const std::string& p, DT dt) {
    conv.load(f, p + ".conv", dt);
    nw = f.get(p + ".norm.weight", tk::F32); nb = f.get(p + ".norm.bias", tk::F32);
    l0.load(f, p + ".mlp.0", dt); l2.load(f, p + ".mlp.2", dt);
  }
  Tensor operator()(const Tensor& x, const Tensor& nbr) const {
    Tensor h = conv(x, nbr);
    h = layer_norm32(h, 1e-6f, &nw, &nb);
    h = l2(silu(l0(h)));
    return add(h, x);
  }
};

// SparseResBlockC2S3d (channels -> out_channels, x2 upsampling by channel-to-spatial).
struct SparseResBlockC2S {
  SparseConvP conv1, conv2; Tensor n1w, n1b; LinearP to_subdiv; bool pred_subdiv = true;
  int64_t channels = 0, out_channels = 0;
  void load(const STFile& f, const std::string& p, DT dt, bool pred) {
    pred_subdiv = pred;
    n1w = f.get(p + ".norm1.weight", tk::F32); n1b = f.get(p + ".norm1.bias", tk::F32);
    conv1.load(f, p + ".conv1", dt); conv2.load(f, p + ".conv2", dt);
    if (pred) to_subdiv.load(f, p + ".to_subdiv", dt);
    channels = conv1.w.shape[1] / 27; out_channels = conv2.w.shape[0];
  }
  // x at level `lv`; subdiv = guide (pred_subdiv false) or receives the predicted logits (pred_subdiv true).
  // Returns the upsampled feats; `fine` receives the level of the output.
  Tensor operator()(const Tensor& x, const SpLevelP& lv, SparseCache& cache, Tensor& subdiv, SpLevelP& fine) const {
    if (pred_subdiv) subdiv = to_subdiv(x);
    Tensor h = layer_norm32(x, 1e-6f, &n1w, &n1b);
    h = silu(h);
    h = conv1(h, neighbor_map(lv));
    Subdiv sd = subdivide(lv->coords, subdiv);          // subdiv.feats > 0
    h = channel2spatial(h, sd);
    Tensor xs = channel2spatial(x, sd);
    fine = cache.level(sd.coords);
    h = layer_norm32(h, 1e-6f);
    h = silu(h);
    h = conv2(h, neighbor_map(fine));
    return add(h, repeat_interleave_dim1(xs, out_channels / (channels / 8)));
  }
};

// ---- SparseUnetVaeDecoder ----------------------------------------------------------------------------------------------
struct SparseUnetVaeDecoder {
  int out_channels = 0, latent_channels = 32;
  std::vector<int> model_channels, num_blocks;
  bool pred_subdiv = true; DT dt = tk::F16;
  LinearP from_latent, output_layer;
  std::vector<std::vector<SparseConvNeXtBlock>> blocks;
  std::vector<SparseResBlockC2S> ups;
  // optional debug tap: called with "blocks.<i>.<j>" and the block output feats
  std::function<void(const std::string&, const Tensor&)> tap;

  void load(const std::string& json, const std::string& st, int default_out_channels) {
    Json cfg = loadJson(json); const Json& a = cfg["args"];
    out_channels = a.has("out_channels") ? (int)a["out_channels"].i() : default_out_channels;
    latent_channels = (int)a["latent_channels"].i();
    for (auto& v : a["model_channels"].arr) model_channels.push_back((int)v.i());
    for (auto& v : a["num_blocks"].arr) num_blocks.push_back((int)v.i());
    pred_subdiv = a.has("pred_subdiv") ? a["pred_subdiv"].b : true;
    dt = (a.has("use_fp16") && a["use_fp16"].b) ? tk::F16 : tk::F32;
    for (auto& v : a["block_type"].arr) if (v.str != "SparseConvNeXtBlock3d") { fprintf(stderr, "decoder: block type %s not ported\n", v.str.c_str()); abort(); }
    for (auto& v : a["up_block_type"].arr) if (v.str != "SparseResBlockC2S3d") { fprintf(stderr, "decoder: up block type %s not ported\n", v.str.c_str()); abort(); }
    STFile f(st);
    from_latent.load(f, "from_latent", tk::F32); output_layer.load(f, "output_layer", tk::F32);
    const int L = (int)num_blocks.size();
    blocks.resize(L); ups.resize(L - 1);
    for (int i = 0; i < L; i++) {
      blocks[i].resize(num_blocks[i]);
      for (int j = 0; j < num_blocks[i]; j++) blocks[i][j].load(f, "blocks." + std::to_string(i) + "." + std::to_string(j), dt);
      if (i < L - 1) ups[i].load(f, "blocks." + std::to_string(i) + "." + std::to_string(num_blocks[i]), dt, pred_subdiv);
    }
  }

  struct Out {
    SparseT h;                  // output_layer result (f32)
    std::vector<SparseT> subs;  // predicted subdivision logits per level (pred_subdiv only)
  };
  // guide_subs: subdivision logits per level (required iff !pred_subdiv). upsample_times >= 0 mirrors
  // SparseUnetVaeDecoder.upsample(): stops when level == upsample_times and returns only h.coords.
  Out forward(const SparseT& x, const std::vector<SparseT>* guide_subs, SparseCache* cache_ = nullptr, int upsample_times = -1) const {
    SparseCache local; SparseCache& cache = cache_ ? *cache_ : local;
    if (pred_subdiv == (guide_subs != nullptr)) { fprintf(stderr, "decoder: guide_subs iff pred_subdiv == false\n"); abort(); }
    Out r;
    Tensor h = from_latent(x.feats);
    h = to(h, dt);
    SpLevelP lv = cache.level(x.coords);
    const int L = (int)blocks.size();
    for (int i = 0; i < L; i++) {
      if (i == upsample_times) { r.h.coords = lv->coords; return r; }
      for (int j = 0; j < (int)blocks[i].size(); j++) @autoreleasepool {
        h = blocks[i][j](h, neighbor_map(lv));
        if (tap) tap("blocks." + std::to_string(i) + "." + std::to_string(j), h);
      }
      if (i < L - 1) @autoreleasepool {
        Tensor sub; SpLevelP fine;
        if (!pred_subdiv) sub = (*guide_subs)[i].feats;
        h = ups[i](h, lv, cache, sub, fine);
        if (pred_subdiv) r.subs.push_back(SparseT{sub, lv->coords});
        lv = fine;
        if (tap) tap("blocks." + std::to_string(i) + "." + std::to_string(blocks[i].size()), h);
      }
    }
    h = to(h, x.feats.dt);
    h = layer_norm(h, 1e-5f);
    h = output_layer(h);
    r.h = SparseT{h, lv->coords};
    return r;
  }
  // SparseUnetVaeDecoder.upsample(x, upsample_times) -> coords [M,4]
  Tensor upsample(const SparseT& x, int upsample_times, SparseCache* cache = nullptr) const {
    return forward(x, nullptr, cache, upsample_times).h.coords;
  }
};

}  // namespace t2
