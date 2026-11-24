// Shared model building blocks for the parity build (torch-MPS kernel replay through t2 ops).
#pragma once
#include "../rt/ops.h"
#include "../rt/st.h"
#include "../rt/rng.h"
#include "consts.h"

namespace t2 {

struct LinearP {
  Tensor w, b;
  void load(const STFile& f, const std::string& p, DT dt) { w = f.get(p + ".weight", dt); if (f.has(p + ".bias")) b = f.get(p + ".bias", dt); }
  Tensor operator()(const Tensor& x) const { return linear(x, w, b.defined() ? &b : nullptr); }
};

// x.expand_as(ref) for keepdim reductions over the last dim
inline Tensor expand_last(const Tensor& x, int64_t n) { Tensor r = x; r.shape.back() = n; r.strides.back() = 0; return r; }
inline Tensor view_as_complex(const Tensor& x) {  // x F32 contiguous [..., 2]
  if (!x.contiguous() || x.shape.back() != 2) { fprintf(stderr, "view_as_complex: bad input\n"); abort(); }
  Tensor r = x; r.dt = tk::C64; r.shape.pop_back(); r.strides = Tensor::cstrides(r.shape); return r;
}
inline Tensor view_as_real(const Tensor& x_) {
  Tensor x = contiguous(x_);
  Tensor r = x; r.dt = tk::F32; r.shape.push_back(2); r.strides = Tensor::cstrides(r.shape); return r;
}

// LayerNorm32: float32 layer norm, cast back
inline Tensor layer_norm32(const Tensor& x, float eps, const Tensor* w = nullptr, const Tensor* b = nullptr) {
  return to(layer_norm(to(x, tk::F32), eps, w, b), x.dt);
}

// MultiHeadRMSNorm: (F.normalize(x.float(), dim=-1) * gamma * scale).to(x.dtype)
inline Tensor rms_norm(const Tensor& x, const Tensor& gamma) {
  Tensor xf = to(x, tk::F32);
  Tensor n = clamp_min(norm_l2(xf, {-1}, true), 1e-12);
  Tensor y = div(xf, expand_last(n, xf.shape.back()));
  y = mul(y, gamma);
  y = mul(y, std::sqrt((double)x.shape.back()));
  return to(y, x.dt);
}

// RotaryPositionEmbedder.apply_rotary_embedding; phases C64 [L, D/2] (already unsqueezed by caller as needed)
inline Tensor apply_rope(const Tensor& x, const Tensor& phases) {
  Tensor xf = contiguous(to(x, tk::F32));
  Shape s = xf.shape; s.back() /= 2; s.push_back(2);
  Tensor xc = view_as_complex(xf.view(s));
  Tensor rot = mul(xc, phases);
  Tensor rr = view_as_real(rot);
  Shape s2 = rot.shape; s2.back() *= 2;
  return to(rr.view(s2), x.dt);
}

struct FFN {
  LinearP l0, l2;
  void load(const STFile& f, const std::string& p, DT dt) { l0.load(f, p + ".mlp.0", dt); l2.load(f, p + ".mlp.2", dt); }
  Tensor operator()(const Tensor& x) const { return l2(gelu(l0(x), true)); }
};

// TimestepEmbedder (frequency_embedding_size 256)
struct TimestepEmbedder {
  LinearP l0, l2; Tensor freqs;
  void load(const STFile& f, const std::string& p) {
    l0.load(f, p + ".mlp.0", tk::F32); l2.load(f, p + ".mlp.2", tk::F32);
    // freqs = torch.exp(-np.log(10000) * torch.arange(0, 128, dtype=f32) / 128) on the CPU (table: consts.h)
    freqs = from_cpu(tk::F32, {1, 128}, kTembFreqBits);
  }
  Tensor operator()(const Tensor& t) const {  // t f32 [B]
    Tensor args = mul(t.unsqueeze(1), freqs);
    Tensor emb = cat({un("cos", args), un("sin", args)}, -1);
    return l2(silu(l0(emb)));
  }
};

// dense RoPE phases for a res^3 grid: C64 [res^3, head_dim/2], computed like the reference init (CPU)
inline Tensor dense_rope_phases(int res, int head_dim) {
  int fd = head_dim / 2 / 3, half = head_dim / 2; int64_t L = (int64_t)res * res * res;
  std::vector<float> fr(fd);
  for (int i = 0; i < fd; i++) { float e = (float)i / (float)fd; fr[i] = 1.0f / std::pow(10000.0f, e); }
  std::vector<float> ph((size_t)L * half * 2);
  for (int64_t l = 0; l < L; l++) {
    int c[3] = {(int)(l / (res * res)), (int)(l / res % res), (int)(l % res)};
    for (int j = 0; j < half; j++) {
      float re = 1.0f, im = 0.0f;
      if (j < 3 * fd) { float a = (float)c[j / fd] * fr[j % fd]; re = std::cos(a); im = std::sin(a); }
      ph[(l * half + j) * 2] = re; ph[(l * half + j) * 2 + 1] = im;
    }
  }
  Tensor r = from_cpu(tk::F32, {L, half, 2}, ph.data());
  return view_as_complex(r);
}

// ---- transformer pieces shared by the dense and sparse flow models ------------------------------------------------
struct AttnP {
  LinearP qkv, q, kv, out; Tensor qg, kg; int heads = 12;
  void loadSelf(const STFile& f, const std::string& p, DT dt) {
    qkv.load(f, p + ".to_qkv", dt); out.load(f, p + ".to_out", dt);
    qg = f.get(p + ".q_rms_norm.gamma", tk::F32); kg = f.get(p + ".k_rms_norm.gamma", tk::F32);
  }
  void loadCross(const STFile& f, const std::string& p, DT dt) {
    q.load(f, p + ".to_q", dt); kv.load(f, p + ".to_kv", dt); out.load(f, p + ".to_out", dt);
    qg = f.get(p + ".q_rms_norm.gamma", tk::F32); kg = f.get(p + ".k_rms_norm.gamma", tk::F32);
  }
};

struct BlockP {
  Tensor modulation, n2w, n2b; AttnP sa, ca; FFN mlp;
  void load(const STFile& f, const std::string& p, DT dt) {
    modulation = f.get(p + ".modulation", tk::F32);
    n2w = f.get(p + ".norm2.weight", tk::F32); n2b = f.get(p + ".norm2.bias", tk::F32);
    sa.loadSelf(f, p + ".self_attn", dt); ca.loadCross(f, p + ".cross_attn", dt); mlp.load(f, p + ".mlp", dt);
  }
};

}  // namespace t2
