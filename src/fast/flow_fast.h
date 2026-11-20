// Fast build of the TRELLIS.2 flow transformers (dense sparse-structure DiT and sparse SLat DiTs).
//
// Same math as the parity build, different execution plan (rounding-level differences only):
//   * the residual stream stays in float32; every elementwise step between two matmuls is ONE fused own Metal kernel
//     (residual add + gate + bias + layer norm + modulation; bias + RMS norm + gamma + RoPE; bias + GELU) instead of
//     10-20 torch dispatches with f32 <-> bf16 round trips,
//   * linears are bias-less bf16 MPSNDArray matmuls (the bias is folded into the next fused kernel),
//   * attention is one un-chunked flash-style prefill dispatch that writes straight into the [N, H*D] layout the
//     output projection wants (no permute copies),
//   * cross-attention K/V of the image condition are computed once per sampler run,
//   * the unconditional branch (negative cond == zeros) uses the closed form of its cross attention: all keys are
//     identical, so the output is the constant row to_out(v_bias) -> a broadcast add, no q-linear / attention at all.
#pragma once
#include "../models/common.h"
#include "../models/sparse_flow.h"
#include "../models/sampler.h"

namespace t2 {
namespace fk {
inline const char* source() {
  return R"MSL(
inline float bf2f(ushort u) { return as_type<float>(((uint)u) << 16); }
inline ushort f2bf(float f) { uint u = as_type<uint>(f); u += 0x7fffu + ((u >> 16) & 1u); return (ushort)(u >> 16); }

// x [N,C] f32 residual stream (in place), o [N,C] bf16 branch output, vec = {ob, g, a, b} f32 [4,C], h [N,C] bf16.
//   flags&1: x += (o + ob) * g      (flags&4: o is absent / zero)
//   flags&2: h = layer_norm(x) * a + b
// one threadgroup (= one SIMD group of 32 threads) per row.
struct RLP { uint C; uint flags; float eps; uint pad; };
static inline float rb(float v) { return bf2f(f2bf(v)); }
kernel void f_resid_ln(device float* x [[buffer(0)]], device const ushort* o [[buffer(1)]], device const float* vec [[buffer(2)]],
                       device ushort* h [[buffer(3)]], constant RLP& p [[buffer(4)]], uint2 tp [[thread_position_in_grid]]) {
  const uint C = p.C, per = C / 32, t = tp.x;
  const ulong base = (ulong)tp.y * C;
  float s = 0;
  if (p.flags & 1u) {
    const bool noo = (p.flags & 4u) != 0;
    for (uint j = 0; j < per; j++) {
      uint c = t + 32 * j;
      float ov = noo ? 0.0f : bf2f(o[base + c]);
      float v = rb(x[base + c] + rb(rb(ov + vec[c]) * vec[C + c]));
      x[base + c] = v; s += v;
    }
  } else {
    for (uint j = 0; j < per; j++) s += x[base + t + 32 * j];
  }
  if (!(p.flags & 2u)) return;
  const float mean = simd_sum(s) / (float)C;
  float ss = 0;
  for (uint j = 0; j < per; j++) { float d = x[base + t + 32 * j] - mean; ss += d * d; }
  const float rstd = 1.0f / sqrt(simd_sum(ss) / (float)C + p.eps);
  for (uint j = 0; j < per; j++) {
    uint c = t + 32 * j;
    float nv = (x[base + c] - mean) * rstd;
    h[base + c] = (p.flags & 8u) ? f2bf(nv * vec[2 * C + c] + vec[3 * C + c]) : f2bf(rb(rb(nv) * vec[2 * C + c]) + vec[3 * C + c]);
  }
}

// src [N, rs] bf16 -> dst [H, N, 128] bf16 for the slice starting at column colOff:
//   v = src + bias; flags&1: v = v / max(|v|, 1e-12) * gamma * sqrt(D); flags&2: complex rotate by ph [N, 64, 2].
// one threadgroup per (head, row); thread t owns elements 4t..4t+3.
struct QP { uint N; uint H; uint rs; uint colOff; uint flags; float scale; uint pad0; uint pad1; };
kernel void f_qprep(device const ushort* src [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* gamma [[buffer(2)]],
                    device const float* ph [[buffer(3)]], device ushort* dst [[buffer(4)]], constant QP& p [[buffer(5)]],
                    uint3 tp [[thread_position_in_grid]]) {
  const uint t = tp.x, hd = tp.y, n = tp.z, d0 = hd * 128 + t * 4;
  const ulong si = (ulong)n * p.rs + p.colOff + d0;
  float4 v = float4(bf2f(src[si]), bf2f(src[si + 1]), bf2f(src[si + 2]), bf2f(src[si + 3]));
  v += float4(bias[p.colOff + d0], bias[p.colOff + d0 + 1], bias[p.colOff + d0 + 2], bias[p.colOff + d0 + 3]);
  v = float4(rb(v.x), rb(v.y), rb(v.z), rb(v.w));
  if (p.flags & 1u) {
    float nrm = max(sqrt(simd_sum(dot(v, v))), 1e-12f);
    v = v / nrm * float4(gamma[d0], gamma[d0 + 1], gamma[d0 + 2], gamma[d0 + 3]) * p.scale;
  }
  if (p.flags & 2u) {
    const ulong pi = ((ulong)n * 64 + t * 2) * 2;
    float4 r = float4(ph[pi], ph[pi + 1], ph[pi + 2], ph[pi + 3]);
    v = float4(v.x * r.x - v.y * r.y, v.x * r.y + v.y * r.x, v.z * r.z - v.w * r.w, v.z * r.w + v.w * r.z);
  }
  const ulong di = ((ulong)hd * p.N + n) * 128 + t * 4;
  dst[di] = f2bf(v.x); dst[di + 1] = f2bf(v.y); dst[di + 2] = f2bf(v.z); dst[di + 3] = f2bf(v.w);
}

// x [N,C] bf16 in place: gelu_tanh(x + bias)
kernel void f_bias_gelu(device ushort* x [[buffer(0)]], device const float* bias [[buffer(1)]], constant uint& C [[buffer(2)]],
                        uint2 g [[thread_position_in_grid]]) {
  const ulong i = (ulong)g.y * C + g.x;
  float v = rb(bf2f(x[i]) + bias[g.x]);
  float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  x[i] = f2bf(0.5f * v * (1.0f + precise::tanh(u)));
}
)MSL";
}
inline float bfRound(float v) {
  uint32_t u; memcpy(&u, &v, 4); u += 0x7fffu + ((u >> 16) & 1u); u &= 0xffff0000u; memcpy(&v, &u, 4); return v;
}
inline void registerKernels() {
  static bool done = false;
  if (!done) { xk::extraSource() += source(); done = true; }
}
enum { RESID = 1, LN = 2, NOO = 4, AFFINE = 8, QNORM = 1, QROPE = 2 };

inline void residLn(const Tensor& x, const Tensor* o, const Tensor& vec, int64_t slot, const Tensor& h, uint32_t flags, float eps, int64_t N, int64_t C) {
  auto p = xk::pso("f_resid_ln"); auto e = tk::enc();
  [e setComputePipelineState:p];
  tk::setBuf(e, x.buf(), 0); tk::setBuf(e, o ? o->buf() : h.buf(), 1);
  tk::Buf vb = vec.buf(); vb.off += (uint64_t)slot * 4 * C * 4;
  tk::setBuf(e, vb, 2); tk::setBuf(e, h.buf(), 3);
  struct { uint32_t C, flags; float eps; uint32_t pad; } rp{(uint32_t)C, flags, eps, 0};
  tk::setVal(e, rp, 4);
  [e dispatchThreads:MTLSizeMake(32, (NSUInteger)N, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  tk::ctx().ndispatch++; pace();
}
inline void qprep(const Tensor& src, int64_t rs, int64_t colOff, const Tensor& bias, const Tensor* gamma, const Tensor* ph, const Tensor& dst,
                  uint32_t flags, int64_t N, int64_t H) {
  auto p = xk::pso("f_qprep"); auto e = tk::enc();
  [e setComputePipelineState:p];
  tk::setBuf(e, src.buf(), 0); tk::setBuf(e, bias.buf(), 1); tk::setBuf(e, gamma ? gamma->buf() : bias.buf(), 2);
  tk::setBuf(e, ph ? ph->buf() : bias.buf(), 3); tk::setBuf(e, dst.buf(), 4);
  struct { uint32_t N, H, rs, colOff, flags; float scale; uint32_t p0, p1; } qp{(uint32_t)N, (uint32_t)H, (uint32_t)rs, (uint32_t)colOff, flags,
                                                                              (float)std::sqrt(128.0), 0, 0};
  tk::setVal(e, qp, 5);
  [e dispatchThreads:MTLSizeMake(32, (NSUInteger)H, (NSUInteger)N) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  tk::ctx().ndispatch++; pace();
}
inline void biasGelu(const Tensor& x, const Tensor& bias, int64_t N, int64_t C) {
  auto p = xk::pso("f_bias_gelu"); auto e = tk::enc();
  [e setComputePipelineState:p];
  tk::setBuf(e, x.buf(), 0); tk::setBuf(e, bias.buf(), 1);
  uint32_t c = (uint32_t)C; tk::setVal(e, c, 2);
  xk::dispatch2D(e, p, C, N);
  tk::ctx().ndispatch++; pace();
}
// Own GEMM on MetalPerformancePrimitives matmul2d (Metal 4, macOS 26): bit-identical to the MPSNDArray matmul, 3-5% faster
// with 64x128 output tiles on 4 simdgroups, and the mlp bias + GELU runs on the result while it is still in registers.
// T2_GEMM=mps (or an OS/GPU without Metal 4) falls back to MPSNDArray + f_bias_gelu.
inline const char* mppSource() {
  return R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct MMP { int M, N, K, pad; };
// X [M,K] bf16, W [N,K] bf16 -> Y [M,N] bf16. GELU: Y = gelu_tanh(bf16(X W^T) + bias)
template <int TM, int TN, int SG, bool GELU>
[[kernel, max_total_threads_per_threadgroup(SG * 32)]] void f_mm(
    device bfloat* X [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    const device float* bias [[buffer(3)]], constant MMP& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto A = tensor(X, dextents<int, 2>(p.K, p.M));
  auto B = tensor(W, dextents<int, 2>(p.K, p.N));
  auto C = tensor(Y, dextents<int, 2>(p.N, p.M));
  const int r0 = tg.y * TM, c0 = tg.x * TN;
  if constexpr (!GELU) {
    if (r0 + TM <= p.M && c0 + TN <= p.N) {
      auto mA = A.template slice<dynamic_extent, TM>(0, r0); auto mB = B.template slice<dynamic_extent, TN>(0, c0); auto mC = C.template slice<TN, TM>(c0, r0);
      op.run(mA, mB, mC);
    } else {
      auto mA = A.slice(0, r0); auto mB = B.slice(0, c0); auto mC = C.slice(c0, r0);
      op.run(mA, mB, mC);
    }
  } else {
    auto mA = A.slice(0, r0); auto mB = B.slice(0, c0); auto mC = C.slice(c0, r0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(mA, mB, cT);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        const int c = min(c0 + (int)idx[0], p.N - 1);
        float v = (float)(bfloat)((float)cT[i] + bias[c]);
        v = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        cT[i] = (bfloat)v;
      }
    }
    cT.store(mC);
  }
}
template [[host_name("f_mm")]] [[kernel]] decltype(f_mm<64, 128, 4, false>) f_mm<64, 128, 4, false>;
template [[host_name("f_mm_gelu")]] [[kernel]] decltype(f_mm<64, 128, 4, true>) f_mm<64, 128, 4, true>;
)MSL";
}
struct MppGemm { id<MTLComputePipelineState> mm = nil, gelu = nil; };
inline bool& gemmForceMps() { static bool v = false; return v; }  // A/B switch (apps/gemm_ab)
inline const MppGemm& mppGemm() {
  static MppGemm g; static bool done = false;
  if (done) return g;
  done = true;
  const char* ev = getenv("T2_GEMM");
  if (ev && !strcmp(ev, "mps")) return g;
  if (@available(macOS 26.0, *)) {
    NSError* err = nil; MTLCompileOptions* o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe; o.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> L = [tk::ctx().dev newLibraryWithSource:[NSString stringWithUTF8String:mppSource()] options:o error:&err];
    if (L) {
      id<MTLComputePipelineState> a = [tk::ctx().dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"f_mm"] error:&err];
      id<MTLComputePipelineState> b = [tk::ctx().dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"f_mm_gelu"] error:&err];
      if (a && b) { g.mm = a; g.gelu = b; }
    }
    if (!g.mm) fprintf(stderr, "fast: Metal 4 GEMM unavailable (%s), using MPSNDArray matmul\n", err ? err.localizedDescription.UTF8String : "?");
  }
  return g;
}
// out = x W^T (bias-less bf16); with geluBias: out = gelu_tanh(bf16(x W^T) + bias)
inline void matmul(const Tensor& x, const Tensor& W, const Tensor& out, int64_t rows, const Tensor* geluBias = nullptr) {
  const MppGemm& g = mppGemm();
  if (!g.mm || gemmForceMps()) {
    tk::linear_ndarray(x.buf(), W.buf(), nullptr, out.buf(), tk::BF16, (uint32_t)rows, (uint32_t)W.shape[1], (uint32_t)W.shape[0]);
    pace();
    if (geluBias) biasGelu(out, *geluBias, rows, W.shape[0]);
    return;
  }
  auto e = tk::enc();
  [e setComputePipelineState:geluBias ? g.gelu : g.mm];
  tk::setBuf(e, x.buf(), 0); tk::setBuf(e, W.buf(), 1); tk::setBuf(e, out.buf(), 2); tk::setBuf(e, geluBias ? geluBias->buf() : W.buf(), 3);
  struct { int32_t M, N, K, pad; } mp{(int32_t)rows, (int32_t)W.shape[0], (int32_t)W.shape[1], 0};
  tk::setVal(e, mp, 4);
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((W.shape[0] + 127) / 128), (NSUInteger)((rows + 63) / 64), 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  tk::ctx().ndispatch++; pace();
}
// q [H,qL,D], k/v [H,kL,D] contiguous bf16 -> out [qL, H*D] (torch's flash-style prefill kernel with custom output strides)
inline void attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, int64_t H, int64_t qL, int64_t kL, int64_t D) {
  const tk::SdpaRoute route = tk::sdpa_route(tk::BF16, H, H, qL, kL, D, D);
  if (route != tk::SdpaRoute::Prefill && route != tk::SdpaRoute::PrefillMpp) { fprintf(stderr, "fast attention: unsupported shape\n"); abort(); }
  const bool mpp = route == tk::SdpaRoute::PrefillMpp;
  tk::PrefillAttnParams p; memset(&p, 0, sizeof p);
  p.B = 1; p.H = (int32_t)H; p.D = (int32_t)D; p.qL = (int32_t)qL; p.kL = (int32_t)kL; p.gqa_factor = 1;
  p.scale = (float)(1.0 / std::sqrt((double)D)); p.softcapping = 1.0f;
  tk::sdpa_contiguous_strides(p.Q_strides, H, qL, D); tk::sdpa_contiguous_strides(p.K_strides, H, kL, D); tk::sdpa_contiguous_strides(p.V_strides, H, kL, D);
  p.O_strides[0] = qL * H * D; p.O_strides[1] = D; p.O_strides[2] = H * D;
  const tk::PrefillBlock blk = tk::sdpa_prefill_block(D, mpp);
  auto e = tk::enc();
  [e setComputePipelineState:tk::pso(tk::sdpa_prefill_name(tk::BF16, D, mpp, false, false, false))];
  tk::setBuf(e, q.buf(), 0); tk::setBuf(e, k.buf(), 1); tk::setBuf(e, v.buf(), 2); tk::setBuf(e, out.buf(), 3);
  tk::setVal(e, p, 4);
  const uint32_t zero = 0; tk::setVal(e, zero, 5); tk::setVal(e, zero, 6);
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((qL + blk.BQ - 1) / blk.BQ), (NSUInteger)H, 1)
      threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)blk.WM, (NSUInteger)blk.WN)];
  tk::ctx().ndispatch++; pace();
}
inline std::vector<float> toVecF32(const Tensor& t) {  // caller synced; bf16 or f32
  int64_t n = t.numel(); std::vector<float> v((size_t)n);
  Tensor c = contiguous_cpu(t);
  if (c.dt == tk::F32) memcpy(v.data(), c.raw(), n * 4);
  else if (c.dt == tk::BF16) { const uint16_t* s = c.ptr<uint16_t>(); for (int64_t i = 0; i < n; i++) { uint32_t u = (uint32_t)s[i] << 16; memcpy(&v[i], &u, 4); } }
  else { fprintf(stderr, "toVecF32: dtype\n"); abort(); }
  return v;
}
}  // namespace fk

struct CondCache {
  std::vector<Tensor> K, V;  // per block: bf16 [H, L, D]
  int64_t L = 0;
};

struct FastFlow {
  int res = 16, C_in = 8, C_out = 8, ch = 1536, heads = 12, nblocks = 30;
  bool dense = false;
  TimestepEmbedder temb; LinearP adaln, in_layer, out_layer;
  struct FB {
    BlockP p;                                     // bf16 weights (+ parity-layout biases, used for the cond K/V cache)
    Tensor qkvB, qB, mlp0B;                       // f32 device biases
    std::vector<float> mod, saOutB, caOutB, mlp2B, n2w, n2b, uncond;
  };
  std::vector<FB> blocks;
  // per-coords state
  Tensor phases, zidx, concat; int64_t N = 0;
  Tensor h, qkv, Q, K, V, O, o, mlp;              // work buffers
  mutable CondCache cc;

  void load(const std::string& json, const std::string& st) {
    Json cfg = loadJson(json); const Json& a = cfg["args"];
    res = (int)a["resolution"].i(); C_in = (int)a["in_channels"].i(); C_out = (int)a["out_channels"].i();
    ch = (int)a["model_channels"].i(); heads = (int)a["num_heads"].i(); nblocks = (int)a["num_blocks"].i();
    if (ch / heads != 128 || ch % 32) { fprintf(stderr, "FastFlow: unsupported head dim\n"); abort(); }
    STFile f(st);
    temb.load(f, "t_embedder"); adaln.load(f, "adaLN_modulation.1", tk::F32);
    in_layer.load(f, "input_layer", tk::F32); out_layer.load(f, "out_layer", tk::F32);
    blocks.resize(nblocks);
    for (int i = 0; i < nblocks; i++) {
      FB& b = blocks[i]; std::string pre = "blocks." + std::to_string(i);
      b.p.load(f, pre, tk::BF16); b.p.sa.heads = b.p.ca.heads = heads;
      b.qkvB = to(f.get(pre + ".self_attn.to_qkv.bias", tk::BF16), tk::F32); b.qB = to(f.get(pre + ".cross_attn.to_q.bias", tk::BF16), tk::F32);
      b.mlp0B = to(f.get(pre + ".mlp.mlp.0.bias", tk::BF16), tk::F32);
    }
    tk::sync();
    parallel_for(nblocks, 1, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; i++) {
        FB& b = blocks[i];
        b.mod = fk::toVecF32(b.p.modulation); b.saOutB = fk::toVecF32(b.p.sa.out.b); b.caOutB = fk::toVecF32(b.p.ca.out.b);
        b.mlp2B = fk::toVecF32(b.p.mlp.l2.b); b.n2w = fk::toVecF32(b.p.n2w); b.n2b = fk::toVecF32(b.p.n2b);
        // unconditional cross attention: every key/value row equals the to_kv bias -> output = to_out(v_bias)
        std::vector<float> kvb = fk::toVecF32(b.p.ca.kv.b);
        const uint16_t* W = b.p.ca.out.w.ptr<uint16_t>();
        b.uncond.resize(ch);
        for (int c = 0; c < ch; c++) {
          double s = 0;
          for (int j = 0; j < ch; j++) { uint32_t u = (uint32_t)W[(size_t)c * ch + j] << 16; float w; memcpy(&w, &u, 4); s += (double)w * kvb[ch + j]; }
          b.uncond[c] = (float)s + b.caOutB[c];
        }
      }
    });
    if (dense) { phases = dense_rope_phases(res, ch / heads); N = (int64_t)res * res * res; allocWork(); }
  }
  void allocWork() {
    int64_t C = ch, H = heads, D = C / H;
    h = Tensor(tk::BF16, {N, C}); qkv = Tensor(tk::BF16, {N, 3 * C}); O = Tensor(tk::BF16, {N, C}); o = Tensor(tk::BF16, {N, C});
    Q = Tensor(tk::BF16, {H, N, D}); K = Tensor(tk::BF16, {H, N, D}); V = Tensor(tk::BF16, {H, N, D});
    mlp = Tensor(tk::BF16, {N, blocks[0].p.mlp.l0.w.shape[0]});
  }
  // sparse: coords I32 [N,4]
  void setCoords(const Tensor& coords, const Tensor& concat_cond = Tensor()) {
    N = coords.shape[0]; concat = concat_cond;
    int hd = ch / heads, fd = hd / 2 / 3;
    std::vector<float> fr(fd);
    for (int i = 0; i < fd; i++) { float e = (float)i / (float)fd; fr[i] = 1.0f / std::pow(10000.0f, e); }
    Tensor freqs = from_cpu(tk::F32, {(int64_t)fd}, fr.data());
    Tensor idx = contiguous(coords.slice(1, 1, 4)).view({N * 3, 1});
    Tensor ph = mul(idx, freqs);
    ph = bin("polar", full(tk::F32, ph.shape, tk::Scal(1.0)), ph).view({N, (int64_t)3 * fd});
    int64_t padn = hd / 2 - 3 * fd;
    if (padn > 0) ph = cat({ph, bin("polar", full(tk::F32, {N, padn}, tk::Scal(1.0)), full(tk::F32, {N, padn}, tk::Scal(0.0)))}, -1);
    phases = contiguous(ph);
    zidx = sp_zero_index(N);
    allocWork();
  }
  void releaseCoords() {
    phases = zidx = concat = Tensor(); h = qkv = Q = K = V = O = o = mlp = Tensor(); N = 0; cc = CondCache();
  }
  // cross-attention keys / values of the image condition, once per sampler run. cond f32 [1,L,1024]
  void setCond(const Tensor& cond) const {
    cc = CondCache(); cc.L = cond.shape[1];
    Tensor c = to(cond, tk::BF16); int64_t H = heads;
    for (auto& b : blocks) {
      Tensor kv = b.p.ca.kv(c).view({1, cc.L, 2, H, -1});
      Tensor k = rms_norm(kv.select(2, 0), b.p.ca.kg);
      cc.K.push_back(contiguous(k.permute({0, 2, 1, 3}))); cc.V.push_back(contiguous(kv.select(2, 1).permute({0, 2, 1, 3})));
    }
  }

  // x f32 [N,C] (updated in place); returns the final residual stream. useCond=false -> zero condition.
  void core(const Tensor& x, const Tensor& t, bool useCond) const {
    const int64_t C = ch, H = heads, D = C / H, nb = nblocks;
    Tensor te = adaln(silu(temb(t)));
    tk::sync();
    const float* tp = te.ptr<float>();
    Tensor vec(tk::F32, {(1 + 3 * nb) * 4, C});
    float* vp = vec.ptr<float>();
    auto slot = [&](int64_t s, int k) { return vp + (s * 4 + k) * C; };
    auto modv = [&](int64_t blk, int k, int c) { return fk::bfRound(blocks[blk].mod[(size_t)k * C + c] + tp[(size_t)k * C + c]); };
    for (int c = 0; c < C; c++) { slot(0, 0)[c] = 0; slot(0, 1)[c] = 0; slot(0, 2)[c] = fk::bfRound(1.0f + modv(0, 1, c)); slot(0, 3)[c] = modv(0, 0, c); }
    for (int64_t i = 0; i < nb; i++) {
      const FB& b = blocks[i];
      float *s1 = slot(1 + 3 * i, 0), *s2 = slot(2 + 3 * i, 0), *s3 = slot(3 + 3 * i, 0);
      for (int c = 0; c < C; c++) {
        s1[c] = b.saOutB[c]; s1[C + c] = modv(i, 2, c); s1[2 * C + c] = b.n2w[c]; s1[3 * C + c] = b.n2b[c];
        s2[c] = useCond ? b.caOutB[c] : b.uncond[c]; s2[C + c] = 1.0f; s2[2 * C + c] = fk::bfRound(1.0f + modv(i, 4, c)); s2[3 * C + c] = modv(i, 3, c);
        s3[c] = b.mlp2B[c]; s3[C + c] = modv(i, 5, c);
        s3[2 * C + c] = i + 1 < nb ? fk::bfRound(1.0f + modv(i + 1, 1, c)) : 1.0f; s3[3 * C + c] = i + 1 < nb ? modv(i + 1, 0, c) : 0.0f;
      }
    }
    using namespace fk;
    residLn(x, nullptr, vec, 0, h, LN, 1e-6f, N, C);
    for (int64_t i = 0; i < nb; i++) {
      const FB& b = blocks[i];
      matmul(h, b.p.sa.qkv.w, qkv, N);
      qprep(qkv, 3 * C, 0, b.qkvB, &b.p.sa.qg, &phases, Q, QNORM | QROPE, N, H);
      qprep(qkv, 3 * C, C, b.qkvB, &b.p.sa.kg, &phases, K, QNORM | QROPE, N, H);
      qprep(qkv, 3 * C, 2 * C, b.qkvB, nullptr, nullptr, V, 0, N, H);
      attention(Q, K, V, O, H, N, N, D);
      matmul(O, b.p.sa.out.w, o, N);
      residLn(x, &o, vec, 1 + 3 * i, h, RESID | LN | AFFINE, 1e-6f, N, C);
      if (useCond) {
        matmul(h, b.p.ca.q.w, o, N);
        qprep(o, C, 0, b.qB, &b.p.ca.qg, nullptr, Q, QNORM, N, H);
        attention(Q, cc.K[i], cc.V[i], O, H, N, cc.L, D);
        matmul(O, b.p.ca.out.w, o, N);
        residLn(x, &o, vec, 2 + 3 * i, h, RESID | LN, 1e-6f, N, C);
      } else {
        residLn(x, nullptr, vec, 2 + 3 * i, h, RESID | LN | NOO, 1e-6f, N, C);
      }
      matmul(h, b.p.mlp.l0.w, mlp, N, &b.mlp0B);
      matmul(mlp, b.p.mlp.l2.w, o, N);
      residLn(x, &o, vec, 3 + 3 * i, h, RESID | (i + 1 < nb ? LN : 0), 1e-6f, N, C);
    }
    tk::flush();  // vec must stay alive until the encoded work is committed (the pool may hand its buffer out again afterwards)
    tk::sync();
  }

  // dense: x f32 [1,C,R,R,R]; sparse: x f32 feats [N,C]. t f32 [1].
  Tensor forward(const Tensor& x, const Tensor& t, bool useCond) const {
    if (dense) {
      Tensor hh = contiguous(x.view({1, (int64_t)C_in, -1}).permute({0, 2, 1}));
      hh = in_layer(hh);
      core(hh.view({N, (int64_t)ch}), t, useCond);
      hh = out_layer(layer_norm(hh, 1e-5f));
      Tensor r(tk::F32, {1, (int64_t)C_out, (int64_t)res, (int64_t)res, (int64_t)res});
      copy_(r.view({1, (int64_t)C_out, -1}), hh.permute({0, 2, 1}));
      return r;
    }
    Tensor hh = in_layer(concat.defined() ? cat({x, concat}, -1) : x);
    core(hh, t, useCond);
    return out_layer(layer_norm(hh, 1e-5f));
  }
};

struct FastDenseFlow : FastFlow { FastDenseFlow() { dense = true; } };
struct FastSparseFlow : FastFlow {};

inline bool allZero(const Tensor& t) {
  tk::sync(); Tensor c = contiguous_cpu(t); const uint8_t* p = c.raw(); int64_t n = c.nbytes();
  for (int64_t i = 0; i < n; i++) if (p[i]) return false;
  return true;
}
// model adapter for flowEulerSample: routes the negative (all-zero) condition to the closed-form branch
struct FastBound {
  const FastFlow& m; const Tensor& neg;
  Tensor operator()(const Tensor& x, const Tensor& t, const Tensor& c) const { return m.forward(x, t, c.st != neg.st); }
};
inline FastBound bindFast(const FastFlow& m, const Tensor& cond, const Tensor& neg) {
  if (!allZero(neg)) { fprintf(stderr, "fast build: negative condition must be zeros\n"); abort(); }
  m.setCond(cond);
  return FastBound{m, neg};
}

}  // namespace t2
