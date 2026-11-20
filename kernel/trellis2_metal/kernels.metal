// Metal kernels of the TRELLIS.2 Metal port's fast flow-transformer path (https://github.com/suvadityamuk/trellis2-metal).
// bf16 tensors are read and written as raw ushort / bfloat; rounding to bf16 is round-to-nearest-even.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline float bf2f(ushort u) { return as_type<float>(((uint)u) << 16); }
inline ushort f2bf(float f) { uint u = as_type<uint>(f); u += 0x7fffu + ((u >> 16) & 1u); return (ushort)(u >> 16); }
static inline float rb(float v) { return bf2f(f2bf(v)); }
static inline float gelu_tanh(float v) { return 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))); }

// ---- GEMM on MetalPerformancePrimitives matmul2d: X [M,K] bf16, W [N,K] bf16 -> Y [M,N] bf16 (Y = X W^T).
// One threadgroup (4 simdgroups) per 64x128 output tile. GELU: Y = gelu_tanh(bf16(X W^T) + bias), bias f32 [N],
// applied on the cooperative destination tensor while the tile is still in registers.
struct MMP { int M, N, K, pad; };
template <int TM, int TN, int SG, bool GELU>
[[kernel, max_total_threads_per_threadgroup(SG * 32)]] void t2_mm(
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
        cT[i] = (bfloat)gelu_tanh((float)(bfloat)((float)cT[i] + bias[c]));
      }
    }
    cT.store(mC);
  }
}
template [[host_name("t2_gemm")]] [[kernel]] decltype(t2_mm<64, 128, 4, false>) t2_mm<64, 128, 4, false>;
template [[host_name("t2_gemm_bias_gelu")]] [[kernel]] decltype(t2_mm<64, 128, 4, true>) t2_mm<64, 128, 4, true>;

// ---- x [N,C] bf16 in place: gelu_tanh(x + bias), bias f32 [C]
kernel void t2_bias_gelu(device ushort* x [[buffer(0)]], device const float* bias [[buffer(1)]], constant uint& C [[buffer(2)]],
                         uint2 g [[thread_position_in_grid]]) {
  const ulong i = (ulong)g.y * C + g.x;
  x[i] = f2bf(gelu_tanh(rb(bf2f(x[i]) + bias[g.x])));
}

// ---- x [N,C] f32 residual stream (in place), o [N,C] bf16 branch output, vec = {ob, g, a, b} f32 [4,C], h [N,C] bf16.
//   flags&1: x += (o + ob) * g      (flags&4: o is absent / zero)
//   flags&2: h = layer_norm(x) * a + b   (flags&8: a single rounding of the affine instead of one per step)
// one threadgroup (= one SIMD group of 32 threads) per row; C must be a multiple of 32.
struct RLP { uint C; uint flags; float eps; uint pad; };
kernel void t2_resid_ln(device float* x [[buffer(0)]], device const ushort* o [[buffer(1)]], device const float* vec [[buffer(2)]],
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

// ---- src [N, rs] bf16 -> dst [H, N, 128] bf16 for the H*128 columns starting at colOff:
//   v = src + bias; flags&1: v = v / max(|v|, 1e-12) * gamma * sqrt(128) (per head); flags&2: rotate pairs by ph [N, 64, 2].
// one threadgroup per (head, row); thread t owns elements 4t..4t+3.
struct QP { uint N; uint H; uint rs; uint colOff; uint flags; float scale; uint pad0; uint pad1; };
kernel void t2_qprep(device const ushort* src [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* gamma [[buffer(2)]],
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
