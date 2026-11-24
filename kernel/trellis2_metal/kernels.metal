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

// =====================================================================================================================
// Mesh post-processing kernels of the port's to_glb (src/glb): nearest triangle on a linear BVH and wavefront TELEA.
// Contraction is switched off from here on: the closest-point region tests and the TELEA sums are written to match a
// CPU implementation operation for operation.
#pragma clang fp contract(off)

struct T2Node { packed_float3 lo0; int l; packed_float3 hi0; int r; packed_float3 lo1; float p0; packed_float3 hi1; float p1; };
struct T2Tri { packed_float3 a; int id; packed_float3 b; float p0; packed_float3 c; float p1; };
struct T2Hit { float d2; int leaf; int id; float3 uvw; };

static inline float t2_box_d2(float3 p, float3 lo, float3 hi) { float3 d = max(max(lo - p, p - hi), float3(0.0f)); return dot(d, d); }
static inline float t2_seg_t(float3 p, float3 a, float3 b) {
  float3 ab = b - a; float l2 = dot(ab, ab);
  return l2 > 0.0f ? clamp(dot(p - a, ab) / l2, 0.0f, 1.0f) : 0.0f;
}
static inline float3 t2_edge_bary(float3 p, float3 a, float3 b, float3 c) {
  float t0 = t2_seg_t(p, a, b), t1 = t2_seg_t(p, b, c), t2 = t2_seg_t(p, c, a);
  float3 q0 = a + (b - a) * t0, q1 = b + (c - b) * t1, q2 = c + (a - c) * t2;
  float e0 = distance_squared(p, q0), e1 = distance_squared(p, q1), e2 = distance_squared(p, q2);
  if (e0 <= e1 && e0 <= e2) return float3(1.0f - t0, t0, 0.0f);
  if (e1 <= e2) return float3(0.0f, 1.0f - t1, t1);
  return float3(t2, 0.0f, 1.0f - t2);
}
// Barycentric coordinates of the point of triangle (a, b, c) closest to p (Ericson, Real-Time Collision Detection
// 5.1.5). Triangles with sin^2 of the angle at a below 1e-7 are treated as their three edges.
static float3 t2_tri_bary(float3 p, float3 a, float3 b, float3 c) {
  float3 ab = b - a, ac = c - a, ap = p - a;
  float3 nrm = cross(ab, ac);
  if (dot(nrm, nrm) <= 1e-7f * dot(ab, ab) * dot(ac, ac)) return t2_edge_bary(p, a, b, c);
  float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return float3(1.0f, 0.0f, 0.0f);
  float3 bp = p - b; float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return float3(0.0f, 1.0f, 0.0f);
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { float v = d1 / max(d1 - d3, 1e-30f); return float3(1.0f - v, v, 0.0f); }
  float3 cp = p - c; float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return float3(0.0f, 0.0f, 1.0f);
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { float w = d2 / max(d2 - d6, 1e-30f); return float3(1.0f - w, 0.0f, w); }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) { float w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-30f); return float3(0.0f, 1.0f - w, w); }
  float s = va + vb + vc;
  if (s > 0.0f) { float inv = 1.0f / s; float v = vb * inv, w = vc * inv; return float3(1.0f - v - w, v, w); }
  return t2_edge_bary(p, a, b, c);
}
static inline void t2_tri_test(const device T2Tri* tris, int leaf, float3 p, thread T2Hit& h) {
  T2Tri t = tris[leaf];
  float3 a = t.a, b = t.b, c = t.c;
  float3 w = t2_tri_bary(p, a, b, c);
  float3 d = p - (a * w.x + b * w.y + c * w.z);
  float d2 = dot(d, d);
  if (d2 < h.d2 || (d2 == h.d2 && t.id < h.id)) { h.d2 = d2; h.leaf = leaf; h.id = t.id; h.uvw = w; }
}

// Nearest triangle to every point: distance, face index (-1 when none within the cap, lowest index on ties) and the
// barycentric coordinates of the closest point. nodes / tris come from the CPU builder (lbvh_build).
struct T2QueryP { int ntri; float cap2; int nq; int pad; };
kernel void t2_mesh_nearest(const device T2Node* nodes [[buffer(0)]], const device T2Tri* tris [[buffer(1)]],
                            const device packed_float3* q [[buffer(2)]], device float* dist [[buffer(3)]], device int* fid [[buffer(4)]],
                            device packed_float3* uvw [[buffer(5)]], constant T2QueryP& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(p.nq)) return;
  float3 pt = float3(q[gid]);
  T2Hit h; h.d2 = p.cap2; h.leaf = -1; h.id = 0x7fffffff; h.uvw = float3(0.0f);
  const int ntri = p.ntri;
  if (ntri == 1) t2_tri_test(tris, 0, pt, h);
  else if (ntri > 1) {
    int stack[64]; float stackd[64]; int sp = 0; int cur = 0;
    while (uint(cur) < uint(ntri - 1)) {
      T2Node nd = nodes[cur];
      float d0 = t2_box_d2(pt, nd.lo0, nd.hi0), d1 = t2_box_d2(pt, nd.lo1, nd.hi1);
      int c0 = nd.l, c1 = nd.r;
      if (d1 < d0) { float td = d0; d0 = d1; d1 = td; int tc = c0; c0 = c1; c1 = tc; }
      bool v0 = d0 <= h.d2;
      if (v0 && c0 < 0) { if (uint(~c0) < uint(ntri)) t2_tri_test(tris, ~c0, pt, h); v0 = false; }
      bool v1 = d1 <= h.d2;
      if (v1 && c1 < 0) { if (uint(~c1) < uint(ntri)) t2_tri_test(tris, ~c1, pt, h); v1 = false; }
      if (v0) {
        if (v1 && sp < 64) { stack[sp] = c1; stackd[sp] = d1; sp++; }
        cur = c0; continue;
      }
      if (v1) { cur = c1; continue; }
      bool found = false;
      while (sp > 0) { sp--; if (stackd[sp] <= h.d2) { cur = stack[sp]; found = true; break; } }
      if (!found) break;
    }
  }
  dist[gid] = h.leaf < 0 ? INFINITY : sqrt(h.d2); fid[gid] = h.leaf < 0 ? -1 : h.id; uvw[gid] = h.uvw;
}

// ---- wavefront TELEA inpaint ----------------------------------------------------------------------------------------
// lv: 4-connected distance level from the known region (0 known, -1 unreachable). One dispatch fills every pixel of
// level L from pixels of lower levels with Telea's weights (direction x distance x level) and gradient term, as
// cv2.inpaint(INPAINT_TELEA) does per pixel; the fill order by level replaces OpenCV's fast-marching order.
// img is planar uint8 [C, H, W], C <= 8.
struct T2TeleaP { int H; int W; int L; int C; int R; int n; int p0; int p1; };
static inline bool t2_kn(const device int* lv, int H, int W, int L, int r, int c) {
  if (r < 0 || c < 0 || r >= H || c >= W) return false;
  int v = lv[r * W + c]; return v >= 0 && v < L;
}
kernel void t2_telea_fill(const device int* lv [[buffer(0)]], device uchar* img [[buffer(1)]], const device int* list [[buffer(2)]],
                          constant T2TeleaP& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(p.n)) return;
  const int H = p.H, W = p.W, L = p.L, C = p.C, R = p.R; const uint P = uint(H) * uint(W);
  const int i = list[gid];
  if (uint(i) >= P || lv[i] != L) return;
  const int y = i / W, x = i % W;
  const float fL = float(L);
  bool xr = t2_kn(lv, H, W, L, y, x + 1), xl = t2_kn(lv, H, W, L, y, x - 1), yd = t2_kn(lv, H, W, L, y + 1, x), yu = t2_kn(lv, H, W, L, y - 1, x);
  float gx = xr ? (xl ? (float(lv[i + 1]) - float(lv[i - 1])) * 0.5f : float(lv[i + 1]) - fL) : (xl ? fL - float(lv[i - 1]) : 0.0f);
  float gy = yd ? (yu ? (float(lv[i + W]) - float(lv[i - W])) * 0.5f : float(lv[i + W]) - fL) : (yu ? fL - float(lv[i - W]) : 0.0f);
  float Ia[8] = {0, 0, 0, 0, 0, 0, 0, 0}, Jx[8] = {0, 0, 0, 0, 0, 0, 0, 0}, Jy[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float s = 1e-20f;
  for (int k = y - R; k <= y + R; k++)
    for (int l = x - R; l <= x + R; l++) {
      int dd = (k - y) * (k - y) + (l - x) * (l - x);
      if (dd > R * R || !t2_kn(lv, H, W, L, k, l)) continue;
      int j = k * W + l;
      float ry = float(y - k), rx = float(x - l), vl = rx * rx + ry * ry;
      float dir = rx * gx + ry * gy; if (fabs(dir) <= 0.01f) dir = 1e-6f;
      float w = fabs(dir / (vl * sqrt(vl) * (1.0f + fabs(float(lv[j]) - fL))));
      bool r1 = t2_kn(lv, H, W, L, k, l + 1), l1 = t2_kn(lv, H, W, L, k, l - 1), d1 = t2_kn(lv, H, W, L, k + 1, l), u1 = t2_kn(lv, H, W, L, k - 1, l);
      for (int c = 0; c < C; c++) {
        uint o = uint(c) * P + uint(j);
        float v = float(img[o]);
        float ix = r1 ? (l1 ? (float(img[o + 1]) - float(img[o - 1])) * 2.0f : float(img[o + 1]) - v) : (l1 ? v - float(img[o - 1]) : 0.0f);
        float iy = d1 ? (u1 ? (float(img[o + uint(W)]) - float(img[o - uint(W)])) * 2.0f : float(img[o + uint(W)]) - v) : (u1 ? v - float(img[o - uint(W)]) : 0.0f);
        Ia[c] += w * v; Jx[c] -= w * ix * rx; Jy[c] -= w * iy * ry;
      }
      s += w;
    }
  for (int c = 0; c < C; c++) {
    float sat = Ia[c] / s + (Jx[c] + Jy[c]) / (sqrt(Jx[c] * Jx[c] + Jy[c] * Jy[c]) + 1e-20f);
    img[uint(c) * P + uint(i)] = uchar(clamp(rint(sat + 0.5f), 0.0f, 255.0f));
  }
}
