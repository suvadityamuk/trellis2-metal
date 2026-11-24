// Linear BVH over triangles (Karras 2012): 63-bit Morton codes of centroids, radix sort, binary radix tree, bottom-up
// boxes. Built with CPU threads directly in shared (unified-memory) buffers; queried on the GPU (kernels.h nearest()).
// Each internal node stores the boxes of both children, so a traversal step reads one node.
#pragma once
#include "kernels.h"

namespace glb {

struct BNode { float lo0[3]; int32_t l; float hi0[3]; int32_t r; float lo1[3]; float p0; float hi1[3]; float p1; };
struct BTri { float a[3]; int32_t id; float b[3]; float p0; float c[3]; float p1; };

struct V3 { float x, y, z; };
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float norm(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 ld(const float* p) { return {p[0], p[1], p[2]}; }

// CPU twin of the MSL tri_bary (reference for tests)
inline float seg_t(V3 p, V3 a, V3 b) { V3 ab = b - a; float l2 = dot(ab, ab); return l2 > 0 ? std::min(std::max(dot(p - a, ab) / l2, 0.f), 1.f) : 0.f; }
inline V3 tri_bary(V3 p, V3 a, V3 b, V3 c) {
  V3 ab = b - a, ac = c - a, ap = p - a;
  V3 nrm = cross(ab, ac);
  if (dot(nrm, nrm) <= 1e-7f * dot(ab, ab) * dot(ac, ac)) {  // (near-)zero-area triangle: closest of the three edges
    float t0 = seg_t(p, a, b), t1 = seg_t(p, b, c), t2 = seg_t(p, c, a);
    V3 q0 = a + (b - a) * t0, q1 = b + (c - b) * t1, q2 = c + (a - c) * t2;
    float e0 = dot(p - q0, p - q0), e1 = dot(p - q1, p - q1), e2 = dot(p - q2, p - q2);
    if (e0 <= e1 && e0 <= e2) return {1 - t0, t0, 0};
    if (e1 <= e2) return {0, 1 - t1, t1};
    return {t2, 0, 1 - t2};
  }
  float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return {1, 0, 0};
  V3 bp = p - b; float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return {0, 1, 0};
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { float v = d1 / std::max(d1 - d3, 1e-30f); return {1 - v, v, 0}; }
  V3 cp = p - c; float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return {0, 0, 1};
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { float w = d2 / std::max(d2 - d6, 1e-30f); return {1 - w, 0, w}; }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { float w = (d4 - d3) / std::max((d4 - d3) + (d5 - d6), 1e-30f); return {0, 1 - w, w}; }
  float s = va + vb + vc;
  if (s > 0) { float inv = 1.f / s; float v = vb * inv, w = vc * inv; return {1 - v - w, v, w}; }
  float t0 = seg_t(p, a, b), t1 = seg_t(p, b, c), t2 = seg_t(p, c, a);
  V3 q0 = a + (b - a) * t0, q1 = b + (c - b) * t1, q2 = c + (a - c) * t2;
  float e0 = dot(p - q0, p - q0), e1 = dot(p - q1, p - q1), e2 = dot(p - q2, p - q2);
  if (e0 <= e1 && e0 <= e2) return {1 - t0, t0, 0};
  if (e1 <= e2) return {0, 1 - t1, t1};
  return {t2, 0, 1 - t2};
}

inline uint64_t spread21(uint64_t x) {
  x &= 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full;
  x = (x | x << 4) & 0x10c30c30c30c30c3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}

struct LBVH {
  int64_t n = 0;
  Tensor nodes, tris;  // BNode[n-1], BTri[n] (Morton order)
  double t_build = 0;

  void build(const float* V, const int32_t* F, int64_t nf) {
    double t0 = now();
    tk::sync();
    n = nf;
    tris = gbuf<BTri>(n); nodes = gbuf<BNode>(std::max<int64_t>(n - 1, 1));
    if (n == 0) return;
    BTri* T = tris.ptr<BTri>(); BNode* N = nodes.ptr<BNode>();
    // centroid bounds
    const int C = 32; float clo[C][3], chi[C][3];
    for (int c = 0; c < C; c++) for (int k = 0; k < 3; k++) { clo[c][k] = INFINITY; chi[c][k] = -INFINITY; }
    par_chunks(n, C, [&](int c, int64_t a, int64_t b) {
      float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
      for (int64_t f = a; f < b; f++) for (int k = 0; k < 3; k++) {
        float x = (V[3 * F[3 * f] + k] + V[3 * F[3 * f + 1] + k] + V[3 * F[3 * f + 2] + k]) * (1.f / 3);
        lo[k] = std::min(lo[k], x); hi[k] = std::max(hi[k], x);
      }
      for (int k = 0; k < 3; k++) { clo[c][k] = lo[k]; chi[c][k] = hi[k]; }
    });
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int c = 0; c < C; c++) for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], clo[c][k]); hi[k] = std::max(hi[k], chi[c][k]); }
    double sc[3]; for (int k = 0; k < 3; k++) sc[k] = hi[k] > lo[k] ? 2097151.0 / ((double)hi[k] - lo[k]) : 0.0;
    Arr<uint64_t> code(n); Arr<uint32_t> idx(n);
    par_for(n, 1 << 16, [&](int64_t a, int64_t b) {
      for (int64_t f = a; f < b; f++) {
        uint64_t m = 0;
        for (int k = 0; k < 3; k++) {
          float x = (V[3 * F[3 * f] + k] + V[3 * F[3 * f + 1] + k] + V[3 * F[3 * f + 2] + k]) * (1.f / 3);
          uint64_t q = (uint64_t)std::min(2097151.0, std::max(0.0, ((double)x - lo[k]) * sc[k]));
          m |= spread21(q) << (2 - k);
        }
        code[f] = m; idx[f] = (uint32_t)f;
      }
    });
    radix_sort_kv(code.p, idx.p, n, 64);
    par_for(n, 1 << 15, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) {
        int32_t f = (int32_t)idx[i]; BTri& t = T[i];
        memcpy(t.a, V + 3 * F[3 * f], 12); memcpy(t.b, V + 3 * F[3 * f + 1], 12); memcpy(t.c, V + 3 * F[3 * f + 2], 12);
        t.id = f; t.p0 = 0; t.p1 = 0;
      }
    });
    if (n == 1) { memset(N, 0, sizeof(BNode)); N[0].l = ~0; N[0].r = ~0; t_build = now() - t0; return; }
    const uint64_t* K = code.p;
    auto delta = [&](int64_t i, int64_t j) -> int {
      if (j < 0 || j >= n) return -1;
      uint64_t a = K[i], b = K[j];
      if (a == b) return 64 + __builtin_clz((uint32_t)i ^ (uint32_t)j);
      return __builtin_clzll(a ^ b);
    };
    Arr<int32_t> parL(n), parI(n - 1);  // parent of leaf / internal node
    parI[0] = -1;
    par_for(n - 1, 1 << 14, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) {
        int d = delta(i, i + 1) - delta(i, i - 1) >= 0 ? 1 : -1;
        int dmin = delta(i, i - d);
        int64_t lmax = 2; while (delta(i, i + lmax * d) > dmin) lmax *= 2;
        int64_t l = 0; for (int64_t t = lmax / 2; t >= 1; t /= 2) if (delta(i, i + (l + t) * d) > dmin) l += t;
        int64_t j = i + l * d; int dn = delta(i, j);
        int64_t s = 0, t = l;
        do { t = (t + 1) / 2; if (delta(i, i + (s + t) * d) > dn) s += t; } while (t > 1);
        int64_t g = i + s * d + std::min(d, 0);
        BNode& nd = N[i];
        if (std::min(i, j) == g) { nd.l = ~(int32_t)g; parL[g] = (int32_t)i; } else { nd.l = (int32_t)g; parI[g] = (int32_t)i; }
        if (std::max(i, j) == g + 1) { nd.r = ~(int32_t)(g + 1); parL[g + 1] = (int32_t)i; } else { nd.r = (int32_t)(g + 1); parI[g + 1] = (int32_t)i; }
      }
    });
    // bottom-up boxes: the second child to arrive at a node computes it and continues upward
    Arr<int32_t> visits(n - 1);
    par_for(n - 1, 1 << 18, [&](int64_t a, int64_t b) { memset(visits.p + a, 0, (b - a) * 4); });
    auto childBox = [&](int32_t c, float* blo, float* bhi) {
      if (c < 0) {
        const BTri& t = T[~c];
        for (int k = 0; k < 3; k++) { blo[k] = std::min(t.a[k], std::min(t.b[k], t.c[k])); bhi[k] = std::max(t.a[k], std::max(t.b[k], t.c[k])); }
      } else {
        const BNode& m = N[c];
        for (int k = 0; k < 3; k++) { blo[k] = std::min(m.lo0[k], m.lo1[k]); bhi[k] = std::max(m.hi0[k], m.hi1[k]); }
      }
    };
    par_for(n, 1 << 14, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) {
        int32_t p = parL[i];
        while (p >= 0) {
          if (__atomic_fetch_add(&visits.p[p], 1, __ATOMIC_ACQ_REL) == 0) break;
          BNode& nd = N[p];
          childBox(nd.l, nd.lo0, nd.hi0); childBox(nd.r, nd.lo1, nd.hi1);
          nd.p0 = 0; nd.p1 = 0;
          p = parI[p];
        }
      }
    });
    t_build = now() - t0;
  }

  // q: packed float3 [m] in a shared buffer. Outputs dist F32 [m], fid I32 [m] (-1 when nothing within cap), uvw F32 [m,3].
  void query(const Tensor& q, int64_t m, Tensor& dist, Tensor& fid, Tensor* uvw = nullptr, float cap = INFINITY) const {
    dist = Tensor(tk::F32, {m}); fid = Tensor(tk::I32, {m});
    Tensor u = uvw ? Tensor(tk::F32, {m, 3}) : Tensor(tk::F32, {1, 3});
    struct { int32_t ntri; float cap2; int32_t want; int32_t pad; } p = {(int32_t)n, std::isinf(cap) ? 3.0e38f : cap * cap, uvw ? 1 : 0, 0};
    const int64_t CH = 1 << 22;
    for (int64_t o = 0; o < m; o += CH) {
      int64_t c = std::min(CH, m - o);
      tk::Buf qb = q.buf(), db = dist.buf(), fb = fid.buf(), ub = u.buf();
      qb.off += o * 12; db.off += o * 4; fb.off += o * 4; if (uvw) ub.off += o * 12;
      run("k_query", c, {nodes.buf(), tris.buf(), qb, db, fb, ub}, p);
    }
    if (uvw) *uvw = u;
  }
};

// brute-force nearest triangle (test reference): lowest face index on ties
inline void brute_nearest(const float* V, const int32_t* F, int64_t nf, V3 p, float& d2, int32_t& fid, V3& uvw) {
  d2 = INFINITY; fid = -1; uvw = {0, 0, 0};
  for (int64_t f = 0; f < nf; f++) {
    V3 a = ld(V + 3 * F[3 * f]), b = ld(V + 3 * F[3 * f + 1]), c = ld(V + 3 * F[3 * f + 2]);
    V3 w = tri_bary(p, a, b, c);
    V3 d = p - (a * w.x + b * w.y + c * w.z);
    float e = dot(d, d);
    if (e < d2) { d2 = e; fid = (int32_t)f; uvw = w; }
  }
}

}  // namespace glb
