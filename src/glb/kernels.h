// Metal Shading Language source of the to_glb compute kernels, compiled at runtime (util.h kernlib()).
//   LBVH nearest-triangle query (stack traversal, exact closest point, lowest face index on ties)
//   narrow-band tests and grid-vertex values for the dual-contouring remesh (fused position generation + query)
//   texture bake: UV-space ids -> surface position -> nearest point on the input mesh -> trilinear sparse-voxel sample
//   wavefront TELEA inpaint over compacted per-level pixel lists, uint8 conversion and glTF texture packing
#pragma once
#include "util.h"

namespace glb {
inline const char* kernel_source() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Node { packed_float3 lo0; int l; packed_float3 hi0; int r; packed_float3 lo1; float p0; packed_float3 hi1; float p1; };
struct Tri { packed_float3 a; int id; packed_float3 b; float p0; packed_float3 c; float p1; };
struct Hit { float d2; int leaf; int id; float3 uvw; };

static inline float box_d2(float3 p, float3 lo, float3 hi) { float3 d = max(max(lo - p, p - hi), float3(0.0f)); return dot(d, d); }

static inline float seg_t(float3 p, float3 a, float3 b) {
  float3 ab = b - a; float l2 = dot(ab, ab);
  return l2 > 0.0f ? clamp(dot(p - a, ab) / l2, 0.0f, 1.0f) : 0.0f;
}
// barycentric coordinates of the point of triangle (a, b, c) closest to p (Ericson, Real-Time Collision Detection 5.1.5)
static float3 tri_bary(float3 p, float3 a, float3 b, float3 c) {
  float3 ab = b - a, ac = c - a, ap = p - a;
  float3 nrm = cross(ab, ac);
  // (near-)zero-area triangle, sin^2 of the angle at a below 1e-7: the region tests below are rounding noise there
  // (and differ with FMA contraction), so take the closest of the three edges; the error is below the sliver's width
  if (dot(nrm, nrm) <= 1e-7f * dot(ab, ab) * dot(ac, ac)) {
    float t0 = seg_t(p, a, b), t1 = seg_t(p, b, c), t2 = seg_t(p, c, a);
    float3 q0 = a + (b - a) * t0, q1 = b + (c - b) * t1, q2 = c + (a - c) * t2;
    float e0 = distance_squared(p, q0), e1 = distance_squared(p, q1), e2 = distance_squared(p, q2);
    if (e0 <= e1 && e0 <= e2) return float3(1.0f - t0, t0, 0.0f);
    if (e1 <= e2) return float3(0.0f, 1.0f - t1, t1);
    return float3(t2, 0.0f, 1.0f - t2);
  }
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
  // degenerate (zero-area) triangle: closest of the three edges
  float t0 = seg_t(p, a, b), t1 = seg_t(p, b, c), t2 = seg_t(p, c, a);
  float3 q0 = a + (b - a) * t0, q1 = b + (c - b) * t1, q2 = c + (a - c) * t2;
  float e0 = distance_squared(p, q0), e1 = distance_squared(p, q1), e2 = distance_squared(p, q2);
  if (e0 <= e1 && e0 <= e2) return float3(1.0f - t0, t0, 0.0f);
  if (e1 <= e2) return float3(0.0f, 1.0f - t1, t1);
  return float3(t2, 0.0f, 1.0f - t2);
}

static inline void tri_test(const device Tri* tris, int leaf, float3 p, thread Hit& h) {
  Tri t = tris[leaf];
  float3 a = t.a, b = t.b, c = t.c;
  float3 w = tri_bary(p, a, b, c);
  float3 d = p - (a * w.x + b * w.y + c * w.z);
  float d2 = dot(d, d);
  if (d2 < h.d2 || (d2 == h.d2 && t.id < h.id)) { h.d2 = d2; h.leaf = leaf; h.id = t.id; h.uvw = w; }
}

// nearest triangle within sqrt(cap2); h.leaf < 0 when there is none
static Hit nearest(const device Node* nodes, const device Tri* tris, int ntri, float3 p, float cap2) {
  Hit h; h.d2 = cap2; h.leaf = -1; h.id = 0x7fffffff; h.uvw = float3(0.0f);
  if (ntri <= 0) return h;
  if (ntri == 1) { tri_test(tris, 0, p, h); return h; }
  int stack[64]; float stackd[64]; int sp = 0; int cur = 0;
  while (true) {
    Node nd = nodes[cur];
    float d0 = box_d2(p, nd.lo0, nd.hi0), d1 = box_d2(p, nd.lo1, nd.hi1);
    int c0 = nd.l, c1 = nd.r;
    if (d1 < d0) { float td = d0; d0 = d1; d1 = td; int tc = c0; c0 = c1; c1 = tc; }
    bool v0 = d0 <= h.d2;
    if (v0 && c0 < 0) { tri_test(tris, ~c0, p, h); v0 = false; }
    bool v1 = d1 <= h.d2;
    if (v1 && c1 < 0) { tri_test(tris, ~c1, p, h); v1 = false; }
    if (v0) {
      if (v1 && sp < 64) { stack[sp] = c1; stackd[sp] = d1; sp++; }
      cur = c0; continue;
    }
    if (v1) { cur = c1; continue; }
    bool found = false;
    while (sp > 0) { sp--; if (stackd[sp] <= h.d2) { cur = stack[sp]; found = true; break; } }
    if (!found) break;
  }
  return h;
}

struct QueryP { int ntri; float cap2; int want_uvw; int pad; };
kernel void k_query(const device Node* nodes [[buffer(0)]], const device Tri* tris [[buffer(1)]], const device packed_float3* q [[buffer(2)]],
                    device float* dist [[buffer(3)]], device int* fid [[buffer(4)]], device packed_float3* uvw [[buffer(5)]],
                    constant QueryP& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  Hit h = nearest(nodes, tris, p.ntri, float3(q[gid]), p.cap2);
  dist[gid] = sqrt(h.d2); fid[gid] = h.leaf < 0 ? -1 : h.id;
  if (p.want_uvw != 0) uvw[gid] = h.uvw;
}

// ---- narrow-band dual-contouring remesh -------------------------------------------------------------------------------
// keys: linear cell index (z * R + y) * R + x at level resolution R. flag = |UDF(center) - eps| < thr
struct BandP { int ntri; uint R; float S; float eps; float thr; float cap2; };
kernel void k_band(const device Node* nodes [[buffer(0)]], const device Tri* tris [[buffer(1)]], const device uint* keys [[buffer(2)]],
                   device uchar* flag [[buffer(3)]], constant BandP& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  uint k = keys[gid];
  uint x = k % p.R, y = (k / p.R) % p.R, z = k / (p.R * p.R);
  float3 pt = ((float3(float(x), float(y), float(z)) + 0.5f) / float(p.R) - 0.5f) * p.S;
  Hit h = nearest(nodes, tris, p.ntri, pt, p.cap2);
  flag[gid] = (h.leaf >= 0 && fabs(sqrt(h.d2) - p.eps) < p.thr) ? 1 : 0;
}
// keys: linear grid-vertex index over (R + 1)^3. val = UDF(vertex) - eps
kernel void k_gridval(const device Node* nodes [[buffer(0)]], const device Tri* tris [[buffer(1)]], const device uint* keys [[buffer(2)]],
                      device float* val [[buffer(3)]], constant BandP& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  uint k = keys[gid]; uint R1 = p.R + 1;
  uint x = k % R1, y = (k / R1) % R1, z = k / (R1 * R1);
  float3 pt = (float3(float(x), float(y), float(z)) / float(p.R) - 0.5f) * p.S;
  Hit h = nearest(nodes, tris, p.ntri, pt, p.cap2);
  val[gid] = sqrt(h.d2) - p.eps;
}

// ---- texture bake -------------------------------------------------------------------------------------------------------
static inline int vox_find(const device uint* bits, const device uint* rank, uint key) {
  uint w = bits[key >> 5], b = key & 31u;
  if (((w >> b) & 1u) == 0u) return -1;
  return int(rank[key >> 5] + popcount(w & ((1u << b) - 1u)));
}
static inline uchar to_u8(float x) { return uchar(int(clamp(x * 255.0f, 0.0f, 255.0f))); }

struct BakeP { int ntri; int W; int H; int res; uint base; uint p0; uint p1; uint p2; };
// ids: face index + 1 per texel (0 = empty). img: planar uint8 [6, P]. lv: 0 for baked texels, -1 for empty ones.
kernel void k_bake(const device Node* nodes [[buffer(0)]], const device Tri* tris [[buffer(1)]], const device uint* ids [[buffer(2)]],
                   const device uint* faces [[buffer(3)]], const device packed_float3* pos [[buffer(4)]], const device packed_float2* uv [[buffer(5)]],
                   const device uint* vbits [[buffer(6)]], const device uint* vrank [[buffer(7)]], const device float* attrs [[buffer(8)]],
                   device uchar* img [[buffer(9)]], device int* lv [[buffer(10)]], constant BakeP& p [[buffer(11)]],
                   uint tid [[thread_position_in_grid]]) {
  uint P = uint(p.W) * uint(p.H);
  uint gid = tid + p.base;
  uint id = ids[gid];
  bool ok = id != 0u;
  float acc[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (ok) {
    uint f = id - 1u;
    uint i0 = faces[3 * f], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
    float2 t = float2((float(gid % uint(p.W)) + 0.5f) / float(p.W), (float(gid / uint(p.W)) + 0.5f) / float(p.H));
    float2 u0 = uv[i0], e1 = float2(uv[i1]) - u0, e2 = float2(uv[i2]) - u0, d = t - u0;
    float den = e1.x * e2.y - e1.y * e2.x;
    if (den == 0.0f) ok = false;
    else {
      float b1 = (d.x * e2.y - d.y * e2.x) / den, b2 = (e1.x * d.y - e1.y * d.x) / den, b0 = 1.0f - b1 - b2;
      float3 x = float3(pos[i0]) * b0 + float3(pos[i1]) * b1 + float3(pos[i2]) * b2;
      Hit h = nearest(nodes, tris, p.ntri, x, 3.0e38f);
      if (h.leaf >= 0) { Tri tr = tris[h.leaf]; x = float3(tr.a) * h.uvw.x + float3(tr.b) * h.uvw.y + float3(tr.c) * h.uvw.z; }
      float3 q = (x + 0.5f) * float(p.res);
      int3 b = int3(floor(q - 0.5f));
      int n[8]; float w[8]; float wsum = 0.0f;
      for (int i = 0; i < 8; i++) {
        int vx = b.x + (i & 1), vy = b.y + ((i >> 1) & 1), vz = b.z + ((i >> 2) & 1);
        n[i] = -1; w[i] = 0.0f;
        if (vx >= 0 && vx < p.res && vy >= 0 && vy < p.res && vz >= 0 && vz < p.res) {
          int j = vox_find(vbits, vrank, (uint(vz) * uint(p.res) + uint(vy)) * uint(p.res) + uint(vx));
          if (j >= 0) {
            n[i] = j;
            w[i] = (1.0f - fabs(q.x - float(vx) - 0.5f)) * (1.0f - fabs(q.y - float(vy) - 0.5f)) * (1.0f - fabs(q.z - float(vz) - 0.5f));
            wsum += w[i];
          }
        }
      }
      float inv = 1.0f / max(wsum, 1e-12f);
      for (int i = 0; i < 8; i++) {
        if (n[i] < 0) continue;
        float wi = w[i] * inv;
        for (int c = 0; c < 6; c++) acc[c] += attrs[uint(n[i]) * 6u + uint(c)] * wi;
      }
    }
  }
  for (uint c = 0; c < 6u; c++) img[c * P + gid] = ok ? to_u8(acc[c]) : uchar(0);
  lv[gid] = ok ? 0 : -1;
}

// ---- wavefront TELEA ------------------------------------------------------------------------------------------------------
// lv: 4-connected distance level from the known region (0 = known). All pixels of level L are filled in parallel from
// pixels of lower levels with TELEA's weights (direction x distance x level); channels 0-2 use radius 3, 3-5 radius 1.
struct TeleaP { int H; int W; int L; int pad; };
static inline bool kn(const device int* lv, int H, int W, int L, int r, int c) {
  if (r < 0 || c < 0 || r >= H || c >= W) return false;
  int v = lv[r * W + c]; return v >= 0 && v < L;
}
kernel void k_telea_fill(const device int* lv [[buffer(0)]], device uchar* img [[buffer(1)]], const device uint* list [[buffer(2)]],
                         constant TeleaP& p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int H = p.H, W = p.W, L = p.L; uint P = uint(H) * uint(W);
  int i = int(list[gid]); int y = i / W, x = i % W;
  float fL = float(L);
  bool xr = kn(lv, H, W, L, y, x + 1), xl = kn(lv, H, W, L, y, x - 1), yd = kn(lv, H, W, L, y + 1, x), yu = kn(lv, H, W, L, y - 1, x);
  float gx = xr ? (xl ? (float(lv[i + 1]) - float(lv[i - 1])) * 0.5f : float(lv[i + 1]) - fL) : (xl ? fL - float(lv[i - 1]) : 0.0f);
  float gy = yd ? (yu ? (float(lv[i + W]) - float(lv[i - W])) * 0.5f : float(lv[i + W]) - fL) : (yu ? fL - float(lv[i - W]) : 0.0f);
  float Ia[6] = {0, 0, 0, 0, 0, 0}, Jx[6] = {0, 0, 0, 0, 0, 0}, Jy[6] = {0, 0, 0, 0, 0, 0};
  float s3 = 1e-20f, s1 = 1e-20f;
  for (int k = y - 3; k <= y + 3; k++)
    for (int l = x - 3; l <= x + 3; l++) {
      int dd = (k - y) * (k - y) + (l - x) * (l - x);
      if (dd > 9 || !kn(lv, H, W, L, k, l)) continue;
      int j = k * W + l;
      float ry = float(y - k), rx = float(x - l), vl = rx * rx + ry * ry;
      float dir = rx * gx + ry * gy; if (fabs(dir) <= 0.01f) dir = 1e-6f;
      float w = fabs(dir / (vl * sqrt(vl) * (1.0f + fabs(float(lv[j]) - fL))));
      bool r1 = kn(lv, H, W, L, k, l + 1), l1 = kn(lv, H, W, L, k, l - 1), d1 = kn(lv, H, W, L, k + 1, l), u1 = kn(lv, H, W, L, k - 1, l);
      int c0 = dd <= 1 ? 6 : 3;
      for (int c = 0; c < c0; c++) {
        uint o = uint(c) * P + uint(j);
        float v = float(img[o]);
        float ix = r1 ? (l1 ? (float(img[o + 1]) - float(img[o - 1])) * 2.0f : float(img[o + 1]) - v) : (l1 ? v - float(img[o - 1]) : 0.0f);
        float iy = d1 ? (u1 ? (float(img[o + uint(W)]) - float(img[o - uint(W)])) * 2.0f : float(img[o + uint(W)]) - v) : (u1 ? v - float(img[o - uint(W)]) : 0.0f);
        Ia[c] += w * v; Jx[c] -= w * ix * rx; Jy[c] -= w * iy * ry;
      }
      s3 += w; if (dd <= 1) s1 += w;
    }
  for (int c = 0; c < 6; c++) {
    float s = c < 3 ? s3 : s1;
    float sat = Ia[c] / s + (Jx[c] + Jy[c]) / (sqrt(Jx[c] * Jx[c] + Jy[c] * Jy[c]) + 1e-20f);
    img[uint(c) * P + uint(i)] = uchar(clamp(rint(sat + 0.5f), 0.0f, 255.0f));
  }
}

// planar [6, P] -> baseColor RGBA [P, 4] (r, g, b, alpha) and metallicRoughness [P, 3] (0, roughness, metallic)
kernel void k_pack(const device uchar* img [[buffer(0)]], device uchar* base [[buffer(1)]], device uchar* mr [[buffer(2)]],
                   constant uint& P [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  base[4 * i] = img[i]; base[4 * i + 1] = img[P + i]; base[4 * i + 2] = img[2 * P + i]; base[4 * i + 3] = img[5 * P + i];
  mr[3 * i] = 0; mr[3 * i + 1] = img[4 * P + i]; mr[3 * i + 2] = img[3 * P + i];
}

// ---- UV-space rasterization (render pipeline): writes face index + 1 --------------------------------------------------
struct RVOut { float4 pos [[position]]; };
vertex RVOut uv_vs(uint vid [[vertex_id]], const device packed_float2* uv [[buffer(0)]]) {
  RVOut o; float2 t = uv[vid];
  o.pos = float4(t.x * 2.0f - 1.0f, 1.0f - t.y * 2.0f, 0.0f, 1.0f);  // texel row 0 <-> v = 0
  return o;
}
fragment uint uv_fs(RVOut in [[stage_in]], uint pid [[primitive_id]]) { return pid + 1u; }
)MSL";
}
}  // namespace glb
