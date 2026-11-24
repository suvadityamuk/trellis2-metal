// fill_holes and the narrow-band dual-contouring remesh of o_voxel.postprocess.to_glb (remesh=True, band=1,
// project_back=0), reimplemented from the CuMesh algorithms (MIT, JeffreyXiang/CuMesh: cumesh/remeshing.py,
// src/remesh/simple_dual_contour.cu, src/clean_up.cu).
//
// Sparse sets of grid cells / grid vertices are bitmaps over the dense index space with a per-word prefix popcount
// (rank), so "is this cell active and what is its row" is two memory reads on the CPU and on the GPU, and rows are in
// key order. The unsigned-distance queries (the cost of the remesh) run on the GPU against the LBVH with the search
// radius capped at the band, the cell refinement and the dual contouring run on CPU threads over shared memory.
#pragma once
#include "lbvh.h"

namespace glb {

struct Mesh {
  Arr<float> V; Arr<int32_t> F;
  int64_t nv = 0, nf = 0;
};

struct BitGrid {
  uint64_t nkeys = 0; int64_t nwords = 0, count = 0;
  Tensor bits, rank;
  uint32_t* B = nullptr; uint32_t* Rk = nullptr;
  void init(uint64_t nk) {
    tk::sync();
    nkeys = nk; nwords = (int64_t)((nk + 31) / 32);
    bits = gbuf<uint32_t>(nwords); rank = gbuf<uint32_t>(nwords);
    B = bits.ptr<uint32_t>(); Rk = rank.ptr<uint32_t>();
    par_for(nwords, 1 << 20, [&](int64_t a, int64_t b) { memset(B + a, 0, (b - a) * 4); });
  }
  void set(uint64_t k) { __atomic_fetch_or(&B[k >> 5], 1u << (k & 31), __ATOMIC_RELAXED); }
  void finalize() {
    par_for(nwords, 1 << 18, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) Rk[i] = (uint32_t)__builtin_popcount(B[i]); });
    count = prefix_sum(Rk, nwords);
  }
  bool has(uint64_t k) const { return (B[k >> 5] >> (k & 31)) & 1u; }
  int64_t find(uint64_t k) const {
    uint32_t w = B[k >> 5], b = (uint32_t)(k & 31);
    if (!((w >> b) & 1u)) return -1;
    return (int64_t)Rk[k >> 5] + __builtin_popcount(w & ((1u << b) - 1u));
  }
  // keys in rank order
  void list(uint32_t* out) const {
    par_for(nwords, 1 << 16, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) { uint32_t w = B[i]; int64_t o = Rk[i]; while (w) { out[o++] = (uint32_t)(i * 32 + __builtin_ctz(w)); w &= w - 1; } }
    });
  }
};

// ---- fill_holes -----------------------------------------------------------------------------------------------------
// Boundary loops (closed chains of edges with one incident face) with perimeter < max_perimeter get a fan around the
// mean of their edge midpoints. Candidates are found without building adjacency: a vertex whose outgoing and incoming
// half-edge neighbour multisets differ (64-bit hash sums) touches a boundary; only half-edges between two such
// vertices are sorted and counted.
inline uint64_t mix64(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }

struct HoleStats { int64_t boundary_edges = 0, loops = 0, filled = 0, new_faces = 0; };

inline HoleStats fill_holes(const float* V, const int32_t* F, int64_t nv, int64_t nf, float max_perimeter, Mesh& out) {
  HoleStats st;
  Arr<uint64_t> D(nv);
  par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memset(D.p + a, 0, (b - a) * 8); });
  par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
    for (int64_t f = a; f < b; f++) for (int k = 0; k < 3; k++) {
      int32_t u = F[3 * f + k], v = F[3 * f + (k + 1) % 3];
      __atomic_fetch_add(&D.p[u], mix64((uint64_t)v), __ATOMIC_RELAXED);
      __atomic_fetch_sub(&D.p[v], mix64((uint64_t)u), __ATOMIC_RELAXED);
    }
  });
  const int C = 32;
  std::vector<std::vector<uint64_t>> part(C);  // (min << 32 | max) keys; direction recovered from the face
  std::vector<std::vector<uint32_t>> partf(C);
  par_chunks(nf, C, [&](int c, int64_t a, int64_t b) {
    for (int64_t f = a; f < b; f++) for (int k = 0; k < 3; k++) {
      int32_t u = F[3 * f + k], v = F[3 * f + (k + 1) % 3];
      if (u != v && D[u] != 0 && D[v] != 0) {
        part[c].push_back(((uint64_t)std::min(u, v) << 32) | (uint32_t)std::max(u, v));
        partf[c].push_back((uint32_t)(f * 3 + k) /* half-edge id; nf < 2^32 / 3 */);
      }
    }
  });
  std::vector<uint64_t> keys; std::vector<uint32_t> hes;
  for (int c = 0; c < C; c++) { keys.insert(keys.end(), part[c].begin(), part[c].end()); hes.insert(hes.end(), partf[c].begin(), partf[c].end()); }
  int64_t m = (int64_t)keys.size();
  radix_sort_kv(keys.data(), hes.data(), m, 64);
  // boundary half-edges (u -> v as in their face)
  std::vector<int32_t> bu, bv;
  for (int64_t i = 0; i < m;) {
    int64_t j = i + 1; while (j < m && keys[j] == keys[i]) j++;
    if (j - i == 1) { uint32_t h = hes[i]; int64_t f = h / 3; int k = h % 3; bu.push_back(F[3 * f + k]); bv.push_back(F[3 * f + (k + 1) % 3]); }
    i = j;
  }
  int64_t nb = (int64_t)bu.size();
  st.boundary_edges = nb;
  std::vector<float> newV; std::vector<int32_t> newF;
  if (nb > 0) {
    // compact boundary vertex ids, union-find components, per-vertex boundary degree
    std::vector<int32_t> vs(bu); vs.insert(vs.end(), bv.begin(), bv.end());
    std::sort(vs.begin(), vs.end()); vs.erase(std::unique(vs.begin(), vs.end()), vs.end());
    auto loc = [&](int32_t v) { return (int32_t)(std::lower_bound(vs.begin(), vs.end(), v) - vs.begin()); };
    int32_t nbv = (int32_t)vs.size();
    std::vector<int32_t> par(nbv), deg(nbv, 0);
    for (int32_t i = 0; i < nbv; i++) par[i] = i;
    auto find = [&](int32_t x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    std::vector<int32_t> lu(nb), lv(nb);
    for (int64_t e = 0; e < nb; e++) {
      lu[e] = loc(bu[e]); lv[e] = loc(bv[e]); deg[lu[e]]++; deg[lv[e]]++;
      int32_t a = find(lu[e]), b = find(lv[e]); if (a != b) par[std::max(a, b)] = std::min(a, b);
    }
    // per component: closed loop iff every vertex has boundary degree 2; perimeter; midpoint mean
    std::vector<uint8_t> isloop(nbv, 1); std::vector<double> perim(nbv, 0.0), cx(nbv, 0.0), cy(nbv, 0.0), cz(nbv, 0.0); std::vector<int32_t> cnt(nbv, 0);
    for (int32_t i = 0; i < nbv; i++) if (deg[i] != 2) isloop[find(i)] = 0;
    for (int64_t e = 0; e < nb; e++) {
      int32_t r = find(lu[e]);
      const float* p = V + 3 * (int64_t)bu[e]; const float* q = V + 3 * (int64_t)bv[e];
      perim[r] += std::sqrt((double)(p[0] - q[0]) * (p[0] - q[0]) + (double)(p[1] - q[1]) * (p[1] - q[1]) + (double)(p[2] - q[2]) * (p[2] - q[2]));
      cx[r] += 0.5 * ((double)p[0] + q[0]); cy[r] += 0.5 * ((double)p[1] + q[1]); cz[r] += 0.5 * ((double)p[2] + q[2]); cnt[r]++;
    }
    std::vector<int32_t> nid(nbv, -1);
    for (int32_t r = 0; r < nbv; r++) {
      if (par[r] != r) continue;
      if (!isloop[r]) continue;
      st.loops++;
      if (!(perim[r] < max_perimeter)) continue;
      st.filled++;
      nid[r] = (int32_t)(nv + (int64_t)newV.size() / 3);
      newV.push_back((float)(cx[r] / cnt[r])); newV.push_back((float)(cy[r] / cnt[r])); newV.push_back((float)(cz[r] / cnt[r]));
    }
    for (int64_t e = 0; e < nb; e++) {
      int32_t r = find(lu[e]);
      if (nid[r] < 0) continue;
      newF.push_back(bv[e]); newF.push_back(bu[e]); newF.push_back(nid[r]);  // opposite to the boundary half-edge
    }
  }
  int64_t av = (int64_t)newV.size() / 3, af = (int64_t)newF.size() / 3;
  st.new_faces = af;
  out.nv = nv + av; out.nf = nf + af;
  out.V.alloc(out.nv * 3); out.F.alloc(out.nf * 3);
  par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memcpy(out.V.p + 3 * a, V + 3 * a, (b - a) * 12); });
  par_for(nf, 1 << 18, [&](int64_t a, int64_t b) { memcpy(out.F.p + 3 * a, F + 3 * a, (b - a) * 12); });
  if (av) memcpy(out.V.p + 3 * nv, newV.data(), av * 12);
  if (af) memcpy(out.F.p + 3 * nf, newF.data(), af * 12);
  return st;
}

// ---- narrow-band dual contouring -------------------------------------------------------------------------------------
struct RemeshStats { int64_t voxels = 0, grid_verts = 0, quads = 0, queries = 0; double t_band = 0, t_grid = 0, t_dc = 0, t_quads = 0; };

// Remesh the zero level set of UDF(mesh) - eps (eps = band * scale / resolution) at `resolution`, centered at 0.
inline RemeshStats remesh_narrow_band_dc(const LBVH& bvh, int resolution, float scale, float band, Mesh& out) {
  RemeshStats st;
  const uint32_t R = (uint32_t)resolution;
  const float eps = band * scale / (float)resolution;
  uint32_t base = R; while (base > 32) { if (base % 2) { fprintf(stderr, "glb: resolution %u has no power-of-two base\n", R); abort(); } base /= 2; }
  double t0 = now();
  tk::sync();
  // level 0: the full base grid
  int64_t nc = (int64_t)base * base * base;
  Tensor keys = gbuf<uint32_t>(nc);
  { uint32_t* k = keys.ptr<uint32_t>(); for (int64_t i = 0; i < nc; i++) k[i] = (uint32_t)i; }
  struct BandP { int32_t ntri; uint32_t R; float S; float eps; float thr; float cap2; };
  auto gpu = [&](const char* kern, const Tensor& in, Tensor& o, int64_t n, int osz, const BandP& p) {
    const int64_t CH = 1 << 22;
    for (int64_t off = 0; off < n; off += CH) {
      tk::Buf ib = in.buf(), ob = o.buf(); ib.off += off * 4; ob.off += off * osz;
      run(kern, std::min(CH, n - off), {bvh.nodes.buf(), bvh.tris.buf(), ib, ob}, p);
      if ((off / CH) % 8 == 7) tk::flush();
    }
    tk::sync();
  };
  uint32_t lr = base;
  while (true) {
    float cell = scale / (float)lr, thr = 0.87f * cell, cap = (eps + thr) * 1.001f;
    BandP p = {(int32_t)bvh.n, lr, scale, eps, thr, cap * cap};
    Tensor flag = gbuf<uint8_t>(nc);
    gpu("k_band", keys, flag, nc, 1, p);
    st.queries += nc;
    const uint32_t* K = keys.ptr<uint32_t>(); const uint8_t* Fl = flag.ptr<uint8_t>();
    const int C = 64; int64_t cnt[C + 1] = {0};
    par_chunks(nc, C, [&](int c, int64_t a, int64_t b) { int64_t s = 0; for (int64_t i = a; i < b; i++) s += Fl[i]; cnt[c + 1] = s; });
    for (int c = 0; c < C; c++) cnt[c + 1] += cnt[c];
    int64_t kept = cnt[C];
    bool last = lr >= R;
    Tensor nk = gbuf<uint32_t>(last ? kept : kept * 8);
    uint32_t* NK = nk.ptr<uint32_t>();
    uint32_t nr = lr * 2;
    par_chunks(nc, C, [&](int c, int64_t a, int64_t b) {
      int64_t o = cnt[c];
      for (int64_t i = a; i < b; i++) {
        if (!Fl[i]) continue;
        uint32_t k = K[i];
        if (last) { NK[o++] = k; continue; }
        uint32_t x = k % lr, y = (k / lr) % lr, z = k / (lr * lr);
        for (int j = 0; j < 8; j++) NK[o * 8 + j] = ((2 * z + ((j >> 2) & 1)) * nr + (2 * y + ((j >> 1) & 1))) * nr + (2 * x + (j & 1));
        o++;
      }
    });
    keys = nk;
    if (last) { nc = kept; break; }
    nc = kept * 8; lr = nr;
  }
  st.t_band = now() - t0; t0 = now();

  // active cells and their corner vertices
  BitGrid vox; vox.init((uint64_t)R * R * R);
  { const uint32_t* K = keys.ptr<uint32_t>(); par_for(nc, 1 << 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) vox.set(K[i]); }); }
  vox.finalize();
  int64_t nvox = vox.count; st.voxels = nvox;
  Arr<uint32_t> vkey(nvox); vox.list(vkey.p);
  keys = Tensor();
  const uint32_t R1 = R + 1;
  BitGrid gv; gv.init((uint64_t)R1 * R1 * R1);
  par_for(nvox, 1 << 15, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      uint32_t k = vkey[i]; uint32_t x = k % R, y = (k / R) % R, z = k / (R * R);
      for (int j = 0; j < 8; j++) gv.set(((uint64_t)(z + ((j >> 2) & 1)) * R1 + (y + ((j >> 1) & 1))) * R1 + (x + (j & 1)));
    }
  });
  gv.finalize();
  int64_t ngv = gv.count; st.grid_verts = ngv;
  Tensor gkeys = gbuf<uint32_t>(ngv); gv.list(gkeys.ptr<uint32_t>());
  Tensor gval = gbuf<float>(ngv);
  {
    float cell = scale / (float)R, cap = (eps + (0.87f + 0.8661f) * cell) * 1.01f;  // a corner is within half a cell diagonal of its cell center
    BandP p = {(int32_t)bvh.n, R, scale, eps, 0.f, cap * cap};
    gpu("k_gridval", gkeys, gval, ngv, 4, p);
    st.queries += ngv;
  }
  st.t_grid = now() - t0; t0 = now();
  const float* GV = gval.ptr<float>();

  // dual vertex and edge crossings per cell (simple_dual_contour)
  Arr<float> dual(nvox * 3); Arr<int8_t> inter(nvox * 3); Arr<uint32_t> qcnt(nvox + 1);
  par_for(nvox, 1 << 14, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      uint32_t k = vkey[i]; uint32_t x = k % R, y = (k / R) % R, z = k / (R * R);
      float c[2][2][2];
      for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) {
        // the two x-neighbours are adjacent keys
        uint64_t g = ((uint64_t)(z + dz) * R1 + (y + dy)) * R1 + x;
        int64_t r0 = gv.find(g);
        c[0][dy][dz] = GV[r0]; c[1][dy][dz] = GV[gv.find(g + 1)];
      }
      float sx = 0, sy = 0, sz = 0; int n = 0; int8_t it[3] = {0, 0, 0};
      auto crossing = [](float v1, float v2) { return (v1 < 0 && v2 >= 0) || (v1 >= 0 && v2 < 0); };
      auto dirOf = [](float v1, float v2) -> int8_t { return (v1 < 0 && v2 >= 0) ? 1 : ((v1 >= 0 && v2 < 0) ? -1 : 0); };
      for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) {
        float v1 = c[0][u][v], v2 = c[1][u][v];
        if (crossing(v1, v2)) { float t = -v1 / (v2 - v1); sx += (float)x + t; sy += (float)(y + u); sz += (float)(z + v); n++; }
        if (u == 1 && v == 1) it[0] = dirOf(v1, v2);
      }
      for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) {
        float v1 = c[u][0][v], v2 = c[u][1][v];
        if (crossing(v1, v2)) { float t = -v1 / (v2 - v1); sx += (float)(x + u); sy += (float)y + t; sz += (float)(z + v); n++; }
        if (u == 1 && v == 1) it[1] = dirOf(v1, v2);
      }
      for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) {
        float v1 = c[u][v][0], v2 = c[u][v][1];
        if (crossing(v1, v2)) { float t = -v1 / (v2 - v1); sx += (float)(x + u); sy += (float)(y + v); sz += (float)z + t; n++; }
        if (u == 1 && v == 1) it[2] = dirOf(v1, v2);
      }
      if (n > 0) { dual[3 * i] = sx / (float)n; dual[3 * i + 1] = sy / (float)n; dual[3 * i + 2] = sz / (float)n; }
      else { dual[3 * i] = (float)x + 0.5f; dual[3 * i + 1] = (float)y + 0.5f; dual[3 * i + 2] = (float)z + 0.5f; }
      // mesh space
      for (int j = 0; j < 3; j++) dual[3 * i + j] = (dual[3 * i + j] / (float)R - 0.5f) * scale;
      // quads: the 4 cells around the (1,1) edge of each axis must all be active
      static const int off[3][3][3] = {{{0, 0, 1}, {0, 1, 1}, {0, 1, 0}}, {{1, 0, 0}, {1, 0, 1}, {0, 0, 1}}, {{0, 1, 0}, {1, 1, 0}, {1, 0, 0}}};
      uint32_t q = 0;
      for (int ax = 0; ax < 3; ax++) {
        if (!it[ax]) continue;
        bool ok = true;
        for (int j = 0; j < 3 && ok; j++) {
          uint32_t nx = x + off[ax][j][0], ny = y + off[ax][j][1], nz = z + off[ax][j][2];
          ok = nx < R && ny < R && nz < R && vox.has(((uint64_t)nz * R + ny) * R + nx);
        }
        if (ok) q++; else it[ax] = 0;
      }
      inter[3 * i] = it[0]; inter[3 * i + 1] = it[1]; inter[3 * i + 2] = it[2];
      qcnt[i] = q;
    }
  });
  st.t_dc = now() - t0; t0 = now();
  int64_t nq = prefix_sum(qcnt.p, nvox); qcnt[nvox] = (uint32_t)nq;
  st.quads = nq;
  // triangles (split along the diagonal whose two triangles are better aligned), then drop unreferenced dual vertices
  Arr<int32_t> tri(nq * 6); Arr<uint32_t> used(nvox + 1);
  par_for(nvox, 1 << 18, [&](int64_t a, int64_t b) { memset(used.p + a, 0, (b - a) * 4); });
  par_for(nvox, 1 << 14, [&](int64_t a, int64_t b) {
    static const int off[3][3][3] = {{{0, 0, 1}, {0, 1, 1}, {0, 1, 0}}, {{1, 0, 0}, {1, 0, 1}, {0, 0, 1}}, {{0, 1, 0}, {1, 1, 0}, {1, 0, 0}}};
    static const int s1n[6] = {0, 1, 2, 0, 2, 3}, s1p[6] = {0, 2, 1, 0, 3, 2}, s2n[6] = {0, 1, 3, 3, 1, 2}, s2p[6] = {0, 3, 1, 3, 2, 1};
    for (int64_t i = a; i < b; i++) {
      int64_t o = qcnt[i];
      if (qcnt[i + 1] == o) continue;
      uint32_t k = vkey[i]; uint32_t x = k % R, y = (k / R) % R, z = k / (R * R);
      for (int ax = 0; ax < 3; ax++) {
        int8_t dir = inter[3 * i + ax];
        if (!dir) continue;
        int32_t q[4]; q[0] = (int32_t)i;
        for (int j = 0; j < 3; j++) q[j + 1] = (int32_t)vox.find(((uint64_t)(z + off[ax][j][2]) * R + (y + off[ax][j][1])) * R + (x + off[ax][j][0]));
        const int* a0 = dir == 1 ? s1p : s1n; const int* a1 = dir == 1 ? s2p : s2n;
        auto align = [&](const int* s) {
          V3 p0 = ld(&dual[3 * (int64_t)q[s[0]]]), p1 = ld(&dual[3 * (int64_t)q[s[1]]]), p2 = ld(&dual[3 * (int64_t)q[s[2]]]), p3 = ld(&dual[3 * (int64_t)q[s[3]]]);
          return std::fabs(dot(cross(p1 - p0, p2 - p0), cross(p2 - p1, p3 - p1)));
        };
        const int* s = align(a0) > align(a1) ? a0 : a1;
        for (int j = 0; j < 6; j++) tri[o * 6 + j] = q[s[j]];
        for (int j = 0; j < 4; j++) used[q[j]] = 1;  // benign race: all writers store 1
        o++;
      }
    }
  });
  int64_t nv = prefix_sum(used.p, nvox); used[nvox] = (uint32_t)nv;
  out.nv = nv; out.nf = nq * 2;
  out.V.alloc(nv * 3); out.F.alloc(nq * 6);
  par_for(nvox, 1 << 16, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) if (used[i + 1] != used[i]) memcpy(out.V.p + 3 * (int64_t)used[i], dual.p + 3 * i, 12);
  });
  par_for(nq * 6, 1 << 18, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) out.F[i] = (int32_t)used[tri[i]]; });
  st.t_quads = now() - t0;
  return st;
}

}  // namespace glb
