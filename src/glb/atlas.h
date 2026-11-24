// UV unwrap: chart clustering (after CuMesh's compute_charts, MIT, JeffreyXiang/CuMesh src/atlas.cu) on CPU threads,
// then xatlas parameterization + packing run as many independent group atlases in parallel and tiled into one atlas
// (the grouped scheme of the ROCm port's csrc/uv_atlas.cpp).
#pragma once
#include "simplify.h"
#include "../../third_party/xatlas/xatlas.h"
#include <numeric>
#include <pthread.h>

extern std::atomic<unsigned> t2r_xa_thread_override;

namespace glb {

// drop faces with a repeated vertex or zero area, then unreferenced vertices
inline void clean_mesh(Mesh& m) {
  const int64_t nv = m.nv, nf = m.nf; const float* V = m.V.p; const int32_t* F = m.F.p;
  Arr<uint32_t> fmap(nf + 1), vmap(nv + 1);
  par_for(nv + 1, 1 << 18, [&](int64_t a, int64_t b) { memset(vmap.p + a, 0, (b - a) * 4); });
  par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
    for (int64_t f = a; f < b; f++) {
      const int32_t* t = F + 3 * f; bool ok = t[0] != t[1] && t[1] != t[2] && t[0] != t[2];
      if (ok) { V3 n = cross(ld(V + 3 * (int64_t)t[1]) - ld(V + 3 * (int64_t)t[0]), ld(V + 3 * (int64_t)t[2]) - ld(V + 3 * (int64_t)t[0])); ok = dot(n, n) > 1e-24f; }
      fmap[f] = ok;
      if (ok) for (int k = 0; k < 3; k++) vmap[t[k]] = 1;
    }
  });
  int64_t nf2 = prefix_sum(fmap.p, nf); fmap[nf] = (uint32_t)nf2;
  int64_t nv2 = prefix_sum(vmap.p, nv); vmap[nv] = (uint32_t)nv2;
  if (nf2 == nf && nv2 == nv) return;
  Arr<float> V2(nv2 * 3); Arr<int32_t> F2(nf2 * 3);
  par_for(nv, 1 << 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) if (vmap[i + 1] != vmap[i]) memcpy(V2.p + 3 * (int64_t)vmap[i], V + 3 * i, 12); });
  par_for(nf, 1 << 16, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) if (fmap[i + 1] != fmap[i]) { int32_t* d = F2.p + 3 * (int64_t)fmap[i]; for (int k = 0; k < 3; k++) d[k] = (int32_t)vmap[F[3 * i + k]]; }
  });
  m.V = std::move(V2); m.F = std::move(F2); m.nv = nv2; m.nf = nf2;
}

// area-weighted vertex normals (sum of face cross products, normalized), as CuMesh compute_vertex_normals
inline void vertex_normals(const float* V, const int32_t* F, int64_t nv, int64_t nf, Arr<float>& N) {
  VF vf; vf.build(F, nv, nf);
  N.alloc(nv * 3);
  par_for(nv, 1 << 13, [&](int64_t a, int64_t b) {
    for (int64_t v = a; v < b; v++) {
      double s[3] = {0, 0, 0};
      for (uint32_t i = vf.off[v]; i < vf.off[v + 1]; i++) {
        const int32_t* t = F + 3 * (int64_t)vf.f[i];
        V3 n = cross(ld(V + 3 * (int64_t)t[1]) - ld(V + 3 * (int64_t)t[0]), ld(V + 3 * (int64_t)t[2]) - ld(V + 3 * (int64_t)t[0]));
        s[0] += n.x; s[1] += n.y; s[2] += n.z;
      }
      double l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
      if (l > 0) { N[3 * v] = (float)(s[0] / l); N[3 * v + 1] = (float)(s[1] / l); N[3 * v + 2] = (float)(s[2] / l); }
      else { N[3 * v] = 0; N[3 * v + 1] = 0; N[3 * v + 2] = 1; }
    }
  });
}

// ---- chart clustering ---------------------------------------------------------------------------------------------------
struct ChartStats { int iters = 0; int64_t charts = 0, adj = 0; double t_adj = 0, t_merge = 0; };

// Bottom-up merging of adjacent charts (every face starts as a chart). Each round prices every pair of adjacent charts
// (half-angle of the merged normal cone + 0.1 * merged area + 1e-4 * merged perimeter^2 / merged area) and merges the
// pairs that are each other's cheapest pair and cost <= threshold; normal cones are recomputed exactly after a merge.
inline int64_t compute_charts(const float* V, const int32_t* F, int64_t nv, int64_t nf, Arr<int32_t>& chart, ChartStats& st,
                              float thresh = 1.5707964f, float w_area = 0.1f, float w_perim = 1e-4f) {
  double t0 = now();
  Arr<float> FN(nf * 3);
  std::vector<double> area(nf), nsum((size_t)nf * 3);
  par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
    for (int64_t f = a; f < b; f++) {
      V3 p0 = ld(V + 3 * (int64_t)F[3 * f]), p1 = ld(V + 3 * (int64_t)F[3 * f + 1]), p2 = ld(V + 3 * (int64_t)F[3 * f + 2]);
      V3 n = cross(p1 - p0, p2 - p0); float l = norm(n);
      if (l > 0) n = n * (1.f / l); else n = {0, 0, 0};
      FN[3 * f] = n.x; FN[3 * f + 1] = n.y; FN[3 * f + 2] = n.z;
      area[f] = 0.5 * l; nsum[3 * f] = n.x; nsum[3 * f + 1] = n.y; nsum[3 * f + 2] = n.z;
    }
  });
  // manifold face adjacency
  std::vector<uint64_t> ek; std::vector<double> el;
  {
    Arr<uint64_t> k(nf * 3); Arr<uint32_t> fv(nf * 3);
    par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
      for (int64_t f = a; f < b; f++) for (int j = 0; j < 3; j++) {
        int32_t u = F[3 * f + j], v = F[3 * f + (j + 1) % 3];
        k[3 * f + j] = ((uint64_t)std::min(u, v) << 32) | (uint32_t)std::max(u, v); fv[3 * f + j] = (uint32_t)f;
      }
    });
    radix_sort_kv(k.p, fv.p, nf * 3, 64);
    int64_t n = nf * 3;
    for (int64_t i = 0; i < n;) {
      int64_t j = i + 1; while (j < n && k[j] == k[i]) j++;
      if (j - i == 2 && fv[i] != fv[i + 1]) {
        uint32_t a = std::min(fv[i], fv[i + 1]), b = std::max(fv[i], fv[i + 1]);
        ek.push_back(((uint64_t)a << 32) | b);
        el.push_back((double)norm(ld(V + 3 * (int64_t)(k[i] >> 32)) - ld(V + 3 * (int64_t)(uint32_t)k[i])));
      }
      i = j;
    }
  }
  // sort the adjacency by (c0, c1); two faces can share more than one edge
  auto reduce = [&]() {
    int64_t n = (int64_t)ek.size();
    std::vector<uint32_t> idx((size_t)n); std::iota(idx.begin(), idx.end(), 0u);
    radix_sort_kv(ek.data(), idx.data(), n, 64);
    std::vector<double> l2((size_t)n); for (int64_t i = 0; i < n; i++) l2[i] = el[idx[i]];
    int64_t o = 0;
    for (int64_t i = 0; i < n;) { int64_t j = i; double s = 0; while (j < n && ek[j] == ek[i]) s += l2[j++]; ek[o] = ek[i]; el[o] = s; o++; i = j; }
    ek.resize((size_t)o); el.resize((size_t)o);
  };
  reduce();
  st.adj = (int64_t)ek.size();
  st.t_adj = now() - t0; t0 = now();

  int64_t nc = nf;
  chart.alloc(nf);
  par_for(nf, 1 << 18, [&](int64_t a, int64_t b) { for (int64_t f = a; f < b; f++) chart[f] = (int32_t)f; });
  std::vector<float> half((size_t)nf, 0.f), axis((size_t)nf * 3);
  for (int64_t i = 0; i < nf * 3; i++) axis[i] = FN[i];
  std::vector<double> perim; std::vector<uint64_t> cmin, pack; std::vector<int32_t> remap; std::vector<uint8_t> dirty;
  while (true) {
    int64_t ne = (int64_t)ek.size();
    perim.assign((size_t)nc, 0.0);
    for (int64_t e = 0; e < ne; e++) { perim[ek[e] >> 32] += el[e]; perim[(uint32_t)ek[e]] += el[e]; }
    cmin.assign((size_t)nc, ~0ull); pack.resize((size_t)ne);
    par_for(ne, 1 << 14, [&](int64_t a, int64_t b) {
      for (int64_t e = a; e < b; e++) {
        uint32_t c0 = (uint32_t)(ek[e] >> 32), c1 = (uint32_t)ek[e];
        float d = axis[3 * (size_t)c0] * axis[3 * (size_t)c1] + axis[3 * (size_t)c0 + 1] * axis[3 * (size_t)c1 + 1] + axis[3 * (size_t)c0 + 2] * axis[3 * (size_t)c1 + 2];
        float ang = std::acos(std::min(std::max(d, -1.f), 1.f));
        float lo = std::min(-half[c0], ang - half[c1]), hi = std::max(half[c0], ang + half[c1]);
        float na = (float)(area[c0] + area[c1]), np = (float)(perim[c0] + perim[c1] - 2.0 * el[e]);
        float c = (hi - lo) * 0.5f + w_area * na + w_perim * np * np / std::max(na, 1e-30f);
        if (!(c >= 0)) c = INFINITY;
        uint32_t bits; memcpy(&bits, &c, 4);
        uint64_t pk = ((uint64_t)bits << 32) | (uint32_t)e; pack[e] = pk;
        amin64(&cmin[c0], pk); amin64(&cmin[c1], pk);
      }
    });
    remap.resize((size_t)nc); for (int64_t c = 0; c < nc; c++) remap[c] = (int32_t)c;
    int64_t merged = 0;
    for (int64_t e = 0; e < ne; e++) {
      uint32_t c0 = (uint32_t)(ek[e] >> 32), c1 = (uint32_t)ek[e];
      float c; uint32_t bits = (uint32_t)(pack[e] >> 32); memcpy(&c, &bits, 4);
      if (c <= thresh && cmin[c0] == pack[e] && cmin[c1] == pack[e]) { remap[c1] = (int32_t)c0; merged++; }
    }
    if (!merged) break;
    st.iters++;
    // merge chart data, compress ids
    dirty.assign((size_t)nc, 0);
    for (int64_t c = 0; c < nc; c++) if (remap[c] != c) {
      int32_t r = remap[c]; area[r] += area[c]; for (int k = 0; k < 3; k++) nsum[3 * (size_t)r + k] += nsum[3 * (size_t)c + k]; dirty[r] = 1;
    }
    int64_t n2 = 0; std::vector<int32_t> nid((size_t)nc);
    for (int64_t c = 0; c < nc; c++) if (remap[c] == c) nid[c] = (int32_t)n2++;
    for (int64_t c = 0; c < nc; c++) if (remap[c] != c) nid[c] = nid[remap[c]];
    for (int64_t c = 0; c < nc; c++) if (remap[c] == c) {
      int32_t d = nid[c]; area[d] = area[c]; half[d] = dirty[c] ? 0.f : half[c]; dirty[d] = dirty[c];
      for (int k = 0; k < 3; k++) { nsum[3 * (size_t)d + k] = nsum[3 * (size_t)c + k]; if (!dirty[c]) axis[3 * (size_t)d + k] = axis[3 * (size_t)c + k]; }
      if (dirty[c]) {
        double x = nsum[3 * (size_t)d], y = nsum[3 * (size_t)d + 1], z = nsum[3 * (size_t)d + 2], l = std::sqrt(x * x + y * y + z * z);
        if (l > 0) { axis[3 * (size_t)d] = (float)(x / l); axis[3 * (size_t)d + 1] = (float)(y / l); axis[3 * (size_t)d + 2] = (float)(z / l); }
        else { axis[3 * (size_t)d] = 0; axis[3 * (size_t)d + 1] = 0; axis[3 * (size_t)d + 2] = 0; }
      }
    }
    nc = n2;
    uint32_t* hb = (uint32_t*)half.data();
    par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
      for (int64_t f = a; f < b; f++) {
        int32_t c = nid[chart[f]]; chart[f] = c;
        if (!dirty[c]) continue;
        float d = axis[3 * (size_t)c] * FN[3 * f] + axis[3 * (size_t)c + 1] * FN[3 * f + 1] + axis[3 * (size_t)c + 2] * FN[3 * f + 2];
        float ang = std::acos(std::min(std::max(d, -1.f), 1.f));
        uint32_t bits; memcpy(&bits, &ang, 4);
        uint32_t cur = __atomic_load_n(&hb[c], __ATOMIC_RELAXED);
        while (bits > cur && !__atomic_compare_exchange_n(&hb[c], &cur, bits, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
      }
    });
    // contract the adjacency
    int64_t o = 0;
    for (int64_t e = 0; e < ne; e++) {
      uint32_t c0 = (uint32_t)nid[ek[e] >> 32], c1 = (uint32_t)nid[(uint32_t)ek[e]];
      if (c0 == c1) continue;
      ek[o] = ((uint64_t)std::min(c0, c1) << 32) | std::max(c0, c1); el[o] = el[e]; o++;
    }
    ek.resize((size_t)o); el.resize((size_t)o);
    reduce();
  }
  st.charts = nc; st.t_merge = now() - t0;
  return nc;
}

// ---- xatlas in parallel groups ---------------------------------------------------------------------------------------------
struct UnwrapStats { ChartStats charts; int64_t pieces = 0; int groups = 0, W = 0, H = 0; double t_charts = 0, t_pieces = 0, t_xatlas = 0, t_pack = 0; };

// -> vmap [NV'] rows of V, faces [NF', 3] into vmap rows, uv [NV', 2] in [0, 1]
inline void uv_unwrap(const float* V, const int32_t* F, int64_t nv, int64_t nf, Arr<int32_t>& vmap, Arr<int32_t>& faces, Arr<float>& uvs, UnwrapStats& st,
                      int ngroups = 64, int64_t max_faces = 1024, int threads = 0) {
  double t0 = now();
  Arr<int32_t> chart; int64_t nc = compute_charts(V, F, nv, nf, chart, st.charts);
  st.t_charts = now() - t0; t0 = now();
  // faces by chart
  std::vector<uint32_t> fo((size_t)nc + 1, 0); std::vector<int32_t> cf((size_t)nf);
  for (int64_t f = 0; f < nf; f++) fo[chart[f] + 1]++;
  for (int64_t c = 0; c < nc; c++) fo[c + 1] += fo[c];
  { std::vector<uint32_t> cur(fo.begin(), fo.end() - 1); for (int64_t f = 0; f < nf; f++) cf[cur[chart[f]]++] = (int32_t)f; }

  struct Piece { std::vector<float> P; std::vector<uint32_t> I; std::vector<int32_t> gmap; double area = 0; };
  std::vector<std::vector<Piece>> per_chart((size_t)nc);
  par_chunks(nc, 64, [&](int, int64_t cb, int64_t ce) {
    std::vector<int32_t> loc((size_t)nv, -1);
    for (int64_t c = cb; c < ce; c++) {
      const int64_t n = fo[c + 1] - fo[c];
      std::vector<std::vector<int32_t>> leaves;
      std::vector<int32_t> all(cf.begin() + fo[c], cf.begin() + fo[c + 1]);
      if (max_faces <= 0 || n <= max_faces) leaves.push_back(std::move(all));
      else {
        std::vector<std::vector<int32_t>> stack; stack.push_back(std::move(all));
        auto cen = [&](int32_t f, int k) { return V[3 * (int64_t)F[3 * (int64_t)f] + k] + V[3 * (int64_t)F[3 * (int64_t)f + 1] + k] + V[3 * (int64_t)F[3 * (int64_t)f + 2] + k]; };
        while (!stack.empty()) {
          auto cur = std::move(stack.back()); stack.pop_back();
          if ((int64_t)cur.size() <= max_faces) { leaves.push_back(std::move(cur)); continue; }
          float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
          for (int32_t f : cur) for (int k = 0; k < 3; k++) { float x = cen(f, k); lo[k] = std::min(lo[k], x); hi[k] = std::max(hi[k], x); }
          int ax = 0; for (int k = 1; k < 3; k++) if (hi[k] - lo[k] > hi[ax] - lo[ax]) ax = k;
          auto mid = cur.begin() + cur.size() / 2;
          std::nth_element(cur.begin(), mid, cur.end(), [&](int32_t x, int32_t y) { float a = cen(x, ax), b = cen(y, ax); return a < b || (a == b && x < y); });
          stack.emplace_back(cur.begin(), mid); stack.emplace_back(mid, cur.end());
        }
      }
      for (auto& lf : leaves) {
        Piece pc;
        for (int32_t f : lf) {
          for (int k = 0; k < 3; k++) {
            int32_t gv = F[3 * (int64_t)f + k]; int32_t& l = loc[gv];
            if (l < 0) { l = (int32_t)pc.gmap.size(); pc.gmap.push_back(gv); pc.P.insert(pc.P.end(), V + 3 * (int64_t)gv, V + 3 * (int64_t)gv + 3); }
            pc.I.push_back((uint32_t)l);
          }
          const float* p0 = V + 3 * (int64_t)F[3 * (int64_t)f]; const float* p1 = V + 3 * (int64_t)F[3 * (int64_t)f + 1]; const float* p2 = V + 3 * (int64_t)F[3 * (int64_t)f + 2];
          double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2], vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
          double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
          pc.area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }
        for (int32_t gv : pc.gmap) loc[gv] = -1;
        per_chart[c].push_back(std::move(pc));
      }
    }
  });
  std::vector<Piece> pieces;
  for (auto& v : per_chart) for (auto& pc : v) pieces.push_back(std::move(pc));
  per_chart.clear();
  const int64_t np = (int64_t)pieces.size();
  st.pieces = np;

  // shared texelsPerUnit: xatlas's packCharts estimate (resolution 1024 at 75% utilization) over the whole mesh
  double total = 0; for (auto& pc : pieces) total += pc.area;
  const float tpu = (float)std::sqrt((1024.0 * 1024.0) / std::max(1.0, total / 0.75));
  // deal pieces to groups round-robin in decreasing area order
  int G = (int)std::max<int64_t>(1, std::min<int64_t>(ngroups, np));
  std::vector<int64_t> order((size_t)np); std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return pieces[a].area > pieces[b].area; });
  std::vector<std::vector<int64_t>> groups((size_t)G);
  for (int64_t i = 0; i < np; i++) groups[i % G].push_back(order[i]);
  st.groups = G;
  st.t_pieces = now() - t0; t0 = now();

  struct GroupOut { std::vector<int32_t> vmap, faces; std::vector<float> uv; int w = 0, h = 0; std::string err; };
  std::vector<GroupOut> out((size_t)G);
  t2r_xa_thread_override.store(2);  // xatlas needs >= 2 (its single worker uses thread index 1)
  {
    std::atomic<int> next{0};
    int T = std::max(1, std::min(threads > 0 ? threads : nthreads(), G));
    std::vector<std::thread> pool;
    for (int t = 0; t < T; t++) pool.emplace_back([&] {
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
      for (int g; (g = next.fetch_add(1)) < G;) {
        auto& o = out[g];
        t2r_xatlas::Atlas* atlas = t2r_xatlas::Create();
        for (int64_t pi : groups[g]) {
          const Piece& pc = pieces[pi];
          t2r_xatlas::MeshDecl d;
          d.vertexCount = (uint32_t)pc.gmap.size();
          d.vertexPositionData = pc.P.data();
          d.vertexPositionStride = sizeof(float) * 3;
          d.indexCount = (uint32_t)pc.I.size();
          d.indexData = pc.I.data();
          d.indexFormat = t2r_xatlas::IndexFormat::UInt32;
          auto r = t2r_xatlas::AddMesh(atlas, d, (uint32_t)groups[g].size());
          if (r != t2r_xatlas::AddMeshError::Success) { o.err = t2r_xatlas::StringForEnum(r); break; }
        }
        if (o.err.empty()) {
          t2r_xatlas::AddMeshJoin(atlas);
          t2r_xatlas::ChartOptions co;
          t2r_xatlas::ComputeCharts(atlas, co);
          t2r_xatlas::PackOptions po;
          po.texelsPerUnit = tpu;
          t2r_xatlas::PackCharts(atlas, po);
          o.w = (int)atlas->width; o.h = (int)atlas->height;
          for (uint32_t m = 0; m < atlas->meshCount; m++) {
            const auto& mesh = atlas->meshes[m];
            int32_t base = (int32_t)(o.vmap.size());
            const Piece& pc = pieces[groups[g][m]];
            for (uint32_t i = 0; i < mesh.vertexCount; i++) {
              const auto& v = mesh.vertexArray[i];
              o.vmap.push_back(pc.gmap[v.xref]);
              o.uv.push_back(v.uv[0]); o.uv.push_back(v.uv[1]);
            }
            for (uint32_t i = 0; i < mesh.indexCount; i++) o.faces.push_back(base + (int32_t)mesh.indexArray[i]);
          }
        }
        t2r_xatlas::Destroy(atlas);
      }
    });
    for (auto& th : pool) th.join();
  }
  t2r_xa_thread_override.store(0);
  for (auto& o : out) if (!o.err.empty()) { fprintf(stderr, "glb: xatlas AddMesh failed: %s\n", o.err.c_str()); exit(1); }
  st.t_xatlas = now() - t0; t0 = now();

  // shelf-pack the group atlases (tallest first), 1 texel apart; try a range of shelf widths, keep the squarest fit
  std::vector<int> gord((size_t)G); std::iota(gord.begin(), gord.end(), 0);
  std::stable_sort(gord.begin(), gord.end(), [&](int a, int b) { return out[a].h != out[b].h ? out[a].h > out[b].h : out[a].w > out[b].w; });
  double sumA = 0; int maxW = 0;
  for (auto& o : out) { sumA += (double)(o.w + 1) * (o.h + 1); maxW = std::max(maxW, o.w + 1); }
  std::vector<int> ox((size_t)G), oy((size_t)G), bx((size_t)G), by((size_t)G); int W = 0, H = 0; double bestA = 1e300;
  for (int k = 0; k <= 64; k++) {
    int rowW = std::max(maxW, (int)std::ceil(std::sqrt(sumA) * (0.7 + 0.02 * k)));
    int x = 0, y = 0, rowH = 0, w = 0;
    for (int g : gord) {
      if (x > 0 && x + out[g].w + 1 > rowW) { y += rowH; x = 0; rowH = 0; }
      bx[g] = x; by[g] = y; x += out[g].w + 1; rowH = std::max(rowH, out[g].h + 1); w = std::max(w, x);
    }
    int h = y + rowH;
    // the texture is square and UVs are normalized per axis: texel density is set by the longer side
    double score = (double)std::max(w, h) * 1e6 + (double)std::min(w, h);
    if (score < bestA) { bestA = score; W = w; H = h; ox = bx; oy = by; }
  }
  st.W = W; st.H = H;
  int64_t NV = 0, NF = 0; std::vector<int64_t> vbase((size_t)G), fbase((size_t)G);
  for (int g = 0; g < G; g++) { vbase[g] = NV; fbase[g] = NF; NV += (int64_t)out[g].vmap.size(); NF += (int64_t)out[g].faces.size() / 3; }
  vmap.alloc(NV); faces.alloc(NF * 3); uvs.alloc(NV * 2);
  par_for(G, 1, [&](int64_t b, int64_t e) {
    for (int64_t g = b; g < e; g++) {
      const auto& o = out[g];
      for (size_t i = 0; i < o.vmap.size(); i++) {
        vmap[vbase[g] + (int64_t)i] = o.vmap[i];
        uvs[2 * (vbase[g] + (int64_t)i)] = (o.uv[2 * i] + (float)ox[g]) / (float)W;
        uvs[2 * (vbase[g] + (int64_t)i) + 1] = (o.uv[2 * i + 1] + (float)oy[g]) / (float)H;
      }
      for (size_t i = 0; i < o.faces.size(); i++) faces[3 * fbase[g] + (int64_t)i] = (int32_t)(o.faces[i] + vbase[g]);
    }
  });
  st.t_pack = now() - t0;
}

}  // namespace glb
