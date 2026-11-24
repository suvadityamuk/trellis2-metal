// Mesh simplification: parallel QEM edge collapse, after CuMesh's simplify (MIT, JeffreyXiang/CuMesh src/simplify.cu):
// every pass prices each edge (plane-distance quadric of the faces around both endpoints + lambda_edge_length * len^2 +
// lambda_skinny * mean skinniness * len^2, infinite if a surrounding triangle would flip), then collapses the edges
// below the threshold that are the cheapest in their neighbourhood, all at once. The threshold starts at 1e-8 and is
// multiplied by 10 whenever a pass removes less than 1% of the faces.
//
// Differences from the CuMesh driver, none of which changes what a single collapse does:
//   - the quadric error is evaluated directly as the sum of squared distances to the planes of the incident faces
//     (same value as v^T (Q0 + Q1) v, without the cancellation of a float quadric);
//   - after the neighbourhood minima are collapsed, the selection is repeated among the candidates whose neighbourhood
//     was not touched (their prices are still exact), so one pricing pass removes several independent sets;
//   - the last pass keeps only the cheapest collapses needed to land on the face target instead of overshooting.
#pragma once
#include "remesh.h"

namespace glb {

struct SimplifyStats { int passes = 0; int64_t collapses = 0; double t_csr = 0, t_cost = 0, t_select = 0, t_apply = 0; };

inline void amin64(uint64_t* p, uint64_t v) {
  uint64_t c = __atomic_load_n(p, __ATOMIC_RELAXED);
  while (v < c && !__atomic_compare_exchange_n(p, &c, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

// vertex -> incident faces (sorted per vertex)
struct VF {
  Arr<uint32_t> off; Arr<int32_t> f;
  void build(const int32_t* F, int64_t nv, int64_t nf) {
    off.alloc(nv + 1);
    par_for(nv + 1, 1 << 18, [&](int64_t a, int64_t b) { memset(off.p + a, 0, (b - a) * 4); });
    par_for(nf * 3, 1 << 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) __atomic_fetch_add(&off.p[F[i]], 1u, __ATOMIC_RELAXED); });
    prefix_sum(off.p, nv); off[nv] = (uint32_t)(nf * 3);
    Arr<uint32_t> cur(nv);
    par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memcpy(cur.p + a, off.p + a, (b - a) * 4); });
    f.alloc(nf * 3);
    par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) for (int k = 0; k < 3; k++) f[__atomic_fetch_add(&cur.p[F[3 * i + k]], 1u, __ATOMIC_RELAXED)] = (int32_t)i;
    });
    par_for(nv, 1 << 14, [&](int64_t a, int64_t b) {
      for (int64_t v = a; v < b; v++) {
        int32_t* s = f.p + off[v]; int n = (int)(off[v + 1] - off[v]);
        if (n <= 16) { for (int i = 1; i < n; i++) { int32_t x = s[i]; int j = i - 1; while (j >= 0 && s[j] > x) { s[j + 1] = s[j]; j--; } s[j + 1] = x; } }
        else std::sort(s, s + n);
      }
    });
  }
};

// sorted neighbour multiset of v (the two other vertices of every incident face)
struct NbrBuf {
  int32_t small[96]; std::vector<int32_t> big; int32_t* p = small; int n = 0;
  void gather(const VF& vf, const int32_t* F, int64_t v) {
    int nfv = (int)(vf.off[v + 1] - vf.off[v]);
    if (nfv * 2 > 96) { big.resize((size_t)nfv * 2); p = big.data(); } else p = small;
    n = 0;
    for (uint32_t i = vf.off[v]; i < vf.off[v + 1]; i++) { const int32_t* t = F + 3 * (int64_t)vf.f[i]; for (int k = 0; k < 3; k++) if (t[k] != v) p[n++] = t[k]; }
    if (n <= 24) { for (int i = 1; i < n; i++) { int32_t x = p[i]; int j = i - 1; while (j >= 0 && p[j] > x) { p[j + 1] = p[j]; j--; } p[j + 1] = x; } }
    else std::sort(p, p + n);
  }
};

template <class T, class Pred> inline int64_t par_compact(const T* in, int64_t n, Pred keep, Arr<T>& out) {
  const int C = 64; int64_t cnt[C + 1] = {0};
  par_chunks(n, C, [&](int c, int64_t a, int64_t b) { int64_t s = 0; for (int64_t i = a; i < b; i++) s += keep(i) ? 1 : 0; cnt[c + 1] = s; });
  for (int c = 0; c < C; c++) cnt[c + 1] += cnt[c];
  out.alloc(cnt[C]);
  par_chunks(n, C, [&](int c, int64_t a, int64_t b) { int64_t o = cnt[c]; for (int64_t i = a; i < b; i++) if (keep(i)) out[o++] = in[i]; });
  return cnt[C];
}

inline void simplify(Mesh& m, int64_t target, SimplifyStats& st, float lam_len = 1e-2f, float lam_skinny = 1e-3f, bool verbose = false) {
  float thresh = 1e-8f;
  while (m.nf > target) {
    double t0 = now();
    const int64_t nv = m.nv, nf = m.nf; float* V = m.V.p; int32_t* F = m.F.p;
    VF vf; vf.build(F, nv, nf);
    Arr<float> FN(nf * 3);
    par_for(nf, 1 << 15, [&](int64_t a, int64_t b) {
      for (int64_t f = a; f < b; f++) {
        V3 p0 = ld(V + 3 * (int64_t)F[3 * f]), p1 = ld(V + 3 * (int64_t)F[3 * f + 1]), p2 = ld(V + 3 * (int64_t)F[3 * f + 2]);
        V3 n = cross(p1 - p0, p2 - p0); float l = norm(n);
        if (l > 0) n = n * (1.f / l); else n = {0, 0, 0};
        FN[3 * f] = n.x; FN[3 * f + 1] = n.y; FN[3 * f + 2] = n.z;
      }
    });
    // edges owned by their lower vertex; boundary vertices
    Arr<uint32_t> eoff(nv + 1); Arr<uint8_t> bnd(nv);
    par_for(nv, 1 << 13, [&](int64_t a, int64_t b) {
      NbrBuf nb;
      for (int64_t v = a; v < b; v++) {
        nb.gather(vf, F, v);
        uint32_t c = 0; uint8_t bd = 0;
        for (int i = 0; i < nb.n;) { int j = i + 1; while (j < nb.n && nb.p[j] == nb.p[i]) j++; if (j - i == 1) bd = 1; if (nb.p[i] > v) c++; i = j; }
        eoff[v] = c; bnd[v] = bd;
      }
    });
    int64_t ne = prefix_sum(eoff.p, nv); eoff[nv] = (uint32_t)ne;
    st.t_csr += now() - t0; t0 = now();

    auto newpos = [&](int32_t a, int32_t b) {
      V3 pa = ld(V + 3 * (int64_t)a), pb = ld(V + 3 * (int64_t)b);
      if (bnd[a] != bnd[b]) return bnd[a] ? pa : pb;
      return (pa + pb) * 0.5f;
    };
    auto cost = [&](int32_t a, int32_t b) -> float {
      V3 p = newpos(a, b);
      float err = 0, sk = 0; int ns = 0;
      for (int side = 0; side < 2; side++) {
        int32_t v = side ? b : a, o = side ? a : b;
        for (uint32_t i = vf.off[v]; i < vf.off[v + 1]; i++) {
          int64_t f = vf.f[i]; const int32_t* t = F + 3 * f;
          V3 n = ld(FN.p + 3 * f);
          float d = dot(n, p - ld(V + 3 * (int64_t)t[0])); err += d * d;
          if (t[0] == o || t[1] == o || t[2] == o) continue;
          V3 q0 = t[0] == v ? p : ld(V + 3 * (int64_t)t[0]), q1 = t[1] == v ? p : ld(V + 3 * (int64_t)t[1]), q2 = t[2] == v ? p : ld(V + 3 * (int64_t)t[2]);
          V3 nn = cross(q1 - q0, q2 - q0);
          if (dot(nn, n) < 0) return INFINITY;
          V3 e0 = q1 - q0, e1 = q2 - q1, e2 = q0 - q2;
          float s = dot(e0, e0) + dot(e1, e1) + dot(e2, e2);
          sk += 1.f - std::min(std::max(6.9282032f * (0.5f * norm(nn)) / std::max(s, 1e-12f), 0.f), 1.f); ns++;
        }
      }
      V3 ab = ld(V + 3 * (int64_t)b) - ld(V + 3 * (int64_t)a); float len2 = dot(ab, ab);
      return err + lam_len * len2 + lam_skinny * (ns ? sk / (float)ns : 0.f) * len2;
    };

    Arr<int32_t> eu(ne), ew(ne); Arr<uint64_t> epack(ne); Arr<uint64_t> vmin(nv);
    par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memset(vmin.p + a, 0xff, (b - a) * 8); });
    par_for(nv, 1 << 12, [&](int64_t a, int64_t b) {
      NbrBuf nb;
      for (int64_t v = a; v < b; v++) {
        if (eoff[v + 1] == eoff[v]) continue;
        nb.gather(vf, F, v);
        int64_t e = eoff[v];
        for (int i = 0; i < nb.n;) {
          int j = i + 1; while (j < nb.n && nb.p[j] == nb.p[i]) j++;
          int32_t w = nb.p[i]; i = j;
          if (w <= v) continue;
          float c = cost((int32_t)v, w);
          uint32_t bits; memcpy(&bits, &c, 4);
          uint64_t pk = ((uint64_t)bits << 32) | (uint32_t)e;
          eu[e] = (int32_t)v; ew[e] = w; epack[e] = pk;
          if (c <= thresh) { amin64(&vmin.p[v], pk); amin64(&vmin.p[w], pk); }
          e++;
        }
      }
    });
    st.t_cost += now() - t0; t0 = now();

    // candidates, then repeated neighbourhood-minimum selection
    Arr<uint32_t> cand;
    { Arr<uint32_t> ids(ne); par_for(ne, 1 << 18, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) ids[i] = (uint32_t)i; });
      par_compact(ids.p, ne, [&](int64_t i) { float c; uint32_t bits = (uint32_t)(epack[i] >> 32); memcpy(&c, &bits, 4); return c <= thresh; }, cand); }
    Arr<uint8_t> touched(nv);
    par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memset(touched.p + a, 0, (b - a)); });
    std::vector<uint32_t> selected;
    auto ringMin = [&](int32_t v, uint64_t pk) {
      for (uint32_t i = vf.off[v]; i < vf.off[v + 1]; i++) { const int32_t* t = F + 3 * (int64_t)vf.f[i]; if (vmin[t[0]] < pk || vmin[t[1]] < pk || vmin[t[2]] < pk) return false; }
      return true;
    };
    auto touchRing = [&](int32_t v) {
      touched[v] = 1;
      for (uint32_t i = vf.off[v]; i < vf.off[v + 1]; i++) { const int32_t* t = F + 3 * (int64_t)vf.f[i]; touched[t[0]] = 1; touched[t[1]] = 1; touched[t[2]] = 1; }
    };
    for (int round = 0; round < 8 && cand.n > 0; round++) {
      int64_t nc = cand.n;
      Arr<uint8_t> sel(nc);
      par_for(nc, 1 << 13, [&](int64_t a, int64_t b) {
        for (int64_t i = a; i < b; i++) { uint32_t e = cand[i]; uint64_t pk = epack[e]; sel[i] = ringMin(eu[e], pk) && ringMin(ew[e], pk); }
      });
      Arr<uint32_t> s; int64_t nsel = par_compact(cand.p, nc, [&](int64_t i) { return sel[i] != 0; }, s);
      if (nsel == 0) break;
      selected.insert(selected.end(), s.p, s.p + nsel);
      par_for(nsel, 1 << 12, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) { touchRing(eu[s[i]]); touchRing(ew[s[i]]); } });
      Arr<uint32_t> live; par_compact(cand.p, nc, [&](int64_t i) { uint32_t e = cand[i]; return !touched[eu[e]] && !touched[ew[e]]; }, live);
      cand = std::move(live);
      if (nsel * 50 < (int64_t)selected.size()) break;  // diminishing returns: reprice instead
      par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memset(vmin.p + a, 0xff, (b - a) * 8); });
      par_for(cand.n, 1 << 14, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) { uint32_t e = cand[i]; amin64(&vmin.p[eu[e]], epack[e]); amin64(&vmin.p[ew[e]], epack[e]); } });
    }
    // land on the target: keep the cheapest collapses (each removes about two faces)
    int64_t need = (nf - target + 1) / 2;
    if ((int64_t)selected.size() > need) {
      std::nth_element(selected.begin(), selected.begin() + need, selected.end(), [&](uint32_t x, uint32_t y) { return epack[x] < epack[y]; });
      selected.resize((size_t)need);
    }
    st.t_select += now() - t0; t0 = now();

    int64_t nsel = (int64_t)selected.size();
    Arr<uint8_t> vdead(nv);
    par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { memset(vdead.p + a, 0, (b - a)); });
    par_for(nsel, 1 << 12, [&](int64_t lo, int64_t hi) {
      for (int64_t i = lo; i < hi; i++) {
        int32_t a = eu[selected[i]], b = ew[selected[i]];
        V3 p = newpos(a, b);
        V[3 * (int64_t)a] = p.x; V[3 * (int64_t)a + 1] = p.y; V[3 * (int64_t)a + 2] = p.z;
        for (uint32_t j = vf.off[b]; j < vf.off[b + 1]; j++) {
          int32_t* t = F + 3 * (int64_t)vf.f[j];
          if (t[0] == a || t[1] == a || t[2] == a) { t[0] = t[1] = t[2] = -1; continue; }
          for (int k = 0; k < 3; k++) if (t[k] == b) t[k] = a;
        }
        vdead[b] = 1;
      }
    });
    // compact
    Arr<uint32_t> vmap(nv + 1), fmap(nf + 1);
    par_for(nv, 1 << 18, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) vmap[i] = (vdead[i] || vf.off[i + 1] == vf.off[i]) ? 0 : 1; });
    int64_t nv2 = prefix_sum(vmap.p, nv); vmap[nv] = (uint32_t)nv2;
    par_for(nf, 1 << 18, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) fmap[i] = F[3 * i] >= 0 ? 1 : 0; });
    int64_t nf2 = prefix_sum(fmap.p, nf); fmap[nf] = (uint32_t)nf2;
    Arr<float> V2(nv2 * 3); Arr<int32_t> F2(nf2 * 3);
    par_for(nv, 1 << 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) if (vmap[i + 1] != vmap[i]) memcpy(V2.p + 3 * (int64_t)vmap[i], V + 3 * i, 12); });
    par_for(nf, 1 << 16, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; i++) if (fmap[i + 1] != fmap[i]) { int32_t* d = F2.p + 3 * (int64_t)fmap[i]; for (int k = 0; k < 3; k++) d[k] = (int32_t)vmap[F[3 * i + k]]; }
    });
    m.V = std::move(V2); m.F = std::move(F2); m.nv = nv2; m.nf = nf2;
    st.t_apply += now() - t0;
    st.passes++; st.collapses += nsel;
    if (verbose) fprintf(stderr, "  simplify pass %d: thresh %.0e edges %lld collapses %lld faces %lld -> %lld\n", st.passes, thresh, (long long)ne, (long long)nsel, (long long)nf, (long long)nf2);
    if ((double)(nf - nf2) < 1e-2 * (double)nf) {
      if (thresh > 1e20f) break;  // nothing left that can collapse
      thresh *= 10.f;
    }
  }
}

}  // namespace glb
