// Unit tests of the to_glb kernels against brute-force / CPU references.
//   build/glb_test [--mesh x.safetensors] [--dump dir] [--texture-size N] [--faces N]
// Synthetic tests always run; with --mesh the atlas / raster / bake / TELEA checks are repeated on a real pipeline
// output at example.py settings. --dump writes the pre- and post-inpaint textures for tools/check_telea.py (OpenCV).
// Exit code 0 only when every gate passes.
#include "pipeline.h"
#include "glb/to_glb.h"
#include <random>
#include <set>
#include <map>
#include <array>
#include <unordered_map>
#include <atomic>
using namespace glb;

static int g_fail = 0;
static void gate(bool ok, const char* name, const char* fmt, ...) {
  char buf[1024]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  printf("%s  %-28s %s\n", ok ? "PASS" : "FAIL", name, buf); fflush(stdout);
  if (!ok) g_fail++;
}

struct Rng { std::mt19937_64 g; explicit Rng(uint64_t s) : g(s) {} float u() { return (float)((g() >> 40) * (1.0 / 16777216.0)); } float r(float a, float b) { return a + (b - a) * u(); } };

// ---- LBVH nearest triangle vs brute force -------------------------------------------------------------------------------
static void lbvh_case(const char* name, const float* V, const int32_t* F, int64_t nf, const std::vector<float>& Q, float cap) {
  const int64_t m = (int64_t)Q.size() / 3;
  LBVH bvh; bvh.build(V, F, nf);
  Tensor q = gbuf<float>(m * 3); memcpy(q.ptr<float>(), Q.data(), (size_t)m * 12);
  Tensor dist, fid, uvw; bvh.query(q, m, dist, fid, &uvw, cap); tk::sync();
  const float* D = dist.ptr<float>(); const int32_t* I = fid.ptr<int32_t>(); const float* U = uvw.ptr<float>();
  std::vector<float> bd((size_t)m); std::vector<int32_t> bi((size_t)m);
  par_for(m, 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) { float d2; V3 w; brute_nearest(V, F, nf, ld(&Q[3 * i]), d2, bi[i], w); bd[i] = std::sqrt(d2); } });
  int64_t same_face = 0, exact = 0, bad = 0, capmiss = 0, cmp = 0; double maxerr = 0, maxrec = 0;
  for (int64_t i = 0; i < m; i++) {
    if (std::isfinite(cap)) {
      bool in = bd[i] <= cap;
      if (std::fabs(bd[i] - cap) < 1e-6f * (1 + cap)) continue;  // on the cap within rounding
      if ((I[i] >= 0) != in) { capmiss++; continue; }
      if (!in) continue;
    }
    if (I[i] < 0 || I[i] >= nf) { bad++; continue; }
    cmp++; double e = std::fabs((double)D[i] - bd[i]); maxerr = std::max(maxerr, e);
    exact += D[i] == bd[i]; same_face += I[i] == bi[i];
    // the returned (face, uvw) must reproduce the returned distance
    V3 a = ld(V + 3 * (int64_t)F[3 * (int64_t)I[i]]), b = ld(V + 3 * (int64_t)F[3 * (int64_t)I[i] + 1]), c = ld(V + 3 * (int64_t)F[3 * (int64_t)I[i] + 2]);
    V3 p = a * U[3 * i] + b * U[3 * i + 1] + c * U[3 * i + 2];
    maxrec = std::max(maxrec, (double)std::fabs(norm(ld(&Q[3 * i]) - p) - bd[i]));
    if (e > 3e-5) bad++;  // float32 with O(1) coordinates: a point on a face has d^2 ~ 1e-11 of rounding noise, so d is only good to ~1e-5 absolute (MSL dot() and the CPU sum in a different order)
  }
  gate(bad == 0 && capmiss == 0 && maxrec < 3e-5, name, "%lld tris, %lld queries with a hit: max |d_gpu - d_brute| %.2e, distance bit-equal %.2f%%, same face %.2f%% (rest are ties), "
       "max |dist(q, face*uvw) - d| %.2e, cap mismatches %lld", (long long)nf, (long long)cmp, maxerr, 100.0 * exact / std::max<int64_t>(cmp, 1), 100.0 * same_face / std::max<int64_t>(cmp, 1), maxrec, (long long)capmiss);
}

static void test_lbvh(const Tensor* rv, const Tensor* rf) {
  Rng r(1);
  // random soup: mixed sizes, some degenerate (zero-area, zero-length) triangles, duplicated triangles
  const int64_t nf = 30000; std::vector<float> V((size_t)nf * 9); std::vector<int32_t> F((size_t)nf * 3);
  for (int64_t f = 0; f < nf; f++) {
    float c[3] = {r.r(-0.5f, 0.5f), r.r(-0.5f, 0.5f), r.r(-0.5f, 0.5f)}; float s = f % 50 == 0 ? 0.3f : (f % 7 == 0 ? 0.002f : 0.03f);
    for (int k = 0; k < 3; k++) for (int j = 0; j < 3; j++) V[9 * f + 3 * k + j] = c[j] + r.r(-s, s);
    if (f % 101 == 0) memcpy(&V[9 * f + 6], &V[9 * f + 3], 12);       // edge
    if (f % 211 == 0) { memcpy(&V[9 * f + 3], &V[9 * f], 12); memcpy(&V[9 * f + 6], &V[9 * f], 12); }  // point
    if (f % 307 == 0 && f > 0) memcpy(&V[9 * f], &V[9 * (f - 1)], 36);  // duplicate
    for (int k = 0; k < 3; k++) F[3 * f + k] = (int32_t)(3 * f + k);
  }
  const int64_t m = 60000; std::vector<float> Q((size_t)m * 3);
  for (int64_t i = 0; i < m; i++) {
    if (i % 3 == 0) for (int j = 0; j < 3; j++) Q[3 * i + j] = r.r(-0.7f, 0.7f);                 // anywhere (also outside the cube)
    else if (i % 3 == 1) { int64_t f = (int64_t)(r.u() * nf) % nf; for (int j = 0; j < 3; j++) Q[3 * i + j] = V[9 * f + j] + r.r(-1e-3f, 1e-3f); }  // near a vertex
    else { int64_t f = (int64_t)(r.u() * nf) % nf; float a = r.u(), b = r.u() * (1 - a); for (int j = 0; j < 3; j++) Q[3 * i + j] = V[9 * f + j] * a + V[9 * f + 3 + j] * b + V[9 * f + 6 + j] * (1 - a - b); }  // on a face
  }
  lbvh_case("lbvh.random_soup", V.data(), F.data(), nf, Q, INFINITY);
  lbvh_case("lbvh.random_soup_cap", V.data(), F.data(), nf, Q, 0.01f);
  { // one and two triangles (degenerate tree shapes)
    std::vector<float> q(Q.begin(), Q.begin() + 3000);
    lbvh_case("lbvh.single_triangle", V.data(), F.data(), 1, q, INFINITY);
    lbvh_case("lbvh.two_triangles", V.data(), F.data(), 2, q, INFINITY);
  }
  if (rv) {  // the real mesh: queries at vertices pushed off the surface (what the remesh and the bake ask)
    const float* MV = rv->ptr<float>(); const int32_t* MF = rf->ptr<int32_t>(); int64_t mnv = rv->shape[0], mnf = rf->shape[0];
    const int64_t mq = 1500; std::vector<float> q((size_t)mq * 3);
    for (int64_t i = 0; i < mq; i++) { int64_t v = (int64_t)(r.u() * mnv) % mnv; float s = i % 2 ? 2e-3f : 2e-2f; for (int j = 0; j < 3; j++) q[3 * i + j] = MV[3 * v + j] + r.r(-s, s); }
    lbvh_case("lbvh.real_mesh", MV, MF, mnf, q, INFINITY);
  }
}

// ---- UV rasterizer vs CPU reference ---------------------------------------------------------------------------------------
static double edge_dist(const float* uv, const int32_t* F, uint32_t id, int W, int H, double cx, double cy) {
  if (!id) return 1e30;
  double best = 1e30; const int32_t* t = F + 3 * (int64_t)(id - 1);
  for (int k = 0; k < 3; k++) {
    double ax = (double)uv[2 * t[k]] * W, ay = (double)uv[2 * t[k] + 1] * H, bx = (double)uv[2 * t[(k + 1) % 3]] * W, by = (double)uv[2 * t[(k + 1) % 3] + 1] * H;
    double ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey, s = l2 > 0 ? std::min(1.0, std::max(0.0, ((cx - ax) * ex + (cy - ay) * ey) / l2)) : 0;
    best = std::min(best, std::hypot(cx - (ax + s * ex), cy - (ay + s * ey)));
  }
  return best;
}
// returns mismatching texels; maxd = largest distance of a mismatching texel center from the nearest edge of either candidate face (texels)
static int64_t raster_compare(const float* uv, int64_t nvert, const int32_t* F, int64_t nf, int W, int H, int64_t& covered, double& maxd, std::vector<uint8_t>* cnt = nullptr) {
  Tensor tuv = gbuf<float>(nvert * 2), tf = gbuf<int32_t>(nf * 3);
  memcpy(tuv.ptr<float>(), uv, (size_t)nvert * 8); memcpy(tf.ptr<int32_t>(), F, (size_t)nf * 12);
  Tensor ids = rasterize_uv(tuv, tf, nf, W, H); tk::sync();
  std::vector<uint32_t> ref((size_t)W * H); if (cnt) cnt->resize((size_t)W * H);
  rasterize_uv_cpu(uv, F, nf, W, H, ref.data(), cnt ? cnt->data() : nullptr);
  const uint32_t* G = ids.ptr<uint32_t>(); int64_t mis = 0; covered = 0; maxd = 0;
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
    size_t i = (size_t)y * W + x; covered += ref[i] != 0;
    if (G[i] == ref[i]) continue;
    mis++;
    maxd = std::max(maxd, std::min(edge_dist(uv, F, G[i], W, H, x + 0.5, y + 0.5), edge_dist(uv, F, ref[i], W, H, x + 0.5, y + 0.5)));
  }
  return mis;
}

static void test_raster() {
  Rng r(2);
  { // exact case: a grid of quads whose vertices sit on texel centers (every edge passes through texel centers -> pure fill-rule test)
    const int W = 256, H = 256, n = 32; std::vector<float> uv; std::vector<int32_t> F;
    for (int j = 0; j <= n; j++) for (int i = 0; i <= n; i++) { uv.push_back((float)(i * 7 + 0.5) / W); uv.push_back((float)(j * 7 + 0.5) / H); }
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
      int a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
      if ((i + j) % 2) { F.insert(F.end(), {a, b, d, a, d, c}); } else { F.insert(F.end(), {a, c, b, b, c, d}); }  // both diagonals, both windings
    }
    int64_t cov; double maxd; std::vector<uint8_t> cnt;
    int64_t mis = raster_compare(uv.data(), (int64_t)uv.size() / 2, F.data(), (int64_t)F.size() / 3, W, H, cov, maxd, &cnt);
    int64_t multi = 0; for (uint8_t c : cnt) multi += c > 1;
    gate(mis == 0 && multi == 0 && cov == (int64_t)(n * 7) * (n * 7), "raster.fill_rule_grid", "%lld faces on texel centers, %dx%d: %lld mismatching texels, %lld covered (expected %d), %lld texels hit twice",
         (long long)F.size() / 3, W, H, (long long)mis, (long long)cov, n * 7 * n * 7, (long long)multi);
  }
  { // random triangles: large, small, slivers, partly outside [0, 1], overlapping (draw order decides)
    const int W = 1024, H = 1024; const int64_t nf = 20000; std::vector<float> uv((size_t)nf * 6); std::vector<int32_t> F((size_t)nf * 3);
    for (int64_t f = 0; f < nf; f++) {
      float cx = r.r(-0.05f, 1.05f), cy = r.r(-0.05f, 1.05f), s = f % 40 == 0 ? 0.2f : (f % 3 == 0 ? 0.002f : 0.02f);
      for (int k = 0; k < 3; k++) { uv[6 * f + 2 * k] = cx + r.r(-s, s); uv[6 * f + 2 * k + 1] = cy + r.r(-s, s) * (f % 11 == 0 ? 0.01f : 1.f); F[3 * f + k] = (int32_t)(3 * f + k); }
    }
    int64_t cov; double maxd; int64_t mis = raster_compare(uv.data(), nf * 3, F.data(), nf, W, H, cov, maxd);
    gate(maxd < 1.0 / 64 && mis < cov / 500, "raster.random_triangles", "%lld faces, %dx%d: %lld of %lld covered texels differ (%.4f%%), all within %.4f texel of a triangle edge (Metal snaps vertices to 1/256 texel)",
         (long long)nf, W, H, (long long)mis, (long long)cov, 100.0 * mis / std::max<int64_t>(cov, 1), maxd);
  }
}

// ---- TELEA: GPU wavefront vs the CPU twin ---------------------------------------------------------------------------------
struct TeleaCmp { int64_t unknown = 0, diff = 0; int maxdiff = 0, levels = 0; double mean = 0; };
static TeleaCmp telea_compare(const uint8_t* img0, const int32_t* lv0, int W, int H, std::vector<uint8_t>* gpu_out = nullptr) {
  const int64_t P = (int64_t)W * H;
  Tensor img = gbuf<uint8_t>(P * 6), lv = gbuf<int32_t>(P);
  memcpy(img.ptr<uint8_t>(), img0, (size_t)P * 6);
  for (int64_t i = 0; i < P; i++) lv.ptr<int32_t>()[i] = lv0[i] == 0 ? 0 : -1;
  TeleaStats st; telea(lv, img, W, H, st);
  std::vector<uint8_t> c(img0, img0 + P * 6); std::vector<int32_t> l((size_t)P); for (int64_t i = 0; i < P; i++) l[i] = lv0[i] == 0 ? 0 : -1;
  std::vector<uint32_t> list; std::vector<int64_t> off; telea_levels(l.data(), W, H, list, off);
  telea_cpu(l.data(), c.data(), W, H, list, off);
  TeleaCmp r; r.unknown = (int64_t)list.size(); r.levels = st.levels;
  const uint8_t* g = img.ptr<uint8_t>(); double s = 0;
  for (int64_t i = 0; i < P * 6; i++) { int d = std::abs((int)g[i] - (int)c[i]); if (d) { r.diff++; r.maxdiff = std::max(r.maxdiff, d); s += d; } }
  r.mean = s / std::max<int64_t>(1, r.unknown * 6);
  bool known_ok = true; for (int64_t i = 0; i < P && known_ok; i++) if (lv0[i] == 0) for (int ch = 0; ch < 6; ch++) known_ok = known_ok && g[ch * P + i] == img0[ch * P + i];
  if (!known_ok) r.maxdiff = 999;  // known texels must be untouched
  bool lv_ok = memcmp(lv.ptr<int32_t>(), l.data(), (size_t)P * 4) == 0;
  if (!lv_ok) r.maxdiff = 998;
  if (gpu_out) gpu_out->assign(g, g + P * 6);
  return r;
}
static void write_bin(const std::string& path, const void* p, size_t n) { FILE* f = fopen(path.c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); } fwrite(p, 1, n, f); fclose(f); }
static void dump_telea(const std::string& dir, const char* tag, const uint8_t* img0, const int32_t* lv, const uint8_t* out, int W, int H) {
  const size_t P = (size_t)W * H; std::string b = dir + "/" + tag;
  write_bin(b + ".in.u8", img0, P * 6); write_bin(b + ".out.u8", out, P * 6); write_bin(b + ".lv.i32", lv, P * 4);
  FILE* f = fopen((b + ".json").c_str(), "w"); fprintf(f, "{\"W\": %d, \"H\": %d}\n", W, H); fclose(f);
}

static void test_telea(const std::string& dump) {
  Rng r(3); const int W = 768, H = 640; const int64_t P = (int64_t)W * H;
  std::vector<uint8_t> img((size_t)P * 6); std::vector<int32_t> lv((size_t)P, -1);
  // known region: random discs and thin strips (chart-like islands), smooth colour field plus noise; everything else unknown
  for (int k = 0; k < 220; k++) {
    float cx = r.r(0, (float)W), cy = r.r(0, (float)H), rad = r.r(2, 30), ex = k % 5 == 0 ? 6.f : 1.f;
    for (int y = std::max(0, (int)(cy - rad)); y < std::min(H, (int)(cy + rad) + 1); y++) for (int x = std::max(0, (int)(cx - rad * ex)); x < std::min(W, (int)(cx + rad * ex) + 1); x++)
      if ((x - cx) * (x - cx) / (ex * ex) + (y - cy) * (y - cy) <= rad * rad) lv[(size_t)y * W + x] = 0;
  }
  for (int64_t i = 0; i < P; i++) for (int c = 0; c < 6; c++) {
    int x = (int)(i % W), y = (int)(i / W);
    float v = 127 + 90 * std::sin(0.031f * x * (c + 1) + 0.017f * y) * std::cos(0.023f * y - 0.4f * c) + r.r(-25, 25);
    img[(size_t)c * P + i] = lv[i] == 0 ? (uint8_t)std::min(255.f, std::max(0.f, v)) : 0;
  }
  std::vector<uint8_t> out; TeleaCmp c = telea_compare(img.data(), lv.data(), W, H, &out);
  gate(c.maxdiff == 0 && c.diff == 0, "telea.gpu_vs_cpu_twin", "%dx%d, %lld unknown texels in %d levels: %lld of %lld channel values differ (%.4f%%), max |diff| %d, known texels untouched",
       W, H, (long long)c.unknown, c.levels, (long long)c.diff, (long long)c.unknown * 6, 100.0 * c.diff / std::max<int64_t>(1, c.unknown * 6), c.maxdiff);
  if (!dump.empty()) { std::vector<int32_t> l(lv); std::vector<uint32_t> list; std::vector<int64_t> off; telea_levels(l.data(), W, H, list, off); dump_telea(dump, "telea_synth", img.data(), l.data(), out.data(), W, H); }
  { // nothing known / everything known
    std::vector<int32_t> none((size_t)P, -1), all((size_t)P, 0);
    TeleaCmp a = telea_compare(img.data(), none.data(), W, H), b = telea_compare(img.data(), all.data(), W, H);
    gate(a.levels == 0 && b.levels == 0 && b.maxdiff == 0, "telea.trivial_masks", "no known texel: %d levels (nothing to propagate); all known: %d levels, max diff %d", a.levels, b.levels, b.maxdiff);
  }
}

// ---- atlas validity ---------------------------------------------------------------------------------------------------
struct AtlasRep { bool ok = true; };
static void check_atlas(const char* tag, const GlbOut& r, int W, int H, int64_t target_faces) {
  char nm[64];
  const Mesh& m = r.simplified; const int64_t NF = r.faces.n / 3, NV = r.vmap.n;
  // every simplified face appears exactly once (same vertices, same orientation)
  auto canon = [](int32_t a, int32_t b, int32_t c) { if (b < a && b <= c) { int32_t t = a; a = b; b = c; c = t; } else if (c < a && c < b) { int32_t t = a; a = c; c = b; b = t; } return std::array<int32_t, 3>{a, b, c}; };
  std::vector<std::array<int32_t, 3>> A((size_t)m.nf), B((size_t)NF);
  for (int64_t f = 0; f < m.nf; f++) A[f] = canon(m.F[3 * f], m.F[3 * f + 1], m.F[3 * f + 2]);
  bool idx_ok = true;
  for (int64_t f = 0; f < NF; f++) { int32_t t[3]; for (int k = 0; k < 3; k++) { int32_t v = r.faces[3 * f + k]; if (v < 0 || v >= NV) { idx_ok = false; v = 0; } t[k] = r.vmap[v]; } B[f] = canon(t[0], t[1], t[2]); }
  std::sort(A.begin(), A.end()); std::sort(B.begin(), B.end());
  float umin = 1e9f, umax = -1e9f; bool finite = true;
  for (int64_t i = 0; i < NV * 2; i++) { umin = std::min(umin, r.uv[i]); umax = std::max(umax, r.uv[i]); finite = finite && std::isfinite(r.uv[i]); }
  snprintf(nm, sizeof nm, "%s.atlas_all_faces_mapped", tag);
  gate(idx_ok && A == B && finite && umin >= 0 && umax <= 1 && m.nf <= target_faces, nm, "%lld simplified faces (target %lld) == %lld atlas faces as a multiset (orientation kept), %lld atlas verts, uv in [%.4f, %.4f]",
       (long long)m.nf, (long long)target_faces, (long long)NF, (long long)NV, umin, umax);
  // overlaps: texel centers covered by more than one face (exact CPU coverage count); and GPU raster vs CPU on the real atlas
  int64_t cov; double maxd; std::vector<uint8_t> cnt;
  int64_t mis = raster_compare(r.uv.p, NV, r.faces.p, NF, W, H, cov, maxd, &cnt);
  int64_t multi = 0; for (uint8_t c : cnt) multi += c > 1;
  snprintf(nm, sizeof nm, "%s.atlas_no_overlap", tag);
  gate(multi * 10000 <= cov, nm, "%dx%d: %lld of %lld covered texels are covered by more than one face (%.5f%%); coverage %.1f%% of the texture", W, H, (long long)multi, (long long)cov,
       100.0 * multi / std::max<int64_t>(1, cov), 100.0 * cov / ((double)W * H));
  snprintf(nm, sizeof nm, "%s.raster_gpu_vs_cpu", tag);
  gate(maxd < 1.0 / 64 && mis * 200 < cov, nm, "%lld of %lld covered texels differ (%.4f%%), all within %.4f texel of a face edge", (long long)mis, (long long)cov, 100.0 * mis / std::max<int64_t>(1, cov), maxd);
  // texel density: texels per unit of surface length, per face (area weighted), and faces that got no texel
  std::vector<std::pair<double, double>> d; d.reserve((size_t)NF); double a3 = 0, a2 = 0; int64_t flipped = 0, zero_uv = 0;
  for (int64_t f = 0; f < NF; f++) {
    const int32_t* t = r.faces.p + 3 * f;
    V3 p0 = ld(m.V.p + 3 * (int64_t)r.vmap[t[0]]), p1 = ld(m.V.p + 3 * (int64_t)r.vmap[t[1]]), p2 = ld(m.V.p + 3 * (int64_t)r.vmap[t[2]]);
    double A3 = 0.5 * norm(cross(p1 - p0, p2 - p0));
    double ux = ((double)r.uv[2 * t[1]] - r.uv[2 * t[0]]) * W, uy = ((double)r.uv[2 * t[1] + 1] - r.uv[2 * t[0] + 1]) * H, vx = ((double)r.uv[2 * t[2]] - r.uv[2 * t[0]]) * W, vy = ((double)r.uv[2 * t[2] + 1] - r.uv[2 * t[0] + 1]) * H;
    double A2 = 0.5 * (ux * vy - uy * vx);
    flipped += A2 < 0; zero_uv += A2 == 0;
    a3 += A3; a2 += std::fabs(A2);
    if (A3 > 0) d.push_back({std::sqrt(std::fabs(A2) / A3), A3});
  }
  std::sort(d.begin(), d.end());
  auto q = [&](double f) { double acc = 0; for (auto& e : d) { acc += e.second; if (acc >= f * a3) return e.first; } return d.back().first; };
  double q05 = q(0.05), q50 = q(0.5), q95 = q(0.95), q01 = q(0.01);
  std::vector<uint8_t> hit((size_t)NF, 0); { std::vector<uint32_t> ids((size_t)W * H); rasterize_uv_cpu(r.uv.p, r.faces.p, NF, W, H, ids.data()); for (uint32_t i : ids) if (i) hit[i - 1] = 1; }
  int64_t nohit = 0; double nohit_area = 0;
  for (int64_t f = 0; f < NF; f++) if (!hit[f]) { nohit++; const int32_t* t = r.faces.p + 3 * f; nohit_area += 0.5 * norm(cross(ld(m.V.p + 3 * (int64_t)r.vmap[t[1]]) - ld(m.V.p + 3 * (int64_t)r.vmap[t[0]]), ld(m.V.p + 3 * (int64_t)r.vmap[t[2]]) - ld(m.V.p + 3 * (int64_t)r.vmap[t[0]]))); }
  snprintf(nm, sizeof nm, "%s.atlas_texel_density", tag);
  gate(q05 > 0.5 * q50 && q95 < 2 * q50 && zero_uv * 1000 < NF, nm, "texels per unit length (area-weighted): p1 %.0f, p5 %.0f, median %.0f, p95 %.0f (p5/median %.2f, p95/median %.2f); mean %.0f; "
       "%lld faces with zero UV area, %lld mirrored; %lld faces (%.3f%% of the surface area) own no texel center",
       q01, q05, q50, q95, q05 / q50, q95 / q50, std::sqrt(a2 / a3), (long long)zero_uv, (long long)flipped, (long long)nohit, 100.0 * nohit_area / a3);
}

// ---- bake: every texel vs a brute-force CPU reference -------------------------------------------------------------------
static void check_bake(const char* tag, const GlbOut& r, const float* V, const int32_t* F, int64_t nf, const int32_t* coords, const float* attrs, int64_t nvox, int res, int W, int H, int64_t samples) {
  char nm[64]; const int64_t P = (int64_t)W * H; const Mesh& m = r.simplified;
  const uint32_t* ids = r.ids.ptr<uint32_t>();
  std::vector<int64_t> tex; for (int64_t i = 0; i < P; i++) if (ids[i]) tex.push_back(i);
  if ((int64_t)tex.size() > samples) { Rng g(5); std::shuffle(tex.begin(), tex.end(), g.g); tex.resize((size_t)samples); }
  std::unordered_map<uint64_t, int64_t> vox; vox.reserve((size_t)nvox * 2);
  for (int64_t i = 0; i < nvox; i++) vox[((uint64_t)coords[3 * i + 2] * res + coords[3 * i + 1]) * res + coords[3 * i]] = i;
  std::atomic<int64_t> d0{0}, d1{0}, dbig{0}; std::atomic<int> maxd{0};
  par_for((int64_t)tex.size(), 4, [&](int64_t a, int64_t b) {
    for (int64_t s = a; s < b; s++) {
      int64_t gid = tex[s]; int64_t f = ids[gid] - 1; const int32_t* t = r.faces.p + 3 * f;
      float tx = ((float)(gid % W) + 0.5f) / (float)W, ty = ((float)(gid / W) + 0.5f) / (float)H;
      float u0x = r.uv[2 * t[0]], u0y = r.uv[2 * t[0] + 1], e1x = r.uv[2 * t[1]] - u0x, e1y = r.uv[2 * t[1] + 1] - u0y, e2x = r.uv[2 * t[2]] - u0x, e2y = r.uv[2 * t[2] + 1] - u0y, dx = tx - u0x, dy = ty - u0y;
      float den = e1x * e2y - e1y * e2x; if (den == 0) continue;
      float b1 = (dx * e2y - dy * e2x) / den, b2 = (e1x * dy - e1y * dx) / den, b0 = 1.f - b1 - b2;
      V3 x = ld(m.V.p + 3 * (int64_t)r.vmap[t[0]]) * b0 + ld(m.V.p + 3 * (int64_t)r.vmap[t[1]]) * b1 + ld(m.V.p + 3 * (int64_t)r.vmap[t[2]]) * b2;
      float d2; int32_t fid; V3 w; brute_nearest(V, F, nf, x, d2, fid, w);
      x = ld(V + 3 * (int64_t)F[3 * (int64_t)fid]) * w.x + ld(V + 3 * (int64_t)F[3 * (int64_t)fid + 1]) * w.y + ld(V + 3 * (int64_t)F[3 * (int64_t)fid + 2]) * w.z;
      float q[3] = {(x.x + 0.5f) * (float)res, (x.y + 0.5f) * (float)res, (x.z + 0.5f) * (float)res};
      int bx = (int)std::floor(q[0] - 0.5f), by = (int)std::floor(q[1] - 0.5f), bz = (int)std::floor(q[2] - 0.5f);
      int64_t n[8]; float wt[8], wsum = 0;
      for (int i = 0; i < 8; i++) {
        int vx = bx + (i & 1), vy = by + ((i >> 1) & 1), vz = bz + ((i >> 2) & 1); n[i] = -1; wt[i] = 0;
        if (vx < 0 || vy < 0 || vz < 0 || vx >= res || vy >= res || vz >= res) continue;
        auto it = vox.find(((uint64_t)vz * res + vy) * res + vx); if (it == vox.end()) continue;
        n[i] = it->second; wt[i] = (1.f - std::fabs(q[0] - (float)vx - 0.5f)) * (1.f - std::fabs(q[1] - (float)vy - 0.5f)) * (1.f - std::fabs(q[2] - (float)vz - 0.5f)); wsum += wt[i];
      }
      float inv = 1.f / std::max(wsum, 1e-12f), acc[6] = {0, 0, 0, 0, 0, 0};
      for (int i = 0; i < 8; i++) if (n[i] >= 0) for (int c = 0; c < 6; c++) acc[c] += attrs[n[i] * 6 + c] * (wt[i] * inv);
      for (int c = 0; c < 6; c++) {
        int ref = (int)std::min(std::max(acc[c] * 255.f, 0.f), 255.f), got = r.img0[(int64_t)c * P + gid], d = std::abs(ref - got);
        if (d == 0) d0++; else if (d == 1) d1++; else dbig++;
        int mx = maxd.load(); while (d > mx && !maxd.compare_exchange_weak(mx, d)) {}
      }
    }
  });
  int64_t tot = d0 + d1 + dbig;
  snprintf(nm, sizeof nm, "%s.bake_vs_brute_force", tag);
  gate(tot > 0 && dbig * 2000 <= tot && (d0 + d1) > 0, nm, "%lld texels x 6 channels (raster id -> barycentric point -> brute-force nearest point on the %lld-face input mesh -> trilinear voxel sample -> uint8): "
       "equal %.3f%%, off by one %.3f%%, off by more %.4f%% (max %d; nearest-point ties / float rounding at uint8 truncation)", (long long)tex.size(), (long long)nf,
       100.0 * d0 / tot, 100.0 * d1 / tot, 100.0 * dbig / tot, maxd.load());
}

// ---- synthetic end-to-end: bumpy sphere with a small hole, analytic voxel colours ----------------------------------------
static void test_synthetic(const std::string& dump, const std::string& tmp) {
  const int nu = 160, nvr = 80, res = 64; std::vector<float> V; std::vector<int32_t> F;
  auto P = [&](int i, int j) { float th = (float)M_PI * (float)j / nvr, ph = 2 * (float)M_PI * (float)i / nu; float rad = 0.33f + 0.04f * std::sin(5 * ph) * std::sin(3 * th);
    return V3{rad * std::sin(th) * std::cos(ph), rad * std::sin(th) * std::sin(ph), rad * std::cos(th) * 0.8f}; };
  for (int j = 1; j < nvr; j++) for (int i = 0; i < nu; i++) { V3 p = P(i, j); V.insert(V.end(), {p.x, p.y, p.z}); }
  int top = (int)V.size() / 3; V.insert(V.end(), {0, 0, P(0, 0).z}); int bot = top + 1; V.insert(V.end(), {0, 0, P(0, nvr).z});
  auto id = [&](int i, int j) { return (j - 1) * nu + (i % nu); };
  int64_t holes_cut = 0;
  for (int j = 1; j < nvr - 1; j++) for (int i = 0; i < nu; i++) {
    if (j == 40 && i == 17) { holes_cut++; continue; }  // one missing quad: perimeter ~0.05 > 3e-2, stays open
    F.insert(F.end(), {id(i, j), id(i, j + 1), id(i + 1, j + 1)});
    if (j == 20 && i == 90) { holes_cut++; continue; }  // one missing triangle near... small hole (perimeter < 3e-2 ?)
    F.insert(F.end(), {id(i, j), id(i + 1, j + 1), id(i + 1, j)});
  }
  for (int i = 0; i < nu; i++) { F.insert(F.end(), {top, id(i, 1), id(i + 1, 1)}); F.insert(F.end(), {bot, id(i + 1, nvr - 1), id(i, nvr - 1)}); }
  const int64_t nv = (int64_t)V.size() / 3, nf = (int64_t)F.size() / 3;
  // expected fill_holes result by brute force: boundary loops and their perimeters
  // voxels: every cell whose center is within 1.2 voxels of the surface (brute force), colours analytic in position
  std::vector<int32_t> coords; std::vector<float> attrs;
  { std::vector<uint8_t> on((size_t)res * res * res, 0);
    par_for((int64_t)res * res * res, 256, [&](int64_t a, int64_t b) { for (int64_t k = a; k < b; k++) {
      int x = (int)(k % res), y = (int)((k / res) % res), z = (int)(k / (res * res)); V3 c = {((float)x + 0.5f) / res - 0.5f, ((float)y + 0.5f) / res - 0.5f, ((float)z + 0.5f) / res - 0.5f};
      if (std::fabs(norm(V3{c.x, c.y, c.z / 0.8f}) - 0.33f) > 0.04f + 2.5f / res) continue;
      float d2; int32_t fi; V3 w; brute_nearest(V.data(), F.data(), nf, c, d2, fi, w); on[k] = std::sqrt(d2) < 1.2f / res; } });
    for (int64_t k = 0; k < (int64_t)res * res * res; k++) if (on[k]) {
      int x = (int)(k % res), y = (int)((k / res) % res), z = (int)(k / (res * res)); coords.insert(coords.end(), {x, y, z});
      float fx = (float)x / res, fy = (float)y / res, fz = (float)z / res;
      attrs.insert(attrs.end(), {0.5f + 0.5f * std::sin(9 * fx), 0.5f + 0.5f * std::cos(7 * fy + 2 * fz), fz * 1.2f - 0.1f, fx, 0.3f + 0.6f * fy, 1.0f});
    } }
  const int64_t nvox = (int64_t)coords.size() / 3;
  // fill_holes against a brute-force boundary walk
  { Mesh hf; HoleStats hs = fill_holes(V.data(), F.data(), nv, nf, 3e-2f, hf);
    std::map<std::pair<int, int>, int> he; for (int64_t f = 0; f < nf; f++) for (int k = 0; k < 3; k++) he[{F[3 * f + k], F[3 * f + (k + 1) % 3]}]++;
    std::map<int, int> nxt; for (auto& e : he) if (!he.count({e.first.second, e.first.first})) nxt[e.first.second] = e.first.first;  // boundary edge reversed: walk the loop
    int loops = 0, small = 0, small_edges = 0; std::set<int> seen;
    for (auto& e : nxt) { if (seen.count(e.first)) continue; int v = e.first, n = 0; double per = 0; do { seen.insert(v); int w = nxt[v]; per += norm(ld(&V[3 * v]) - ld(&V[3 * w])); v = w; n++; } while (v != e.first && n < 100000); loops++; if (per < 3e-2) { small++; small_edges += n; } }
    gate(hs.loops == loops && hs.filled == small && hs.new_faces == small_edges && hf.nf == nf + small_edges, "synthetic.fill_holes", "%lld boundary loops (brute force %d), %lld filled with perimeter < 3e-2 (brute force %d), %lld new faces (expected %d)",
         (long long)hs.loops, loops, (long long)hs.filled, small, (long long)hs.new_faces, small_edges);
  }
  GlbOpts o; o.faces = 6000; o.texture_size = 512; GlbOut r; std::string path = tmp + "/synthetic.glb";
  to_glb(V.data(), F.data(), nv, nf, coords.data(), attrs.data(), nvox, res, path, o, r, true);
  // remesh + simplify: the surface must sit eps = band * scale / res off the input surface (both sides), within a fraction of a voxel
  { const Mesh& m = r.simplified; const float eps = (float)(res + 3) / res / res; double sum = 0, mx = 0;
    std::vector<float> dv((size_t)m.nv);
    par_for(m.nv, 16, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) { float d2; int32_t fi; V3 w; brute_nearest(V.data(), F.data(), nf, ld(m.V.p + 3 * i), d2, fi, w); dv[i] = std::sqrt(d2); } });
    for (int64_t i = 0; i < m.nv; i++) { double e = std::fabs(dv[i] - eps) * res; sum += e; mx = std::max(mx, e); }
    // edge manifoldness of the simplified mesh
    std::map<std::pair<int, int>, int> he; for (int64_t f = 0; f < m.nf; f++) for (int k = 0; k < 3; k++) he[{m.F[3 * f + k], m.F[3 * f + (k + 1) % 3]}]++;
    int64_t dup = 0, open = 0; for (auto& e : he) { dup += e.second > 1; open += !he.count({e.first.second, e.first.first}); }
    gate(m.nf <= o.faces && m.nf > o.faces * 0.9 && sum / m.nv < 0.25 && mx < 1.0 && dup == 0, "synthetic.remesh_simplify", "%lld -> %lld remeshed -> %lld faces (target %lld): vertex distance to the input surface is eps = %.4f "
         "(1 band voxel) with mean |d - eps| %.3f voxel, max %.3f voxel (brute force); %lld non-manifold directed edges, %lld boundary edges",
         (long long)nf, (long long)r.remesh_faces, (long long)m.nf, (long long)o.faces, eps, sum / m.nv, mx, (long long)dup, (long long)open);
  }
  check_atlas("synthetic", r, o.texture_size, o.texture_size, o.faces);
  { Mesh hf; fill_holes(V.data(), F.data(), nv, nf, 3e-2f, hf);
    check_bake("synthetic", r, hf.V.p, hf.F.p, hf.nf, coords.data(), attrs.data(), nvox, res, o.texture_size, o.texture_size, 1 << 30); }
  { TeleaCmp c = telea_compare(r.img0.p, r.lv.ptr<int32_t>(), o.texture_size, o.texture_size);
    gate(c.maxdiff == 0 && c.diff == 0, "synthetic.telea_gpu_vs_cpu", "%lld unknown texels, %d levels: %lld channel values differ (max |diff| %d)", (long long)c.unknown, c.levels, (long long)c.diff, c.maxdiff); }
  { // packed textures = planar result in glTF layout
    const int64_t Pn = (int64_t)o.texture_size * o.texture_size; const uint8_t* b = r.base.ptr<uint8_t>(); const uint8_t* mr = r.mr.ptr<uint8_t>(); int64_t bad = 0; const uint32_t* ids = r.ids.ptr<uint32_t>();
    for (int64_t i = 0; i < Pn; i++) if (ids[i]) { bad += b[4 * i] != r.img0[i] || b[4 * i + 1] != r.img0[Pn + i] || b[4 * i + 2] != r.img0[2 * Pn + i] || b[4 * i + 3] != r.img0[5 * Pn + i] || mr[3 * i] != 0 || mr[3 * i + 1] != r.img0[4 * Pn + i] || mr[3 * i + 2] != r.img0[3 * Pn + i]; }
    gate(bad == 0, "synthetic.pack_layout", "base = (r, g, b, alpha), metallicRoughness = (0, roughness, metallic) at every baked texel: %lld mismatches", (long long)bad); }
  printf("      synthetic GLB: %s (check with tools/check_glb.py)\n", path.c_str());
  (void)dump; (void)holes_cut;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string mesh, dump, tmp = "."; GlbOpts o;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
      if (a == "--mesh") mesh = nx(); else if (a == "--dump") dump = nx(); else if (a == "--tmp") tmp = nx();
      else if (a == "--faces") o.faces = std::stoll(nx()); else if (a == "--texture-size") o.texture_size = std::stoi(nx());
      else { fprintf(stderr, "usage: glb_test [--mesh x.safetensors] [--dump dir] [--tmp dir] [--faces N] [--texture-size N]\n"); return 2; }
    }
    tk::init(nullptr);
    Tensor v, fc, co, at;
    if (!mesh.empty()) { STFile f(mesh); v = f.get("vertices", tk::F32); fc = f.get("faces", tk::I32); co = f.get("coords", tk::I32); at = f.get("attrs", tk::F32); tk::sync(); }
    test_lbvh(mesh.empty() ? nullptr : &v, mesh.empty() ? nullptr : &fc);
    test_raster();
    test_telea(dump);
    test_synthetic(dump, tmp);
    if (!mesh.empty()) {
      int64_t n = co.shape[0]; int32_t mx = 0; const int32_t* c = co.ptr<int32_t>(); for (int64_t i = 0; i < n * 3; i++) mx = std::max(mx, c[i]);
      int res = 64; while (res <= mx) res *= 2;
      GlbOut r; to_glb(v.ptr<float>(), fc.ptr<int32_t>(), v.shape[0], fc.shape[0], c, at.ptr<float>(), n, res, tmp + "/real.glb", o, r, true);
      check_atlas("real", r, o.texture_size, o.texture_size, o.faces);
      { Mesh hf; fill_holes(v.ptr<float>(), fc.ptr<int32_t>(), v.shape[0], fc.shape[0], o.hole_perimeter, hf);
        check_bake("real", r, hf.V.p, hf.F.p, hf.nf, c, at.ptr<float>(), n, res, o.texture_size, o.texture_size, 3000); }
      std::vector<uint8_t> out; TeleaCmp tc = telea_compare(r.img0.p, r.lv.ptr<int32_t>(), o.texture_size, o.texture_size, &out);
      gate(tc.maxdiff == 0 && tc.diff == 0, "real.telea_gpu_vs_cpu", "%dx%d, %lld unknown texels, %d levels: %lld of %lld channel values differ (%.4f%%), max |diff| %d", o.texture_size, o.texture_size,
           (long long)tc.unknown, tc.levels, (long long)tc.diff, (long long)tc.unknown * 6, 100.0 * tc.diff / std::max<int64_t>(1, tc.unknown * 6), tc.maxdiff);
      if (!dump.empty()) dump_telea(dump, "telea_real", r.img0.p, r.lv.ptr<int32_t>(), out.data(), o.texture_size, o.texture_size);
    }
    printf("%s (%d failed)\n", g_fail ? "GLB TESTS FAILED" : "GLB TESTS OK", g_fail);
  }
  return g_fail ? 1 : 0;
}
