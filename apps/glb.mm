// Standalone to_glb: MeshWithVoxel safetensors (build/t2 --out) -> textured GLB.
//   build/glb --mesh x.safetensors --out x.glb [--res 512|1024] [--faces 1000000] [--texture-size 4096]
//             [--texture-format webp|png] [--webp-method 0..6] [--repeat N] [--bench out.json] [-v]
#include "pipeline.h"
#include "glb/to_glb.h"
using namespace glb;

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string mesh, out, bench; int res = 0, repeat = 1; GlbOpts o;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
      if (a == "--mesh") mesh = nx(); else if (a == "--out") out = nx(); else if (a == "--res") res = std::stoi(nx());
      else if (a == "--faces") o.faces = std::stoll(nx()); else if (a == "--texture-size") o.texture_size = std::stoi(nx());
      else if (a == "--texture-format") o.tex.png = nx() == "png"; else if (a == "--webp-method") o.tex.method = std::stoi(nx()); else if (a == "--webp-alpha-quality") o.tex.alpha_quality = std::stoi(nx());
      else if (a == "--groups") o.xatlas_groups = std::stoi(nx()); else if (a == "--threads") o.xatlas_threads = std::stoi(nx());
      else if (a == "--repeat") repeat = std::stoi(nx()); else if (a == "--bench") bench = nx(); else if (a == "-v") o.verbose = true;
      else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mesh.empty()) { fprintf(stderr, "usage: glb --mesh x.safetensors --out x.glb [--res N] [--faces N] [--texture-size N] [--texture-format webp|png] [-v]\n"); return 2; }
    tk::init(nullptr);
    STFile f(mesh);
    Tensor v = f.get("vertices", tk::F32), fc = f.get("faces", tk::I32), co = f.get("coords", tk::I32), at = f.get("attrs", tk::F32);
    tk::sync();
    int64_t n = co.shape[0];
    if (!res) { int32_t mx = 0; const int32_t* c = co.ptr<int32_t>(); for (int64_t i = 0; i < n * 3; i++) mx = std::max(mx, c[i]); res = 64; while (res <= mx) res *= 2; }
    GlbOut best; double bt = 1e30;
    for (int it = 0; it < repeat; it++) {
      GlbOut r;
      to_glb(v.ptr<float>(), fc.ptr<int32_t>(), v.shape[0], fc.shape[0], co.ptr<int32_t>(), at.ptr<float>(), n, res, out, o, r);
      if (r.t["total"] < bt) { bt = r.t["total"]; best = std::move(r); }
    }
    GlbOut& r = best;
    double fpCur, fpPeak; footprint(fpCur, fpPeak);
    printf("to_glb %s res=%d:", mesh.substr(mesh.find_last_of('/') + 1).c_str(), res);
    for (auto& kv : r.t) printf(" %s=%.3f", kv.first.c_str(), kv.second);
    printf("\n  in_faces=%lld holes_filled=%lld remesh_faces=%lld simp_faces=%lld simp_passes=%d charts=%lld chart_iters=%d pieces=%lld atlas=%dx%d out_verts=%lld out_faces=%lld"
           " baked=%.1f%% telea_levels=%d glb_mb=%.1f footprint_peak_gb=%.2f\n",
           (long long)r.in_faces, (long long)r.holes.filled, (long long)r.remesh_faces, (long long)r.simp_faces, r.simp.passes, (long long)r.unwrap.charts.charts, r.unwrap.charts.iters,
           (long long)r.unwrap.pieces, r.unwrap.W, r.unwrap.H, (long long)r.out_verts, (long long)r.out_faces, 100.0 * (double)r.baked_texels / ((double)o.texture_size * o.texture_size),
           r.telea.levels, (double)r.file.bytes / 1e6, fpPeak);
    printf("  detail: remesh band %.3f grid %.3f dc %.3f quads %.3f | simplify csr %.3f cost %.3f select %.3f apply %.3f | unwrap charts %.3f (adj %.3f merge %.3f) pieces %.3f xatlas %.3f pack %.3f\n",
           r.remesh.t_band, r.remesh.t_grid, r.remesh.t_dc, r.remesh.t_quads, r.simp.t_csr, r.simp.t_cost, r.simp.t_select, r.simp.t_apply, r.unwrap.t_charts, r.unwrap.charts.t_adj,
           r.unwrap.charts.t_merge, r.unwrap.t_pieces, r.unwrap.t_xatlas, r.unwrap.t_pack);
    if (!bench.empty()) {
      std::string js = "{"; char buf[256]; bool first = true;
      for (auto& kv : r.t) { snprintf(buf, sizeof buf, "%s\"%s\": %.4f", first ? "" : ", ", kv.first.c_str(), kv.second); js += buf; first = false; }
      snprintf(buf, sizeof buf, ", \"res\": %d, \"in_faces\": %lld, \"out_faces\": %lld, \"out_verts\": %lld, \"peak_footprint_gb\": %.3f}\n", res, (long long)r.in_faces, (long long)r.out_faces, (long long)r.out_verts, fpPeak);
      js += buf;
      FILE* fo = fopen(bench.c_str(), "w"); if (fo) { fputs(js.c_str(), fo); fclose(fo); }
    }
  }
  return 0;
}
