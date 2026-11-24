// TRELLIS.2 image -> 3D on Metal.
//   build/t2 --image <png|webp> [--type 512|1024|1024_cascade|1536_cascade] [--seed 42] [--mode parity]
//            [--out mesh.safetensors] [--ref refdump/X.safetensors] [--bench out.json] [--runs N] [--warmup N] [--lowmem] [--pool-gb G]
//            [--glb out.glb] [--faces 1000000] [--texture-size 4096] [--texture-format webp|png] [--webp-alpha-quality 0..100]
// --glb runs o_voxel.postprocess.to_glb (src/glb) on the generated mesh and writes a textured GLB. With several
// images --out / --glb name a directory. Extra timings: to_glb (remesh + simplify + uv_unwrap + bake + textures),
// export, and total_glb = image -> GLB file.
#include "../src/pipeline.h"
#include "../src/glb/to_glb.h"
struct GlbArgs { std::string path; glb::GlbOpts o; };
// Returns the stage times; adds them to r.t under glb.* plus to_glb / export / total_glb.
static void runGlb(PipeOut& r, const std::string& path, const glb::GlbOpts& o) {
  tk::sync(); pool().trim();
  glb::GlbOut g;
  glb::to_glb(r.mesh.vertices.ptr<float>(), r.mesh.faces.ptr<int32_t>(), r.mesh.vertices.shape[0], r.mesh.faces.shape[0], r.mesh.coords.ptr<int32_t>(),
              r.mesh.attrs.ptr<float>(), r.mesh.coords.shape[0], r.res, path, o, g);
  pool().trim();
  for (const char* k : {"remesh", "simplify", "uv_unwrap", "bake", "textures"}) r.t[std::string("glb.") + k] = g.t[k];
  r.t["export"] = g.t["export"]; r.t["to_glb"] = g.t["total"] - g.t["export"]; r.t["total_glb"] = r.t["total"] + g.t["total"];
  r.glb_faces = g.out_faces; r.glb_verts = g.out_verts; r.glb_bytes = g.file.bytes;
}
template <class I>
static int runAll(const PipeOpts& o, const std::vector<std::string>& images, const std::string& out, const std::string& bench, int runs, int warmup, const GlbArgs& ga) {
  Pipeline<I> p(o);
  bool ok = true;
  std::string js = "{\n";
  for (int w = 0; w < warmup; w++) @autoreleasepool { PipeOut r = p.run(load_image(images[0]));
    if (!ga.path.empty()) runGlb(r, "", ga.o);  // compiles the to_glb Metal kernels; writes nothing
    fprintf(stderr, "warmup total %.2fs (+load %.2fs)\n", r.t["total"], r.t["load"]); }
  for (size_t ii = 0; ii < images.size(); ii++) {
    PipeOut best; double bt = 1e30;
    std::string base = images[ii].substr(images[ii].find_last_of('/') + 1), stem = base.substr(0, base.find_last_of('.'));
    for (int k = 0; k < runs; k++) @autoreleasepool {
      Image8 raw = load_image(images[ii]);
      PipeOut r = p.run(raw);
      ok &= r.parity_ok;
      if (!ga.path.empty()) runGlb(r, images.size() > 1 ? ga.path + "/" + stem + "_" + o.type + ".glb" : ga.path, ga.o);
      const char* key = ga.path.empty() ? "total" : "total_glb";
      if (r.t[key] < bt) { bt = r.t[key]; best = r; }
    }
    PipeOut& r = best;
    printf("%s %s %s:", I::name(), o.type.c_str(), base.c_str());
    for (auto& kv : r.t) printf(" %s=%.2f", kv.first.c_str(), kv.second);
    double fpCur, fpPeak; footprint(fpCur, fpPeak);
    if (r.glb_bytes) printf(" glb_faces=%lld glb_verts=%lld glb_mb=%.1f", (long long)r.glb_faces, (long long)r.glb_verts, (double)r.glb_bytes / 1e6);
    printf(" tokens=%lld verts=%lld faces=%lld voxels=%lld res=%d peak_gb=%.2f footprint_gb=%.2f footprint_peak_gb=%.2f\n", (long long)r.tokens,
           (long long)r.mesh.vertices.shape[0], (long long)r.mesh.faces.shape[0], (long long)r.mesh.coords.shape[0], r.res,
           (double)pool().peak / (1 << 30), fpCur, fpPeak);
    fflush(stdout);
    char buf[512];
    js += std::string(ii ? ",\n" : "") + "  \"" + base + "\": {";
    for (auto& kv : r.t) { snprintf(buf, sizeof buf, "\"%s\": %.4f, ", kv.first.c_str(), kv.second); js += buf; }
    if (r.glb_bytes) { snprintf(buf, sizeof buf, "\"glb_faces\": %lld, \"glb_vertices\": %lld, \"glb_bytes\": %lld, ", (long long)r.glb_faces, (long long)r.glb_verts, (long long)r.glb_bytes); js += buf; }
    snprintf(buf, sizeof buf, "\"tokens\": %lld, \"vertices\": %lld, \"faces\": %lld, \"voxels\": %lld, \"res\": %d, \"peak_pool_gb\": %.3f, \"peak_footprint_gb\": %.3f}",
             (long long)r.tokens, (long long)r.mesh.vertices.shape[0], (long long)r.mesh.faces.shape[0], (long long)r.mesh.coords.shape[0], r.res,
             (double)pool().peak / (1 << 30), fpPeak);
    js += buf;
    if (!out.empty()) {
      std::string path = out;
      if (images.size() > 1) { path = out + "/" + stem + "_" + o.type + ".safetensors"; }
      saveSafetensors(path, {{"vertices", r.mesh.vertices}, {"faces", r.mesh.faces}, {"coords", r.mesh.coords}, {"attrs", r.mesh.attrs},
                             {"shape_slat", r.shape_slat}, {"tex_slat", r.tex_slat}, {"slat_coords", r.slat_coords}});
    }
    if (!bench.empty()) { FILE* f = fopen(bench.c_str(), "w"); if (f) { fputs((js + "\n}\n").c_str(), f); fclose(f); } }
  }
  if (o.ref && I::exact) printf(ok ? "PARITY OK\n" : "PARITY FAILED\n");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    PipeOpts o; std::vector<std::string> images; std::string out, bench, mode = "parity", refp;
    int runs = 1, warmup = 0; double poolGb = -1; GlbArgs ga;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
      if (a == "--root") o.root = nx(); else if (a == "--type") o.type = nx(); else if (a == "--seed") o.seed = strtoull(nx().c_str(), 0, 10);
      else if (a == "--image") images.push_back(nx()); else if (a == "--out") out = nx(); else if (a == "--bench") bench = nx();
      else if (a == "--mode") mode = nx(); else if (a == "--ref") refp = nx(); else if (a == "--runs") runs = atoi(nx().c_str());
      else if (a == "--warmup") warmup = atoi(nx().c_str()); else if (a == "--lowmem") o.keep_models = false;
      else if (a == "--pool-gb") poolGb = atof(nx().c_str());
      else if (a == "--max-tokens") o.max_num_tokens = atoll(nx().c_str()); else if (a == "--quiet") o.verbose = false;
      else if (a == "--glb") ga.path = nx(); else if (a == "--faces") ga.o.faces = atoll(nx().c_str()); else if (a == "--texture-size") ga.o.texture_size = atoi(nx().c_str());
      else if (a == "--texture-format") ga.o.tex.png = nx() == "png"; else if (a == "--glb-verbose") ga.o.verbose = true;
      else if (a == "--webp-alpha-quality") ga.o.tex.alpha_quality = atoi(nx().c_str());
      else if (a[0] != '-') images.push_back(a);
      else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (poolGb >= 0) pool().cap = (size_t)(poolGb * (1 << 30)); else if (!o.keep_models) pool().cap = (size_t)1 << 30;  // low-memory mode: keep few freed buffers
    if (images.empty()) { fprintf(stderr, "usage: t2 --image <file> [--type 512|1024|1024_cascade] [--mode parity] [--out x.safetensors] [--glb x.glb]\n"); return 2; }
    fk::registerKernels();
    tk::init((o.root + "/build/torchlibs/lib_01.metallib").c_str());
    initOps();
    std::unique_ptr<STFile> ref; if (!refp.empty()) { ref.reset(new STFile(refp)); o.ref = ref.get(); }
    if (mode == "fast") return runAll<FastImpl>(o, images, out, bench, runs, warmup, ga);
    if (mode == "parity") return runAll<ParityImpl>(o, images, out, bench, runs, warmup, ga);
    fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2;
  }
}
