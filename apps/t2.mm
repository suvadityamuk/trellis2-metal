// TRELLIS.2 image -> 3D on Metal.
//   build/t2 --image <png|webp> [--type 512|1024|1024_cascade|1536_cascade] [--seed 42] [--mode parity]
//            [--out mesh.safetensors] [--ref refdump/X.safetensors] [--bench out.json] [--runs N] [--warmup N] [--lowmem] [--pool-gb G]
#include "../src/pipeline.h"
template <class I>
static int runAll(const PipeOpts& o, const std::vector<std::string>& images, const std::string& out, const std::string& bench, int runs, int warmup) {
  Pipeline<I> p(o);
  bool ok = true;
  std::string js = "{\n";
  for (int w = 0; w < warmup; w++) @autoreleasepool { PipeOut r = p.run(load_image(images[0])); fprintf(stderr, "warmup total %.2fs (+load %.2fs)\n", r.t["total"], r.t["load"]); }
  for (size_t ii = 0; ii < images.size(); ii++) {
    PipeOut best; double bt = 1e30;
    for (int k = 0; k < runs; k++) @autoreleasepool {
      Image8 raw = load_image(images[ii]);
      PipeOut r = p.run(raw);
      ok &= r.parity_ok;
      if (r.t["total"] < bt) { bt = r.t["total"]; best = r; }
    }
    PipeOut& r = best;
    std::string base = images[ii].substr(images[ii].find_last_of('/') + 1);
    printf("%s %s %s:", I::name(), o.type.c_str(), base.c_str());
    for (auto& kv : r.t) printf(" %s=%.2f", kv.first.c_str(), kv.second);
    double fpCur, fpPeak; footprint(fpCur, fpPeak);
    printf(" tokens=%lld verts=%lld faces=%lld voxels=%lld res=%d peak_gb=%.2f footprint_gb=%.2f footprint_peak_gb=%.2f\n", (long long)r.tokens,
           (long long)r.mesh.vertices.shape[0], (long long)r.mesh.faces.shape[0], (long long)r.mesh.coords.shape[0], r.res,
           (double)pool().peak / (1 << 30), fpCur, fpPeak);
    fflush(stdout);
    char buf[512];
    js += std::string(ii ? ",\n" : "") + "  \"" + base + "\": {";
    for (auto& kv : r.t) { snprintf(buf, sizeof buf, "\"%s\": %.4f, ", kv.first.c_str(), kv.second); js += buf; }
    snprintf(buf, sizeof buf, "\"tokens\": %lld, \"vertices\": %lld, \"faces\": %lld, \"voxels\": %lld, \"res\": %d, \"peak_pool_gb\": %.3f, \"peak_footprint_gb\": %.3f}",
             (long long)r.tokens, (long long)r.mesh.vertices.shape[0], (long long)r.mesh.faces.shape[0], (long long)r.mesh.coords.shape[0], r.res,
             (double)pool().peak / (1 << 30), fpPeak);
    js += buf;
    if (!out.empty()) {
      std::string path = out;
      if (images.size() > 1) { std::string stem = base.substr(0, base.find_last_of('.')); path = out + "/" + stem + "_" + o.type + ".safetensors"; }
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
    int runs = 1, warmup = 0; double poolGb = -1;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
      if (a == "--root") o.root = nx(); else if (a == "--type") o.type = nx(); else if (a == "--seed") o.seed = strtoull(nx().c_str(), 0, 10);
      else if (a == "--image") images.push_back(nx()); else if (a == "--out") out = nx(); else if (a == "--bench") bench = nx();
      else if (a == "--mode") mode = nx(); else if (a == "--ref") refp = nx(); else if (a == "--runs") runs = atoi(nx().c_str());
      else if (a == "--warmup") warmup = atoi(nx().c_str()); else if (a == "--lowmem") o.keep_models = false;
      else if (a == "--pool-gb") poolGb = atof(nx().c_str());
      else if (a == "--max-tokens") o.max_num_tokens = atoll(nx().c_str()); else if (a == "--quiet") o.verbose = false;
      else if (a[0] != '-') images.push_back(a);
      else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (poolGb >= 0) pool().cap = (size_t)(poolGb * (1 << 30)); else if (!o.keep_models) pool().cap = (size_t)1 << 30;  // low-memory mode: keep few freed buffers
    if (images.empty()) { fprintf(stderr, "usage: t2 --image <file> [--type 512|1024|1024_cascade] [--mode parity] [--out x.safetensors]\n"); return 2; }
    fk::registerKernels();
    tk::init((o.root + "/build/torchlibs/lib_01.metallib").c_str());
    initOps();
    std::unique_ptr<STFile> ref; if (!refp.empty()) { ref.reset(new STFile(refp)); o.ref = ref.get(); }
    if (mode == "fast") return runAll<FastImpl>(o, images, out, bench, runs, warmup);
    if (mode == "parity") return runAll<ParityImpl>(o, images, out, bench, runs, warmup);
    fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2;
  }
}
