// Single-call check of the fast flow transformers against the parity (bitwise-reference) ones, plus timing.
#include "../src/pipeline.h"
using namespace t2;
static void rel(const char* n, const Tensor& a, const Tensor& b) { relReport(n, a, b); }
int main(int argc, char** argv) {
  @autoreleasepool {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    fk::registerKernels(); tk::init("build/torchlibs/lib_01.metallib"); initOps();
    STFile ref(argc > 1 ? argv[1] : "refdump/T_512.safetensors");
    std::string root = "models/trellis2/ckpts/";
    Json pj = loadJson("models/trellis2/pipeline.json");
    Tensor cond = ref.get("cond.0.cond"), neg = zeros(tk::F32, cond.shape);
    float t1000 = 1000.0f; Tensor tt = from_cpu(tk::F32, {1}, &t1000);
    {
      std::string c = "models/trellis2/" + pj["args"]["models"]["sparse_structure_flow_model"].str;
      DenseFlow p; p.load(c + ".json", c + ".safetensors"); FastDenseFlow f; f.load(c + ".json", c + ".safetensors");
      Tensor x = ref.get("ss_s.0.noise");
      f.setCond(cond);
      Tensor pc = p(x, tt, cond), pn = p(x, tt, neg); tk::sync();
      Tensor fc = f.forward(x, tt, true), fn = f.forward(x, tt, false); tk::sync();
      rel("dense cond", fc, pc); rel("dense uncond", fn, pn); rel("(cond vs uncond)", pc, pn);
      for (int k = 0; k < 2; k++) {
        double t0 = wallNow(); for (int i = 0; i < 3; i++) f.forward(x, tt, true); tk::sync(); double t1 = wallNow();
        for (int i = 0; i < 3; i++) f.forward(x, tt, false); tk::sync(); double t2 = wallNow();
        for (int i = 0; i < 3; i++) p(x, tt, cond); tk::sync(); double t3 = wallNow();
        printf("dense per call: fast cond %.3fs uncond %.3fs parity %.3fs\n", (t1 - t0) / 3, (t2 - t1) / 3, (t3 - t2) / 3);
      }
    }
    pool().trim();
    {
      std::string c = "models/trellis2/" + pj["args"]["models"]["shape_slat_flow_model_512"].str;
      SparseFlow p; p.load(c + ".json", c + ".safetensors"); FastSparseFlow f; f.load(c + ".json", c + ".safetensors");
      Tensor coords = ref.get("ss"), x = ref.get("shape_s.0.noise.feats");
      p.setCoords(coords); f.setCoords(coords); f.setCond(cond);
      Tensor pc = p(x, tt, cond), pn = p(x, tt, neg); tk::sync();
      Tensor fc = f.forward(x, tt, true), fn = f.forward(x, tt, false); tk::sync();
      rel("sparse cond", fc, pc); rel("sparse uncond", fn, pn); rel("(cond vs uncond)", pc, pn);
      double t0 = wallNow(); for (int i = 0; i < 3; i++) f.forward(x, tt, true); tk::sync(); double t1 = wallNow();
      for (int i = 0; i < 3; i++) f.forward(x, tt, false); tk::sync(); double t2 = wallNow();
      for (int i = 0; i < 3; i++) p(x, tt, cond); tk::sync(); double t3 = wallNow();
      printf("sparse N=%lld per call: fast cond %.3fs uncond %.3fs parity %.3fs\n", (long long)coords.shape[0], (t1 - t0) / 3, (t2 - t1) / 3, (t3 - t2) / 3);
    }
  }
}
