// Interleaved A/B of the fast sparse flow transformer with the MPSNDArray matmul vs the own Metal 4 GEMM (same process).
#include "../src/pipeline.h"
using namespace t2;
int main(int argc, char** argv) {
  @autoreleasepool {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    fk::registerKernels(); tk::init("build/torchlibs/lib_01.metallib"); initOps();
    STFile ref("refdump/T_1024c.safetensors");
    Json pj = loadJson("models/trellis2/pipeline.json");
    float t1000 = 1000.0f; Tensor tt = from_cpu(tk::F32, {1}, &t1000);
    struct Case { const char* model; const char* coords; const char* feats; const char* cond; };
    for (Case cs : {Case{"shape_slat_flow_model_512", "ss", "shape_s.0.noise.feats", "cond.0.cond"}, Case{"shape_slat_flow_model_1024", "shape_s.1.noise.coords", "shape_s.1.noise.feats", "cond.1.cond"}}) {
      std::string c = "models/trellis2/" + pj["args"]["models"][cs.model].str;
      FastSparseFlow f; f.load(c + ".json", c + ".safetensors");
      Tensor coords = ref.get(cs.coords), x = ref.get(cs.feats), cond = ref.get(cs.cond);
      f.setCoords(coords); f.setCond(cond);
      Tensor a, b;
      for (int w = 0; w < 2; w++) { fk::gemmForceMps() = w; (w ? a : b) = f.forward(x, tt, true); f.forward(x, tt, false); tk::sync(); }
      relReport("mpp vs mps output", b, a);
      double sum[2][2] = {};
      const int R = 6;
      for (int r = 0; r < R; r++) for (int w = 0; w < 2; w++) {
        fk::gemmForceMps() = w;
        double t0 = wallNow(); for (int i = 0; i < 2; i++) f.forward(x, tt, true); tk::sync(); double t1 = wallNow();
        for (int i = 0; i < 2; i++) f.forward(x, tt, false); tk::sync(); double t2 = wallNow();
        sum[w][0] += (t1 - t0) / 2; sum[w][1] += (t2 - t1) / 2;
      }
      printf("N=%lld per call (mean of %d interleaved rounds): cond mps %.4fs mpp %.4fs (%+.1f%%) | uncond mps %.4fs mpp %.4fs (%+.1f%%)\n", (long long)coords.shape[0], R,
             sum[1][0] / R, sum[0][0] / R, 100 * (sum[0][0] / sum[1][0] - 1), sum[1][1] / R, sum[0][1] / R, 100 * (sum[0][1] / sum[1][1] - 1));
      pool().trim();
    }
  }
}
