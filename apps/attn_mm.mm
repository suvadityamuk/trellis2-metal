// Experiment: per-head attention as two MPS matmuls (QK^T, PV), without the softmax pass.
#include "../src/models/common.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Tensor rnd(DT dt, Shape s) { Tensor r(tk::F32, s); TorchRng g(1); tk::sync(); g.randn(r.ptr<float>(), r.numel()); return to(r, dt); }
int main(int argc, char** argv) {
  @autoreleasepool {
    tk::init("build/torchlibs/lib_01.metallib"); initOps();
    for (int64_t N : {4096, 16384}) {
      Tensor q = rnd(tk::BF16, {N, 128}), k = rnd(tk::BF16, {N, 128}), vt = rnd(tk::BF16, {128, N}), b1 = rnd(tk::BF16, {N}), b2 = rnd(tk::BF16, {128});
      double best = 1e9, bq = 1e9;
      for (int r = 0; r < 3; r++) {
        tk::sync(); double t0 = now(); Tensor s = linear(q, k, &b1); tk::sync(); double t1 = now(); Tensor o = linear(s, vt, &b2); tk::sync(); double t2 = now();
        best = std::min(best, t2 - t0); bq = std::min(bq, t1 - t0);
      }
      printf("N=%lld per head: QK^T %.2f ms + PV %.2f ms = %.2f ms -> x12 heads %.1f ms (%.1f TFLOPS, softmax pass not included)\n", (long long)N, bq * 1e3, (best - bq) * 1e3, best * 1e3, best * 12e3, 4.0 * N * N * 128 / best / 1e12);
    }
  }
}
