// Throughput probes for the fast build (matmul / attention).
#include "../src/models/common.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Tensor rnd(DT dt, Shape s) { Tensor r(tk::F32, s); TorchRng g(1); tk::sync(); g.randn(r.ptr<float>(), r.numel()); return to(r, dt); }
int main(int argc, char** argv) {
  @autoreleasepool {
    tk::init("build/torchlibs/lib_01.metallib"); initOps();
    int64_t N = argc > 1 ? atoll(argv[1]) : 4096;
    for (DT dt : {tk::BF16, tk::F16, tk::F32}) {
      for (auto io : std::vector<std::pair<int64_t,int64_t>>{{1536, 8192}, {8192, 1536}, {1536, 4608}, {1536, 1536}}) {
        Tensor x = rnd(dt, {N, io.first}), W = rnd(dt, {io.second, io.first}), b = rnd(dt, {io.second});
        double best = 1e9;
        for (int r = 0; r < 4; r++) { tk::sync(); double t0 = now(); for (int i = 0; i < 5; i++) linear(x, W, &b); tk::sync(); best = std::min(best, (now() - t0) / 5); }
        printf("linear %-5s N=%lld %lld->%lld: %.2f ms  %.1f TFLOPS\n", tk::dtName(dt), (long long)N, (long long)io.first, (long long)io.second, best * 1e3, 2.0 * N * io.first * io.second / best / 1e12);
      }
      Tensor q = rnd(dt, {1, 12, N, 128}), k = rnd(dt, {1, 12, N, 128}), v = rnd(dt, {1, 12, N, 128});
      double best = 1e9;
      for (int r = 0; r < 4; r++) { tk::sync(); double t0 = now(); for (int i = 0; i < 3; i++) sdpa(q, k, v); tk::sync(); best = std::min(best, (now() - t0) / 3); }
      printf("sdpa   %-5s N=%lld: %.2f ms  %.1f TFLOPS\n", tk::dtName(dt), (long long)N, best * 1e3, 4.0 * 12 * N * N * 128 / best / 1e12);
    }
  }
}
