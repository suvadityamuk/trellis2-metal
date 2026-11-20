// Experiment: own GEMM on MetalPerformancePrimitives matmul2d (Metal 4), with an optional in-register
// bias + GELU epilogue (cooperative destination tensor), against MPSNDArray matmul.
#include "../src/models/common.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Tensor rnd(DT dt, Shape s, float sc = 1) { Tensor r(tk::F32, s); TorchRng g(1); tk::sync(); g.randn(r.ptr<float>(), r.numel()); float* p = r.ptr<float>(); for (int64_t i = 0; i < r.numel(); i++) p[i] *= sc; return to(r, dt); }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static const char* SRC = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct MMP { int M, N, K, pad; };
template <int TM, int TN, int SG, int MODE, bool RELAX>
[[kernel, max_total_threads_per_threadgroup(SG * 32)]] void mm(
    device bfloat* X [[buffer(0)]], device bfloat* W [[buffer(1)]], device bfloat* Y [[buffer(2)]],
    const device bfloat* bias [[buffer(3)]], constant MMP& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, RELAX);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto A = tensor(X, dextents<int, 2>(p.K, p.M));
  auto B = tensor(W, dextents<int, 2>(p.K, p.N));
  auto C = tensor(Y, dextents<int, 2>(p.N, p.M));
  const int r0 = tg.y * TM, c0 = tg.x * TN;
  if constexpr (MODE == 0) {
    auto mA = A.slice(0, r0); auto mB = B.slice(0, c0); auto mC = C.slice(c0, r0);
    op.run(mA, mB, mC);
  } else if constexpr (MODE == 1) {
    if (r0 + TM <= p.M && c0 + TN <= p.N) {
      auto mA = A.template slice<dynamic_extent, TM>(0, r0); auto mB = B.template slice<dynamic_extent, TN>(0, c0); auto mC = C.template slice<TN, TM>(c0, r0);
      op.run(mA, mB, mC);
    } else {
      auto mA = A.slice(0, r0); auto mB = B.slice(0, c0); auto mC = C.slice(c0, r0);
      op.run(mA, mB, mC);
    }
  } else {
    auto mA = A.slice(0, r0); auto mB = B.slice(0, c0); auto mC = C.slice(c0, r0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), bfloat>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(mA, mB, cT);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
      if (cT.is_valid_element(i)) {
        auto idx = cT.get_multidimensional_index(i);
        int c = min(c0 + (int)idx[0], p.N - 1);
        float v = (float)(bfloat)((float)cT[i] + (float)bias[c]);
        if constexpr (MODE == 3) v = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        cT[i] = (bfloat)v;
      }
    }
    cT.store(mC);
  }
}
#define INST(tm, tn, sg, mode, rl) template [[host_name("mm_" #tm "_" #tn "_" #sg "_" #mode "_" #rl)]] [[kernel]] decltype(mm<tm, tn, sg, mode, rl>) mm<tm, tn, sg, mode, rl>;
)MSL";
struct V { int tm, tn, sg, mode, rl; };
int main(int argc, char** argv) {
  @autoreleasepool {
    tk::init("build/torchlibs/lib_01.metallib"); initOps();
    std::vector<V> vs = {{64, 128, 4, 1, 0}, {64, 128, 4, 0, 0}, {64, 256, 4, 1, 0}, {128, 128, 4, 1, 0}, {128, 128, 8, 1, 0}, {64, 128, 2, 1, 0}, {32, 128, 2, 1, 0}, {32, 128, 1, 1, 0}, {64, 128, 8, 1, 0}, {32, 256, 4, 1, 0}, {128, 256, 8, 1, 0}, {64, 192, 4, 1, 0},
                         {64, 128, 4, 2, 0}, {64, 128, 4, 3, 0}};
    std::string src = SRC; char b[256];
    for (auto& v : vs) { snprintf(b, sizeof b, "INST(%d, %d, %d, %d, %s)\n", v.tm, v.tn, v.sg, v.mode, v.rl ? "true" : "false"); src += b; }
    NSError* err = nil; MTLCompileOptions* o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe; o.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> L = [tk::ctx().dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!L) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    int64_t N = argc > 1 ? atoll(argv[1]) : 16384;
    for (auto io : std::vector<std::pair<int64_t, int64_t>>{{1536, 8192}, {8192, 1536}, {1536, 4608}, {1536, 1536}}) {
      const int64_t K = io.first, O = io.second;
      Tensor x = rnd(tk::BF16, {N, K}), W = rnd(tk::BF16, {O, K}, 0.02f), bias = rnd(tk::BF16, {O});
      Tensor ref = linear(x, W, nullptr), refb = linear(x, W, &bias); tk::sync();
      double best = 1e9;
      for (int r = 0; r < 4; r++) { tk::sync(); double t = now(); for (int i = 0; i < 3; i++) linear(x, W, nullptr); tk::sync(); best = std::min(best, (now() - t) / 3); }
      printf("N=%lld %lld->%lld  MPS: %.2f ms %.1f TFLOPS\n", (long long)N, (long long)K, (long long)O, best * 1e3, 2.0 * N * K * O / best / 1e12);
      for (auto& v : vs) {
        snprintf(b, sizeof b, "mm_%d_%d_%d_%d_%s", v.tm, v.tn, v.sg, v.mode, v.rl ? "true" : "false");
        id<MTLFunction> fn = [L newFunctionWithName:[NSString stringWithUTF8String:b]];
        id<MTLComputePipelineState> p = fn ? [tk::ctx().dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!p) { printf("  %s: failed %s\n", b, err ? err.localizedDescription.UTF8String : "missing"); continue; }
        struct { int M, N, K, pad; } pr = {(int)N, (int)O, (int)K, 0};
        Tensor out(tk::BF16, {N, O});
        auto run = [&]() {
          auto e = tk::enc(); [e setComputePipelineState:p];
          tk::setBuf(e, x.buf(), 0); tk::setBuf(e, W.buf(), 1); tk::setBuf(e, out.buf(), 2); tk::setBuf(e, bias.buf(), 3); tk::setVal(e, pr, 4);
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((O + v.tn - 1) / v.tn), (NSUInteger)((N + v.tm - 1) / v.tm), 1) threadsPerThreadgroup:MTLSizeMake(32 * v.sg, 1, 1)];
        };
        run(); tk::sync();
        const Tensor& rr = v.mode == 2 ? refb : ref;
        const uint16_t* a = rr.ptr<uint16_t>(); const uint16_t* c = out.ptr<uint16_t>(); int64_t nd = 0; double num = 0, den = 0;
        for (int64_t i = 0; i < rr.numel(); i++) { nd += a[i] != c[i]; float xx = bf(a[i]), y = bf(c[i]); num += (double)(xx - y) * (xx - y); den += (double)xx * xx; }
        best = 1e9;
        for (int r = 0; r < 4; r++) { tk::sync(); double t = now(); for (int i = 0; i < 3; i++) run(); tk::sync(); best = std::min(best, (now() - t) / 3); }
        printf("  %-22s %.2f ms %.1f TFLOPS  diff=%lld relL2=%.2e%s\n", b, best * 1e3, 2.0 * N * K * O / best / 1e12, (long long)nd, std::sqrt(num / std::max(den, 1e-30)), v.mode == 3 ? " (gelu: diff expected)" : "");
      }
    }
  }
}
