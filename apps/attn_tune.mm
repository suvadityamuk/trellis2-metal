// Experiment: recompile torch's MPP flash-attention template at runtime (Metal 4) with other block shapes.
#include "../src/models/common.h"
#include <chrono>
#include <fstream>
#include <sstream>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Tensor rnd(DT dt, Shape s) { Tensor r(tk::F32, s); TorchRng g(1); tk::sync(); g.randn(r.ptr<float>(), r.numel()); return to(r, dt); }
struct V { int bq, bk, wm; };
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char** argv) {
  @autoreleasepool {
    tk::init("build/torchlibs/lib_01.metallib"); initOps();
    std::ifstream f("src/fast/mpp/attn_mpp_base.metal"); std::stringstream ss; ss << f.rdbuf(); std::string src = ss.str();
    std::vector<V> vs = {{64, 32, 4}};
    auto rep = [](std::string s, const std::string& a, const std::string& b, bool last = true) { size_t p = last ? s.rfind(a) : s.find(a); if (p == std::string::npos) { fprintf(stderr, "pattern not found: %s\n", a.c_str()); exit(1); } s.replace(p, a.size(), b); return s; };
    const std::string base = src; int variant = argc > 1 ? atoi(argv[1]) : 0;
    if (variant == 1) src = rep(base, "using AccumType = c10::metal::accum_t<T>;", "using AccumType = half;");
    if (variant == 2) src = rep(base, "          if (id == 4) {\n            threadgroup_barrier(mem_flags::mem_none);\n          }", "");
    if (variant == 3) src = rep(base, "    simdgroup_barrier(mem_flags::mem_none);\n\n    // O += P @ V.", "    // O += P @ V.");
    if (variant == 4) src = rep(rep(base, "    simdgroup_barrier(mem_flags::mem_none);\n\n    // O += P @ V.", "    // O += P @ V."), "          if (id == 4) {\n            threadgroup_barrier(mem_flags::mem_none);\n          }", "");
    printf("variant %d\n", variant);
    char b[512];
    for (auto& v : vs) { snprintf(b, sizeof b, "instantiate_kernel(\"att_bq%d_bk%d_wm%d\", prefill_attention_mpp, bfloat, %d, %d, 128, %d, 1, 0, 0, true, bfloat)\n", v.bq, v.bk, v.wm, v.bq, v.bk, v.wm); src += b; }
    NSError* err = nil; MTLCompileOptions* o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe; o.languageVersion = MTLLanguageVersion4_0;
    double t0 = now();
    id<MTLLibrary> L = [tk::ctx().dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!L) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("compiled in %.1fs\n", now() - t0);
    for (int64_t N : {4096, 16384}) {
      const int64_t H = 12, D = 128;
      Tensor q = rnd(tk::BF16, {1, H, N, D}), k = rnd(tk::BF16, {1, H, N, D}), v = rnd(tk::BF16, {1, H, N, D});
      Tensor ref = sdpa(q, k, v); tk::sync();
      double best = 1e9;
      for (int r = 0; r < 4; r++) { tk::sync(); double t = now(); for (int i = 0; i < 3; i++) sdpa(q, k, v); tk::sync(); best = std::min(best, (now() - t) / 3); }
      printf("N=%lld stock sdpa: %.2f ms %.1f TFLOPS\n", (long long)N, best * 1e3, 4.0 * H * N * N * D / best / 1e12);
      for (auto& vv : vs) {
        snprintf(b, sizeof b, "att_bq%d_bk%d_wm%d", vv.bq, vv.bk, vv.wm);
        id<MTLFunction> fn = [L newFunctionWithName:[NSString stringWithUTF8String:b]];
        if (!fn) { printf("%s: missing\n", b); continue; }
        id<MTLComputePipelineState> p = [tk::ctx().dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) { printf("%s: pso failed %s\n", b, err.localizedDescription.UTF8String); continue; }
        tk::PrefillAttnParams pr; memset(&pr, 0, sizeof pr);
        pr.B = 1; pr.H = H; pr.D = D; pr.qL = (int)N; pr.kL = (int)N; pr.gqa_factor = 1; pr.scale = (float)(1.0 / std::sqrt((double)D)); pr.softcapping = 1.0f;
        tk::sdpa_contiguous_strides(pr.Q_strides, H, N, D); tk::sdpa_contiguous_strides(pr.K_strides, H, N, D); tk::sdpa_contiguous_strides(pr.V_strides, H, N, D); tk::sdpa_contiguous_strides(pr.O_strides, H, N, D);
        Tensor out(tk::BF16, {1, H, N, D});
        auto run = [&]() {
          auto e = tk::enc(); [e setComputePipelineState:p];
          tk::setBuf(e, q.buf(), 0); tk::setBuf(e, k.buf(), 1); tk::setBuf(e, v.buf(), 2); tk::setBuf(e, out.buf(), 3); tk::setVal(e, pr, 4);
          const uint32_t zero = 0; tk::setVal(e, zero, 5); tk::setVal(e, zero, 6);
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + vv.bq - 1) / vv.bq), (NSUInteger)H, 1) threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)vv.wm, 1)];
        };
        run(); tk::sync();
        const uint16_t* a = ref.ptr<uint16_t>(); const uint16_t* c = out.ptr<uint16_t>(); int64_t nd = 0; double num = 0, den = 0;
        for (int64_t i = 0; i < ref.numel(); i++) { nd += a[i] != c[i]; float x = bf(a[i]), y = bf(c[i]); num += (double)(x - y) * (x - y); den += (double)x * x; }
        best = 1e9;
        for (int r = 0; r < 4; r++) { tk::sync(); double t = now(); for (int i = 0; i < 3; i++) run(); tk::sync(); best = std::min(best, (now() - t) / 3); }
        printf("N=%lld %-22s %.2f ms %.1f TFLOPS  diff=%lld relL2=%.2e\n", (long long)N, b, best * 1e3, 4.0 * H * N * N * D / best / 1e12, (long long)nd, std::sqrt(num / std::max(den, 1e-30)));
      }
    }
  }
}
