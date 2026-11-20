// Experiment: flash attention built on threadgroup-scope MPP matmul2d (large tiles) instead of torch's
// simdgroup-scope 16x32x16 fragments. S = Q K^T goes to threadgroup memory, online softmax there, O += P V
// into a cooperative accumulator. Aligned shapes only (qL % BQ == 0, kL % 128 == 0).
#include "../src/models/common.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static Tensor rnd(DT dt, Shape s) { Tensor r(tk::F32, s); TorchRng g(1); tk::sync(); g.randn(r.ptr<float>(), r.numel()); return to(r, dt); }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static const char* SRC = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct AP { int qL, kL, H; float scale; };
template <int BQ, int BK, int SG>
[[kernel, max_total_threads_per_threadgroup(SG * 32)]] void attn_tg(
    device bfloat* Q [[buffer(0)]], device bfloat* K [[buffer(1)]], device bfloat* V [[buffer(2)]], device bfloat* O [[buffer(3)]],
    constant AP& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint ts [[thread_index_in_simdgroup]]) {
  constexpr int D = 128;
  constexpr int TPR = SG * 32 / BQ;       // threads per query row
  constexpr int CPT = BK / TPR;           // score columns per thread
  threadgroup float S[BQ * BK];
  threadgroup float fac[BQ];
  const int q0 = tg.x * BQ, head = tg.y;
  Q += ((ulong)head * p.qL + q0) * D; O += ((ulong)head * p.qL + q0) * D;
  K += (ulong)head * p.kL * D; V += (ulong)head * p.kL * D;
  constexpr auto dS = matmul2d_descriptor(BQ, BK, D, false, true, false);
  constexpr auto dO = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroups<SG>> opS;
  matmul2d<dO, execution_simdgroups<SG>> opO;
  auto tQ = tensor(Q, extents<int, D, BQ>());
  auto tS = tensor((threadgroup float*)S, extents<int, BK, BQ>());
  auto tV0 = tensor(V, extents<int, D, BK>());
  auto cO = opO.template get_destination_cooperative_tensor<decltype(tS), decltype(tV0), float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) cO[i] = 0;
  const uint t = sg * 32 + ts;
  const int r = t / TPR, cb = (t % TPR) * CPT;
  const float scale2 = p.scale * 1.44269504089f;
  float max_score = -INFINITY, sum_score = 0;
  for (int k0 = 0; k0 < p.kL; k0 += BK) {
    auto tK = tensor(K + (ulong)k0 * D, extents<int, D, BK>());
    auto tV = tensor(V + (ulong)k0 * D, extents<int, D, BK>());
    opS.run(tQ, tK, tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (int j = 0; j < CPT; j++) m = max(m, S[r * BK + cb + j]);
    #pragma clang loop unroll(full)
    for (ushort s = 1; s < TPR; s <<= 1) m = max(m, simd_shuffle_xor(m, s));
    const float new_max = max(max_score, m * scale2);
    float sum = 0;
    for (int j = 0; j < CPT; j++) { float e = fast::exp2(S[r * BK + cb + j] * scale2 - new_max); S[r * BK + cb + j] = e; sum += e; }
    #pragma clang loop unroll(full)
    for (ushort s = 1; s < TPR; s <<= 1) sum += simd_shuffle_xor(sum, s);
    const float factor = fast::exp2(max_score - new_max);
    sum_score = sum_score * factor + sum; max_score = new_max;
    if (t % TPR == 0) fac[r] = factor;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) { auto idx = cO.get_multidimensional_index(i); cO[i] *= fac[idx[1]]; }
    opO.run(tS, tV, cO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t % TPR == 0) fac[r] = 1.0f / sum_score;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO.get_capacity(); ++i) if (cO.is_valid_element(i)) { auto idx = cO.get_multidimensional_index(i); O[idx[1] * D + idx[0]] = (bfloat)(cO[i] * fac[idx[1]]); }
}
#define INST(bq, bk, sg) template [[host_name("attn_tg_" #bq "_" #bk "_" #sg)]] [[kernel]] decltype(attn_tg<bq, bk, sg>) attn_tg<bq, bk, sg>;
)MSL";
struct V { int bq, bk, sg; };
int main(int argc, char** argv) {
  @autoreleasepool {
    tk::init("build/torchlibs/lib_01.metallib"); initOps();
    std::vector<V> vs = {{32, 128, 2}, {32, 64, 2}, {32, 128, 4}, {16, 128, 1}, {32, 256, 2}, {16, 256, 1}, {16, 128, 2}, {64, 64, 4}};
    std::string src = SRC; char b[256];
    for (auto& v : vs) { snprintf(b, sizeof b, "INST(%d, %d, %d)\n", v.bq, v.bk, v.sg); src += b; }
    NSError* err = nil; MTLCompileOptions* o = [MTLCompileOptions new]; o.mathMode = MTLMathModeSafe; o.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> L = [tk::ctx().dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!L) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    for (int64_t N : {4096, 16384}) {
      const int64_t H = 12, D = 128;
      Tensor q = rnd(tk::BF16, {1, H, N, D}), k = rnd(tk::BF16, {1, H, N, D}), v = rnd(tk::BF16, {1, H, N, D});
      Tensor ref = sdpa(q, k, v); tk::sync();
      double best = 1e9;
      for (int r = 0; r < 4; r++) { tk::sync(); double t = now(); for (int i = 0; i < 3; i++) sdpa(q, k, v); tk::sync(); best = std::min(best, (now() - t) / 3); }
      printf("N=%lld stock sdpa: %.2f ms %.1f TFLOPS\n", (long long)N, best * 1e3, 4.0 * H * N * N * D / best / 1e12);
      for (auto& vv : vs) {
        snprintf(b, sizeof b, "attn_tg_%d_%d_%d", vv.bq, vv.bk, vv.sg);
        id<MTLFunction> fn = [L newFunctionWithName:[NSString stringWithUTF8String:b]];
        id<MTLComputePipelineState> p = fn ? [tk::ctx().dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!p) { printf("  %s: failed %s\n", b, err ? err.localizedDescription.UTF8String : "missing"); continue; }
        struct { int qL, kL, H; float scale; } pr = {(int)N, (int)N, (int)H, (float)(1.0 / std::sqrt((double)D))};
        Tensor out(tk::BF16, {1, H, N, D});
        auto run = [&]() {
          auto e = tk::enc(); [e setComputePipelineState:p];
          tk::setBuf(e, q.buf(), 0); tk::setBuf(e, k.buf(), 1); tk::setBuf(e, v.buf(), 2); tk::setBuf(e, out.buf(), 3); tk::setVal(e, pr, 4);
          [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(N / vv.bq), (NSUInteger)H, 1) threadsPerThreadgroup:MTLSizeMake(32 * vv.sg, 1, 1)];
        };
        run(); tk::sync();
        const uint16_t* a = ref.ptr<uint16_t>(); const uint16_t* c = out.ptr<uint16_t>(); int64_t nd = 0; double num = 0, den = 0;
        for (int64_t i = 0; i < ref.numel(); i++) { nd += a[i] != c[i]; float x = bf(a[i]), y = bf(c[i]); num += (double)(x - y) * (x - y); den += (double)x * x; }
        best = 1e9;
        for (int r = 0; r < 4; r++) { tk::sync(); double t = now(); for (int i = 0; i < 3; i++) run(); tk::sync(); best = std::min(best, (now() - t) / 3); }
        printf("  %-18s %.2f ms %.1f TFLOPS  diff=%lld/%lld relL2=%.2e\n", b, best * 1e3, 4.0 * H * N * N * D / best / 1e12, (long long)nd, (long long)ref.numel(), std::sqrt(num / std::max(den, 1e-30)));
      }
    }
  }
}
