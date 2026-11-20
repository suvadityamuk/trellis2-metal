// Experiment: cost of the fast build's elementwise passes in isolation (per block, N tokens).
#include "../src/pipeline.h"
using namespace t2;
static Tensor rnd(DT dt, Shape s) { Tensor r(tk::F32, s); TorchRng g(1); tk::sync(); g.randn(r.ptr<float>(), r.numel()); return to(r, dt); }
template <class F> static double tm(F f) { double best = 1e9; for (int r = 0; r < 4; r++) { tk::sync(); double t = wallNow(); for (int i = 0; i < 5; i++) f(); tk::sync(); best = std::min(best, (wallNow() - t) / 5); } return best * 1e3; }
int main(int argc, char** argv) {
  @autoreleasepool {
    fk::registerKernels(); tk::init("build/torchlibs/lib_01.metallib"); initOps();
    int64_t N = argc > 1 ? atoll(argv[1]) : 16384, C = 1536, H = 12;
    Tensor x = rnd(tk::F32, {N, C}), o = rnd(tk::BF16, {N, C}), h(tk::BF16, {N, C}), vec = rnd(tk::F32, {4, C}), qkv = rnd(tk::BF16, {N, 3 * C}), qb = rnd(tk::F32, {3 * C}), g = rnd(tk::F32, {C});
    Tensor ph = rnd(tk::F32, {N, 64, 2}), Q(tk::BF16, {H, N, 128}), mlp = rnd(tk::BF16, {N, 8192}), mb = rnd(tk::F32, {8192}), W0 = rnd(tk::BF16, {8192, C}), W2 = rnd(tk::BF16, {C, 8192}), Wq = rnd(tk::BF16, {3 * C, C}), Wo = rnd(tk::BF16, {C, C}), O = rnd(tk::BF16, {N, C});
    Tensor K = rnd(tk::BF16, {H, N, 128}), V = rnd(tk::BF16, {H, N, 128}); Tensor Kc = rnd(tk::BF16, {H, 1029, 128}), Vc = rnd(tk::BF16, {H, 1029, 128});
    double a = tm([&] { fk::residLn(x, &o, vec, 0, h, fk::RESID | fk::LN | fk::AFFINE, 1e-6f, N, C); });
    double b = tm([&] { fk::qprep(qkv, 3 * C, 0, qb, &g, &ph, Q, fk::QNORM | fk::QROPE, N, H); });
    double b2 = tm([&] { fk::qprep(qkv, 3 * C, 2 * C, qb, nullptr, nullptr, Q, 0, N, H); });
    double c = tm([&] { fk::biasGelu(mlp, mb, N, 8192); });
    double m0 = tm([&] { fk::matmul(h, W0, mlp, N); }), m2 = tm([&] { fk::matmul(mlp, W2, o, N); }), mq = tm([&] { fk::matmul(h, Wq, qkv, N); }), mo = tm([&] { fk::matmul(O, Wo, o, N); });
    double at = tm([&] { fk::attention(Q, K, V, O, H, N, N, 128); }), ac = tm([&] { fk::attention(Q, Kc, Vc, O, H, N, 1029, 128); });
    double epi = 3 * a + 2 * b + b2 + b + c, mm = m0 + m2 + mq + 3 * mo, tot = epi + mm + at + ac;
    printf("N=%lld per block (cond call): residLn %.2f x3, qprep(norm+rope) %.2f x3, qprep(v) %.2f, biasGelu %.2f -> elementwise %.1f ms (%.0f%%)\n", (long long)N, a, b, b2, c, epi, 100 * epi / tot);
    printf("  matmul: mlp0 %.2f mlp2 %.2f qkv %.2f out %.2f x3 -> %.1f ms (%.0f%%); self-attn %.2f (%.0f%%); cross-attn %.2f (%.0f%%); total %.1f ms -> x30 = %.2f s\n", m0, m2, mq, mo, mm, 100 * mm / tot, at, 100 * at / tot, ac, 100 * ac / tot, tot, tot * 30 / 1e3);
  }
}
