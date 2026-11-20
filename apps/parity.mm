// Stage-by-stage bitwise parity checks of the Metal runtime against tensors dumped from the reference pipeline.
#include "../src/models/dense_flow.h"
#include "../src/models/sampler.h"
#include "../src/models/sparse_flow.h"
#include <chrono>
using namespace t2;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string root = ".", dump = "refdump/T_512.safetensors", stage = "all";
    int steps = -1;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--root") root = argv[++i]; else if (a == "--dump") dump = argv[++i];
      else if (a == "--stage") stage = argv[++i]; else if (a == "--steps") steps = atoi(argv[++i]);
    }
    tk::init((root + "/build/torchlibs/lib_01.metallib").c_str());
    initOps();
    STFile ref(root + "/" + dump);
    Json pj = loadJson(root + "/models/trellis2/pipeline.json");
    const Json& pa = pj["args"];
    bool ok = true;
    if (stage == "consts" || stage == "all") {
      STFile c(root + "/refdump/consts.safetensors");
      ok &= cmp("temb_freqs", from_cpu(tk::F32, {128}, kTembFreqBits), c.get("temb_freqs"));
      Tensor ph = dense_rope_phases(16, 128);
      ok &= cmp("rope_dense16", view_as_real(ph), c.get("rope_dense16"));
    }
    if (stage == "ss" || stage == "all") {
      double t0 = now();
      DenseFlow m; m.load(root + "/models/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.json", root + "/models/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors");
      printf("ss flow loaded in %.2fs\n", now() - t0);
      SamplerParams sp = SamplerParams::fromJson(pa["sparse_structure_sampler"]);
      if (steps > 0) sp.steps = steps;
      Tensor cond = ref.get("cond.0.cond"), neg = ref.get("cond.0.neg_cond"), noise = ref.get("ss_s.0.noise");
      auto stdf = [](const Tensor& x) { return std_dims(x, {1, 2, 3, 4}, true); };
      t0 = now();
      flowEulerSample(m, noise, cond, neg, sp, stdf, [&](int i, const Tensor& pred, const Tensor& xn) {
        std::string s = std::to_string(i);
        if (ref.has("ss_s.0.pred." + s)) { ok &= cmp("ss pred " + s, pred, ref.get("ss_s.0.pred." + s)); ok &= cmp("ss x " + s, xn, ref.get("ss_s.0.x." + s)); }
      });
      tk::sync();
      printf("ss sampler: %.2fs\n", now() - t0);
    }
    auto sparseStage = [&](const std::string& tag, const std::string& ckpt, const char* samp, const Tensor& concat, const char* normKey) {
      double t0 = now();
      SparseFlow m; m.load(root + "/models/trellis2/ckpts/" + ckpt + ".json", root + "/models/trellis2/ckpts/" + ckpt + ".safetensors");
      printf("%s flow loaded in %.2fs\n", tag.c_str(), now() - t0);
      SamplerParams sp = SamplerParams::fromJson(pa[samp]);
      if (steps > 0) sp.steps = steps;
      Tensor cond = ref.get("cond.0.cond"), neg = ref.get("cond.0.neg_cond");
      Tensor noise = ref.get(tag + "_s.0.noise.feats"), coords = ref.get(tag + "_s.0.noise.coords");
      m.setCoords(coords, concat);
      t0 = now();
      Tensor x = flowEulerSample(m, noise, cond, neg, sp, sp_std, [&](int i, const Tensor& pred, const Tensor& xn) {
        std::string s = std::to_string(i);
        if (ref.has(tag + "_s.0.pred." + s + ".feats")) { ok &= cmp(tag + " pred " + s, pred, ref.get(tag + "_s.0.pred." + s + ".feats")); ok &= cmp(tag + " x " + s, xn, ref.get(tag + "_s.0.x." + s + ".feats")); }
      }, [&](const Tensor& r) { return sp_bcast(r, m.zidx); });
      x = sp_denorm(x, pa[normKey]);
      tk::sync();
      printf("%s sampler: %.2fs (N=%lld)\n", tag.c_str(), now() - t0, (long long)m.N);
      if (steps <= 0) ok &= cmp(tag + " slat", x, ref.get(tag + ".feats"));
    };
    if (stage == "shape" || stage == "all")
      sparseStage("shape", "slat_flow_img2shape_dit_1_3B_512_bf16", "shape_slat_sampler", Tensor(), "shape_slat_normalization");
    if (stage == "tex" || stage == "all")
      sparseStage("tex", "slat_flow_imgshape2tex_dit_1_3B_512_bf16", "tex_slat_sampler", sp_norm(ref.get("shape.feats"), pa["shape_slat_normalization"]), "tex_slat_normalization");
    printf(ok ? "PARITY OK\n" : "PARITY FAILED\n");
    return ok ? 0 : 1;
  }
}
