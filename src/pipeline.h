// Trellis2ImageTo3DPipeline.run (image -> MeshWithVoxel), pipeline types '512', '1024' and '1024_cascade' / '1536_cascade'.
// Templated on an implementation bundle so the parity build (torch kernel replay) and the fast build share the driver.
#pragma once
#include "models/dense_flow.h"
#include "models/sampler.h"
#include "models/sparse_flow.h"
#include "models/dino.h"
#include "models/ss_decoder.h"
#include "models/tex_decoder.h"
#include "fast/flow_fast.h"
#include <chrono>
#include <map>
#include <memory>

#include <mach/mach.h>
// process memory as Activity Monitor counts it (includes Metal buffers, MPS internals, mapped weights that were touched)
inline void footprint(double& cur, double& peak) {
  task_vm_info_data_t ti; mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &c) != KERN_SUCCESS) { cur = peak = 0; return; }
  cur = (double)ti.phys_footprint / (1 << 30); peak = (double)ti.ledger_phys_footprint_peak / (1 << 30);
}
using namespace t2;

namespace t2 {
inline double fpc() { double a, b; footprint(a, b); return a; }
inline double fpp() { double a, b; footprint(a, b); return b; }

inline double wallNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct PipeOpts {
  std::string root = ".", type = "512";
  uint64_t seed = 42;
  int64_t max_num_tokens = 49152;
  bool keep_models = true;      // keep weights resident between stages / runs (false: load per stage, free after)
  bool verbose = true;
  const STFile* ref = nullptr;  // optional reference dump: every stage boundary is compared bit for bit
};

struct PipeOut {
  MeshWithVoxel mesh; int res = 512;
  Tensor shape_slat, tex_slat, slat_coords;       // denormalised latents (for tolerance gates)
  std::map<std::string, double> t;                // stage seconds (weight loading excluded, reported as load_*)
  int64_t tokens = 0, tokens_lr = 0; bool parity_ok = true;
  int64_t glb_faces = 0, glb_verts = 0, glb_bytes = 0;  // set by build/t2 --glb
};

// The parity implementation bundle.
struct ParityImpl {
  using DenseFlowT = DenseFlow; using SparseFlowT = SparseFlow; using DinoT = Dino; using SSDecoderT = SSDecoder;
  using ShapeDecoderT = ShapeDecoder; using TexDecoderT = TexDecoder;
  static constexpr bool exact = true;
  static const char* name() { return "parity"; }
  template <class Hook> static Tensor sampleDense(const DenseFlow& m, const Tensor& noise, const Tensor& cond, const Tensor& neg, const SamplerParams& sp, Hook hook) {
    return flowEulerSample(m, noise, cond, neg, sp, [](const Tensor& x) { return std_dims(x, {1, 2, 3, 4}, true); }, hook);
  }
  template <class Hook> static Tensor sampleSparse(const SparseFlow& m, const Tensor& noise, const Tensor& cond, const Tensor& neg, const SamplerParams& sp, Hook hook) {
    return flowEulerSample(m, noise, cond, neg, sp, sp_std, hook, [&](const Tensor& r) { return sp_bcast(r, m.zidx); });
  }
  static void release(SparseFlow& m) { m.phases = Tensor(); m.zidx = Tensor(); m.concat = Tensor(); m.N = 0; }
  static Tensor denorm(const Tensor& x, const Json& n) { return sp_denorm(x, n); }
  static Tensor norm(const Tensor& x, const Json& n) { return sp_norm(x, n); }
};

// Fast build: fused-kernel flow transformers (src/fast/flow_fast.h); everything else as in the parity build.
struct FastImpl {
  using DenseFlowT = FastDenseFlow; using SparseFlowT = FastSparseFlow; using DinoT = Dino; using SSDecoderT = SSDecoder;
  using ShapeDecoderT = ShapeDecoder; using TexDecoderT = TexDecoder;
  static constexpr bool exact = false;
  static const char* name() { return "fast"; }
  template <class Hook> static Tensor sampleDense(const FastFlow& m, const Tensor& noise, const Tensor& cond, const Tensor& neg, const SamplerParams& sp, Hook hook) {
    Tensor r = flowEulerSample(bindFast(m, cond, neg), noise, cond, neg, sp, [](const Tensor& x) { return std_dims(x, {1, 2, 3, 4}, true); }, hook);
    m.cc = CondCache(); return r;
  }
  template <class Hook> static Tensor sampleSparse(const FastFlow& m, const Tensor& noise, const Tensor& cond, const Tensor& neg, const SamplerParams& sp, Hook hook) {
    Tensor r = flowEulerSample(bindFast(m, cond, neg), noise, cond, neg, sp, sp_std, hook, [&](const Tensor& r) { return sp_bcast(r, m.zidx); });
    m.cc = CondCache(); return r;
  }
  static void release(FastFlow& m) { m.releaseCoords(); }
  static Tensor denorm(const Tensor& x, const Json& n) { return sp_denorm(x, n); }
  static Tensor norm(const Tensor& x, const Json& n) { return sp_norm(x, n); }
};

// tolerance report for the fast build: relative L2 / max abs error against a reference tensor
inline bool relReport(const std::string& name, const Tensor& a_, const Tensor& b_) {
  tk::sync();
  if (a_.shape != b_.shape || a_.dt != b_.dt) {
    printf("  %-28s shape differs (", name.c_str()); for (auto s : a_.shape) printf("%lld ", (long long)s); printf("vs "); for (auto s : b_.shape) printf("%lld ", (long long)s); printf(")\n");
    return true;
  }
  Tensor a = contiguous_cpu(a_), b = contiguous_cpu(b_); int64_t n = a.numel();
  if (a.dt != tk::F32) { bool same = memcmp(a.raw(), b.raw(), a.nbytes()) == 0; printf("  %-28s %s\n", name.c_str(), same ? "identical" : "differs"); return true; }
  const float *x = a.ptr<float>(), *y = b.ptr<float>(); double num = 0, den = 0, mx = 0;
  for (int64_t i = 0; i < n; i++) { double d = (double)x[i] - y[i]; num += d * d; den += (double)y[i] * y[i]; mx = std::max(mx, std::fabs(d)); }
  printf("  %-28s rel_l2=%.3e maxabs=%.3e\n", name.c_str(), std::sqrt(num / std::max(den, 1e-30)), mx);
  return true;
}

// quant_coords = cat([hr[:, :1], ((hr[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int()]).unique(dim=0)
// (+0.5, /2^k and *int are exact in float32, so the CPU evaluation is bit identical; unique(dim=0) sorts rows lexicographically)
inline Tensor quantUniqueCoords(const Tensor& hr_, int lr_resolution, int hr_resolution) {
  tk::sync();
  Tensor hr = contiguous_cpu(hr_); int64_t n = hr.shape[0]; const int32_t* p = hr.ptr<int32_t>();
  std::vector<uint64_t> keys((size_t)n);
  float s = (float)(hr_resolution / 16);
  parallel_for(n, 1 << 16, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      uint64_t k = (uint64_t)(uint32_t)p[i * 4];
      for (int j = 1; j < 4; j++) { float v = ((float)p[i * 4 + j] + 0.5f) / (float)lr_resolution * s; k = (k << 16) | (uint64_t)(uint16_t)(int32_t)v; }
      keys[i] = k;
    }
  });
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  int64_t m = (int64_t)keys.size();
  Tensor r(tk::I32, {m, 4}); int32_t* o = r.ptr<int32_t>();
  for (int64_t i = 0; i < m; i++) { uint64_t k = keys[i]; o[i * 4] = (int32_t)(k >> 48); o[i * 4 + 1] = (int32_t)((k >> 32) & 0xffff); o[i * 4 + 2] = (int32_t)((k >> 16) & 0xffff); o[i * 4 + 3] = (int32_t)(k & 0xffff); }
  return r;
}

template <class I>
struct Pipeline {
  PipeOpts o; Json pj; TorchRng rng;
  std::unique_ptr<typename I::DinoT> dino;
  std::unique_ptr<typename I::DenseFlowT> ssflow;
  std::unique_ptr<typename I::SSDecoderT> ssdec;
  std::map<std::string, std::unique_ptr<typename I::SparseFlowT>> flows;
  std::unique_ptr<typename I::ShapeDecoderT> shapedec;
  std::unique_ptr<typename I::TexDecoderT> texdec;
  double tload = 0;

  explicit Pipeline(const PipeOpts& opts) : o(opts) { pj = loadJson(o.root + "/models/trellis2/pipeline.json"); }
  const Json& pa() const { return pj["args"]; }
  std::string ckpt(const char* key) const {
    std::string v = pa()["models"][key].str;
    return o.root + "/models/trellis2/" + v;
  }
  template <class M, class F> M& need(std::unique_ptr<M>& p, F loadf) {
    if (!p) { double t0 = wallNow(); p.reset(new M()); loadf(*p); tk::sync(); tload += wallNow() - t0; }
    return *p;
  }
  template <class M> void done(std::unique_ptr<M>& p) { if (!o.keep_models) { tk::sync(); p.reset(); pool().trim(); } }
  typename I::SparseFlowT& flow(const char* key) {
    return need(flows[key], [&](typename I::SparseFlowT& m) { std::string c = ckpt(key); m.load(c + ".json", c + ".safetensors"); });
  }
  void flowDone(const char* key) { done(flows[key]); }
  Tensor randn(const Shape& s) {
    int64_t n = 1; for (auto d : s) n *= d;
    std::vector<float> v((size_t)n); rng.randn(v.data(), (size_t)n);
    return from_cpu(tk::F32, s, v.data());
  }
  bool check(PipeOut& r, const std::string& name, const Tensor& a) {
    if (!o.ref || !o.ref->has(name)) return true;
    bool ok = I::exact ? cmp(name, a, o.ref->get(name)) : relReport(name, a, o.ref->get(name)); r.parity_ok &= ok; return ok;
  }

  // sample_shape_slat / sample_tex_slat body: noise -> sampler -> slat * std + mean
  Tensor sampleSlat(PipeOut& r, const char* key, const char* samp, const char* normKey, const Tensor& coords, const Tensor& cond,
                    const Tensor& neg, const Tensor& concat, const std::string& tag) {
    auto& m = flow(key);
    Tensor noise = randn({coords.shape[0], (int64_t)m.C_in - (concat.defined() ? concat.shape[1] : 0)});
    check(r, tag + ".noise.feats", noise);
    m.setCoords(coords, concat);
    SamplerParams sp = SamplerParams::fromJson(pa()[samp]);
    Tensor x = I::sampleSparse(m, noise, cond, neg, sp, [&](int i, const Tensor& pred, const Tensor& xn) {
      check(r, tag + ".x." + std::to_string(i) + ".feats", xn);
    });
    x = I::denorm(x, pa()[normKey]);
    tk::sync();
    I::release(m);
    flowDone(key);
    return x;
  }

  PipeOut run(const Image8& input, bool preprocess = true) {
    PipeOut r; auto& T = r.t;
    const std::string& ty = o.type;
    if (ty != "512" && ty != "1024" && ty != "1024_cascade" && ty != "1536_cascade") { fprintf(stderr, "invalid pipeline type %s\n", ty.c_str()); abort(); }
    double t00 = wallNow(), t0 = t00, l0 = 0; tload = 0;
    size_t gpeak = 0;
    auto lap = [&](const char* k) { tk::sync(); double t = wallNow(); T[k] = t - t0 - (tload - l0); t0 = t; l0 = tload;
      if (o.verbose) fprintf(stderr, "[%s] %-16s %.2fs  (stage pool peak %.2f GB, footprint %.2f GB, peak %.2f GB)\n", I::name(), k, T[k], (double)pool().peak / (1 << 30), fpc(), fpp());
      gpeak = std::max(gpeak, pool().peak); pool().peak = pool().live + pool().pooled; };

    Image8 img = preprocess ? preprocess_image(input) : input;
    lap("preprocess");
    rng.seed(o.seed);
    Tensor cond512, cond1024, neg512, neg1024;
    {
      auto& d = need(dino, [&](typename I::DinoT& m) { m.load(o.root + "/models/dino/config.json", o.root + "/models/dino/model.safetensors"); });
      cond512 = d(img, 512); neg512 = I::DinoT::negCond(cond512);
      if (ty != "512") { cond1024 = d(img, 1024); neg1024 = I::DinoT::negCond(cond1024); }
      tk::sync(); done(dino);
      check(r, "cond.0.cond", cond512); if (cond1024.defined()) check(r, "cond.1.cond", cond1024);
    }
    lap("cond");

    // sample_sparse_structure
    int ss_res = ty == "1024" ? 64 : 32;
    Tensor coords;
    {
      auto& m = need(ssflow, [&](typename I::DenseFlowT& m) { std::string c = ckpt("sparse_structure_flow_model"); m.load(c + ".json", c + ".safetensors"); });
      Tensor noise = randn({1, (int64_t)m.C_in, (int64_t)m.res, (int64_t)m.res, (int64_t)m.res});
      check(r, "ss_s.0.noise", noise);
      SamplerParams sp = SamplerParams::fromJson(pa()["sparse_structure_sampler"]);
      Tensor z = I::sampleDense(m, noise, cond512, neg512, sp, [&](int i, const Tensor&, const Tensor& xn) { check(r, "ss_s.0.x." + std::to_string(i), xn); });
      tk::sync(); done(ssflow);
      auto& d = need(ssdec, [&](typename I::SSDecoderT& m) { m.load(o.root + "/models/ss_dec/ss_dec_conv3d_16l8_fp16.json", o.root + "/models/ss_dec/ss_dec_conv3d_16l8_fp16.safetensors"); });
      coords = d.run(z, ss_res);
      tk::sync(); done(ssdec);
      check(r, "ss", coords);
    }
    lap("sparse_structure");
    if (coords.shape[0] == 0) { fprintf(stderr, "empty sparse structure\n"); abort(); }

    auto shapeDecoder = [&]() -> typename I::ShapeDecoderT& {
      return need(shapedec, [&](typename I::ShapeDecoderT& m) { std::string c = ckpt("shape_slat_decoder"); m.load(c + ".json", c + ".safetensors"); });
    };
    Tensor shape_slat; int res = 512;
    const Tensor& condHR = ty == "512" ? cond512 : cond1024; const Tensor& negHR = ty == "512" ? neg512 : neg1024;
    const char* texKey = ty == "512" ? "tex_slat_flow_model_512" : "tex_slat_flow_model_1024";
    if (ty == "512" || ty == "1024") {
      res = ty == "512" ? 512 : 1024;
      shape_slat = sampleSlat(r, ty == "512" ? "shape_slat_flow_model_512" : "shape_slat_flow_model_1024", "shape_slat_sampler",
                              "shape_slat_normalization", coords, condHR, negHR, Tensor(), "shape_s.0");
    } else {
      const int lr_resolution = 512, resolution = ty == "1024_cascade" ? 1024 : 1536;
      Tensor lr = sampleSlat(r, "shape_slat_flow_model_512", "shape_slat_sampler", "shape_slat_normalization", coords, cond512, neg512, Tensor(), "shape_s.0");
      r.tokens_lr = coords.shape[0];
      Tensor hr_coords;
      {
        auto& d = shapeDecoder();
        SparseCache cache;
        hr_coords = d.dec.upsample(SparseT{lr, coords}, 4, &cache);
        tk::sync();
      }
      pool().trim();
      int hr_resolution = resolution;
      while (true) {
        coords = quantUniqueCoords(hr_coords, lr_resolution, hr_resolution);
        if (coords.shape[0] < o.max_num_tokens || hr_resolution == 1024) break;
        hr_resolution -= 128;
      }
      if (hr_resolution != resolution && o.verbose) fprintf(stderr, "Due to the limited number of tokens, the resolution is reduced to %d.\n", hr_resolution);
      res = hr_resolution;
      check(r, "shape_s.1.noise.coords", coords);
      shape_slat = sampleSlat(r, "shape_slat_flow_model_1024", "shape_slat_sampler", "shape_slat_normalization", coords, cond1024, neg1024, Tensor(), "shape_s.1");
    }
    check(r, "shape.feats", shape_slat); check(r, "shape.coords", coords);
    r.tokens = coords.shape[0];
    lap("shape_slat");

    Tensor tex_slat = sampleSlat(r, texKey, "tex_slat_sampler", "tex_slat_normalization", coords, condHR, negHR,
                                 I::norm(shape_slat, pa()["shape_slat_normalization"]), "tex_s.0");
    check(r, "tex.feats", tex_slat);
    lap("tex_slat");

    // decode_latent
    SparseCache cache;
    typename I::ShapeDecoderT::Out so;
    {
      auto& d = shapeDecoder();
      d.set_resolution(res);
      so = d.run(SparseT{shape_slat, coords}, &cache);
      tk::sync(); done(shapedec);
      check(r, "mesh.vertices", so.mesh.vertices); check(r, "mesh.faces", so.mesh.faces);
    }
    so.h = SparseT();
    pool().trim();
    lap("decode_shape");
    {
      auto& d = need(texdec, [&](typename I::TexDecoderT& m) { std::string c = ckpt("tex_slat_decoder"); m.load(c + ".json", c + ".safetensors"); });
      SparseT vox = d.run(SparseT{tex_slat, coords}, so.subs, &cache);
      tk::sync(); done(texdec);
      r.mesh = make_mesh_with_voxel(so.mesh, vox);
    }
    so = typename I::ShapeDecoderT::Out(); cache.clear(); pool().trim();
    lap("decode_tex");
    check(r, "out.vertices", r.mesh.vertices); check(r, "out.faces", r.mesh.faces);
    check(r, "out.attrs", r.mesh.attrs); check(r, "out.coords", r.mesh.coords);
    pool().peak = std::max(gpeak, pool().peak);
    r.res = res; r.shape_slat = shape_slat; r.tex_slat = tex_slat; r.slat_coords = coords;
    T["load"] = tload;
    T["total"] = wallNow() - t00 - tload;
    return r;
  }
};

}  // namespace t2
