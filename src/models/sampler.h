// FlowEulerGuidanceIntervalSampler — parity build. Latents are plain tensors (dense [B,C,D,H,W] or sparse feats [N,C]).
#pragma once
#include "common.h"

namespace t2 {

struct SamplerParams {
  int steps = 12; double gs = 7.5, rescale = 0.7, lo = 0.6, hi = 1.0, rescale_t = 5.0, sigma_min = 1e-5;
  static SamplerParams fromJson(const Json& j) {
    SamplerParams p; const Json& a = j["params"];
    p.steps = (int)a["steps"].i(); p.gs = a["guidance_strength"].d(); p.rescale = a["guidance_rescale"].d();
    p.lo = a["guidance_interval"][0].d(); p.hi = a["guidance_interval"][1].d(); p.rescale_t = a["rescale_t"].d();
    p.sigma_min = j["args"]["sigma_min"].d();
    return p;
  }
};

inline std::vector<double> tSchedule(int steps, double rescale_t) {
  std::vector<double> t(steps + 1);
  double step = (0.0 - 1.0) / steps;
  for (int i = 0; i <= steps; i++) t[i] = i * step + 1.0;  // np.linspace(1, 0, steps + 1)
  t[steps] = 0.0;
  for (auto& v : t) v = rescale_t * v / (1 + (rescale_t - 1) * v);
  return t;
}

// model(x, t f32 [B], cond) -> pred. std_fn(x) -> per-sample std keepdim (dense: dims 1..; sparse: handled by caller).
template <class Model, class Hook>
Tensor flowEulerSample(const Model& model, Tensor x, const Tensor& cond, const Tensor& neg, const SamplerParams& sp,
                       const std::function<Tensor(const Tensor&)>& std_fn, Hook hook,
                       const std::function<Tensor(const Tensor&)>& bcast = nullptr) {
  std::vector<double> ts = tSchedule(sp.steps, sp.rescale_t);
  double sm = sp.sigma_min;
  for (int i = 0; i < sp.steps; i++) @autoreleasepool {  // drop Metal/MPS temporaries every step
    double t = ts[i], tp = ts[i + 1];
    float t1000 = (float)(1000 * t);
    Tensor tt = from_cpu(tk::F32, {x.shape[0] > 0 && x.ndim() == 5 ? x.shape[0] : 1}, &t1000);
    double gs = (sp.lo <= t && t <= sp.hi) ? sp.gs : 1.0;
    Tensor pred;
    if (gs == 1) pred = model(x, tt, cond);
    else if (gs == 0) pred = model(x, tt, neg);
    else {
      Tensor pp = model(x, tt, cond), pn = model(x, tt, neg);
      pred = add(mul(gs, pp), mul(1 - gs, pn));
      if (sp.rescale > 0) {
        auto xstart = [&](const Tensor& p) { return sub(mul(1 - sm, x), mul(sm + (1 - sm) * t, p)); };
        Tensor x0p = xstart(pp), x0c = xstart(pred);
        Tensor sdp = std_fn(x0p), sdc = std_fn(x0c);
        Tensor ratio = div(sdp, sdc);
        Tensor x0r = mul(x0c, bcast ? bcast(ratio) : ratio);
        Tensor x0 = add(mul(sp.rescale, x0r), mul(1 - sp.rescale, x0c));
        pred = div(sub(mul(1 - sm, x), x0), sm + (1 - sm) * t);
      }
    }
    Tensor xn = sub(x, mul(t - tp, pred));
    hook(i, pred, xn);
    x = xn;
  }
  return x;
}

}  // namespace t2
