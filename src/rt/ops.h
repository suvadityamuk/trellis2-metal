// Torch-like op layer for the parity build. Every float computation goes through a torch-MPS kernel replay
// (src/torchk); pure data movement / integer bookkeeping uses our own kernels (xk) or CPU code.
#pragma once
#include "tensor.h"
#include "../torchk/elem.h"
#include "../torchk/reduce.h"
#include "../torchk/heavy.h"

namespace t2 {

// ---- own runtime-compiled kernels (data movement only in the parity build) ---------------------------------------
namespace xk {
inline std::string& extraSource() { static std::string s; return s; }
inline const char* baseSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct GP { uint C; uint N; uint rowOff; uint pad; };
#define GATHER(NAME, T, IT) \
kernel void NAME(device const T* src [[buffer(0)]], device const IT* idx [[buffer(1)]], device T* dst [[buffer(2)]], \
                 constant GP& p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) { \
  ulong i = (ulong)idx[g.y + p.rowOff]; T v = 0; if (i < (ulong)p.N) v = src[i * p.C + g.x]; \
  dst[(ulong)g.y * p.C + g.x] = v; }
GATHER(gather_u8_i32, uchar, int)   GATHER(gather_u8_i64, uchar, long)
GATHER(gather_u16_i32, ushort, int) GATHER(gather_u16_i64, ushort, long)
GATHER(gather_u32_i32, uint, int)   GATHER(gather_u32_i64, uint, long)
// im2col gather for sparse conv: nbr [rows,27] -> dst [rows, 27*C]
#define GATHER27(NAME, T) \
kernel void NAME(device const T* src [[buffer(0)]], device const int* nbr [[buffer(1)]], device T* dst [[buffer(2)]], \
                 constant GP& p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) { \
  uint k = g.x / p.C, c = g.x - k * p.C; \
  uint i = (uint)nbr[(ulong)(g.y + p.rowOff) * 27 + k]; T v = 0; if (i < p.N) v = src[(ulong)i * p.C + c]; \
  dst[(ulong)g.y * 27 * p.C + g.x] = v; }
GATHER27(gather27_u16, ushort) GATHER27(gather27_u32, uint)
)MSL";
}
struct Lib {
  id<MTLLibrary> lib = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos;
};
inline Lib& lib() {
  static Lib L;
  if (!L.lib) {
    NSError* err = nil;
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.mathMode = MTLMathModeSafe;
    std::string src = std::string(baseSource()) + extraSource();
    L.lib = [tk::ctx().dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!L.lib) { fprintf(stderr, "xk: compile failed: %s\n", err.localizedDescription.UTF8String); abort(); }
  }
  return L;
}
inline id<MTLComputePipelineState> pso(const std::string& name) {
  Lib& L = lib();
  auto it = L.psos.find(name);
  if (it != L.psos.end()) return it->second;
  id<MTLFunction> fn = [L.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!fn) { fprintf(stderr, "xk: kernel '%s' not found\n", name.c_str()); abort(); }
  NSError* err = nil;
  id<MTLComputePipelineState> p = [tk::ctx().dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) { fprintf(stderr, "xk: pso '%s' failed: %s\n", name.c_str(), err.localizedDescription.UTF8String); abort(); }
  L.psos[name] = p;
  return p;
}
inline void dispatch2D(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, uint64_t w, uint64_t h) {
  NSUInteger tw = p.threadExecutionWidth, th = p.maxTotalThreadsPerThreadgroup / tw;
  [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint64_t>(tw, w), std::min<uint64_t>(th, h), 1)];
}
}  // namespace xk

// ---- command stream pacing -------------------------------------------------------------------------------------------
inline void pace() {
  static uint64_t last = 0;
  if (tk::ctx().ndispatch - last >= 48) { tk::flush(); last = tk::ctx().ndispatch; }
}

inline tk::Opnd O(const Tensor& t) { return tk::Opnd::tensor(t.buf(), t.dt, t.shape, t.strides); }

struct Arg {
  const Tensor* t = nullptr;
  int kind = 0;  // 0 tensor, 1 python float, 2 python int
  double d = 0; int64_t i = 0;
  Arg(const Tensor& x) : t(&x) {}
  Arg(double v) : kind(1), d(v) {}
  Arg(float v) : kind(1), d(v) {}
  Arg(int v) : kind(2), i(v) {}
  Arg(int64_t v) : kind(2), i(v) {}
  tk::Opnd op() const { return kind == 0 ? O(*t) : kind == 1 ? tk::Opnd::pyFloat(d) : tk::Opnd::pyInt(i); }
};

inline Shape broadcastShape(const Shape& a, const Shape& b) {
  size_t n = std::max(a.size(), b.size()); Shape r(n);
  for (size_t i = 0; i < n; i++) {
    int64_t x = i + a.size() >= n ? a[i + a.size() - n] : 1, y = i + b.size() >= n ? b[i + b.size() - n] : 1;
    if (x != y && x != 1 && y != 1) { fprintf(stderr, "t2: broadcast mismatch\n"); abort(); }
    r[i] = x == 1 ? y : x;
  }
  return r;
}

// out-of-place binary op (name in add/sub/mul/div_true/div_floor/remainder/polar/eq/ne/lt/le/gt/ge)
inline Tensor bin(const char* name, const Arg& a, const Arg& b) {
  tk::Opnd oa = a.op(), ob = b.op();
  std::string n = name;
  bool cmp = n == "eq" || n == "ne" || n == "lt" || n == "le" || n == "gt" || n == "ge";
  DT rdt = cmp ? tk::BOOL : n == "polar" ? tk::C64 : tk::resultType({&oa, &ob});
  Tensor r(rdt, broadcastShape(oa.shape, ob.shape));
  tk::Opnd out = tk::Opnd::output(r.buf(), rdt);
  if (n == "add") tk::add(out, oa, ob);
  else if (n == "sub") tk::sub(out, oa, ob);
  else if (cmp) tk::compare(n, out, oa, ob);
  else tk::binary_iter(n, out, oa, ob);
  r.shape = out.shape; r.strides = out.strides;
  pace();
  return r;
}
inline Tensor add(const Arg& a, const Arg& b) { return bin("add", a, b); }
inline Tensor sub(const Arg& a, const Arg& b) { return bin("sub", a, b); }
inline Tensor mul(const Arg& a, const Arg& b) { return bin("mul", a, b); }
inline Tensor div(const Arg& a, const Arg& b) { return bin("div_true", a, b); }
inline Tensor gt(const Arg& a, const Arg& b) { return bin("gt", a, b); }
// in-place
inline void bin_(const char* name, Tensor& a, const Arg& b) {
  tk::Opnd out = O(a), ob = b.op();
  std::string n = name;
  if (n == "add") tk::add(out, O(a), ob);
  else if (n == "sub") tk::sub(out, O(a), ob);
  else tk::binary_iter(n, out, O(a), ob);
  pace();
}
inline void add_(Tensor& a, const Arg& b) { bin_("add", a, b); }
inline void sub_(Tensor& a, const Arg& b) { bin_("sub", a, b); }
inline void mul_(Tensor& a, const Arg& b) { bin_("mul", a, b); }
inline void div_(Tensor& a, const Arg& b) { bin_("div_true", a, b); }

inline Tensor finishOut(Tensor r, const tk::Opnd& out) { r.shape = out.shape; r.strides = out.strides; pace(); return r; }
inline Tensor un(const char* name, const Tensor& x) {
  Tensor r(x.dt, x.shape); tk::Opnd out = tk::Opnd::output(r.buf(), x.dt);
  tk::unary(name, out, O(x)); return finishOut(r, out);
}
inline Tensor silu(const Tensor& x) {
  Tensor r(x.dt, x.shape); tk::Opnd out = tk::Opnd::output(r.buf(), x.dt);
  tk::silu(out, O(x)); return finishOut(r, out);
}
inline Tensor gelu(const Tensor& x, bool tanh_approx) {
  Tensor r(x.dt, x.shape); tk::Opnd out = tk::Opnd::output(r.buf(), x.dt);
  tk::gelu(out, O(x), tanh_approx); return finishOut(r, out);
}
inline Tensor softplus(const Tensor& x) {
  Tensor r(x.dt, x.shape); tk::Opnd out = tk::Opnd::output(r.buf(), x.dt);
  tk::softplus(out, O(x)); return finishOut(r, out);
}
inline Tensor clamp_min(const Tensor& x, double mn) {
  Tensor r(x.dt, x.shape); tk::Opnd out = tk::Opnd::output(r.buf(), x.dt);
  tk::clamp_min(out, O(x), tk::Scal(mn)); return finishOut(r, out);
}

// ---- copies ------------------------------------------------------------------------------------------------------------
inline void copy_(const Tensor& dst, const Tensor& src) {
  tk::Opnd d = O(dst); tk::copy_(d, O(src)); pace();
}
// x.to(dtype) / x.clone(): _to_copy / clone with preserve_format
inline Tensor to(const Tensor& x, DT dt, bool force_copy = false) {
  if (x.dt == dt && !force_copy) return x;
  Tensor r(dt, x.shape);
  tk::Opnd ox = O(x);
  if (ox.nonOverlappingAndDense()) r.strides = x.strides;
  copy_(r, x);
  return r;
}
inline Tensor clone(const Tensor& x) { return to(x, x.dt, true); }
inline Tensor contiguous(const Tensor& x) {
  if (x.contiguous()) return x;
  Tensor r(x.dt, x.shape); copy_(r, x); return r;
}
inline Tensor reshape_g(const Tensor& x, const Shape& s) { return contiguous(x).view(s); }
inline Tensor full(DT dt, const Shape& s, tk::Scal v) {
  Tensor r(dt, s); tk::Opnd o = O(r); tk::fill(o, v); pace(); return r;
}
inline Tensor zeros(DT dt, const Shape& s) { return full(dt, s, tk::Scal((int64_t)0)); }
// torch.cat along dim (data movement via copy kernels onto slice views)
inline Tensor cat(const std::vector<Tensor>& xs, int dim) {
  Shape s = xs[0].shape; if (dim < 0) dim += (int)s.size();
  int64_t tot = 0; for (auto& x : xs) tot += x.shape[dim];
  s[dim] = tot;
  Tensor r(xs[0].dt, s);
  int64_t o = 0;
  for (auto& x : xs) { if (x.shape[dim]) copy_(r.slice(dim, o, o + x.shape[dim]), x); o += x.shape[dim]; }
  return r;
}
// torch.stack
inline Tensor stack(const std::vector<Tensor>& xs, int dim) {
  std::vector<Tensor> u; for (auto& x : xs) u.push_back(x.unsqueeze(dim));
  return cat(u, dim);
}

// row gather: src [N, ...] contiguous, idx int32/int64 [M] -> [M, ...]; out-of-range index -> zero row.
inline Tensor gather_rows(const Tensor& src, const Tensor& idx) {
  int64_t N = src.shape[0], M = idx.shape[0];
  Shape s = src.shape; s[0] = M;
  Tensor r(src.dt, s);
  if (M == 0) return r;
  if (!src.contiguous() || !idx.contiguous()) { fprintf(stderr, "gather_rows: non-contiguous\n"); abort(); }
  uint32_t es = src.esize(); int64_t C = N ? src.numel() / N : 0;
  uint32_t ue = es == 8 ? 4 : es; if (es == 8) C *= 2;
  std::string name = std::string("gather_u") + std::to_string(ue * 8) + (idx.dt == tk::I32 ? "_i32" : "_i64");
  auto p = xk::pso(name);
  auto e = tk::enc();
  [e setComputePipelineState:p];
  tk::setBuf(e, src.buf(), 0); tk::setBuf(e, idx.buf(), 1); tk::setBuf(e, r.buf(), 2);
  struct { uint32_t C, N, rowOff, pad; } gp{(uint32_t)C, (uint32_t)N, 0, 0};
  tk::setVal(e, gp, 3);
  xk::dispatch2D(e, p, C, M);
  tk::ctx().ndispatch++; pace();
  return r;
}

// ---- reductions / norms ------------------------------------------------------------------------------------------------
inline std::vector<std::shared_ptr<Storage>>& reduceScratch() { static std::vector<std::shared_ptr<Storage>> v; return v; }
inline void initOps() {
  tk::reduce_alloc() = [](size_t bytes) {
    auto st = std::make_shared<Storage>();
    st->buf = pool().get(std::max<size_t>(bytes, 16));
    reduceScratch().push_back(st);
    return tk::Buf{st->buf, 0};
  };
}
inline tk::RTensor R(const Tensor& t) { return tk::RTensor{t.buf(), t.dt, t.shape, t.strides}; }
inline Shape reducedShape(const Shape& s, const std::vector<int64_t>& dims, bool keepdim) {
  Shape r; int n = (int)s.size();
  std::vector<bool> m(n, dims.empty());
  for (auto d : dims) m[d < 0 ? d + n : d] = true;
  for (int i = 0; i < n; i++) { if (!m[i]) r.push_back(s[i]); else if (keepdim) r.push_back(1); }
  return r;
}
inline Tensor norm_l2(const Tensor& x, const std::vector<int64_t>& dims, bool keepdim) {
  Tensor r(x.dt, reducedShape(x.shape, dims, keepdim));
  tk::norm_dim(R(x), r.buf(), 2.0, dims, keepdim); reduceScratch().clear(); pace(); return r;
}
inline Tensor std_dims(const Tensor& x, const std::vector<int64_t>& dims, bool keepdim) {
  Tensor r(x.dt, reducedShape(x.shape, dims, keepdim));
  tk::std_var_dim(R(x), r.buf(), dims, keepdim); reduceScratch().clear(); pace(); return r;
}
inline Tensor mean_dims(const Tensor& x, const std::vector<int64_t>& dims, bool keepdim) {
  Tensor r(x.dt, reducedShape(x.shape, dims, keepdim));
  tk::mean_dim(R(x), r.buf(), dims, keepdim); reduceScratch().clear(); pace(); return r;
}
// F.layer_norm over the last dim
inline Tensor layer_norm(const Tensor& x_, float eps, const Tensor* w = nullptr, const Tensor* b = nullptr) {
  Tensor x = contiguous(x_);
  uint32_t C = (uint32_t)x.shape.back(), rows = (uint32_t)(x.numel() / C);
  Tensor r(x.dt, x.shape), mean(x.dt, {rows}), rstd(x.dt, {rows});
  tk::Buf wb, bb;
  if (w) wb = w->buf();
  if (b) bb = b->buf();
  tk::layer_norm(x.buf(), r.buf(), mean.buf(), rstd.buf(), rows, C, eps, w ? &wb : nullptr, b ? &bb : nullptr, x.dt);
  pace();
  return r;
}

// ---- heavy ops ---------------------------------------------------------------------------------------------------------
// F.linear(x, W, b): x [..., in] ; W [out, in]
inline Tensor linear(const Tensor& x_, const Tensor& W, const Tensor* b) {
  Tensor x = contiguous(x_);
  if (x.dt != W.dt) { fprintf(stderr, "linear: dtype mismatch %s vs %s\n", tk::dtName(x.dt), tk::dtName(W.dt)); abort(); }
  int64_t in_f = W.shape[1], out_f = W.shape[0];
  if (x.shape.back() != in_f) { fprintf(stderr, "linear: shape mismatch\n"); abort(); }
  Shape s = x.shape; s.back() = out_f;
  Tensor r(x.dt, s);
  int64_t rows = x.numel() / in_f;
  if (rows == 0) return r;
  tk::Buf bb; if (b) bb = b->buf();
  bool done = tk::linear(x.buf(), W.buf(), b ? &bb : nullptr, r.buf(), x.dt, (uint32_t)rows, (uint32_t)in_f, (uint32_t)out_f);
  if (!done && b) add_(r, *b);
  pace();
  return r;
}
// F.scaled_dot_product_attention(q, k, v): [B,H,L,D] views (stride(-1) == 1)
inline Tensor sdpa(const Tensor& q, const Tensor& k, const Tensor& v) {
  tk::SdpaArgs a;
  a.B = q.shape[0]; a.H = q.shape[1]; a.Hkv = k.shape[1]; a.qL = q.shape[2]; a.kL = k.shape[2]; a.D = q.shape[3];
  if (q.strides[3] != 1 || k.strides[3] != 1 || v.strides[3] != 1) { fprintf(stderr, "sdpa: last-dim stride != 1\n"); abort(); }
  for (int i = 0; i < 3; i++) { a.qs[i] = q.strides[i]; a.ks[i] = k.strides[i]; a.vs[i] = v.strides[i]; }
  Tensor r(q.dt, {a.B, a.H, a.qL, a.D});
  tk::sdpa_prefill(q.buf(), k.buf(), v.buf(), r.buf(), q.dt, a);
  pace();
  return r;
}
// F.conv3d(x, W, b, padding=1) with 3x3x3 kernels; x [N,C,D,H,W]
inline Tensor conv3d_333(const Tensor& x_, const Tensor& W, const Tensor* b) {
  Tensor x = contiguous(x_);
  int64_t N = x.shape[0], Cin = x.shape[1], D = x.shape[2], H = x.shape[3], Wd = x.shape[4], Cout = W.shape[0];
  Tensor r(x.dt, {N, Cout, D, H, Wd}), ta(x.dt, x.shape), tw(W.dt, W.shape);
  tk::Buf bb; if (b) bb = b->buf();
  tk::conv3d_333(x.buf(), W.buf(), b ? &bb : nullptr, r.buf(), ta.buf(), tw.buf(), x.dt, N, Cin, D, H, Wd, Cout);
  pace();
  return r;
}

// ---- parity helpers ----------------------------------------------------------------------------------------------------
struct CmpResult { int64_t n = 0, mism = 0; double maxabs = 0; };
inline CmpResult compare_bits(const Tensor& a_, const Tensor& b_) {
  tk::sync();
  Tensor a = contiguous_cpu(a_), b = contiguous_cpu(b_);
  CmpResult r; r.n = a.numel();
  if (a.dt != b.dt || a.shape != b.shape) { r.mism = -1; return r; }
  uint32_t es = a.esize();
  const uint8_t* pa = a.raw(); const uint8_t* pb = b.raw();
  for (int64_t i = 0; i < r.n; i++) {
    if (memcmp(pa + i * es, pb + i * es, es)) {
      r.mism++;
      double x = 0, y = 0;
      if (a.dt == tk::F32) { x = ((const float*)pa)[i]; y = ((const float*)pb)[i]; }
      else if (a.dt == tk::F16) { x = (float)((const __fp16*)pa)[i]; y = (float)((const __fp16*)pb)[i]; }
      else if (a.dt == tk::BF16) { uint32_t u = (uint32_t)((const uint16_t*)pa)[i] << 16, w = (uint32_t)((const uint16_t*)pb)[i] << 16; float fx, fy; memcpy(&fx, &u, 4); memcpy(&fy, &w, 4); x = fx; y = fy; }
      else if (a.dt == tk::I32) { x = ((const int32_t*)pa)[i]; y = ((const int32_t*)pb)[i]; }
      else if (a.dt == tk::I64) { x = (double)((const int64_t*)pa)[i]; y = (double)((const int64_t*)pb)[i]; }
      else { x = pa[i]; y = pb[i]; }
      double d = std::fabs(x - y); if (d > r.maxabs || d != d) r.maxabs = d;
    }
  }
  return r;
}
inline bool cmp(const std::string& name, const Tensor& a, const Tensor& ref) {
  CmpResult r = compare_bits(a, ref);
  if (r.mism < 0) {
    printf("  %-28s SHAPE/DTYPE MISMATCH (%s vs %s; ", name.c_str(), tk::dtName(a.dt), tk::dtName(ref.dt));
    for (auto s : a.shape) printf("%lld ", (long long)s); printf("vs "); for (auto s : ref.shape) printf("%lld ", (long long)s); printf(")\n");
    return false;
  }
  printf("  %-28s %s  n=%lld mismatches=%lld maxabs=%.3g\n", name.c_str(), r.mism == 0 ? "BITWISE OK" : "DIFF      ",
         (long long)r.n, (long long)r.mism, r.maxabs);
  fflush(stdout);
  return r.mism == 0;
}

}  // namespace t2
