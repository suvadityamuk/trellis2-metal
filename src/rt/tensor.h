// Tensor = strided view over a shared-storage MTLBuffer (unified memory), with the subset of torch view semantics
// the pipeline needs. Data movement (copies, gathers, cats) is bit-exact by construction, so it is done here with
// plain CPU code after a sync, or by callers with GPU kernels; float arithmetic never happens in this file.
#pragma once
#include "../torchk/base.h"
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <dispatch/dispatch.h>

namespace t2 {
using tk::DT;
typedef std::vector<int64_t> Shape;

struct Pool {
  std::multimap<size_t, id<MTLBuffer>> free_;
  size_t live = 0, pooled = 0, peak = 0;
  size_t cap = (size_t)6 << 30;  // max bytes kept idle
  static size_t roundUp(size_t n) {
    if (n < 4096) return 4096;
    if (n < (1u << 20)) { size_t p = 4096; while (p < n) p <<= 1; return p; }
    return (n + (1u << 20) - 1) & ~(size_t)((1u << 20) - 1);
  }
  id<MTLBuffer> get(size_t n) {
    n = roundUp(n);
    // best fit with bounded slack: a fresh shared buffer costs a zero-fill page fault per page on first touch
    auto it = free_.lower_bound(n);
    id<MTLBuffer> b;
    if (it != free_.end() && it->first <= n + std::max(n / 4, (size_t)1 << 20)) { b = it->second; n = it->first; free_.erase(it); pooled -= n; }
    else {
      b = [tk::ctx().dev newBufferWithLength:n options:MTLResourceStorageModeShared];
      if (!b) { fprintf(stderr, "t2: out of memory allocating %zu bytes\n", n); abort(); }
    }
    live += n; if (live + pooled > peak) peak = live + pooled;
    return b;
  }
  void put(id<MTLBuffer> b) {
    size_t n = b.length; live -= n;
    if (pooled + n > cap) return;
    free_.emplace(n, b); pooled += n;
  }
  void trim() { free_.clear(); pooled = 0; }
};
inline Pool& pool() { static Pool p; return p; }

struct Storage {
  id<MTLBuffer> buf = nil;
  bool pooled = true;
  ~Storage() { if (buf && pooled) pool().put(buf); }
};

struct Tensor {
  std::shared_ptr<Storage> st;
  int64_t off = 0;  // bytes
  DT dt = tk::F32;
  Shape shape, strides;  // strides in elements

  Tensor() {}
  Tensor(DT d, const Shape& s) : dt(d), shape(s) {
    strides = cstrides(s);
    st = std::make_shared<Storage>();
    st->buf = pool().get(std::max<int64_t>(numel() * tk::dtSize(d), 16));
  }
  static Shape cstrides(const Shape& s) {
    Shape r(s.size()); int64_t k = 1;
    for (int i = (int)s.size() - 1; i >= 0; i--) { r[i] = k; k *= std::max<int64_t>(s[i], 1); }
    return r;
  }
  bool defined() const { return st != nullptr; }
  int ndim() const { return (int)shape.size(); }
  int64_t size(int d) const { return shape[d < 0 ? d + ndim() : d]; }
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  uint32_t esize() const { return tk::dtSize(dt); }
  int64_t nbytes() const { return numel() * esize(); }
  bool contiguous() const {
    int64_t k = 1;
    for (int i = ndim() - 1; i >= 0; i--) { if (shape[i] != 1) { if (strides[i] != k) return false; k *= shape[i]; } }
    return true;
  }
  tk::Buf buf() const { return tk::Buf{st->buf, (uint64_t)off}; }
  // CPU pointer (caller must have synced)
  template <class T> T* ptr() const { return (T*)((uint8_t*)st->buf.contents + off); }
  uint8_t* raw() const { return (uint8_t*)st->buf.contents + off; }

  // ---- views ----
  Tensor view(const Shape& s_) const {
    Shape s = s_; int64_t known = 1; int neg = -1;
    for (int i = 0; i < (int)s.size(); i++) { if (s[i] < 0) neg = i; else known *= s[i]; }
    if (neg >= 0) s[neg] = known ? numel() / known : 0;
    if (!contiguous()) { fprintf(stderr, "t2: view of non-contiguous tensor\n"); abort(); }
    int64_t n = 1; for (auto v : s) n *= v;
    if (n != numel()) { fprintf(stderr, "t2: bad view %lld vs %lld\n", (long long)n, (long long)numel()); abort(); }
    Tensor r = *this; r.shape = s; r.strides = cstrides(s); return r;
  }
  Tensor permute(const std::vector<int>& p) const {
    Tensor r = *this;
    for (int i = 0; i < ndim(); i++) { r.shape[i] = shape[p[i]]; r.strides[i] = strides[p[i]]; }
    return r;
  }
  Tensor transpose(int a, int b) const {
    Tensor r = *this; if (a < 0) a += ndim(); if (b < 0) b += ndim();
    std::swap(r.shape[a], r.shape[b]); std::swap(r.strides[a], r.strides[b]); return r;
  }
  Tensor slice(int d, int64_t start, int64_t end, int64_t step = 1) const {
    if (d < 0) d += ndim();
    Tensor r = *this; r.off += start * strides[d] * esize();
    r.shape[d] = (end - start + step - 1) / step; r.strides[d] = strides[d] * step; return r;
  }
  Tensor select(int d, int64_t i) const {
    if (d < 0) d += ndim();
    Tensor r = *this; r.off += i * strides[d] * esize();
    r.shape.erase(r.shape.begin() + d); r.strides.erase(r.strides.begin() + d); return r;
  }
  Tensor unsqueeze(int d) const {
    if (d < 0) d += ndim() + 1;
    Tensor r = *this;
    int64_t st_ = d < ndim() ? strides[d] * shape[d] : 1;
    r.shape.insert(r.shape.begin() + d, 1); r.strides.insert(r.strides.begin() + d, st_); return r;
  }
  Tensor squeeze(int d) const {
    if (d < 0) d += ndim();
    Tensor r = *this; r.shape.erase(r.shape.begin() + d); r.strides.erase(r.strides.begin() + d); return r;
  }
  std::vector<Tensor> chunk(int n, int d) const {
    if (d < 0) d += ndim();
    std::vector<Tensor> r; int64_t c = shape[d] / n;
    for (int i = 0; i < n; i++) r.push_back(slice(d, i * c, (i + 1) * c));
    return r;
  }
  // reinterpret dtype with same element size (e.g. view_as_real / view_as_complex helpers do their own shapes)
  Tensor as(DT d) const { Tensor r = *this; r.dt = d; return r; }
};

// ---- CPU strided copy (bit exact) ---------------------------------------------------------------------------------
inline void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& f) {
  if (n <= grain) { if (n > 0) f(0, n); return; }
  int64_t chunks = std::min<int64_t>((n + grain - 1) / grain, 64);
  int64_t per = (n + chunks - 1) / chunks;
  dispatch_apply((size_t)chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
    int64_t a = (int64_t)c * per, b = std::min(n, a + per);
    if (a < b) f(a, b);
  });
}

// copies src (any strides) into dst (contiguous, same shape & dtype). Caller synced.
inline void copy_strided_cpu(const Tensor& src, uint8_t* dst) {
  int nd = src.ndim(); uint32_t es = src.esize();
  if (src.numel() == 0) return;
  if (src.contiguous()) { memcpy(dst, src.raw(), src.nbytes()); return; }
  // find innermost contiguous run
  int k = nd; int64_t run = 1;
  while (k > 0 && src.strides[k - 1] == run) { run *= src.shape[k - 1]; k--; }
  if (k == 0) { memcpy(dst, src.raw(), src.nbytes()); return; }
  int64_t outer = 1; for (int i = 0; i < k; i++) outer *= src.shape[i];
  const uint8_t* base = src.raw();
  int64_t runb = run * es;
  parallel_for(outer, std::max<int64_t>(1, (1 << 20) / std::max<int64_t>(runb, 1)), [&](int64_t a, int64_t b) {
    std::vector<int64_t> idx(k);
    int64_t t = a;
    for (int i = k - 1; i >= 0; i--) { idx[i] = t % src.shape[i]; t /= src.shape[i]; }
    for (int64_t o = a; o < b; o++) {
      int64_t so = 0; for (int i = 0; i < k; i++) so += idx[i] * src.strides[i];
      if (runb == 2) memcpy(dst + o * 2, base + so * es, 2);
      else if (runb == 4) memcpy(dst + o * 4, base + so * es, 4);
      else memcpy(dst + o * runb, base + so * es, runb);
      for (int i = k - 1; i >= 0; i--) { if (++idx[i] < src.shape[i]) break; idx[i] = 0; }
    }
  });
}

inline Tensor contiguous_cpu(const Tensor& x) {
  if (x.contiguous()) return x;
  tk::sync();
  Tensor r(x.dt, x.shape);
  copy_strided_cpu(x, r.raw());
  return r;
}
inline Tensor clone_cpu(const Tensor& x) {
  tk::sync();
  Tensor r(x.dt, x.shape);
  copy_strided_cpu(x, r.raw());
  return r;
}
// torch.reshape: view when contiguous, else copy
inline Tensor reshape(const Tensor& x, const Shape& s) { return contiguous_cpu(x).view(s); }

inline Tensor from_cpu(DT dt, const Shape& s, const void* data) {
  tk::sync();  // the pooled buffer may still be read by pending GPU work
  Tensor r(dt, s); memcpy(r.raw(), data, r.nbytes()); return r;
}

}  // namespace t2
