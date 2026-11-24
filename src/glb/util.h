// to_glb support: uninitialized arrays, chunked parallel loops, parallel prefix sum / compaction / radix sort, and the
// runtime-compiled Metal library that holds the to_glb compute kernels (kernels.h).
#pragma once
#include "../rt/tensor.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <unordered_map>
#include <dispatch/dispatch.h>
#include <vector>

namespace glb {
using t2::Tensor;

inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// malloc-backed array without value initialization (first touch happens in the parallel loop that fills it)
template <class T> struct Arr {
  T* p = nullptr; int64_t n = 0;
  Arr() {}
  explicit Arr(int64_t n_) { alloc(n_); }
  Arr(const Arr&) = delete; Arr& operator=(const Arr&) = delete;
  Arr(Arr&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  Arr& operator=(Arr&& o) noexcept { if (this != &o) { free(p); p = o.p; n = o.n; o.p = nullptr; o.n = 0; } return *this; }
  ~Arr() { free(p); }
  void alloc(int64_t n_) { free(p); n = n_; p = (T*)malloc(std::max<int64_t>(n_, 1) * sizeof(T)); if (!p) { fprintf(stderr, "glb: out of memory\n"); abort(); } }
  void clear() { free(p); p = nullptr; n = 0; }
  T& operator[](int64_t i) { return p[i]; }
  const T& operator[](int64_t i) const { return p[i]; }
  T* data() { return p; } const T* data() const { return p; }
  int64_t size() const { return n; }
};

inline int nthreads() { static int n = std::max(1u, std::thread::hardware_concurrency()); return n; }

// f(chunk, begin, end) over `chunks` equal ranges of [0, n)
template <class Fn> inline void par_chunks(int64_t n, int chunks, Fn f) {
  if (n <= 0) return;
  chunks = (int)std::max<int64_t>(1, std::min<int64_t>(chunks, n));
  int64_t per = (n + chunks - 1) / chunks;
  if (chunks == 1) { f(0, (int64_t)0, n); return; }
  auto* fp = &f;
  dispatch_apply((size_t)chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
    int64_t a = (int64_t)c * per, b = std::min(n, a + per);
    if (a < b) (*fp)((int)c, a, b);
  });
}
// f(begin, end) with dynamic scheduling over blocks of `grain`
template <class Fn> inline void par_for(int64_t n, int64_t grain, Fn f) {
  if (n <= 0) return;
  if (n <= grain) { f((int64_t)0, n); return; }
  int64_t blocks = (n + grain - 1) / grain;
  auto* fp = &f;
  dispatch_apply((size_t)blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
    int64_t a = (int64_t)c * grain, b = std::min(n, a + grain);
    (*fp)(a, b);
  });
}

// exclusive prefix sum in place over cnt[0..n); returns the total. cnt may alias an array of n+1 (caller writes [n]).
template <class T> inline int64_t prefix_sum(T* cnt, int64_t n) {
  const int C = 32;
  if (n < (1 << 16)) { int64_t s = 0; for (int64_t i = 0; i < n; i++) { int64_t c = (int64_t)cnt[i]; cnt[i] = (T)s; s += c; } return s; }
  int64_t part[C + 1] = {0};
  par_chunks(n, C, [&](int c, int64_t a, int64_t b) { int64_t s = 0; for (int64_t i = a; i < b; i++) s += (int64_t)cnt[i]; part[c + 1] = s; });
  for (int c = 0; c < C; c++) part[c + 1] += part[c];
  par_chunks(n, C, [&](int c, int64_t a, int64_t b) { int64_t s = part[c]; for (int64_t i = a; i < b; i++) { int64_t k = (int64_t)cnt[i]; cnt[i] = (T)s; s += k; } });
  return part[C];
}

// LSD radix sort of (key, value) pairs by 64-bit key, stable, parallel. `bits` = number of significant key bits.
inline void radix_sort_kv(uint64_t* k, uint32_t* v, int64_t n, int bits = 64) {
  if (n < 2) return;
  Arr<uint64_t> tk_(n); Arr<uint32_t> tv_(n);
  uint64_t* ks = k; uint32_t* vs = v; uint64_t* kd = tk_.p; uint32_t* vd = tv_.p;
  const int C = n < (1 << 16) ? 1 : 32;
  std::vector<int64_t> hist((size_t)C * 256);
  for (int sh = 0; sh < bits; sh += 8) {
    std::fill(hist.begin(), hist.end(), 0);
    par_chunks(n, C, [&](int c, int64_t a, int64_t b) { int64_t* h = &hist[(size_t)c * 256]; for (int64_t i = a; i < b; i++) h[(ks[i] >> sh) & 255]++; });
    int64_t tot[256] = {0}; int used = 0;
    for (int d = 0; d < 256; d++) { for (int c = 0; c < C; c++) tot[d] += hist[(size_t)c * 256 + d]; used += tot[d] != 0; }
    if (used <= 1) continue;
    int64_t s = 0;
    for (int d = 0; d < 256; d++) for (int c = 0; c < C; c++) { int64_t h = hist[(size_t)c * 256 + d]; hist[(size_t)c * 256 + d] = s; s += h; }
    par_chunks(n, C, [&](int c, int64_t a, int64_t b) {
      int64_t* h = &hist[(size_t)c * 256];
      for (int64_t i = a; i < b; i++) { int64_t o = h[(ks[i] >> sh) & 255]++; kd[o] = ks[i]; vd[o] = vs[i]; }
    });
    std::swap(ks, kd); std::swap(vs, vd);
  }
  if (ks != k) { par_for(n, 1 << 18, [&](int64_t a, int64_t b) { memcpy(k + a, ks + a, (b - a) * 8); memcpy(v + a, vs + a, (b - a) * 4); }); }
}

// ---- Metal library for to_glb -----------------------------------------------------------------------------------------
const char* kernel_source();  // kernels.h
struct KernLib {
  id<MTLLibrary> lib = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos;
};
inline KernLib& kernlib() {
  static KernLib k;
  if (!k.lib) {
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.fastMathEnabled = NO;  // IEEE semantics: distances / barycentrics must not be reassociated
    NSError* err = nil;
    k.lib = [tk::ctx().dev newLibraryWithSource:[NSString stringWithUTF8String:kernel_source()] options:o error:&err];
    if (!k.lib) { fprintf(stderr, "glb: MSL compile failed: %s\n", err.localizedDescription.UTF8String); exit(1); }
  }
  return k;
}
inline id<MTLComputePipelineState> pso(const std::string& name) {
  KernLib& k = kernlib();
  auto it = k.psos.find(name);
  if (it != k.psos.end()) return it->second;
  id<MTLFunction> fn = [k.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!fn) { fprintf(stderr, "glb: kernel '%s' not found\n", name.c_str()); abort(); }
  NSError* err = nil;
  id<MTLComputePipelineState> p = [tk::ctx().dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) { fprintf(stderr, "glb: pso '%s' failed: %s\n", name.c_str(), err.localizedDescription.UTF8String); abort(); }
  k.psos[name] = p;
  return p;
}
// 1-D dispatch of n threads: buffers at indices 0.., then the params struct.
template <class P> inline void run(const std::string& name, uint64_t n, std::initializer_list<tk::Buf> bufs, const P& params) {
  if (n == 0) return;
  id<MTLComputePipelineState> ps = pso(name);
  id<MTLComputeCommandEncoder> e = tk::enc();
  [e setComputePipelineState:ps];
  unsigned i = 0;
  for (auto& b : bufs) { [e setBuffer:b.buf offset:b.off atIndex:i]; i++; }
  [e setBytes:&params length:sizeof(P) atIndex:i];
  NSUInteger w = std::min<NSUInteger>(ps.maxTotalThreadsPerThreadgroup, 256);
  [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}

template <class T> inline Tensor gbuf(int64_t n) { return Tensor(tk::U8, {std::max<int64_t>(n, 1) * (int64_t)sizeof(T)}); }

}  // namespace glb
