#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

namespace {

inline id<MTLBuffer> mtlBuffer(const torch::Tensor &t) { return __builtin_bit_cast(id<MTLBuffer>, t.storage().data()); }
inline NSUInteger mtlOffset(const torch::Tensor &t) { return (NSUInteger)(t.storage_offset() * t.element_size()); }

id<MTLComputePipelineState> pipeline(const char *name) {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  static id<MTLLibrary> library = nil;
  static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *cache = [NSMutableDictionary new];
  @synchronized(cache) {
    NSString *key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> pso = cache[key];
    if (pso) return pso;
    NSError *error = nil;
    if (!library) {
      library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
      TORCH_CHECK(library, "Failed to create Metal library from embedded data: ", error.localizedDescription.UTF8String);
    }
    id<MTLFunction> fn = [library newFunctionWithName:key];
    TORCH_CHECK(fn, "Metal function not found: ", name);
    pso = [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);
    cache[key] = pso;
    return pso;
  }
}

// Encodes one compute pass on torch's MPS stream. `encode` binds the arguments and dispatches.
template <class F> void encodePass(const char *kernel, F encode) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = pipeline(kernel);
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
      TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");
      id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute command encoder");
      [enc setComputePipelineState:pso];
      encode(enc);
      [enc endEncoding];
      torch::mps::commit();
    });
  }
}

inline void bind(id<MTLComputeCommandEncoder> enc, const torch::Tensor &t, NSUInteger index) {
  [enc setBuffer:mtlBuffer(t) offset:mtlOffset(t) atIndex:index];
}

void check(const torch::Tensor &t, const char *name, torch::ScalarType dtype, int64_t dim) {
  TORCH_CHECK(t.device().is_mps(), name, " must be an MPS tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.dim() == dim, name, " must have ", dim, " dimensions, got ", t.dim());
}

void gemmImpl(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor *bias) {
  check(x, "x", torch::kBFloat16, 2);
  check(weight, "weight", torch::kBFloat16, 2);
  check(out, "out", torch::kBFloat16, 2);
  const int64_t M = x.size(0), K = x.size(1), N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K, "weight must be [N, ", K, "], got ", weight.sizes());
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out must be [", M, ", ", N, "], got ", out.sizes());
  TORCH_CHECK(M < INT32_MAX && N < INT32_MAX && K < INT32_MAX, "dimension too large");
  if (bias) {
    check(*bias, "bias", torch::kFloat, 1);
    TORCH_CHECK(bias->size(0) == N, "bias must be [", N, "], got ", bias->sizes());
  }
  if (M == 0 || N == 0) return;
  TORCH_CHECK(K > 0, "x must have at least one column");
  struct { int32_t M, N, K, pad; } p{(int32_t)M, (int32_t)N, (int32_t)K, 0};
  encodePass(bias ? "t2_gemm_bias_gelu" : "t2_gemm", [&](id<MTLComputeCommandEncoder> enc) {
    bind(enc, x, 0); bind(enc, weight, 1); bind(enc, out, 2); bind(enc, bias ? *bias : weight, 3);
    [enc setBytes:&p length:sizeof p atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 127) / 128), (NSUInteger)((M + 63) / 64), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  });
}

}  // namespace

void gemm(torch::Tensor &out, torch::Tensor const &x, torch::Tensor const &weight) { gemmImpl(out, x, weight, nullptr); }

void gemm_bias_gelu(torch::Tensor &out, torch::Tensor const &x, torch::Tensor const &weight, torch::Tensor const &bias) {
  gemmImpl(out, x, weight, &bias);
}

void bias_gelu(torch::Tensor &x, torch::Tensor const &bias) {
  check(x, "x", torch::kBFloat16, 2);
  check(bias, "bias", torch::kFloat, 1);
  const int64_t N = x.size(0), C = x.size(1);
  TORCH_CHECK(bias.size(0) == C, "bias must be [", C, "], got ", bias.sizes());
  TORCH_CHECK(C < UINT32_MAX, "dimension too large");
  if (N == 0 || C == 0) return;
  const uint32_t c = (uint32_t)C;
  encodePass("t2_bias_gelu", [&](id<MTLComputeCommandEncoder> enc) {
    bind(enc, x, 0); bind(enc, bias, 1);
    [enc setBytes:&c length:sizeof c atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)C, (NSUInteger)N, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  });
}

// flags: 1 residual update, 2 layer norm into `out`, 4 no branch tensor, 8 fused affine
void resid_ln(torch::Tensor &x, torch::Tensor const &branch, torch::Tensor const &vec, torch::Tensor &out, int64_t flags, double eps) {
  check(x, "x", torch::kFloat, 2);
  check(vec, "vec", torch::kFloat, 2);
  const int64_t N = x.size(0), C = x.size(1);
  TORCH_CHECK(C % 32 == 0 && C > 0, "the number of channels must be a positive multiple of 32, got ", C);
  TORCH_CHECK(vec.size(0) == 4 && vec.size(1) == C, "vec must be [4, ", C, "], got ", vec.sizes());
  TORCH_CHECK((flags & ~15) == 0, "invalid flags");
  const bool useBranch = (flags & 1) && !(flags & 4), useOut = (flags & 2) != 0;
  if (useBranch) {
    check(branch, "branch", torch::kBFloat16, 2);
    TORCH_CHECK(branch.sizes() == x.sizes(), "branch must have the shape of x, got ", branch.sizes());
  }
  if (useOut) {
    check(out, "out", torch::kBFloat16, 2);
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have the shape of x, got ", out.sizes());
  }
  if (N == 0 || !(flags & 3)) return;
  struct { uint32_t C, flags; float eps; uint32_t pad; } p{(uint32_t)C, (uint32_t)flags, (float)eps, 0};
  encodePass("t2_resid_ln", [&](id<MTLComputeCommandEncoder> enc) {
    bind(enc, x, 0); bind(enc, useBranch ? branch : x, 1); bind(enc, vec, 2); bind(enc, useOut ? out : x, 3);
    [enc setBytes:&p length:sizeof p atIndex:4];
    [enc dispatchThreads:MTLSizeMake(32, (NSUInteger)N, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  });
}

// flags: 1 RMS norm with gamma, 2 rotary phases
void qprep(torch::Tensor &out, torch::Tensor const &src, torch::Tensor const &bias, torch::Tensor const &gamma,
           torch::Tensor const &phases, int64_t col_offset, int64_t flags) {
  check(src, "src", torch::kBFloat16, 2);
  check(out, "out", torch::kBFloat16, 3);
  check(bias, "bias", torch::kFloat, 1);
  const int64_t N = src.size(0), rs = src.size(1), H = out.size(0), HD = H * 128;
  TORCH_CHECK(out.size(1) == N && out.size(2) == 128, "out must be [heads, ", N, ", 128], got ", out.sizes());
  TORCH_CHECK(col_offset >= 0 && col_offset + HD <= rs, "columns [", col_offset, ", ", col_offset + HD, ") are outside src ", src.sizes());
  TORCH_CHECK(bias.size(0) == rs, "bias must be [", rs, "], got ", bias.sizes());
  TORCH_CHECK((flags & ~3) == 0, "invalid flags");
  if (flags & 1) {
    check(gamma, "gamma", torch::kFloat, 1);
    TORCH_CHECK(gamma.size(0) == HD, "gamma must be [", HD, "], got ", gamma.sizes());
  }
  if (flags & 2) {
    check(phases, "phases", torch::kFloat, 3);
    TORCH_CHECK(phases.size(0) == N && phases.size(1) == 64 && phases.size(2) == 2, "phases must be [", N, ", 64, 2], got ", phases.sizes());
  }
  TORCH_CHECK(N < UINT32_MAX && rs < UINT32_MAX, "dimension too large");
  if (N == 0 || H == 0) return;
  struct { uint32_t N, H, rs, colOff, flags; float scale; uint32_t p0, p1; } p{
      (uint32_t)N, (uint32_t)H, (uint32_t)rs, (uint32_t)col_offset, (uint32_t)flags, (float)std::sqrt(128.0), 0, 0};
  encodePass("t2_qprep", [&](id<MTLComputeCommandEncoder> enc) {
    bind(enc, src, 0); bind(enc, bias, 1); bind(enc, (flags & 1) ? gamma : bias, 2); bind(enc, (flags & 2) ? phases : bias, 3); bind(enc, out, 4);
    [enc setBytes:&p length:sizeof p atIndex:5];
    [enc dispatchThreads:MTLSizeMake(32, (NSUInteger)H, (NSUInteger)N) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  });
}

// =====================================================================================================================
// Mesh post-processing ops (the port's to_glb): linear BVH build (CPU), nearest triangle (Metal), TELEA inpaint.

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace {

struct BNode { float lo0[3]; int32_t l; float hi0[3]; int32_t r; float lo1[3]; float p0; float hi1[3]; float p1; };
struct BTri { float a[3]; int32_t id; float b[3]; float p0; float c[3]; float p1; };
static_assert(sizeof(BNode) == 64 && sizeof(BTri) == 48, "layout shared with kernels.metal");

inline uint64_t spread21(uint64_t x) {
  x &= 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full;
  x = (x | x << 4) & 0x10c30c30c30c30c3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}

void checkCpu(const torch::Tensor &t, const char *name, torch::ScalarType dtype, int64_t dim) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.dim() == dim, name, " must have ", dim, " dimensions, got ", t.dim());
}

}  // namespace

// Linear BVH (Karras 2012) over the triangles: 63-bit Morton codes of the centroids, sort, binary radix tree, boxes
// bottom-up. Returns (nodes int32 [max(F - 1, 1), 16], tris int32 [F, 12]): raw 64-byte nodes holding the boxes of both
// children and 48-byte triangles in Morton order, the layout t2_mesh_nearest reads.
std::tuple<torch::Tensor, torch::Tensor> lbvh_build(torch::Tensor const &vertices, torch::Tensor const &faces) {
  checkCpu(vertices, "vertices", torch::kFloat, 2);
  checkCpu(faces, "faces", torch::kInt, 2);
  TORCH_CHECK(vertices.size(1) == 3, "vertices must be [V, 3], got ", vertices.sizes());
  TORCH_CHECK(faces.size(1) == 3, "faces must be [F, 3], got ", faces.sizes());
  const int64_t nv = vertices.size(0), n = faces.size(0);
  TORCH_CHECK(n < INT32_MAX && nv < INT32_MAX, "mesh too large");
  const float *V = vertices.data_ptr<float>();
  const int32_t *F = faces.data_ptr<int32_t>();
  for (int64_t i = 0; i < n * 3; i++) TORCH_CHECK(F[i] >= 0 && F[i] < nv, "face index ", F[i], " is outside [0, ", nv, ")");
  for (int64_t i = 0; i < nv * 3; i++) TORCH_CHECK(std::isfinite(V[i]), "vertices must be finite");
  torch::Tensor nodesT = torch::zeros({std::max<int64_t>(n - 1, 1), 16}, torch::kInt), trisT = torch::zeros({n, 12}, torch::kInt);
  if (n == 0) return {nodesT, trisT};
  BNode *N = (BNode *)nodesT.data_ptr<int32_t>();
  BTri *T = (BTri *)trisT.data_ptr<int32_t>();
  auto centroid = [&](int64_t f, int k) { return (V[3 * F[3 * f] + k] + V[3 * F[3 * f + 1] + k] + V[3 * F[3 * f + 2] + k]) * (1.f / 3); };
  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (int64_t f = 0; f < n; f++) for (int k = 0; k < 3; k++) { float x = centroid(f, k); lo[k] = std::min(lo[k], x); hi[k] = std::max(hi[k], x); }
  double sc[3];
  for (int k = 0; k < 3; k++) sc[k] = hi[k] > lo[k] ? 2097151.0 / ((double)hi[k] - lo[k]) : 0.0;
  std::vector<std::pair<uint64_t, uint32_t>> code((size_t)n);
  at::parallel_for(0, n, 1 << 14, [&](int64_t a, int64_t b) {
    for (int64_t f = a; f < b; f++) {
      uint64_t m = 0;
      for (int k = 0; k < 3; k++) {
        uint64_t q = (uint64_t)std::min(2097151.0, std::max(0.0, ((double)centroid(f, k) - lo[k]) * sc[k]));
        m |= spread21(q) << (2 - k);
      }
      code[f] = {m, (uint32_t)f};
    }
  });
  std::sort(code.begin(), code.end());
  at::parallel_for(0, n, 1 << 14, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      const int32_t f = (int32_t)code[i].second; BTri &t = T[i];
      memcpy(t.a, V + 3 * F[3 * f], 12); memcpy(t.b, V + 3 * F[3 * f + 1], 12); memcpy(t.c, V + 3 * F[3 * f + 2], 12);
      t.id = f; t.p0 = 0; t.p1 = 0;
    }
  });
  if (n == 1) { N[0].l = ~0; N[0].r = ~0; return {nodesT, trisT}; }
  auto delta = [&](int64_t i, int64_t j) -> int {
    if (j < 0 || j >= n) return -1;
    const uint64_t a = code[i].first, b = code[j].first;
    if (a == b) return 64 + __builtin_clz((uint32_t)i ^ (uint32_t)j);
    return __builtin_clzll(a ^ b);
  };
  std::vector<int32_t> parL((size_t)n), parI((size_t)n - 1), visits((size_t)n - 1, 0);
  parI[0] = -1;
  at::parallel_for(0, n - 1, 1 << 12, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      const int d = delta(i, i + 1) - delta(i, i - 1) >= 0 ? 1 : -1;
      const int dmin = delta(i, i - d);
      int64_t lmax = 2; while (delta(i, i + lmax * d) > dmin) lmax *= 2;
      int64_t l = 0; for (int64_t t = lmax / 2; t >= 1; t /= 2) if (delta(i, i + (l + t) * d) > dmin) l += t;
      const int64_t j = i + l * d; const int dn = delta(i, j);
      int64_t s = 0, t = l;
      do { t = (t + 1) / 2; if (delta(i, i + (s + t) * d) > dn) s += t; } while (t > 1);
      const int64_t g = i + s * d + std::min(d, 0);
      BNode &nd = N[i];
      if (std::min(i, j) == g) { nd.l = ~(int32_t)g; parL[g] = (int32_t)i; } else { nd.l = (int32_t)g; parI[g] = (int32_t)i; }
      if (std::max(i, j) == g + 1) { nd.r = ~(int32_t)(g + 1); parL[g + 1] = (int32_t)i; } else { nd.r = (int32_t)(g + 1); parI[g + 1] = (int32_t)i; }
    }
  });
  auto childBox = [&](int32_t c, float *blo, float *bhi) {
    if (c < 0) {
      const BTri &t = T[~c];
      for (int k = 0; k < 3; k++) { blo[k] = std::min(t.a[k], std::min(t.b[k], t.c[k])); bhi[k] = std::max(t.a[k], std::max(t.b[k], t.c[k])); }
    } else {
      const BNode &m = N[c];
      for (int k = 0; k < 3; k++) { blo[k] = std::min(m.lo0[k], m.lo1[k]); bhi[k] = std::max(m.hi0[k], m.hi1[k]); }
    }
  };
  // the second child to arrive at a node computes its boxes and continues upward
  at::parallel_for(0, n, 1 << 12, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i++) {
      int32_t p = parL[i];
      while (p >= 0) {
        if (__atomic_fetch_add(&visits[p], 1, __ATOMIC_ACQ_REL) == 0) break;
        BNode &nd = N[p];
        childBox(nd.l, nd.lo0, nd.hi0); childBox(nd.r, nd.lo1, nd.hi1);
        p = parI[p];
      }
    }
  });
  return {nodesT, trisT};
}

void mesh_nearest(torch::Tensor &dist, torch::Tensor &face, torch::Tensor &bary, torch::Tensor const &nodes, torch::Tensor const &tris,
                  torch::Tensor const &points, double max_distance) {
  check(nodes, "nodes", torch::kInt, 2);
  check(tris, "tris", torch::kInt, 2);
  check(points, "points", torch::kFloat, 2);
  check(dist, "dist", torch::kFloat, 1);
  check(face, "face", torch::kInt, 1);
  check(bary, "bary", torch::kFloat, 2);
  const int64_t n = tris.size(0), m = points.size(0);
  TORCH_CHECK(tris.size(1) == 12, "tris must be [F, 12] from lbvh_build, got ", tris.sizes());
  TORCH_CHECK(nodes.size(1) == 16 && nodes.size(0) == std::max<int64_t>(n - 1, 1), "nodes must be [max(F - 1, 1), 16] from lbvh_build, got ", nodes.sizes());
  TORCH_CHECK(points.size(1) == 3, "points must be [M, 3], got ", points.sizes());
  TORCH_CHECK(dist.size(0) == m && face.size(0) == m && bary.size(0) == m && bary.size(1) == 3, "outputs must be [M], [M] and [M, 3]");
  TORCH_CHECK(n < INT32_MAX && m < INT32_MAX, "too many triangles or points");
  TORCH_CHECK(!(max_distance < 0), "max_distance must be non-negative");
  if (m == 0) return;
  const double c2 = max_distance * max_distance;
  struct { int32_t ntri; float cap2; int32_t nq, pad; } p{(int32_t)n, c2 < 3.0e38 ? (float)c2 : 3.0e38f, (int32_t)m, 0};
  encodePass("t2_mesh_nearest", [&](id<MTLComputeCommandEncoder> enc) {
    bind(enc, nodes, 0); bind(enc, tris, 1); bind(enc, points, 2); bind(enc, dist, 3); bind(enc, face, 4); bind(enc, bary, 5);
    [enc setBytes:&p length:sizeof p atIndex:6];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)m, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  });
}

// mask uint8 [H, W] (non-zero = pixel to fill) -> (lv int32 [H, W], list int32 [n], offsets int64 [levels + 1]):
// lv is the 4-connected distance level from the known region (0 known, -1 if there is no known pixel to reach it from),
// list holds the linear indices of the pixels to fill ordered by level, offsets the start of each level in list.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> telea_levels(torch::Tensor const &mask) {
  checkCpu(mask, "mask", torch::kByte, 2);
  const int64_t H = mask.size(0), W = mask.size(1), P = H * W;
  TORCH_CHECK(P < INT32_MAX, "mask too large");
  torch::Tensor lvT = torch::empty({H, W}, torch::kInt);
  int32_t *lv = lvT.data_ptr<int32_t>();
  const uint8_t *mk = mask.data_ptr<uint8_t>();
  for (int64_t i = 0; i < P; i++) lv[i] = mk[i] ? -1 : 0;
  std::vector<int32_t> list; std::vector<int64_t> off(1, 0);
  for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) {
    const int64_t i = y * W + x;
    if (lv[i] != -1) continue;
    if ((x > 0 && lv[i - 1] == 0) || (x + 1 < W && lv[i + 1] == 0) || (y > 0 && lv[i - W] == 0) || (y + 1 < H && lv[i + W] == 0)) list.push_back((int32_t)i);
  }
  for (int32_t i : list) lv[i] = 1;
  off.push_back((int64_t)list.size());
  for (int32_t L = 2;; L++) {
    const int64_t a = off[off.size() - 2], b = off.back();
    if (a == b) { off.pop_back(); break; }
    for (int64_t j = a; j < b; j++) {
      const int32_t i = list[j]; const int64_t x = i % W, y = i / W;
      if (x > 0 && lv[i - 1] == -1) { lv[i - 1] = L; list.push_back(i - 1); }
      if (x + 1 < W && lv[i + 1] == -1) { lv[i + 1] = L; list.push_back(i + 1); }
      if (y > 0 && lv[i - W] == -1) { lv[i - W] = L; list.push_back((int32_t)(i - W)); }
      if (y + 1 < H && lv[i + W] == -1) { lv[i + W] = L; list.push_back((int32_t)(i + W)); }
    }
    off.push_back((int64_t)list.size());
  }
  torch::Tensor listT = torch::empty({(int64_t)list.size()}, torch::kInt), offT = torch::empty({(int64_t)off.size()}, torch::kLong);
  if (!list.empty()) memcpy(listT.data_ptr<int32_t>(), list.data(), list.size() * 4);
  memcpy(offT.data_ptr<int64_t>(), off.data(), off.size() * 8);
  return {lvT, listT, offT};
}

// img uint8 [C, H, W] (planar, C <= 8), filled in place level by level. lv / list / offsets from telea_levels.
void telea_fill(torch::Tensor &img, torch::Tensor const &lv, torch::Tensor const &list, std::vector<int64_t> offsets, int64_t radius) {
  check(img, "img", torch::kByte, 3);
  check(lv, "lv", torch::kInt, 2);
  check(list, "list", torch::kInt, 1);
  const int64_t C = img.size(0), H = img.size(1), W = img.size(2), n = list.size(0), nl = (int64_t)offsets.size() - 1;
  TORCH_CHECK(C >= 1 && C <= 8, "img must have 1 to 8 channels, got ", C);
  TORCH_CHECK(lv.size(0) == H && lv.size(1) == W, "lv must be [", H, ", ", W, "], got ", lv.sizes());
  TORCH_CHECK(radius >= 1 && radius <= 16, "radius must be in [1, 16], got ", radius);
  TORCH_CHECK(H * W < INT32_MAX, "image too large");
  TORCH_CHECK(nl >= 0, "offsets must have at least one element");
  const int64_t *off = offsets.data();
  TORCH_CHECK(off[0] == 0 && off[nl] == n, "offsets must start at 0 and end at len(list)");
  for (int64_t l = 0; l < nl; l++) TORCH_CHECK(off[l] <= off[l + 1], "offsets must be non-decreasing");
  if (n == 0) return;
  encodePass("t2_telea_fill", [&](id<MTLComputeCommandEncoder> enc) {
    bind(enc, lv, 0); bind(enc, img, 1);
    for (int64_t l = 0; l < nl; l++) {
      const int64_t cnt = off[l + 1] - off[l];
      if (cnt == 0) continue;
      struct { int32_t H, W, L, C, R, n, p0, p1; } p{(int32_t)H, (int32_t)W, (int32_t)(l + 1), (int32_t)C, (int32_t)radius, (int32_t)cnt, 0, 0};
      [enc setBuffer:mtlBuffer(list) offset:mtlOffset(list) + (NSUInteger)off[l] * 4 atIndex:2];
      [enc setBytes:&p length:sizeof p atIndex:3];
      [enc dispatchThreads:MTLSizeMake((NSUInteger)cnt, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
  });
}
