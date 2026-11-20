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
