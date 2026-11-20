// Linear / convolution / attention (+ the MPSGraph parts of repeat and bincount) replayed exactly the
// way torch 2.15 MPS dispatches them. Header-only; encodes onto tk::enc() / tk::cmdbuf().
//
// Torch sources mirrored (ref/pytorch-src/aten/src/ATen/...):
//   native/mps/operations/Linear.mm            _mps_linear, _mps_linear_nograph
//   native/mps/operations/LinearAlgebra.mm     try_mps_gemv, dispatch_gemv, gemv_kernel_name
//   native/mps/operations/GemmHeuristics.cpp   GemvPolicy::pick_nt / clamp_vec
//   native/mps/operations/Convolution.mm       _mps_convolution_impl, conv3d_metal_forward, ...
//   native/mps/operations/ConvolutionHeuristics.cpp   select_conv3d_mpp_tile, conv3d_prefer_im2col
//   native/mps/operations/Attention.mm         _scaled_dot_product_attention_math_mps, sdpa_prefill_mps
//   native/mps/operations/Repeat.mm            repeat_mps
//   native/mps/operations/SummaryOps.mm        bincount_mps_impl
//   native/mps/OperationUtils.{h,mm}           getMPSNDArray, getStridedMPSNDArray, Placeholder, mtl_dispatch2DJob
//   mps/MPSStream.mm                           executeMPSGraph, commandEncoder, endKernelCoalescing
//
// Conventions: all Buf offsets are bytes (== torch storage_offset * element_size). "Contiguous" means
// torch-contiguous in the stated layout. Temporaries that torch allocates with at::empty are passed in
// by the caller (their previous contents are irrelevant; every element is overwritten).
#pragma once
#import "base.h"
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "elem.h"
#include <IOKit/IOKitLib.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>

// MPSStream.mm:14 declares this private property and sets it to YES on the stream's execution descriptor.
@interface MPSGraphExecutionDescriptor ()
@property(readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace tk {

// ---------------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------------

[[noreturn]] inline void heavyFail(const char* what) {
  fprintf(stderr, "tk/heavy: %s\n", what);
  abort();
}

// getMPSDataType (OperationUtils.mm:52). BF16 = MPSDataTypeAlternateEncodingBit | MPSDataTypeFloat16.
inline MPSDataType mpsDT(DT d) {
  switch (d) {
    case F32: return MPSDataTypeFloat32;
    case F16: return MPSDataTypeFloat16;
    case BF16: return (MPSDataType)(MPSDataTypeAlternateEncodingBit | MPSDataTypeFloat16);
    case I32: return MPSDataTypeInt32;
    case I64: return MPSDataTypeInt64;
    case I16: return MPSDataTypeInt16;
    case I8: return MPSDataTypeInt8;
    case U8: return MPSDataTypeUInt8;
    case BOOL: return MPSDataTypeBool;
    default: heavyFail("mpsDT: unsupported dtype");
  }
}

inline NSArray<NSNumber*>* mpsShape(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray arrayWithCapacity:v.size()];
  for (int64_t x : v) [a addObject:@(x)];
  return a;
}

inline int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// MPSDevice::getCoreCount (mps/MPSDevice.mm:79): IORegistry AGXAccelerator "gpu-core-count".
inline unsigned gpuCoreCount() {
  static const unsigned n = [] {
    io_iterator_t it = 0;
    int cores = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) != KERN_SUCCESS) return 0u;
    io_registry_entry_t e = 0;
    while ((e = IOIteratorNext(it)) != 0) {
      CFTypeRef p = IORegistryEntryCreateCFProperty(e, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      bool found = p && CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &cores);
      if (p) CFRelease(p);
      IOObjectRelease(e);
      if (found) break;
    }
    IOObjectRelease(it);
    return (unsigned)cores;
  }();
  return n;
}

// is_apple_family_or_newer (mps/MPSDevice.mm:111). 1007 = M1, 1008 = M2, 1009 = M3/M4, 1010 = M5.
inline bool appleFamilyOrNewer(int family) { return [ctx().dev supportsFamily:(MTLGPUFamily)family]; }

// at::mps::has_mpp() (mps/MPSDevice.mm:151): macOS >= 26.2, Metal 4.0 kernels, Apple7+. The metallib this
// runtime loads IS torch's metal_40 section, so the default is true; override for other setups.
inline bool& hasMpp() {
  static bool v = true;
  return v;
}

// torch's stream command buffer is an MPSCommandBuffer (MPSStream::commandBuffer:
// [MPSCommandBuffer commandBufferFromCommandQueue:]). MPSGraph encoding requires one and the MPSNDArray
// kernels get handed it. base.mm creates a plain MTLCommandBuffer, so wrap it in place (MPSCommandBuffer
// conforms to MTLCommandBuffer; later tk::enc()/flush()/sync() go through the wrapper). Must be called
// with no open encoder.
inline MPSCommandBuffer* mpsCmdBuf() {
  id<MTLCommandBuffer> cb = cmdbuf();
  if ([(id)cb isKindOfClass:[MPSCommandBuffer class]]) return (MPSCommandBuffer*)cb;
  if (ctx().ce) heavyFail("mpsCmdBuf: encoder must be ended before upgrading the command buffer");
  MPSCommandBuffer* m = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
  ctx().cb = m;
  return m;
}

// getStridedMPSNDArray + permuteNDArray (OperationUtils.mm:323-470). `nd` is the freshly wrapped array.
inline MPSNDArray* stridedNDArrayView(MPSNDArray* nd, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
  const size_t n = strides.size();
  // getSortedStrides: indices sorted by stride, descending (std::sort, same comparator as torch).
  std::vector<int64_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return strides[a] > strides[b]; });
  NSMutableArray<NSNumber*>* sStrides = [NSMutableArray arrayWithCapacity:n];
  NSMutableArray<NSNumber*>* sShape = [NSMutableArray arrayWithCapacity:n];
  for (size_t i = 0; i < n; i++) {
    [sStrides addObject:@(strides[idx[i]])];
    [sShape addObject:@(sizes[idx[i]])];
  }
  NSArray<NSNumber*>* origShape = sShape;
  const bool hasNonUnitInner = n == 0 ? false : strides[idx[n - 1]] != 1;
  if (hasNonUnitInner) {
    origShape = [sShape copy];
    [sStrides addObject:@1];
    [sShape addObject:@1];
  }
  if (n == 0) origShape = mpsShape(sizes);
  nd = [nd arrayViewWithShape:sShape strides:sStrides];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  if (hasNonUnitInner) {
    MPSNDArrayIdentity* identity = [[MPSNDArrayIdentity alloc] initWithDevice:ctx().dev];
    nd = [identity reshapeWithCommandBuffer:nil sourceArray:nd shape:origShape destinationArray:nil];
  }
  if (!nd) heavyFail("stridedNDArrayView: view failed");
  // permuteNDArray(nd, idx)
  std::vector<int64_t> inv(n);
  for (size_t i = 0; i < n; i++) inv[idx[i]] = (int64_t)i;
  const NSUInteger rank = [nd numberOfDimensions];
  if (rank != n) heavyFail("stridedNDArrayView: rank mismatch");
  std::vector<NSUInteger> order(rank);
  std::iota(order.begin(), order.end(), 0);
  MPSNDArrayDescriptor* desc = [nd descriptor];
  for (int64_t i = (int64_t)rank - 1; i >= 0; i--) {
    NSUInteger axis = (NSUInteger)inv[i];
    auto it = std::find(order.begin(), order.end(), axis);
    NSUInteger axis1 = rank - i - 1;
    NSUInteger axis2 = order.end() - it - 1;
    std::iter_swap(order.begin() + i, it);
    if (axis1 != axis2) [desc transposeDimension:axis1 withDimension:axis2];
  }
  MPSNDArray* r = [nd arrayViewWithCommandBuffer:nil descriptor:desc aliasing:MPSAliasingStrategyShallAlias];
#pragma clang diagnostic pop
  if (!r) heavyFail("stridedNDArrayView: permute view failed");
  return r;
}

// getMPSNDArray(t, t.sizes(), t.strides()) (OperationUtils.mm:409). strides in elements.
inline MPSNDArray* ndArray(const Buf& b, DT dt, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:mpsDT(dt) shape:mpsShape(sizes)];
  d.preferPackedRows = YES;
  MPSNDArray* nd = [[MPSNDArray alloc] initWithBuffer:b.buf offset:b.off descriptor:d];
  if (!strides.empty()) nd = stridedNDArrayView(nd, sizes, strides);
  return nd;
}

inline std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> s(sizes.size(), 1);
  for (int64_t i = (int64_t)sizes.size() - 2; i >= 0; i--) s[i] = s[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
  return s;
}

// Placeholder (OperationUtils.mm:478) for a contiguous tensor with storage_offset == 0:
// [[MPSGraphTensorData alloc] initWithMTLBuffer:shape:dataType:].
inline MPSGraphTensorData* graphData(const Buf& b, DT dt, const std::vector<int64_t>& shape) {
  if (b.off != 0) heavyFail("graphData: storage offset != 0 (torch gathers or uses the strided NDArray path)");
  return [[MPSGraphTensorData alloc] initWithMTLBuffer:b.buf shape:mpsShape(shape) dataType:mpsDT(dt)];
}

// Placeholder strided-API branch (non-contiguous or offset tensor, useMPSStridedAPI = true, no gather):
// 1-D NDArray over the remaining storage, then getStridedMPSNDArray. storageNumel = elements from the
// tensor's storage offset to the end of its storage (src.storage().nbytes()/elsize - storage_offset).
inline MPSGraphTensorData* graphDataStrided(const Buf& b, DT dt, const std::vector<int64_t>& sizes,
                                            const std::vector<int64_t>& strides, int64_t storageNumel) {
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:mpsDT(dt) shape:@[ @(storageNumel) ]];
  d.preferPackedRows = YES;
  MPSNDArray* nd = [[MPSNDArray alloc] initWithBuffer:b.buf offset:b.off descriptor:d];
  nd = stridedNDArrayView(nd, sizes, strides);
  return [[MPSGraphTensorData alloc] initWithMPSNDArray:nd];
}

// MPSStream::executeMPSGraph (mps/MPSStream.mm): endKernelCoalescing, encodeToCommandBuffer with the
// stream's execution descriptor (enableCommitAndContinue = YES, optimizationLevel0), then
// synchronize(COMMIT_ADAPTIVE) which only ends the encoder (and commits under memory pressure).
inline MPSGraphExecutionDescriptor* graphExecDesc() {
  static MPSGraphExecutionDescriptor* d = [] {
    MPSGraphExecutionDescriptor* e = [MPSGraphExecutionDescriptor new];
    e.enableCommitAndContinue = YES;
    MPSGraphCompilationDescriptor* c = [MPSGraphCompilationDescriptor new];
    c.optimizationLevel = MPSGraphOptimizationLevel0;
    e.compilationDescriptor = c;
    return e;
  }();
  return d;
}
inline void runGraph(MPSGraph* g, NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                     NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results) {
  endEncoder();
  [g encodeToCommandBuffer:mpsCmdBuf() feeds:feeds targetOperations:nil resultsDictionary:results executionDescriptor:graphExecDesc()];
  ctx().ndispatch++;
}

// Cached graphs, like torch's MPSGraphCache (keyed by op string key). Same graph object is reused for the
// same key; a new MPSGraph would compile to the same program, so caching only affects speed.
struct CachedGraph {
  MPSGraph* graph = nil;
  MPSGraphTensor* in0 = nil;
  MPSGraphTensor* in1 = nil;
  MPSGraphTensor* in2 = nil;
  MPSGraphTensor* out = nil;
};
inline std::unordered_map<std::string, CachedGraph>& graphCache() {
  static std::unordered_map<std::string, CachedGraph> c;
  return c;
}

// ---------------------------------------------------------------------------------------------------
// aten::linear
// ---------------------------------------------------------------------------------------------------

enum class LinearRoute {
  Trivial,   // output numel == 0 or in_features == 0 (bias copy / zero fill)
  Gemv,      // rows == 1 (or Apple7/8 K > 32768 overflow fallback): at::addmm / at::mm -> gemv_* kernels
  NDArray,   // all operands contiguous: MPSNDArrayMatrixMultiplication (_mps_linear_nograph)
  Graph,     // otherwise (non-contiguous or complex): MPSGraph transpose + matmul + add (not implemented)
};

// _mps_linear route selection (Linear.mm:92-199). rows = product of all leading dims of the input.
inline LinearRoute linear_route(int64_t rows, int64_t in_f, int64_t out_f, bool allContiguous) {
  if (rows * out_f == 0 || in_f == 0) return LinearRoute::Trivial;
  // needs_mm_overflow_fallback: only on GPUs older than Apple9.
  const bool overflow = !appleFamilyOrNewer(1009) && in_f > 32768 && rows >= 16 && out_f >= 16;
  if (rows == 1 || overflow) return LinearRoute::Gemv;  // (overflow with rows > 1 lands in the metal GEMM, not handled here)
  return allContiguous ? LinearRoute::NDArray : LinearRoute::Graph;
}

// GemvDims (native/mps/kernels/LinearAlgebra.h)
struct GemvDims {
  int64_t n, K, ld, ms, xs, bias_r, bias_c;
};

struct GemvPick {
  int nsimd, vec;
};
// GemvPolicy::pick_nt + clamp_vec (GemmHeuristics.cpp:6,102). align in elements.
inline GemvPick gemv_pick_nt(DT dt, int64_t outlen, int64_t align) {
  GemvPick c = dt == F32 ? GemvPick{outlen >= 2048 ? 4 : 16, 4} : GemvPick{outlen >= 8192 ? 8 : 4, 8};
  while (c.vec > 1 && (align & (c.vec - 1))) c.vec >>= 1;
  return c;
}

// rows == 1 linear: Linear.mm:158 -> at::addmm(bias, x2d, W.t()) (or at::mm without bias) ->
// addmm_out_mps_impl -> try_mps_gemv -> dispatch_gemv (LinearAlgebra.mm:215-323).
// A = x (1,K) contiguous, B = W.t() (K,N) with strides (1,K) -> resolve_matrix: transposed, ld = K,
// stride = 1 -> gemv_nt (m_is_one && transposed). W must be the contiguous [out_f,in_f] weight.
// Kernel: gemv_nt_{dt}_{nsimd}_{vec}_{ab|none}_{xc|xs}; args 0=W 1=x 2=out 3=GemvDims 4=bias (W as dummy)
// 5={alpha,beta}={1,1}; dispatchThreadgroups (ceil(N/nsimd),1,1) x (nsimd*32,1,1).
// Not covered: 64-bit offset variants (_i64, operands > 2^31 elements) and the K == 1 corner where
// resolve_matrix reports the matrix as non-transposed (gemv_t).
inline std::string linear_gemv_name(DT dt, int64_t in_f, int64_t out_f, bool hasBias, int64_t wOffElems, int64_t xOffElems, GemvPick* pickOut = nullptr) {
  if (dt != F32 && dt != F16 && dt != BF16) heavyFail("linear_gemv: dtype takes the MPSGraph mm path");
  if (in_f == 1) heavyFail("linear_gemv: in_features == 1 resolves to gemv_t (not mirrored)");
  const int64_t align = in_f | wOffElems;  // matrix.ld | matrix.tensor.storage_offset()
  // pick_nt then normalize_gemv_config (clamp_vec again; idempotent)
  GemvPick c = gemv_pick_nt(dt, out_f, align);
  const bool xc = c.vec > 1 && (xOffElems % c.vec) == 0;  // vec_xs == 1 always here
  if (pickOut) *pickOut = c;
  return std::string("gemv_nt_") + dtName(dt) + "_" + std::to_string(c.nsimd) + "_" + std::to_string(c.vec) + "_" +
         (hasBias ? "ab" : "none") + "_" + (xc ? "xc" : "xs");
}
inline void linear_gemv(Buf x, Buf W, const Buf* bias, Buf out, DT dt, uint32_t in_f, uint32_t out_f) {
  const uint32_t es = dtSize(dt);
  GemvPick c;
  const std::string name = linear_gemv_name(dt, in_f, out_f, bias != nullptr, (int64_t)(W.off / es), (int64_t)(x.off / es), &c);
  GemvDims dims;
  dims.n = out_f;
  dims.K = in_f;
  dims.ld = in_f;
  dims.ms = 1;
  dims.xs = 1;
  dims.bias_r = bias ? 1 : 0;  // expanded_bias.stride(1) of the 1-D bias expanded to (1,N); 0 for mm
  dims.bias_c = 0;
  const std::array<float, 2> alphaBeta = {1.0f, 1.0f};
  auto e = enc();
  [e setComputePipelineState:pso(name)];
  setBuf(e, W, 0);
  setBuf(e, x, 1);
  setBuf(e, out, 2);
  setVal(e, dims, 3);
  setBuf(e, bias ? *bias : W, 4);
  setVal(e, alphaBeta, 5);
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger)ceilDiv(out_f, c.nsimd), 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)c.nsimd * 32, 1, 1)];
  ctx().ndispatch++;
}

// LookUpOrCreateCachedKernel("mps_linear" + getTensorsStringKey({input, weight, bias}, true, true)):
// torch keeps one MPSNDArrayMatrixMultiplication per (dtype, input shape, weight shape, bias shape).
inline MPSNDArrayMatrixMultiplication* linearKernel(DT dt, uint32_t rows, uint32_t in_f, uint32_t out_f, bool hasBias) {
  static std::unordered_map<std::string, MPSNDArrayMatrixMultiplication*> cache;
  char key[128];
  snprintf(key, sizeof key, "mps_linear:%u:%u,%u:%u,%u:%d", (unsigned)dt, rows, in_f, out_f, in_f, hasBias ? (int)out_f : -1);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  MPSNDArrayMatrixMultiplication* k = [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:ctx().dev sourceCount:hasBias ? 3 : 2];
  cache[key] = k;
  return k;
}

// _mps_linear_nograph (Linear.mm:38). x: contiguous [rows,in_f] (a >2-D input is flattened to 2-D by torch
// before it gets here: input.flatten(0,-2)), W: contiguous [out_f,in_f], bias: [out_f], out: [rows,out_f].
// Sequence: endKernelCoalescing -> fresh compute encoder -> NDArrays (input/out/bias through
// getMPSNDArray + strided view, weight through a transposed descriptor) -> cached kernel
// (sourceCount 3 with bias, 2 without; alpha/beta untouched) -> encodeToCommandEncoder:commandBuffer:
// sourceArrays:destinationArray:. Everything the kernel dispatches (incl. MPS's own 8192-row chunking of
// very tall inputs, e.g. 599343x64 -> 7) happens inside that single call.
inline void linear_ndarray(Buf x, Buf W, const Buf* bias, Buf out, DT dt, uint32_t rows, uint32_t in_f, uint32_t out_f) {
  @autoreleasepool {
  endEncoder();
  MPSCommandBuffer* cb = mpsCmdBuf();
  id<MTLComputeCommandEncoder> e = enc();
  MPSNDArray* inND = ndArray(x, dt, {rows, in_f}, {in_f, 1});
  MPSNDArray* outND = ndArray(out, dt, {rows, out_f}, {out_f, 1});
  MPSNDArrayDescriptor* wDesc = [MPSNDArrayDescriptor descriptorWithDataType:mpsDT(dt) shape:@[ @(out_f), @(in_f) ]];
  wDesc.preferPackedRows = YES;
  [wDesc transposeDimension:0 withDimension:1];
  MPSNDArray* wND = [[MPSNDArray alloc] initWithBuffer:W.buf offset:W.off descriptor:wDesc];
  MPSNDArrayMatrixMultiplication* k = linearKernel(dt, rows, in_f, out_f, bias != nullptr);
  if (bias) {
    MPSNDArray* bND = ndArray(*bias, dt, {out_f}, {1});
    [k encodeToCommandEncoder:e commandBuffer:cb sourceArrays:@[ inND, wND, bND ] destinationArray:outND];
  } else {
    [k encodeToCommandEncoder:e commandBuffer:cb sourceArrays:@[ inND, wND ] destinationArray:outND];
  }
  ctx().ndispatch++;
  }
}

// aten::linear for contiguous float operands. rows = product of the input's leading dims.
// Returns false when torch would add the bias separately afterwards (Apple7 + macOS 26.x "decompose_bias":
// the caller must then run output.add_(bias) with the elementwise family); true otherwise.
inline bool linear(Buf x, Buf W, const Buf* bias, Buf out, DT dt, uint32_t rows, uint32_t in_f, uint32_t out_f) {
  switch (linear_route(rows, in_f, out_f, true)) {
    case LinearRoute::Gemv:
      if (rows != 1) heavyFail("linear: K>32768 overflow fallback (metal GEMM) not mirrored");
      linear_gemv(x, W, bias, out, dt, in_f, out_f);
      return true;
    case LinearRoute::NDArray: {
      // decompose_bias: Apple7 only (M1) on macOS 26.x; bias-free kernel + separate add_.
      static const bool decompose = [] {
        NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
        return appleFamilyOrNewer(1007) && !appleFamilyOrNewer(1008) && v.majorVersion == 26;
      }();
      const bool addAfter = bias && decompose;
      linear_ndarray(x, W, addAfter ? nullptr : bias, out, dt, rows, in_f, out_f);
      return !addAfter;
    }
    default:
      heavyFail("linear: trivial / MPSGraph route not mirrored");
  }
}

// ---------------------------------------------------------------------------------------------------
// aten::convolution, 3-D (conv3d_metal_forward)
// ---------------------------------------------------------------------------------------------------

// kernels/Convolution.h
struct ConvWeightPermuteParams {
  uint32_t output_channels, input_channels_per_group, kernel_height, kernel_width;
  uint32_t output_channel_stride, input_channel_stride, depth_stride, height_stride, width_stride;
};
struct Conv2DParams {
  int32_t N, C_in, C_out, H, W, outH, outW, kH, kW, sH, sW, padH, padW, dH, dW, C_in_per_group, C_out_per_group;
  bool has_bias;
};
struct Conv3DParams {
  Conv2DParams conv2d;
  int32_t D, outD, kD, sD, padD, dD;
  bool out_ncdhw;
};
static_assert(sizeof(Conv2DParams) == 72 && sizeof(Conv3DParams) == 100);

struct Conv3dShape {
  int64_t N, Cin, D, H, W;     // input [N,Cin,D,H,W]
  int64_t Cout;                // weight [Cout, Cin/groups, kD,kH,kW]
  int64_t k[3], stride[3], pad[3], dil[3];  // (depth, height, width)
  int64_t groups = 1;
  bool hasBias = true;
  int64_t outDim(int i) const {
    const int64_t in = i == 0 ? D : i == 1 ? H : W;
    return (in + 2 * pad[i] - dil[i] * (k[i] - 1) - 1) / stride[i] + 1;
  }
};

enum class Conv3dRoute { PointwiseMatmul, Im2colMatmul, Metal };
// _mps_convolution_impl 3-D dispatch (Convolution.mm:880-889) + conv3d_is_pointwise +
// conv3d_prefer_im2col (ConvolutionHeuristics.cpp:83).
inline Conv3dRoute conv3d_route(const Conv3dShape& s, DT dt) {
  const bool unitStride = s.stride[0] == 1 && s.stride[1] == 1 && s.stride[2] == 1;
  const bool noPad = s.pad[0] == 0 && s.pad[1] == 0 && s.pad[2] == 0;
  if (s.groups == 1 && s.k[0] == 1 && s.k[1] == 1 && s.k[2] == 1 && unitStride && noPad) return Conv3dRoute::PointwiseMatmul;
  if (s.groups != 1 || s.dil[0] != 1 || s.dil[1] != 1 || s.dil[2] != 1) return Conv3dRoute::Metal;
  const int64_t cinpg = s.Cin / s.groups;
  const int64_t reduction = cinpg * s.k[0] * s.k[1] * s.k[2];
  const int64_t oD = s.outDim(0), oH = s.outDim(1), oW = s.outDim(2);
  constexpr int64_t kI32 = std::numeric_limits<int32_t>::max();
  if (s.Cin * s.H * s.W > kI32 || s.Cout * oH * oW > kI32 || s.N * oD * oH * oW > kI32 / std::max<int64_t>(reduction, 1))
    return Conv3dRoute::Metal;
  const bool patchEmbed = noPad && s.stride[0] == s.k[0] && s.stride[1] == s.k[1] && s.stride[2] == s.k[2];
  if (patchEmbed) return reduction >= 256 ? Conv3dRoute::Im2colMatmul : Conv3dRoute::Metal;
  return (dt == F32 && s.N == 1 && oD * oH * oW <= 256 && reduction >= 4096) ? Conv3dRoute::Im2colMatmul : Conv3dRoute::Metal;
}


// conv3d_to_ndhwc (Convolution.mm:125). src: contiguous NCDHW [N,C,D,H,W]; dst: NDHWC temp of the same
// numel (torch: at::empty(..., ChannelsLast3d)).
//  * C <= 8 (or spatial > INT32_MAX): tensor.contiguous(ChannelsLast3d) = generic strided copy
//    (exec_unary_kernel "identity", OperationUtils.mm:1075): kernel copy_identity_strided_{dt}_{dt},
//    args 0=dst 1=src 2=iter.shape() 3=src byte strides 4=dst byte strides 5=uint32 ndim, 2-D dispatch.
//    TensorIterator presents (after reorder by the output's strides + coalescing) dims
//    (C, D*H*W[, N]): N == 1 coalesces into the spatial dim (ndim 2, as traced); N > 1 stays a third dim
//    (derived from the coalescing rules, not traced).
//  * else nchw_to_nhwc_{dt}_{ctile}_{stile}_{vread}_{vwrite}: args 0=src 1=dst 2={int32 C, int32 spatial},
//    dispatchThreadgroups (ceil(spatial/stile), ceil(C/ctile), N) x (256,1,1).
inline void conv3d_to_ndhwc(Buf src, Buf dst, DT dt, int64_t N, int64_t C, int64_t spatial) {
  auto e = enc();
  const int64_t es = dtSize(dt);
  if (C <= 8 || spatial > std::numeric_limits<int32_t>::max()) {
    if (C == 1) heavyFail("conv3d_to_ndhwc: C == 1 is already channels-last (torch passes the input through)");
    auto p = pso(std::string("copy_identity_strided_") + dtName(dt) + "_" + dtName(dt));
    [e setComputePipelineState:p];
    setBuf(e, dst, 0);
    setBuf(e, src, 1);
    int64_t shape[3] = {C, spatial, N};
    int64_t srcStr[3] = {spatial * es, es, C * spatial * es};
    int64_t dstStr[3] = {es, C * es, C * spatial * es};
    const uint32_t ndim = N == 1 ? 2 : 3;
    if (N == 1) shape[1] = spatial;
    setRaw(e, shape, ndim * 8, 2);
    setRaw(e, srcStr, ndim * 8, 3);
    setRaw(e, dstStr, ndim * 8, 4);
    setVal(e, ndim, 5);
    dispatch2D(e, p, (NSUInteger)C, (NSUInteger)(spatial * N));
    return;
  }
  const int ctile = C <= 16 ? 16 : 32;
  const int stile = C <= 16 ? 64 : 32;
  const bool vread = spatial % 2 == 0 && (src.off / es) % 2 == 0;
  const bool vwrite = C % 2 == 0;
  const std::string name = std::string("nchw_to_nhwc_") + dtName(dt) + "_" + std::to_string(ctile) + "_" + std::to_string(stile) + "_" +
                           (vread ? "true" : "false") + "_" + (vwrite ? "true" : "false");
  [e setComputePipelineState:pso(name)];
  setBuf(e, src, 0);
  setBuf(e, dst, 1);
  const std::array<int32_t, 2> dims = {(int32_t)C, (int32_t)spatial};
  setVal(e, dims, 2);
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger)ceilDiv(spatial, stile), (NSUInteger)ceilDiv(C, ctile), (NSUInteger)N) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  ctx().ndispatch++;
}

// conv3d_weights_to_dhwio (Convolution.mm:170). W: contiguous [Cout,Cin_pg,kD,kH,kW]; dst: temp
// [kD,kH,kW,Cin_pg,Cout]. Kernel conv_weight_to_dhwio_{dt}: args 0=W 1=dst 2=ConvWeightPermuteParams;
// dispatchThreads (Cout, Cin_pg, kD*kH) tg (min(Cout,256),1,1).
inline void conv3d_weights_to_dhwio(Buf W, Buf dst, DT dt, int64_t Cout, int64_t CinPg, int64_t kD, int64_t kH, int64_t kW) {
  ConvWeightPermuteParams p;
  p.output_channels = (uint32_t)Cout;
  p.input_channels_per_group = (uint32_t)CinPg;
  p.kernel_height = (uint32_t)kH;
  p.kernel_width = (uint32_t)kW;
  p.output_channel_stride = (uint32_t)(CinPg * kD * kH * kW);
  p.input_channel_stride = (uint32_t)(kD * kH * kW);
  p.depth_stride = (uint32_t)(kH * kW);
  p.height_stride = (uint32_t)kW;
  p.width_stride = 1;
  auto e = enc();
  [e setComputePipelineState:pso(std::string("conv_weight_to_dhwio_") + dtName(dt))];
  setBuf(e, W, 0);
  setBuf(e, dst, 1);
  setVal(e, p, 2);
  [e dispatchThreads:MTLSizeMake((NSUInteger)Cout, (NSUInteger)CinPg, (NSUInteger)(kD * kH)) threadsPerThreadgroup:MTLSizeMake((NSUInteger)std::min<int64_t>(Cout, 256), 1, 1)];
  ctx().ndispatch++;
}

// Conv3dMppTile + select_conv3d_mpp_tile (ConvolutionHeuristics.cpp:13-66). The arithmetic is kept
// expression-for-expression (float/int promotions included) because the winner is picked by a strict
// float compare. `cores` = gpuCoreCount(), 16 when that reports 0.
struct Conv3dMppTile {
  int output_channels, output_width, output_height, simdgroups;
  float cost(const Conv3DParams& params, int64_t groups, int64_t cores) const {
    constexpr float kOccupancyFill = 300.0f;
    constexpr float kSimdgroupOverhead = 8.0f;
    constexpr float kOutputChannelOverhead = 90.0f;
    constexpr float kWeightLoadCost = 0.15f;
    constexpr float kStagingOverhead = 0.4f;
    const auto& conv2d = params.conv2d;
    const auto output_channel_tiles = ceilDiv(int64_t(conv2d.C_out_per_group), int64_t(output_channels));
    const auto width_tiles = ceilDiv(int64_t(conv2d.outW), int64_t(output_width));
    const auto height_tiles = ceilDiv(int64_t(conv2d.outH), int64_t(output_height));
    const auto threadgroups = int64_t(output_channel_tiles) * groups * width_tiles * height_tiles * conv2d.N * params.outD;
    const auto kernel_elements = params.kD * conv2d.kH * conv2d.kW;
    const auto padded_output = float(threadgroups) * output_channels * output_width * output_height;
    const auto outputs_per_simdgroup = float(output_width * output_height) / simdgroups;
    const auto input_tile_width = int64_t(output_width - 1) * conv2d.sW + (conv2d.kW - 1) * conv2d.dW + 1;
    const auto input_tile_height = int64_t(output_height - 1) * conv2d.sH + (conv2d.kH - 1) * conv2d.dH + 1;
    const auto staging_ratio = float(output_width * output_height) /
        (float(input_tile_width * input_tile_height) + kWeightLoadCost * output_channels * kernel_elements);
    const auto tile_efficiency = outputs_per_simdgroup / (outputs_per_simdgroup + kSimdgroupOverhead) *
        (output_channels / (output_channels + kOutputChannelOverhead)) * (staging_ratio / (staging_ratio + kStagingOverhead));
    const auto device_utilization = float(threadgroups) / (threadgroups + cores * kOccupancyFill);
    return padded_output / (tile_efficiency * device_utilization);
  }
};
inline Conv3dMppTile select_conv3d_mpp_tile(const Conv3DParams& params, int64_t groups, int64_t cores) {
  const Conv3dMppTile candidates[] = {{32, 8, 8, 2}, {32, 16, 8, 4}, {64, 8, 8, 2}, {64, 16, 8, 4}, {128, 8, 8, 4}};
  auto best = candidates[0];
  auto bestCost = std::numeric_limits<float>::max();
  for (const auto& c : candidates) {
    const auto cost = c.cost(params, groups, cores);
    if (cost < bestCost) {
      bestCost = cost;
      best = c;
    }
  }
  return best;
}
inline Conv3dMppTile select_conv3d_mpp_tile(const Conv3DParams& params, int64_t groups) {
  const unsigned c = gpuCoreCount();
  return select_conv3d_mpp_tile(params, groups, c > 0 ? (int64_t)c : 16);
}

// Conv3DParams as conv3d_metal_forward fills it (Convolution.mm:331-357). Padding bytes are zeroed here;
// torch leaves them uninitialised (the trace shows stray 0x01 there) and the kernel never reads them.
inline Conv3DParams conv3d_params(const Conv3dShape& s) {
  Conv3DParams p;
  memset(&p, 0, sizeof p);
  p.conv2d.N = (int32_t)s.N;
  p.conv2d.C_in = (int32_t)s.Cin;
  p.conv2d.C_out = (int32_t)s.Cout;
  p.conv2d.H = (int32_t)s.H;
  p.conv2d.W = (int32_t)s.W;
  p.conv2d.outH = (int32_t)s.outDim(1);
  p.conv2d.outW = (int32_t)s.outDim(2);
  p.conv2d.kH = (int32_t)s.k[1];
  p.conv2d.kW = (int32_t)s.k[2];
  p.conv2d.sH = (int32_t)s.stride[1];
  p.conv2d.sW = (int32_t)s.stride[2];
  p.conv2d.padH = (int32_t)s.pad[1];
  p.conv2d.padW = (int32_t)s.pad[2];
  p.conv2d.dH = (int32_t)s.dil[1];
  p.conv2d.dW = (int32_t)s.dil[2];
  p.conv2d.C_in_per_group = (int32_t)(s.Cin / s.groups);
  p.conv2d.C_out_per_group = (int32_t)(s.Cout / s.groups);
  p.conv2d.has_bias = s.hasBias;
  p.D = (int32_t)s.D;
  p.outD = (int32_t)s.outDim(0);
  p.kD = (int32_t)s.k[0];
  p.sD = (int32_t)s.stride[0];
  p.padD = (int32_t)s.pad[0];
  p.dD = (int32_t)s.dil[0];
  p.out_ncdhw = true;
  return p;
}

// conv3d_mpp_pipeline name (Convolution.mm:215): tries the channel-specialised name (Cin_pg <= 64) and
// falls back to "_cdyn_" when the library lacks it (torch's shipped metallib only has cdyn for 3x3x3).
// Returns "" when no MPP kernel exists (torch then uses conv3d_simd_*; not mirrored).
inline std::string conv3d_mpp_name(const Conv3dShape& s, DT dt, const Conv3dMppTile& t) {
  if (std::max<int64_t>(s.W, 16384) != 16384 || std::max<int64_t>(s.H, 16384) != 16384) return "";
  auto build = [&](const std::string& ch) {
    char b[256];
    snprintf(b, sizeof b, "conv3d_mpp_%s_b%d_w%d_h%d_s%d_k%d%d%d_s%d%d%d_d%d%d%d_c%s_%s_%s_%s", dtName(dt), t.output_channels,
             t.output_width, t.output_height, t.simdgroups, (int)s.k[0], (int)s.k[1], (int)s.k[2], (int)s.stride[0],
             (int)s.stride[1], (int)s.stride[2], (int)s.dil[0], (int)s.dil[1], (int)s.dil[2], ch.c_str(),
             s.hasBias ? "bias" : "nobias", "ncdhw", s.groups > 1 ? "grouped" : "ungrouped");
    return std::string(b);
  };
  const int64_t cinpg = s.Cin / s.groups;
  if (cinpg <= 64) {
    const auto n = build(std::to_string(cinpg));
    if (hasFn(n)) return n;
  }
  const auto n = build("dyn");
  return hasFn(n) ? n : std::string();
}

// aten::convolution, 5-D input, direct Metal path: conv3d_metal_forward (Convolution.mm:285).
//   x      contiguous NCDHW [N,Cin,D,H,W]          W   contiguous [Cout,Cin/groups,kD,kH,kW]
//   bias   [Cout] or nullptr (must match s.hasBias) out contiguous NCDHW [N,Cout,oD,oH,oW]
//   tmpAct NDHWC temp, numel(x) elements            tmpW DHWIO temp, numel(W) elements
// Dispatch order (as traced): activation transpose, weight permute, conv3d_mpp.
// conv3d_mpp args: 0=tmpAct 1=tmpW 2=out 3=Conv3DParams 4=bias (tmpAct as dummy); dispatchThreadgroups
// (ceil(Cout_pg/tile.oc)*groups, ceil(oW/tile.ow), ceil(oH/tile.oh)*N*oD) x (tile.simdgroups*32,1,1).
// Not mirrored: channels-last input/output, long-index (plane > INT32_MAX) and non-MPP simd fallback.
inline void conv3d(Buf x, Buf W, const Buf* bias, Buf out, Buf tmpAct, Buf tmpW, DT dt, const Conv3dShape& s) {
  if ((bias != nullptr) != s.hasBias) heavyFail("conv3d: bias/hasBias mismatch");
  if (conv3d_route(s, dt) != Conv3dRoute::Metal) heavyFail("conv3d: pointwise/im2col matmul route not mirrored");
  if (!hasMpp()) heavyFail("conv3d: non-MPP (conv3d_simd) path not mirrored");
  const int64_t cinpg = s.Cin / s.groups;
  conv3d_to_ndhwc(x, tmpAct, dt, s.N, s.Cin, s.D * s.H * s.W);
  conv3d_weights_to_dhwio(W, tmpW, dt, s.Cout, cinpg, s.k[0], s.k[1], s.k[2]);
  const Conv3DParams params = conv3d_params(s);
  const Conv3dMppTile tile = select_conv3d_mpp_tile(params, s.groups);
  const std::string name = conv3d_mpp_name(s, dt, tile);
  if (name.empty()) heavyFail("conv3d: no conv3d_mpp kernel for this configuration (torch falls back to conv3d_simd)");
  auto e = enc();
  [e setComputePipelineState:pso(name)];
  setBuf(e, tmpAct, 0);
  setBuf(e, tmpW, 1);
  setBuf(e, out, 2);
  setVal(e, params, 3);
  setBuf(e, bias ? *bias : tmpAct, 4);
  const auto& c = params.conv2d;
  const int64_t ocTiles = ceilDiv(c.C_out_per_group, tile.output_channels);
  const int64_t wTiles = ceilDiv(c.outW, tile.output_width);
  const int64_t hTiles = ceilDiv(c.outH, tile.output_height);
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(ocTiles * s.groups), (NSUInteger)wTiles, (NSUInteger)hTiles * c.N * params.outD)
      threadsPerThreadgroup:MTLSizeMake((NSUInteger)tile.simdgroups * 32, 1, 1)];
  ctx().ndispatch++;
}

// The only conv3d configuration in the pipeline: 3x3x3, stride 1, padding 1, dilation 1, groups 1, cubic
// or non-cubic volume. tmpAct: N*Cin*D*H*W elements, tmpW: Cout*Cin*27 elements.
inline void conv3d_333(Buf x, Buf W, const Buf* bias, Buf out, Buf tmpAct, Buf tmpW, DT dt, int64_t N, int64_t Cin, int64_t D,
                       int64_t H, int64_t Wd, int64_t Cout) {
  Conv3dShape s;
  s.N = N; s.Cin = Cin; s.D = D; s.H = H; s.W = Wd; s.Cout = Cout;
  for (int i = 0; i < 3; i++) { s.k[i] = 3; s.stride[i] = 1; s.pad[i] = 1; s.dil[i] = 1; }
  s.groups = 1;
  s.hasBias = bias != nullptr;
  conv3d(x, W, bias, out, tmpAct, tmpW, dt, s);
}

// ---------------------------------------------------------------------------------------------------
// aten::convolution, 2-D (MPSGraph) -- DINOv3 patch embedding
// ---------------------------------------------------------------------------------------------------

// _mps_convolution_impl 2-D branch (Convolution.mm:895-990) for a contiguous NCHW input, contiguous OIHW
// weight and 1-D bias, all with storage offset 0 (torch gathers offset/strided weights first).
// Preconditions for this branch: kH,kW < 256 (else conv2d_r*_ metal kernel), not the conv1d shape
// (H == 1 && kH == 1 && padH == 0), not depthwise (groups > 1 && Cin/groups == 1 uses the depthwise op).
// Graph: x = placeholder([N,Cin,H,W]); w = placeholder([Cout,Cin/g,kH,kW]);
//        y = convolution2D(x, w, desc{strideInX=sW, strideInY=sH, dilationRateInX=dW, dilationRateInY=dH,
//            paddingStyle=Explicit, left=right=padW, top=bottom=padH, NCHW, OIHW, groups});
//        b = placeholder(shape nil);  y = addition(y, b)   (fed with the bias viewed as [1,Cout,1,1]).
// Run: encodeToCommandBuffer:feeds:targetOperations:nil resultsDictionary:{y: out}.
inline void conv2d_graph(Buf x, Buf W, const Buf* bias, Buf out, DT dt, int64_t N, int64_t Cin, int64_t H, int64_t Wd, int64_t Cout,
                         int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t padH, int64_t padW, int64_t dH = 1, int64_t dW = 1,
                         int64_t groups = 1) {
  if (kH >= 256 || kW >= 256) heavyFail("conv2d_graph: large-kernel metal path not mirrored");
  if (H == 1 && kH == 1 && padH == 0) heavyFail("conv2d_graph: conv1d metal path not mirrored");
  if (groups > 1 && Cin / groups == 1) heavyFail("conv2d_graph: depthwise graph not mirrored");
  const int64_t oH = (H + 2 * padH - dH * (kH - 1) - 1) / sH + 1;
  const int64_t oW = (Wd + 2 * padW - dW * (kW - 1) - 1) / sW + 1;
  char key[256];
  snprintf(key, sizeof key, "mps_convolution:%lld,%lld:%lld,%lld:%lld,%lld:%lld:%u:[%lld,%lld,%lld,%lld]:[%lld,%lld,%lld,%lld]:%d",
           (long long)sH, (long long)sW, (long long)dH, (long long)dW, (long long)padH, (long long)padW, (long long)groups, (unsigned)dt,
           (long long)N, (long long)Cin, (long long)H, (long long)Wd, (long long)Cout, (long long)(Cin / groups), (long long)kH,
           (long long)kW, bias ? 1 : 0);
  CachedGraph& cg = graphCache()[key];
  if (!cg.graph) {
    MPSGraph* g = [MPSGraph new];
    cg.in0 = [g placeholderWithShape:mpsShape({N, Cin, H, Wd}) dataType:mpsDT(dt) name:nil];
    cg.in1 = [g placeholderWithShape:mpsShape({Cout, Cin / groups, kH, kW}) dataType:mpsDT(dt) name:nil];
    MPSGraphConvolution2DOpDescriptor* d = [MPSGraphConvolution2DOpDescriptor new];
    d.strideInX = (NSUInteger)sW;
    d.strideInY = (NSUInteger)sH;
    d.dilationRateInX = (NSUInteger)dW;
    d.dilationRateInY = (NSUInteger)dH;
    d.paddingStyle = MPSGraphPaddingStyleExplicit;
    d.paddingLeft = (NSUInteger)padW;
    d.paddingRight = (NSUInteger)padW;
    d.paddingTop = (NSUInteger)padH;
    d.paddingBottom = (NSUInteger)padH;
    d.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    d.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    d.groups = (NSUInteger)groups;
    MPSGraphTensor* y = [g convolution2DWithSourceTensor:cg.in0 weightsTensor:cg.in1 descriptor:d name:nil];
    if (bias) {
      cg.in2 = [g placeholderWithShape:nil dataType:mpsDT(dt) name:nil];
      y = [g additionWithPrimaryTensor:y secondaryTensor:cg.in2 name:nil];
    }
    cg.out = y;
    cg.graph = g;
  }
  NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[NSMutableDictionary alloc] initWithCapacity:3];
  feeds[cg.in0] = graphData(x, dt, {N, Cin, H, Wd});
  feeds[cg.in1] = graphData(W, dt, {Cout, Cin / groups, kH, kW});
  if (bias) feeds[cg.in2] = graphData(*bias, dt, {1, Cout, 1, 1});
  runGraph(cg.graph, feeds, @{cg.out : graphData(out, dt, {N, Cout, oH, oW})});
}

// DINOv3 patch embedding: kernel == stride == patch, no padding. x [N,Cin,H,W] -> out [N,Cout,H/p,W/p].
inline void conv2d_patch(Buf x, Buf W, const Buf* bias, Buf out, DT dt, int64_t N, int64_t Cin, int64_t H, int64_t Wd, int64_t Cout, int64_t patch) {
  conv2d_graph(x, W, bias, out, dt, N, Cin, H, Wd, Cout, patch, patch, patch, patch, 0, 0);
}

// ---------------------------------------------------------------------------------------------------
// aten::_scaled_dot_product_attention_math_for_mps
// ---------------------------------------------------------------------------------------------------

enum class SdpaRoute {
  Graph,        // sdpa_general_mps (MPSGraph) -- none of the kernel paths apply
  VectorFast,   // qL <= 8, kL < 1024 (sdpa_vector_*)
  Vector2Pass,  // qL <= 8, kL >= 1024 (or GQA with kL >= 4096)
  Prefill,      // prefill_attention_{float32|float16|bfloat16}_...
  PrefillMpp,   // prefill_attention_mpp_{a9|a10}_{float16|bfloat16}_...
};

enum class SdpaMask { None, Bool, SameDtype, Other };

// Path selection of _scaled_dot_product_attention_math_mps (Attention.mm:742-866). dropout_p must be 0.
// Dq/Dv: head dims of query/value. H/Hkv: query / key head counts (GQA when different).
inline SdpaRoute sdpa_route(DT dt, int64_t H, int64_t Hkv, int64_t qL, int64_t kL, int64_t Dq, int64_t Dv, SdpaMask mask = SdpaMask::None,
                            bool isCausal = false) {
  const bool hasMask = mask != SdpaMask::None;
  const bool maskOk = mask != SdpaMask::Other;
  const bool vecHead = Dq == Dv && (Dq == 64 || Dq == 96 || Dq == 128 || Dq == 256);
  const bool supportsVector = qL <= 8 && qL <= kL && maskOk && vecHead;
  const bool fast = supportsVector && !(isCausal && hasMask);
  static const bool prefillDisabled = [] {
    const char* v = getenv("PYTORCH_MPS_DISABLE_PREFILL_ATTENTION");
    return v && strcmp(v, "0") != 0;
  }();
  const bool prefillDtype = dt == F32 || dt == F16 || dt == BF16;
  const bool prefillHead = Dq == Dv && (Dq == 32 || Dq == 64 || Dq == 72 || Dq == 80 || Dq == 96 || Dq == 128 || Dq == 256);
  const bool prefill = !prefillDisabled && !fast && prefillDtype && maskOk && prefillHead && qL > 8 && kL > 0;
  // can_use_mpp_prefill
  const bool mpp = !prefillDisabled && hasMpp() && !fast && qL > 8 && kL > 0 && (dt == F16 || dt == BF16) && vecHead && maskOk;
  if (!fast && !prefill && !mpp) return SdpaRoute::Graph;
  if (mpp) return SdpaRoute::PrefillMpp;
  if (prefill) return SdpaRoute::Prefill;
  return (kL >= 1024 || (Hkv < H && kL >= 4096)) ? SdpaRoute::Vector2Pass : SdpaRoute::VectorFast;
}

// PrefillAttnParamsHost / PrefillAttnMaskParamsHost (Attention.mm:487-504), 128 / 32 bytes.
struct PrefillAttnParams {
  int32_t B, H, D, qL, kL, gqa_factor;
  float scale, softcapping;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
struct PrefillAttnMaskParams {
  int64_t M_strides[4];
};
static_assert(sizeof(PrefillAttnParams) == 128);

struct SdpaArgs {
  int64_t B = 1, H = 1, Hkv = 1;  // batch, query heads, key/value heads
  int64_t qL = 0, kL = 0, D = 0;  // query length, key length, head dim (q, k and v share it)
  // Element strides of dims 0..2 of the 4-D [B,H,L,D] views exactly as torch sees them (stride(-1) must be
  // 1; views with another last-dim stride are made contiguous by torch first). A transposed
  // [B,L,H,D] -> [B,H,L,D] view has strides {L*H*D, D, H*D}.
  int64_t qs[3] = {0, 0, 0}, ks[3] = {0, 0, 0}, vs[3] = {0, 0, 0};
  std::optional<double> scale;    // python `scale=` kwarg; default 1/sqrt(D)
  bool isCausal = false;
  // Optional mask already expanded to 4-D [B,H,qL,kL] with stride(-1) == 1: bool or the query dtype.
  const Buf* mask = nullptr;
  bool maskIsBool = false;
  int64_t ms[4] = {0, 0, 0, 0};
};

inline void sdpa_contiguous_strides(int64_t out[3], int64_t H, int64_t L, int64_t D) {
  out[0] = H * L * D;
  out[1] = L * D;
  out[2] = D;
}

// Kernel name of sdpa_prefill_mps (Attention.mm:650-710):
//   prefill_attention[_mpp_{a9|a10}]_{float32|float16|bfloat16}_bq{BQ}_bk{BK}_bd{D}_wm{WM}_wn{WN}
//       _hm{has_mask}_dc{is_causal}_mask{bool_|<dtype>}
// Block shapes: plain D=256 -> {16,8,2,1}; D=128 -> {32,16,4,1}; D in {96,80,72,64,32} -> {32,32,4,1};
// MPP (D in {64,96,128,256}) -> {64,32,4,1}. a10 = Apple10+ GPU family (M5), else a9.
struct PrefillBlock {
  int BQ, BK, WM, WN;
};
inline PrefillBlock sdpa_prefill_block(int64_t D, bool mpp) {
  if (mpp) return {64, 32, 4, 1};
  if (D == 256) return {16, 8, 2, 1};
  if (D == 128) return {32, 16, 4, 1};
  return {32, 32, 4, 1};
}
inline std::string sdpa_prefill_name(DT dt, int64_t D, bool mpp, bool hasMask, bool maskIsBool, bool isCausal) {
  const char* d = dt == F32 ? "float32" : dt == F16 ? "float16" : dt == BF16 ? "bfloat16" : nullptr;
  if (!d) heavyFail("sdpa_prefill: unsupported dtype");
  const PrefillBlock b = sdpa_prefill_block(D, mpp);
  char buf[256];
  if (mpp) {
    static const bool a10 = appleFamilyOrNewer(1010);
    snprintf(buf, sizeof buf, "prefill_attention_mpp_%s_%s_bq%d_bk%d_bd%d_wm%d_wn%d_hm%d_dc%d_mask%s", a10 ? "a10" : "a9", d, b.BQ, b.BK,
             (int)D, b.WM, b.WN, hasMask ? 1 : 0, isCausal ? 1 : 0, (hasMask && maskIsBool) ? "bool_" : d);
  } else {
    snprintf(buf, sizeof buf, "prefill_attention_%s_bq%d_bk%d_bd%d_wm%d_wn%d_hm%d_dc%d_mask%s", d, b.BQ, b.BK, (int)D, b.WM, b.WN,
             hasMask ? 1 : 0, isCausal ? 1 : 0, (hasMask && maskIsBool) ? "bool_" : d);
  }
  return buf;
}

// sdpa_prefill_mps (Attention.mm:585). q/k/v are bound at their own storage offsets with their own
// strides (no copies as long as stride(-1) == 1); out is a fresh contiguous [B,H,qL,D] tensor.
// Args: 0=q 1=k 2=v 3=out 4=PrefillAttnParams; without mask 5=uint32(0) 6=uint32(0) (dummy setBytes);
// with mask 5=PrefillAttnMaskParams 6=mask buffer. dispatchThreadgroups (ceil(qL/BQ), H, B) x (32, WM, WN).
// softcapping is always 1.0. scale = float(scale kwarg) or float(1.0 / sqrt(double(D))).
inline void sdpa_prefill(Buf q, Buf k, Buf v, Buf out, DT dt, const SdpaArgs& a) {
  const SdpaMask mk = !a.mask ? SdpaMask::None : a.maskIsBool ? SdpaMask::Bool : SdpaMask::SameDtype;
  const SdpaRoute route = sdpa_route(dt, a.H, a.Hkv, a.qL, a.kL, a.D, a.D, mk, a.isCausal);
  if (route != SdpaRoute::Prefill && route != SdpaRoute::PrefillMpp) heavyFail("sdpa_prefill: torch takes a non-prefill path for this shape");
  const bool mpp = route == SdpaRoute::PrefillMpp;
  PrefillAttnParams p;
  memset(&p, 0, sizeof p);
  p.B = (int32_t)a.B;
  p.H = (int32_t)a.H;
  p.D = (int32_t)a.D;
  p.qL = (int32_t)a.qL;
  p.kL = (int32_t)a.kL;
  p.gqa_factor = (int32_t)(a.H / a.Hkv);
  p.scale = a.scale.has_value() ? (float)*a.scale : (float)(1.0 / std::sqrt((double)a.D));
  p.softcapping = 1.0f;
  for (int i = 0; i < 3; i++) {
    p.Q_strides[i] = a.qs[i];
    p.K_strides[i] = a.ks[i];
    p.V_strides[i] = a.vs[i];
  }
  sdpa_contiguous_strides(p.O_strides, a.H, a.qL, a.D);
  const PrefillBlock blk = sdpa_prefill_block(a.D, mpp);
  auto e = enc();
  [e setComputePipelineState:pso(sdpa_prefill_name(dt, a.D, mpp, a.mask != nullptr, a.maskIsBool, a.isCausal))];
  setBuf(e, q, 0);
  setBuf(e, k, 1);
  setBuf(e, v, 2);
  setBuf(e, out, 3);
  setVal(e, p, 4);
  if (a.mask) {
    PrefillAttnMaskParams mp;
    for (int i = 0; i < 4; i++) mp.M_strides[i] = a.ms[i];
    setVal(e, mp, 5);
    setBuf(e, *a.mask, 6);
  } else {
    const uint32_t zero = 0;
    setVal(e, zero, 5);
    setVal(e, zero, 6);
  }
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger)ceilDiv(a.qL, blk.BQ), (NSUInteger)a.H, (NSUInteger)a.B)
      threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)blk.WM, (NSUInteger)blk.WN)];
  ctx().ndispatch++;
}

// ---------------------------------------------------------------------------------------------------
// aten::repeat (MPSGraph tile) and the MPSGraph part of aten::bincount
// ---------------------------------------------------------------------------------------------------

// repeat_mps (Repeat.mm:41). x: contiguous, offset 0, viewed with leading 1-dims so its rank equals
// repeats.size() (torch: self.expand(padded_size); a strided/offset input is gathered by a copy first).
// Graph: placeholder(padded shape) -> tileTensor:withMultiplier:repeats. Pure data movement.
inline void repeat_graph(Buf x, Buf out, DT dt, const std::vector<int64_t>& paddedSizes, const std::vector<int64_t>& repeats) {
  if (paddedSizes.size() != repeats.size()) heavyFail("repeat_graph: rank mismatch");
  std::string key = "repeat_mps:" + std::to_string((unsigned)dt);
  std::vector<int64_t> target(repeats.size());
  for (size_t i = 0; i < repeats.size(); i++) {
    key += ":" + std::to_string(paddedSizes[i]) + "x" + std::to_string(repeats[i]);
    target[i] = paddedSizes[i] * repeats[i];
  }
  CachedGraph& cg = graphCache()[key];
  if (!cg.graph) {
    MPSGraph* g = [MPSGraph new];
    cg.in0 = [g placeholderWithShape:mpsShape(paddedSizes) dataType:mpsDT(dt) name:nil];
    cg.out = [g tileTensor:cg.in0 withMultiplier:mpsShape(repeats) name:nil];
    cg.graph = g;
  }
  runGraph(cg.graph, @{cg.in0 : graphData(x, dt, paddedSizes)}, @{cg.out : graphData(out, dt, target)});
}

// bincount_mps_impl without weights (SummaryOps.mm:9). Before this, _bincount_mps runs aminmax(self)
// (+ stack + .cpu() readback) to get nbins = max(max+1, minlength) and at::zeros({nbins}, int64) -- those
// dispatches belong to other families. Graph: idx = placeholder([n], self dtype); data = placeholder(nil,
// int64); upd = constant(1.0, shape [n], int64); out = scatter(data, upd, idx, axis 0, mode Add).
// Feeds: idx <- self, data <- out (the zero-filled output buffer); result written back into out.
// self: int32/int64 1-D of n elements with element stride `stride`; when it is not (contiguous, offset 0)
// torch feeds it through the strided NDArray view and needs storageNumel (elements from self's storage
// offset to the end of its storage). Integer scatter-add: exact.
inline void bincount_graph(Buf self, DT selfDt, int64_t n, int64_t stride, int64_t storageNumel, Buf out, int64_t nbins) {
  if (selfDt == U8 || selfDt == BOOL) heavyFail("bincount_graph: uint8 (cast) / bool input not mirrored");
  const std::string key = "bincount_mps_impl:" + std::to_string((unsigned)selfDt) + ":" + std::to_string(n);
  CachedGraph& cg = graphCache()[key];
  if (!cg.graph) {
    MPSGraph* g = [MPSGraph new];
    cg.in0 = [g placeholderWithShape:mpsShape({n}) dataType:mpsDT(selfDt) name:nil];
    cg.in1 = [g placeholderWithShape:nil dataType:MPSDataTypeInt64 name:nil];
    MPSGraphTensor* upd = [g constantWithScalar:1.0f shape:mpsShape({n}) dataType:MPSDataTypeInt64];
    cg.out = [g scatterWithDataTensor:cg.in1 updatesTensor:upd indicesTensor:cg.in0 axis:0 mode:MPSGraphScatterModeAdd name:nil];
    cg.graph = g;
  }
  const bool direct = stride == 1 && self.off == 0;  // (n == 1 is also contiguous whatever its stride)
  MPSGraphTensorData* idx = (direct || (n == 1 && self.off == 0)) ? graphData(self, selfDt, {n}) : graphDataStrided(self, selfDt, {n}, {stride}, storageNumel);
  runGraph(cg.graph, @{cg.in0 : idx, cg.in1 : graphData(out, I64, {nbins})}, @{cg.out : graphData(out, I64, {nbins})});
}

}  // namespace tk
