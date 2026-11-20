// Elementwise torch-MPS kernels (unary / binary / copy-cast / fill), replayed bit-for-bit.
//
// Mirrors (torch 2.15, ref/pytorch-src):
//   aten/src/ATen/native/mps/OperationUtils.mm   MetalShaderLibrary::exec_unary_kernel (l.1004),
//                                                exec_unary_kernel_raw (l.1200), exec_binary_kernel (l.1245)
//   aten/src/ATen/native/mps/OperationUtils.h    exec_unary_kernel_with_params (l.710), bind_iter_tensors (l.534),
//                                                mtl_dispatch1DJob/2DJob (l.598/619), is_dense_broadcastable (l.868)
//   aten/src/ATen/native/mps/operations/Copy.mm  copy_kernel_mps / copy_cast_kernel_mps / contiguous_copy_kernel_mps
//   aten/src/ATen/native/mps/operations/ConstantOps.mm  fill_mps_kernel
//   aten/src/ATen/TensorIterator.cpp             build(): compute_shape, fast_set_up, compute_strides,
//                                                reorder_dimensions, allocate_or_resize_outputs, coalesce_dimensions
//
// ---------------------------------------------------------------------------------------------------------------
// TensorIterator conventions (what the kernels receive as "sizes"/"strides" payloads)
// ---------------------------------------------------------------------------------------------------------------
//  * Operand numbering: operand 0 is the OUTPUT, operands 1.. are the inputs in call order. bind_iter_tensors binds
//    operand i at buffer index i, at byte offset (data_ptr - storage base) == storage_offset*itemsize (== Buf.off here).
//    A CPU 0-dim operand (python number wrapped as a tensor) is bound with setBytes instead of setBuffer:
//    python float -> 4 bytes float (double is downcast), python int -> 8 bytes int64, bool -> 1 byte.
//  * iter.shape() is int64[ndim]; iter.strides(i) is int64[ndim] in BYTES (element stride * itemsize of operand i).
//    Both are uploaded raw (8*ndim bytes). Broadcast dims (operand size 1 vs iterator size != 1, or missing leading
//    dims) have stride 0.
//  * Dim order: index 0 is the FASTEST moving (innermost) dim. Without fast setup the dims start reversed
//    (perm = n-1..0) and are then insertion-sorted by ascending stride; operands are consulted in order 0,1,2..
//    (an output that the iterator itself allocates is skipped), a 0 stride in either dim defers to the next operand,
//    equal strides swap only if shape[dim0] > shape[dim1]; an ambiguous comparison keeps scanning further left.
//  * An output allocated by the iterator gets strides that are contiguous in the *iteration* order (so permuted
//    inputs give a permuted output; C-contiguous inputs give a C-contiguous output).
//  * Coalescing: adjacent dims d,d+1 merge when either has size 1, or when for EVERY operand
//    shape[d]*stride[d] == stride[d+1]. So a contiguous tensor against a broadcast one stays multi-dim.
//  * Fast setup (all operands have identical shape, no broadcasting, no 0-dim-vs-tensor mix) collapses to 1-D
//    shape {numel}, strides {itemsize(i)} when all operands are (a) C-contiguous, (b) channels-last contiguous (4-D),
//    or (c) non-overlapping-and-dense with identical element strides. In (b)/(c) memory order is iterated, so a
//    "dense" kernel is used on permuted tensors.
//  * iter.is_contiguous(): numel==1, or (ndim==1 and strides(i)[0]==itemsize(i) for every operand). A CPU scalar
//    operand has stride 0, so tensor-vs-python-scalar iterators are NOT contiguous (the dense_scalar kernels are
//    chosen from tensor-level properties before that matters).
//  * isInnerContiguous(): ndim>=2, !is_contiguous, strides(i)[0]==itemsize(i) for every operand; the
//    inner_contiguous kernels are used only when additionally shape[0] >= 16.
//  * Unary strided/inner_contiguous kernels get strides in the order (input, output); binary ones (out, in, other).
//  * Unary strided kernels are dispatched 2-D (shape[0], numel/shape[0]); binary strided kernels 1-D (numel).
//  * Not handled here: iterators that fail can_use_32bit_indexing (numel > INT32_MAX or any operand spanning more
//    than INT32_MAX bytes) - torch splits those into several dispatches; we abort.
#pragma once
#include "base.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <optional>
#include <string>

namespace tk {

// ---------------------------------------------------------------------------------------------------------------
// dtype helpers (c10/core/ScalarType.h)
// ---------------------------------------------------------------------------------------------------------------
inline bool dtIsFloat(DT d) { return d == F16 || d == F32 || d == F64 || d == BF16; }
inline bool dtIsComplex(DT d) { return d == C64; }
inline bool dtIsIntegral(DT d, bool includeBool) {
  return d == U8 || d == I8 || d == I16 || d == I32 || d == I64 || (includeBool && d == BOOL);
}
constexpr DT DT_UNDEF = (DT)0xff;

// c10::promoteTypes restricted to the DT set of base.h.
inline DT dtPromote(DT a, DT b) {
  if (a == DT_UNDEF) return b;
  if (b == DT_UNDEF) return a;
  if (a == b) return a;
  if (a == C64 || b == C64) return C64;
  const bool fa = dtIsFloat(a), fb = dtIsFloat(b);
  if (fa && fb) {
    if (a == F64 || b == F64) return F64;
    if (a == F32 || b == F32) return F32;
    return F32;  // half + bfloat16 -> float
  }
  if (fa) return a;
  if (fb) return b;
  if (a == BOOL) return b;
  if (b == BOOL) return a;
  if (a == U8 || b == U8) {  // uint8 with a signed type -> at least int16
    const DT o = a == U8 ? b : a;
    return o == I8 ? I16 : o;
  }
  return dtSize(a) >= dtSize(b) ? a : b;
}

// ---------------------------------------------------------------------------------------------------------------
// Scalars (c10::Scalar) and getMPSScalar (OperationUtils.mm:600)
// ---------------------------------------------------------------------------------------------------------------
struct Scal {
  bool isF = true;
  double d = 0;
  int64_t i = 0;
  Scal() = default;
  Scal(double v) : isF(true), d(v), i((int64_t)v) {}
  Scal(float v) : Scal((double)v) {}
  Scal(int64_t v) : isF(false), d((double)v), i(v) {}
  Scal(int v) : Scal((int64_t)v) {}
  Scal(bool v) : Scal((int64_t)v) {}
};

// c10::BFloat16(float): round-to-nearest-even on the float bit pattern.
inline uint16_t f32ToBf16Bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if (f != f) return 0x7FC0;
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

// getMPSScalar(scalar, type): returns payload size; bytes written to `out` (>= 8 bytes).
inline size_t scalBytes(const Scal& s, DT type, uint8_t* out) {
  switch (type) {
    case F64:
    case F32: { float v = s.isF ? (float)s.d : (float)s.i; memcpy(out, &v, 4); return 4; }
    case F16: { _Float16 v = (_Float16)(s.isF ? (float)s.d : (float)s.i); memcpy(out, &v, 2); return 2; }
    case BF16: { uint16_t v = f32ToBf16Bits(s.isF ? (float)s.d : (float)s.i); memcpy(out, &v, 2); return 2; }
    case C64: { float v[2] = {s.isF ? (float)s.d : (float)s.i, 0.f}; memcpy(out, v, 8); return 8; }
    case I64: { int64_t v = s.isF ? (int64_t)s.d : s.i; memcpy(out, &v, 8); return 8; }
    case I32: { int32_t v = (int32_t)(s.isF ? (int64_t)s.d : s.i); memcpy(out, &v, 4); return 4; }
    case I16: { int16_t v = (int16_t)(s.isF ? (int64_t)s.d : s.i); memcpy(out, &v, 2); return 2; }
    case I8: { int8_t v = (int8_t)(s.isF ? (int64_t)s.d : s.i); memcpy(out, &v, 1); return 1; }
    case U8: { uint8_t v = (uint8_t)(s.isF ? (int64_t)s.d : s.i); memcpy(out, &v, 1); return 1; }
    case BOOL: { uint8_t v = s.isF ? (s.d != 0) : (s.i != 0); memcpy(out, &v, 1); return 1; }
    default: fprintf(stderr, "tk::scalBytes: unsupported dtype %u\n", (unsigned)type); abort();
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Operand description
// ---------------------------------------------------------------------------------------------------------------
struct Opnd {
  Buf b;                         // buffer + byte offset of element [0,0,..] (storage_offset * itemsize)
  DT dt = F32;
  std::vector<int64_t> shape;    // tensor sizes (outermost first, as in python)
  std::vector<int64_t> strides;  // tensor strides in ELEMENTS; empty => C-contiguous
  bool undef = false;            // output only: let the iterator choose the layout; shape/strides are written back
                                 // (the buffer must hold numel*itemsize bytes; layout is contiguous in iteration order)
  bool cpu = false;              // CPU 0-dim scalar (python number): bound with setBytes
  bool wrapped = false;          // python number (does not participate in dtype promotion across categories)
  Scal sv;                       // value when cpu

  static Opnd tensor(Buf b, DT dt, std::vector<int64_t> shape, std::vector<int64_t> strides = {}) {
    Opnd o; o.b = b; o.dt = dt; o.shape = std::move(shape); o.strides = std::move(strides); o.normalize(); return o;
  }
  // Output to be allocated by the iterator (dtype decided by the caller, see resultType()).
  static Opnd output(Buf b, DT dt) { Opnd o; o.b = b; o.dt = dt; o.undef = true; return o; }
  // Python float: wrapped double tensor; every MPS path downcasts it to float32 first (convert_double_scalar /
  // mtl_setBuffer), and the wrapped-number result type is the default dtype float32, so it is modelled as F32.
  static Opnd pyFloat(double v) { Opnd o; o.dt = F32; o.cpu = o.wrapped = true; o.sv = Scal(v); return o; }
  static Opnd pyInt(int64_t v) { Opnd o; o.dt = I64; o.cpu = o.wrapped = true; o.sv = Scal(v); return o; }
  static Opnd pyBool(bool v) { Opnd o; o.dt = BOOL; o.cpu = o.wrapped = true; o.sv = Scal(v); return o; }

  void normalize() {
    if (strides.empty() && !shape.empty()) {
      strides.resize(shape.size());
      int64_t z = 1;
      for (int d = (int)shape.size() - 1; d >= 0; d--) { strides[d] = z; z *= shape[d]; }
    }
  }
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  // TensorImpl::compute_contiguous
  bool contiguous() const {
    int64_t z = 1;
    for (int d = (int)shape.size() - 1; d >= 0; d--) {
      if (shape[d] != 1) { if (strides[d] != z) return false; z *= shape[d]; }
    }
    return true;
  }
  // compute_channels_last_contiguous_2d (4-D only)
  bool channelsLastContiguous() const {
    if (shape.size() != 4) return false;
    int64_t z = 1;
    for (int d : {1, 3, 2, 0}) {
      if (shape[d] != 1) { if (strides[d] != z) return false; z *= shape[d]; }
    }
    return true;
  }
  // compute_non_overlapping_and_dense
  bool nonOverlappingAndDense() const {
    const int n = (int)shape.size();
    if (n == 1) return shape[0] < 2 || strides[0] == 1;
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
      if (shape[a] < 2) return false;
      if (shape[b] < 2) return true;
      return strides[a] < strides[b];
    });
    int64_t req = 1;
    for (int i = 0; i < n; i++) {
      if (shape[perm[i]] < 2) return true;
      if (strides[perm[i]] != req) return false;
      req *= shape[perm[i]];
    }
    return true;
  }
};

// at::result_type over input operands (ResultTypeState: dim / zero-dim / wrapped categories + combine_categories).
inline DT resultType(std::initializer_list<const Opnd*> ins) {
  DT dimR = DT_UNDEF, zeroR = DT_UNDEF, wrapR = DT_UNDEF;
  for (const Opnd* o : ins) {
    if (o->wrapped) wrapR = dtPromote(wrapR, o->dt);
    else if (!o->shape.empty()) dimR = dtPromote(dimR, o->dt);
    else zeroR = dtPromote(zeroR, o->dt);
  }
  auto combine = [](DT higher, DT lower) {
    if (higher != DT_UNDEF && dtIsComplex(higher)) return higher;
    if (higher != DT_UNDEF && dtIsFloat(higher)) return (lower != DT_UNDEF && dtIsComplex(lower)) ? C64 : higher;
    if (higher == BOOL || (lower != DT_UNDEF && (dtIsFloat(lower) || dtIsComplex(lower)))) return dtPromote(higher, lower);
    if (higher != DT_UNDEF) return higher;
    return lower;
  };
  return combine(dimR, combine(zeroR, wrapR));
}

// ---------------------------------------------------------------------------------------------------------------
// TensorIterator view
// ---------------------------------------------------------------------------------------------------------------
struct Iter {
  std::vector<int64_t> shape;                 // iter.shape(), [0] = innermost
  std::vector<std::vector<int64_t>> strides;  // iter.strides(i) in bytes
  std::vector<int64_t> esize;                 // itemsize per operand
  int ndim() const { return (int)shape.size(); }
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  bool firstDimContiguous() const {  // has_contiguous_first_dim
    if (ndim() == 0) return true;
    for (size_t i = 0; i < strides.size(); i++) if (strides[i][0] != esize[i]) return false;
    return true;
  }
  bool contiguous() const {  // TensorIteratorBase::is_contiguous
    if (numel() == 1) return true;
    if (ndim() != 1) return false;
    return firstDimContiguous();
  }
  bool innerContiguous() const { return ndim() >= 2 && !contiguous() && firstDimContiguous(); }  // isInnerContiguous
};

// TensorIteratorBase::build for ops = {output, input0, input1, ...}. `resize_outputs` mirrors the config flag
// (true for unary/binary/comparison ops, false for copy): when true a defined output does not take part in shape
// inference (it must already have the broadcast shape; torch would resize it otherwise).
// If ops[0]->undef, its shape/strides are filled in exactly as allocate_or_resize_outputs would lay it out.
inline Iter buildIter(const std::vector<Opnd*>& ops, bool resize_outputs = true) {
  const int nt = (int)ops.size();
  Iter it;
  it.esize.resize(nt);
  for (int i = 0; i < nt; i++) it.esize[i] = dtSize(ops[i]->dt);
  for (int i = 0; i < nt; i++) if (!ops[i]->undef) ops[i]->normalize();

  // compute_shape
  std::vector<int64_t> shape;
  bool all_same = true, has_scalars = false, has_tensors = false, first = true;
  for (int i = 0; i < nt; i++) {
    const Opnd& o = *ops[i];
    if (o.undef) continue;
    if (i == 0 && resize_outputs) continue;
    if (o.shape.empty()) has_scalars = true; else has_tensors = true;
    if (has_scalars && has_tensors) all_same = false;
    if (first || shape.empty()) {
      if (!first && !o.shape.empty()) all_same = all_same && false;
      if (first || !o.shape.empty()) shape = o.shape;
      first = false;
    } else if (o.shape != shape) {
      all_same = false;
      const size_t n = std::max(shape.size(), o.shape.size());
      std::vector<int64_t> r(n);
      for (size_t k = 0; k < n; k++) {
        const int64_t a = k < n - shape.size() ? 1 : shape[k - (n - shape.size())];
        const int64_t b = k < n - o.shape.size() ? 1 : o.shape[k - (n - o.shape.size())];
        if (a != b && a != 1 && b != 1) { fprintf(stderr, "tk::buildIter: shapes not broadcastable\n"); abort(); }
        r[k] = a == 1 ? b : a;
      }
      shape = r;
    }
  }
  if (!ops[0]->undef && resize_outputs && ops[0]->shape != shape) {
    fprintf(stderr, "tk::buildIter: defined output does not have the broadcast shape\n");
    abort();
  }
  const int nd = (int)shape.size();
  int64_t numel = 1;
  for (auto s : shape) numel *= s;

  // fast_set_up / compute_fast_setup_type
  if (all_same) {
    bool contig = true, cl = true, nod = true;
    for (Opnd* o : ops) {
      if (o->undef) continue;
      contig = contig && o->contiguous();
      cl = cl && o->channelsLastContiguous();
      nod = nod && o->nonOverlappingAndDense();
    }
    int type = 0;  // 1 contiguous, 2 channels_last, 3 non_overlapping_dense
    const Opnd* last_defined = nullptr;
    for (int i = nt - 1; i >= 0; i--) if (!ops[i]->undef) { last_defined = ops[i]; break; }
    if (contig) type = 1;
    else if (cl) type = 2;
    else if (nod) {
      type = 3;
      for (Opnd* o : ops) if (!o->undef && o->strides != last_defined->strides) { type = 0; break; }
    }
    if (type) {
      if (ops[0]->undef) {
        Opnd& out = *ops[0];
        out.shape = shape;
        out.strides.clear();
        if (type == 1) out.normalize();
        else if (type == 2) { const int64_t C = shape[1], H = shape[2], W = shape[3]; out.strides = {H * W * C, 1, W * C, C}; }
        else out.strides = last_defined->strides;
        out.undef = false;
      }
      if (nd >= 1) it.shape = {numel};
      it.strides.resize(nt);
      for (int i = 0; i < nt; i++) if (nd > 0) it.strides[i] = {it.esize[i]};
      return it;
    }
  }

  // compute_strides (bytes, tensor dim order, right-aligned)
  std::vector<std::vector<int64_t>> sb(nt);
  for (int i = 0; i < nt; i++) {
    const Opnd& o = *ops[i];
    if (o.undef) continue;
    sb[i].assign(nd, 0);
    const int off = nd - (int)o.shape.size();
    for (int k = 0; k < (int)o.shape.size(); k++)
      sb[i][off + k] = (o.shape[k] == 1 && shape[off + k] != 1) ? 0 : o.strides[k] * it.esize[i];
  }

  // reorder_dimensions
  std::vector<int> perm(nd);
  if (nd == 1) {
    perm[0] = 0;
  } else if (nd > 1) {
    for (int k = 0; k < nd; k++) perm[k] = nd - 1 - k;
    auto should_swap = [&](int dim0, int dim1) {
      for (int a = 0; a < nt; a++) {
        if (sb[a].empty()) continue;
        const int64_t s0 = sb[a][dim0], s1 = sb[a][dim1];
        if (s0 == 0 || s1 == 0) continue;
        if (s0 < s1) return -1;
        if (s0 > s1) return 1;
        if (shape[dim0] > shape[dim1]) return 1;
      }
      return 0;
    };
    for (int i = 1; i < nd; i++) {
      int dim1 = i;
      for (int dim0 = i - 1; dim0 >= 0; dim0--) {
        const int c = should_swap(perm[dim0], perm[dim1]);
        if (c > 0) { std::swap(perm[dim0], perm[dim1]); dim1 = dim0; }
        else if (c < 0) break;
      }
    }
  }
  it.shape.resize(nd);
  for (int k = 0; k < nd; k++) it.shape[k] = shape[perm[k]];
  it.strides.resize(nt);
  for (int i = 0; i < nt; i++) {
    if (sb[i].empty()) continue;
    it.strides[i].resize(nd);
    for (int k = 0; k < nd; k++) it.strides[i][k] = sb[i][perm[k]];
  }

  // allocate_or_resize_outputs: compatible_stride in iteration order, written back through invert_perm
  if (ops[0]->undef) {
    Opnd& out = *ops[0];
    it.strides[0].resize(nd);
    int64_t next = it.esize[0];
    for (int k = 0; k < nd; k++) { it.strides[0][k] = next; next *= it.shape[k]; }
    out.shape = shape;
    out.strides.assign(nd, 0);
    for (int k = 0; k < nd; k++) out.strides[perm[k]] = it.strides[0][k] / it.esize[0];
    out.undef = false;
  }

  // coalesce_dimensions
  if (nd > 1) {
    auto can_coalesce = [&](int d0, int d1) {
      if (it.shape[d0] == 1 || it.shape[d1] == 1) return true;
      for (int i = 0; i < nt; i++) if (it.shape[d0] * it.strides[i][d0] != it.strides[i][d1]) return false;
      return true;
    };
    auto replace_stride = [&](int d0, int d1) { for (int i = 0; i < nt; i++) it.strides[i][d0] = it.strides[i][d1]; };
    int prev = 0;
    for (int d = 1; d < nd; d++) {
      if (can_coalesce(prev, d)) {
        if (it.shape[prev] == 1) replace_stride(prev, d);
        it.shape[prev] *= it.shape[d];
      } else {
        prev++;
        if (prev != d) { replace_stride(prev, d); it.shape[prev] = it.shape[d]; }
      }
    }
    it.shape.resize(prev + 1);
    for (int i = 0; i < nt; i++) it.strides[i].resize(prev + 1);
  }
  return it;
}

// can_use_32bit_indexing: torch splits iterators that fail this (TensorIterator::with_32bit_indexing) and runs the
// kernel once per piece; elementwise results do not depend on where the split falls.
inline bool iterFits32(const Iter& it) {
  bool ok = it.numel() <= INT32_MAX;
  for (size_t i = 0; ok && i < it.strides.size(); i++) {
    int64_t ext = 0;
    for (int d = 0; d < it.ndim(); d++) ext += (it.shape[d] - 1) * it.strides[i][d];
    ok = ext <= INT32_MAX;
  }
  return ok;
}
inline void iterCheck32(const Iter& it) {
  if (!iterFits32(it)) { fprintf(stderr, "tk elem: iterator needs 64-bit splitting (with_32bit_indexing), not supported\n"); abort(); }
}
// If the iterator over ops = {out (defined), inputs...} needs splitting, halves the widest output dim, calls
// fn(sliced operands) on each half (which may split again) and returns true.
template <class F>
inline bool iterSplit32(const Iter& it, const std::vector<const Opnd*>& ops, F&& fn) {
  if (iterFits32(it)) return false;
  const Opnd& out = *ops[0];
  const int nd = (int)out.shape.size();
  int sd = -1; int64_t best = 0;
  for (int d = 0; d < nd; d++) { const int64_t w = out.shape[d] * std::abs(out.strides[d]); if (out.shape[d] > 1 && w > best) { best = w; sd = d; } }
  if (sd < 0) iterCheck32(it);
  const int64_t n = out.shape[sd], half = n / 2;
  for (int h = 0; h < 2; h++) {
    const int64_t s = h ? half : 0, len = h ? n - half : half;
    std::vector<Opnd> sub;
    for (const Opnd* o : ops) {
      Opnd c = *o;
      const int d = sd - (nd - (int)c.shape.size());
      if (!c.cpu && d >= 0 && c.shape[d] != 1) { c.normalize(); c.b.off += (uint64_t)(s * c.strides[d]) * dtSize(c.dt); c.shape[d] = len; }
      sub.push_back(std::move(c));
    }
    fn(sub);
  }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Dispatch + binding helpers
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kILP = 4;                // c10::metal::ILP_PER_THREAD
constexpr int64_t kInnerContigMinExtent = 16;  // INNER_CONTIGUOUS_MIN_EXTENT

// mtl_dispatch1DJob (OperationUtils.h:598)
inline void dispatch1D(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, NSUInteger length) {
  const NSUInteger maxTG = [p maxTotalThreadsPerThreadgroup];
  [e dispatchThreads:MTLSizeMake(length, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(std::clamp<NSUInteger>(length, 1, maxTG), 1, 1)];
  ctx().ndispatch++;
}
// mtl_dispatch2DJob (OperationUtils.h:619)
inline void dispatch2D(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, NSUInteger inner, NSUInteger outer) {
  const NSUInteger maxTG = [p maxTotalThreadsPerThreadgroup];
  const NSUInteger tg_x = std::min(maxTG, inner);
  const NSUInteger tg_y = std::clamp<NSUInteger>(outer, 1, maxTG / tg_x);
  [e dispatchThreads:MTLSizeMake(inner, outer, 1) threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
  ctx().ndispatch++;
}
inline void setI64s(id<MTLComputeCommandEncoder> e, const std::vector<int64_t>& v, size_t from, unsigned idx) {
  setRaw(e, v.data() + from, (v.size() - from) * sizeof(int64_t), idx);
}
// One operand of bind_iter_tensors / mtl_setBuffer (OperationUtils.h:480,534). `as` = dtype a CPU scalar is uploaded as.
inline void bindOpnd(id<MTLComputeCommandEncoder> e, const Opnd& o, unsigned idx, std::optional<DT> as = std::nullopt) {
  if (o.cpu) {
    uint8_t tmp[16];
    const size_t n = scalBytes(o.sv, as.value_or(o.dt), tmp);
    setRaw(e, tmp, n, idx);
  } else {
    setBuf(e, o.b, idx);
  }
}
// bindInnerContiguousOuter (OperationUtils.mm:992): shape[1:] at base, then strides(arg)[1:]; returns next index.
inline unsigned bindInnerOuter(id<MTLComputeCommandEncoder> e, const Iter& it, unsigned base, std::initializer_list<int> order) {
  setI64s(e, it.shape, 1, base);
  unsigned idx = base + 1;
  for (int a : order) setI64s(e, it.strides[a], 1, idx++);
  return idx;
}

// ---------------------------------------------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------------------------------------------
// MetalShaderLibrary::exec_unary_kernel (OperationUtils.mm:1004) on an already built iterator {out, in}.
// alpha_type must be given when alpha is (torch: scalar_arg_type or iter.common_dtype()). Returns the kernel name.
inline std::string exec_unary(const Iter& it, const Opnd& out, const Opnd& in, const std::string& name,
                              std::optional<Scal> alpha = std::nullopt, std::optional<DT> alpha_type = std::nullopt,
                              std::optional<uint32_t> ilp_threshold = std::nullopt) {
  iterCheck32(it);
  const uint32_t length = (uint32_t)it.numel();
  if (length == 0) return "";
  const bool is_contig = it.contiguous();
  const bool ilp_eligible = dtIsFloat(out.dt) || ilp_threshold.has_value();
  bool dense_ilp = is_contig && !alpha && ilp_eligible && length >= ilp_threshold.value_or(1u << 18);
  const DT atype = alpha_type.value_or(in.dt);
  const bool inner_contig = !is_contig && it.shape[0] >= kInnerContigMinExtent && it.innerContiguous();
  const bool byte_copy = inner_contig && name == "copy_identity" && out.dt == in.dt;
  const char* suffix = inner_contig ? "inner_contiguous" : !is_contig ? "strided" : (dense_ilp || alpha) ? "dense" : "dense_scalar";
  std::string kname = name + "_" + suffix + "_" + dtName(out.dt) + "_" + dtName(in.dt) + (alpha ? std::string("_") + dtName(atype) : "");
  bool cast_needed = false, cast_ilp = false;
  if (!alpha && !hasFn(kname)) {
    cast_needed = true;
    cast_ilp = is_contig && ilp_threshold.has_value() && length >= *ilp_threshold;
    suffix = cast_ilp ? "dense_castout_ilp" : inner_contig ? "inner_contiguous_castout" : is_contig ? "dense_castout" : "strided_castout";
    dense_ilp = false;
    kname = name + "_" + suffix + "_" + dtName(in.dt);
  }
  if (byte_copy) kname = "inner_contiguous_copy";

  auto e = enc();
  auto p = pso(kname);
  [e setComputePipelineState:p];
  bindOpnd(e, out, 0);
  bindOpnd(e, in, 1);
  auto strided2d = [&] { const NSUInteger inner = (NSUInteger)it.shape[0]; dispatch2D(e, p, inner, (NSUInteger)length / inner); };
  uint8_t ab[16];
  if (cast_needed) {
    const uint32_t out_type = (uint32_t)out.dt, esz = dtSize(out.dt);
    if (cast_ilp) {
      const uint32_t v[4] = {esz, out_type, length, 0};  // c10::metal::vec3<uint32_t> is alignas(16): 16 bytes
      setRaw(e, v, 16, 2);
      dispatch1D(e, p, ((NSUInteger)length + kILP - 1) / kILP);
    } else if (is_contig) {
      const uint32_t v[2] = {esz, out_type};
      setRaw(e, v, 8, 2);
      dispatch1D(e, p, length);
    } else if (inner_contig) {
      const uint32_t inner = (uint32_t)it.shape[0];
      const uint32_t v[4] = {(uint32_t)(it.ndim() - 1), inner, esz, out_type};
      setRaw(e, v, 16, bindInnerOuter(e, it, 2, {1, 0}));
      dispatch2D(e, p, ((NSUInteger)inner + kILP - 1) / kILP, (NSUInteger)length / inner);
    } else {
      const uint32_t v[2] = {(uint32_t)it.ndim(), out_type};
      setI64s(e, it.shape, 0, 2);
      setI64s(e, it.strides[1], 0, 3);
      setI64s(e, it.strides[0], 0, 4);
      setRaw(e, v, 8, 5);
      strided2d();
    }
  } else if (dense_ilp) {
    setVal(e, length, 2);
    dispatch1D(e, p, ((NSUInteger)length + kILP - 1) / kILP);
  } else if (inner_contig) {
    const uint32_t inner = (uint32_t)it.shape[0];
    const NSUInteger outer = (NSUInteger)length / inner;
    if (byte_copy) {
      const uint32_t inner_bytes = inner * (uint32_t)it.esize[0];
      const uint32_t v[2] = {(uint32_t)(it.ndim() - 1), inner_bytes};
      setRaw(e, v, 8, bindInnerOuter(e, it, 2, {1, 0}));
      dispatch2D(e, p, ((NSUInteger)inner_bytes + 15) / 16, outer);
    } else {
      const uint32_t v[2] = {(uint32_t)(it.ndim() - 1), inner};
      const unsigned pidx = bindInnerOuter(e, it, 2, {1, 0});
      setRaw(e, v, 8, pidx);
      if (alpha) setRaw(e, ab, scalBytes(*alpha, atype, ab), pidx + 1);
      dispatch2D(e, p, ((NSUInteger)inner + kILP - 1) / kILP, outer);
    }
  } else {
    if (!is_contig) {
      setI64s(e, it.shape, 0, 2);
      setI64s(e, it.strides[1], 0, 3);
      setI64s(e, it.strides[0], 0, 4);
      setVal(e, (uint32_t)it.ndim(), 5);
    }
    if (alpha) setRaw(e, ab, scalBytes(*alpha, atype, ab), is_contig ? 2 : 6);
    if (!is_contig) strided2d(); else dispatch1D(e, p, length);
  }
  return kname;
}

// TensorIterator::unary_op / unary_float_op + exec_unary_kernel. `out` may be Opnd::output(buf, dt) (layout is
// written back) or a defined tensor (in-place: pass the same description as `in`).
inline std::string unary_iter(const std::string& name, Opnd& out, Opnd in, std::optional<Scal> alpha = std::nullopt,
                              std::optional<DT> alpha_type = std::nullopt, std::optional<uint32_t> ilp_threshold = std::nullopt) {
  Iter it = buildIter({&out, &in});
  std::string kn;
  if (iterSplit32(it, {&out, &in}, [&](std::vector<Opnd>& o) { kn = unary_iter(name, o[0], o[1], alpha, alpha_type, ilp_threshold); })) return kn;
  return exec_unary(it, out, in, name, alpha, alpha_type, ilp_threshold);
}

// MetalShaderLibrary::exec_unary_kernel_with_params (OperationUtils.h:710): params struct of n bytes.
inline std::string unary_params_iter(const std::string& name, Opnd& out, Opnd in, const void* params, size_t n,
                                     const std::string& params_type_name, std::optional<uint32_t> ilp_threshold = std::nullopt) {
  Iter it = buildIter({&out, &in});
  std::string kn;
  if (iterSplit32(it, {&out, &in}, [&](std::vector<Opnd>& o) { kn = unary_params_iter(name, o[0], o[1], params, n, params_type_name, ilp_threshold); })) return kn;
  const uint32_t length = (uint32_t)it.numel();
  if (length == 0) return "";
  const bool is_contig = it.contiguous();
  const bool dense_ilp = is_contig && ilp_threshold.has_value() && length >= *ilp_threshold;
  const std::string kname = name + "_" + (dense_ilp ? "dense_ilp" : is_contig ? "dense" : "strided") + "_" + dtName(out.dt) +
                            "_" + dtName(in.dt) + "_" + params_type_name;
  auto e = enc();
  auto p = pso(kname);
  [e setComputePipelineState:p];
  bindOpnd(e, out, 0);
  bindOpnd(e, in, 1);
  if (!is_contig) {
    setI64s(e, it.shape, 0, 2);
    setI64s(e, it.strides[1], 0, 3);
    setI64s(e, it.strides[0], 0, 4);
    setVal(e, (uint32_t)it.ndim(), 5);
  }
  setRaw(e, params, n, is_contig ? 2 : 6);
  if (!is_contig) {
    const NSUInteger inner = (NSUInteger)it.shape[0];
    dispatch2D(e, p, inner, (NSUInteger)length / inner);
  } else if (dense_ilp) {
    setVal(e, length, 3);
    dispatch1D(e, p, ((NSUInteger)length + kILP - 1) / kILP);
  } else {
    dispatch1D(e, p, length);
  }
  return kname;
}

// Low-level: contiguous unary on raw buffers (same selection as exec_unary_kernel with a 1-D contiguous iterator).
inline std::string unary_dense(const std::string& name, DT odt, DT idt, Buf out, Buf in, uint32_t n,
                               std::optional<Scal> alpha = std::nullopt, std::optional<DT> alpha_type = std::nullopt,
                               std::optional<uint32_t> ilp_threshold = std::nullopt) {
  Opnd o = Opnd::tensor(out, odt, {(int64_t)n}), i = Opnd::tensor(in, idt, {(int64_t)n});
  return unary_iter(name, o, i, alpha, alpha_type, ilp_threshold);
}

// MetalShaderLibrary::exec_unary_kernel_raw (OperationUtils.mm:1200): `<name>_dense_castout[_ilp]_<src>`.
inline std::string unary_castout_raw(const std::string& name, Buf dst, DT ddt, Buf src, DT sdt, uint32_t numel,
                                     std::optional<uint32_t> ilp_threshold = std::nullopt) {
  if (numel == 0) return "";
  const bool use_ilp = ilp_threshold.has_value() && numel >= *ilp_threshold;
  const std::string kname = name + (use_ilp ? "_dense_castout_ilp_" : "_dense_castout_") + dtName(sdt);
  auto e = enc();
  auto p = pso(kname);
  [e setComputePipelineState:p];
  setBuf(e, dst, 0);
  setBuf(e, src, 1);
  if (use_ilp) {
    const uint32_t v[4] = {dtSize(ddt), (uint32_t)ddt, numel, 0};
    setRaw(e, v, 16, 2);
    dispatch1D(e, p, ((NSUInteger)numel + kILP - 1) / kILP);
  } else {
    const uint32_t v[2] = {dtSize(ddt), (uint32_t)ddt};
    setRaw(e, v, 8, 2);
    dispatch1D(e, p, numel);
  }
  return kname;
}

// ---------------------------------------------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------------------------------------------
// is_dense_broadcastable (OperationUtils.h:868)
inline bool isDenseBroadcastable(const Opnd& from, const Opnd& into) {
  if (!from.contiguous() || !into.contiguous()) return false;
  bool squeezable = false;
  const int fn = (int)from.shape.size(), in = (int)into.shape.size();
  for (int d = 0; d < fn; d++) {
    const int64_t fs = from.shape[fn - 1 - d];
    if (squeezable) { if (fs == 1) continue; return false; }
    squeezable = fs != into.shape[in - 1 - d];
  }
  return true;
}

// TensorIterator::binary_op / comparison_op + MetalShaderLibrary::exec_binary_kernel (OperationUtils.mm:1245).
//   out: Opnd::output(buf, dt) or a defined tensor (in-place ops: same description as `a`).
//   natural: kernel-natural output dtype (comparisons pass BOOL); ilp_threshold: comparisons pass 1<<18.
// Returns the kernel name.
inline std::string binary_iter(const std::string& name, Opnd& out, Opnd a, Opnd b, std::optional<Scal> alpha = std::nullopt,
                               std::optional<DT> alpha_type = std::nullopt, std::optional<DT> natural_out = std::nullopt,
                               std::optional<uint32_t> ilp_threshold = std::nullopt) {
  Iter it = buildIter({&out, &a, &b});
  if (it.numel() == 0) return "";
  std::string kn;
  if (iterSplit32(it, {&out, &a, &b}, [&](std::vector<Opnd>& o) { kn = binary_iter(name, o[0], o[1], o[2], alpha, alpha_type, natural_out, ilp_threshold); })) return kn;
  const DT common = resultType({&a, &b});
  const DT natural = natural_out.value_or(out.dt);
  const bool output_cast_needed = out.dt != natural;

  // `a`/`b` below play torch's `input`/`other` (already convert_double_scalar'ed: python floats are F32).
  bool cast_needed = a.dt != b.dt;
  bool use_broadcast = false, use_scalar = false, broadcast_on_lhs = false, scalar_on_lhs = false;
  int64_t broadcast_numel = 0;
  const bool a_full = a.numel() == out.numel(), b_full = b.numel() == out.numel();
  const bool a_bc = isDenseBroadcastable(a, out) && !a_full;
  const bool b_bc = isDenseBroadcastable(b, out) && !b_full;
  if (!output_cast_needed) {
    if (a_bc && b_full && b.contiguous() && out.contiguous()) {
      broadcast_numel = a.numel();
      if (broadcast_numel == 1) { use_scalar = true; scalar_on_lhs = true; }
      else { use_broadcast = true; broadcast_on_lhs = true; }
    } else if (b_bc && a_full && a.contiguous() && out.contiguous()) {
      broadcast_numel = b.numel();
      if (broadcast_numel == 1) { use_scalar = true; scalar_on_lhs = false; }
      else { use_broadcast = true; broadcast_on_lhs = false; }
    }
  }
  // Safe host promote of an integral scalar to a float32-backed common dtype.
  DT a_dt = a.dt, b_dt = b.dt;  // dtypes as seen by the kernel (after host promotion)
  if (use_scalar && cast_needed) {
    const DT tensor_dt = scalar_on_lhs ? b.dt : a.dt, scalar_dt = scalar_on_lhs ? a.dt : b.dt;
    const DT common_real = common == C64 ? F32 : common;
    if (tensor_dt == common && dtIsIntegral(scalar_dt, true) && common_real == F32) {
      const Opnd& s = scalar_on_lhs ? a : b;
      if (!s.cpu) {
        // torch does `scalar.to(common)` on the MPS 0-dim tensor (an extra copy_identity_dense_castout dispatch
        // into a fresh buffer). The caller has to replay that copy and pass the promoted operand.
        fprintf(stderr, "tk::binary_iter(%s): host promotion of an MPS scalar tensor must be done by the caller\n", name.c_str());
        abort();
      }
      (scalar_on_lhs ? a_dt : b_dt) = common;
      cast_needed = false;
    }
  }
  const DT atype = alpha_type.value_or(cast_needed ? out.dt : common);
  const std::string alpha_suffix = alpha ? std::string("_") + dtName(atype) : "";
  const bool is_contig = it.contiguous();
  const bool ilp_applicable = !use_scalar && !use_broadcast && is_contig && !cast_needed && !alpha && !output_cast_needed;
  const uint32_t threshold = ilp_threshold.value_or(dtIsFloat(out.dt) ? (1u << 18) : UINT32_MAX);
  const bool dense_ilp = ilp_applicable && (uint32_t)it.numel() >= threshold;
  const bool ic_applicable = !use_scalar && !use_broadcast && !cast_needed && !alpha && !output_cast_needed && !is_contig && it.innerContiguous();
  const bool inner_contig = ic_applicable && it.shape[0] >= kInnerContigMinExtent;

  const std::string o_s = dtName(out.dt);
  const std::string cast_suffix = o_s + "_" + dtName(out.dt == BOOL ? common : out.dt);
  std::string kname;
  if (output_cast_needed) {
    kname = name + "_strided_castout_" + dtName(natural) + "_" + dtName(common) + alpha_suffix;
  } else if (use_scalar) {
    const DT tensor_dt = scalar_on_lhs ? b_dt : a_dt;
    const std::string lhs = scalar_on_lhs ? "_lhs" : "";
    kname = cast_needed ? name + "_dense_scalar" + lhs + "_cast_" + cast_suffix + alpha_suffix
                        : name + "_dense_scalar" + lhs + "_" + o_s + "_" + dtName(tensor_dt) + alpha_suffix;
  } else if (use_broadcast) {
    const DT tensor_dt = broadcast_on_lhs ? b_dt : a_dt;
    const std::string rhs = broadcast_on_lhs ? "_rhs" : "";
    kname = cast_needed ? name + "_dense_broadcast" + rhs + "_cast_" + cast_suffix + alpha_suffix
                        : name + "_dense_broadcast" + rhs + "_" + o_s + "_" + dtName(tensor_dt) + alpha_suffix;
  } else {
    const std::string suffix = is_contig ? (dense_ilp ? "dense_ilp4" : "dense") : (inner_contig ? "inner_contiguous" : "strided");
    kname = cast_needed ? name + "_" + suffix + "_cast_" + cast_suffix + alpha_suffix
                        : name + "_" + suffix + "_" + o_s + "_" + dtName(a_dt) + alpha_suffix;
  }

  auto e = enc();
  auto p = pso(kname);
  [e setComputePipelineState:p];
  // bind_iter_tensors; the scalar kernel re-binds its (possibly host-promoted) scalar at the same index.
  bindOpnd(e, out, 0);
  bindOpnd(e, a, 1, a_dt);
  bindOpnd(e, b, 2, b_dt);
  uint8_t ab[16];
  size_t an = 0;
  if (alpha) an = scalBytes(*alpha, atype, ab);
  const int nd_types[4] = {it.ndim(), (int)a_dt, (int)b_dt, (int)out.dt};
  const uint32_t sizes_types[4] = {dtSize(a_dt), dtSize(b_dt), (uint32_t)a_dt, (uint32_t)b_dt};
  auto set_strided = [&](unsigned base) {
    setI64s(e, it.shape, 0, base);
    setI64s(e, it.strides[0], 0, base + 1);
    setI64s(e, it.strides[1], 0, base + 2);
    setI64s(e, it.strides[2], 0, base + 3);
    setRaw(e, nd_types, 16, base + 4);
  };
  if (output_cast_needed) {
    set_strided(3);
  } else if (use_scalar) {
    if (cast_needed) {
      if (alpha) { setRaw(e, ab, an, 3); setRaw(e, sizes_types, 16, 4); }
      else setRaw(e, sizes_types, 16, 3);
    } else if (alpha) {
      setRaw(e, ab, an, 3);
    }
  } else if (use_broadcast) {
    setVal(e, broadcast_numel, 3);  // int64
    if (cast_needed) {
      if (alpha) { setRaw(e, ab, an, 4); setRaw(e, sizes_types, 16, 5); }
      else setRaw(e, sizes_types, 16, 4);
    } else if (alpha) {
      setRaw(e, ab, an, 4);
    }
  } else if (is_contig) {
    if (dense_ilp) setVal(e, (uint32_t)it.numel(), 3);
    else if (alpha) setRaw(e, ab, an, 3);
    if (cast_needed) setRaw(e, sizes_types, 16, alpha ? 4 : 3);
  } else if (inner_contig) {
    const uint32_t v[2] = {(uint32_t)(it.ndim() - 1), (uint32_t)it.shape[0]};
    setRaw(e, v, 8, bindInnerOuter(e, it, 3, {0, 1, 2}));
  } else {
    if (alpha) { setRaw(e, ab, an, 3); set_strided(4); }
    else set_strided(3);
  }
  if (inner_contig) {
    const uint32_t inner = (uint32_t)it.shape[0];
    dispatch2D(e, p, ((NSUInteger)inner + kILP - 1) / kILP, (NSUInteger)it.numel() / inner);
  } else {
    dispatch1D(e, p, dense_ilp ? ((NSUInteger)it.numel() + kILP - 1) / kILP : (NSUInteger)it.numel());
  }
  return kname;
}

// Low-level: same-shape contiguous binary on raw buffers (dense / dense_ilp4 / dense_cast selection).
inline std::string binary_dense(const std::string& name, DT odt, DT adt, DT bdt, Buf out, Buf a, Buf b, uint32_t n,
                                std::optional<DT> natural_out = std::nullopt, std::optional<uint32_t> ilp_threshold = std::nullopt) {
  Opnd o = Opnd::tensor(out, odt, {(int64_t)n});
  return binary_iter(name, o, Opnd::tensor(a, adt, {(int64_t)n}), Opnd::tensor(b, bdt, {(int64_t)n}), std::nullopt, std::nullopt,
                     natural_out, ilp_threshold);
}

// ---------------------------------------------------------------------------------------------------------------
// Copy / cast (Copy.mm)
// ---------------------------------------------------------------------------------------------------------------
// contiguous_copy_kernel_mps (Copy.mm:70): byte-erased copy, one dispatch per <= 2GB chunk,
// MetalKernelFunction::dispatch(length) => tg = min(length, maxTG).
inline void contiguous_byte_copy(Buf dst, Buf src, uint64_t nbytes) {
  auto e = enc();
  auto p = pso("contiguous_byte_copy");
  [e setComputePipelineState:p];
  setBuf(e, dst, 0);
  setBuf(e, src, 1);
  const uint64_t max_chunk = 0x80000000ull;
  for (uint64_t base = 0; base < nbytes;) {
    const uint32_t chunk = (uint32_t)std::min(max_chunk, nbytes - base);
    setVal(e, chunk, 2);
    setVal(e, (uint64_t)base, 3);
    const NSUInteger len = ((NSUInteger)chunk + 15) / 16;
    [e dispatchThreads:MTLSizeMake(len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(len, [p maxTotalThreadsPerThreadgroup]), 1, 1)];
    ctx().ndispatch++;
    base += chunk;
  }
}

// copy_cast_kernel_mps (Copy.mm:32): iterator {dst, src} with resize_outputs(false), exec_unary_kernel
// "copy_identity" with ilp_threshold 131072. (conj/neg variants are not modelled.)
inline std::string copy_cast(Opnd& dst, Opnd src) {
  Iter it = buildIter({&dst, &src}, /*resize_outputs=*/false);
  std::string kn;
  if (iterSplit32(it, {&dst, &src}, [&](std::vector<Opnd>& o) { kn = copy_cast(o[0], o[1]); })) return kn;
  return exec_unary(it, dst, src, "copy_identity", std::nullopt, std::nullopt, 131072u);
}

// mps_copy_ / copy_kernel_mps (Copy.mm:355,377) for MPS->MPS: dst.copy_(src), also the kernel behind clone(),
// contiguous(), to(dtype) (_to_copy = empty + copy_). dst must be a defined tensor.
//   both C-contiguous (or both channels-last contiguous) and same dtype -> contiguous_byte_copy
//   otherwise copy_identity via exec_unary_kernel:
//     same dtype, strided              -> inner_contiguous_copy (unit inner stride, shape[0]>=16) / copy_identity_strided_T_T
//     dtype differs, contiguous        -> copy_identity_dense_castout_ilp_<src> (numel >= 131072) / _dense_castout_<src>
//     dtype differs, strided           -> copy_identity_inner_contiguous_castout_<src> / _strided_castout_<src>
inline std::string copy_(Opnd& dst, Opnd src) {
  dst.normalize();
  src.normalize();
  if (src.numel() == 0) return "";
  if (src.shape != dst.shape) {  // src.expand_as(dst)
    Opnd ex = src;
    const size_t n = dst.shape.size(), off = n - src.shape.size();
    ex.shape = dst.shape;
    ex.strides.assign(n, 0);
    for (size_t k = 0; k < src.shape.size(); k++)
      ex.strides[off + k] = (src.shape[k] == 1 && dst.shape[off + k] != 1) ? 0 : src.strides[k];
    src = ex;
  }
  // dst.suggest_memory_format(): channels_last only when dst is channels-last-contiguous but not C-contiguous
  // (simplification of is_strides_like_channels_last, see report).
  const bool cl = dst.shape.size() == 4 && !dst.contiguous() && dst.channelsLastContiguous();
  const bool same_fmt = cl ? (src.channelsLastContiguous() && dst.channelsLastContiguous()) : (src.contiguous() && dst.contiguous());
  const bool same_dt = src.dt == dst.dt;
  if (same_fmt && same_dt) {
    contiguous_byte_copy(dst.b, src.b, (uint64_t)src.numel() * dtSize(src.dt));
    return "contiguous_byte_copy";
  }
  return copy_cast(dst, src);
}

// ---------------------------------------------------------------------------------------------------------------
// Fill (ConstantOps.mm fill_mps_kernel)
// ---------------------------------------------------------------------------------------------------------------
inline std::string fill(Opnd& out, Scal value) {
  out.normalize();
  Iter it = buildIter({&out}, /*resize_outputs=*/false);
  if (it.numel() == 0) return "";
  std::string kn;
  if (iterSplit32(it, {&out}, [&](std::vector<Opnd>& o) { kn = fill(o[0], value); })) return kn;
  uint8_t sb[16];
  const size_t sn = scalBytes(value, out.dt, sb);
  auto e = enc();
  if (!out.nonOverlappingAndDense()) {
    const std::string kname = std::string("fill_scalar_strided_") + dtName(out.dt);
    auto p = pso(kname);
    const int64_t dim0 = it.ndim() > 0 ? it.shape[0] : 1;
    const int64_t inner_numel = it.numel() / dim0;
    [e setComputePipelineState:p];
    setBuf(e, out.b, 0);
    setRaw(e, sb, sn, 1);
    setI64s(e, it.shape, 0, 2);
    setI64s(e, it.strides[0], 0, 3);
    setVal(e, (uint32_t)it.ndim(), 4);
    const NSUInteger maxTG = [p maxTotalThreadsPerThreadgroup];
    [e dispatchThreads:MTLSizeMake((NSUInteger)inner_numel, (NSUInteger)dim0, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(maxTG, (NSUInteger)inner_numel), 1, 1)];
    ctx().ndispatch++;
    return kname;
  }
  const uint32_t numel = (uint32_t)it.numel();
  const bool is_byte = dtSize(out.dt) == 1;
  const uint32_t offs = is_byte ? (uint32_t)(out.b.off & 3) : 0;
  const NSUInteger threads = is_byte ? (numel + offs + 3) / 4 : numel;
  const std::string kname = std::string("fill_scalar_dense_") + dtName(out.dt);
  auto p = pso(kname);
  [e setComputePipelineState:p];
  setBuf(e, out.b, 0);
  setRaw(e, sb, sn, 1);
  setVal(e, numel, 2);
  dispatch1D(e, p, threads);
  return kname;
}

// ---------------------------------------------------------------------------------------------------------------
// aten-level wrappers (kernel-name selection of the individual ops)
// ---------------------------------------------------------------------------------------------------------------
// BinaryKernel.mm add_mps_kernel: alpha==1 -> "add"; alpha==-1 (non-bool) -> "sub"; else "add_alpha" + alpha
// (alpha typed as the kernel's alpha_type). aten::sub(a,b,alpha) == add(a,b,-alpha).
inline std::string add(Opnd& out, const Opnd& a, const Opnd& b, Scal alpha = Scal((int64_t)1)) {
  const bool one = alpha.isF ? alpha.d == 1.0 : alpha.i == 1, mone = alpha.isF ? alpha.d == -1.0 : alpha.i == -1;
  if (one) return binary_iter("add", out, a, b);
  if (mone && resultType({&a, &b}) != BOOL) return binary_iter("sub", out, a, b);
  return binary_iter("add_alpha", out, a, b, alpha);
}
inline std::string sub(Opnd& out, const Opnd& a, const Opnd& b, Scal alpha = Scal((int64_t)1)) {
  return add(out, a, b, alpha.isF ? Scal(-alpha.d) : Scal(-alpha.i));
}
// BinaryKernel.mm: mul -> "mul"; div (true) -> "div_true"; floor_divide / div(rounding_mode='floor') -> "div_floor";
// remainder(.Tensor/.Scalar) -> "remainder"; polar -> "polar" (out complex64 from float32 abs, angle).
inline std::string mul(Opnd& out, const Opnd& a, const Opnd& b) { return binary_iter("mul", out, a, b); }
inline std::string div_true(Opnd& out, const Opnd& a, const Opnd& b) { return binary_iter("div_true", out, a, b); }
inline std::string div_floor(Opnd& out, const Opnd& a, const Opnd& b) { return binary_iter("div_floor", out, a, b); }
inline std::string remainder(Opnd& out, const Opnd& a, const Opnd& b) { return binary_iter("remainder", out, a, b); }
inline std::string polar(Opnd& out, const Opnd& abs, const Opnd& angle) { return binary_iter("polar", out, abs, angle); }
// BinaryKernel.mm comparison stubs: op in {"eq","ne","lt","le","gt","ge"}; natural output bool, ILP threshold 1<<18.
// out.dt must be BOOL for the normal (non-castout) kernels.
inline std::string compare(const std::string& op, Opnd& out, const Opnd& a, const Opnd& b) {
  return binary_iter(op, out, a, b, std::nullopt, std::nullopt, BOOL, 1u << 18);
}

// UnaryKernel.mm REGISTER_UNARY_TI_DISPATCH: name in {"sin","cos","sqrt","exp","neg","sigmoid","tanh","abs","log",...}
inline std::string unary(const std::string& name, Opnd& out, const Opnd& in) { return unary_iter(name, out, in); }
// UnaryKernel.mm pow_tensor_scalar_kernel: 2 -> "sqr"; integral dtype -> "pow_scalar"(int alpha); -1 -> "reciprocal";
// -0.5 -> "rsqrt"; 0.5 -> "sqrt"; else "pow_scalar"(float alpha).
inline std::string pow_scalar(Opnd& out, const Opnd& in, Scal ex) {
  const double v = ex.isF ? ex.d : (double)ex.i;
  if (v == 2.0) return unary_iter("sqr", out, in);
  if (dtIsIntegral(in.dt, true)) return unary_iter("pow_scalar", out, in, ex, I32);
  if (v == -1.0) return unary_iter("reciprocal", out, in);
  if (v == -0.5) return unary_iter("rsqrt", out, in);
  if (v == 0.5) return unary_iter("sqrt", out, in);
  return unary_iter("pow_scalar", out, in, ex, F32);
}
// ActivationKernel.mm: silu -> "silu"; gelu(approximate) -> "gelu_tanh" if approximate=='tanh' else "gelu".
inline std::string silu(Opnd& out, const Opnd& in) { return unary_iter("silu", out, in); }
inline std::string gelu(Opnd& out, const Opnd& in, bool approximate_tanh) {
  return unary_iter(approximate_tanh ? "gelu_tanh" : "gelu", out, in);
}
// ActivationKernel.mm softplus: exec_unary_kernel_with_params(iter, "softplus", SoftplusParams{beta, threshold},
// "SoftplusParams") - struct { float beta; float threshold; } (8 bytes), no ILP variant.
inline std::string softplus(Opnd& out, const Opnd& in, float beta = 1.0f, float threshold = 20.0f) {
  const float params[2] = {beta, threshold};
  return unary_params_iter("softplus", out, in, params, sizeof(params), "SoftplusParams");
}
// TensorCompare.mm clamp_min_scalar: exec_unary_kernel(iter, "clamp_min_scalar", min, iter.common_dtype())
// -> clamp_min_scalar_{dense|strided|inner_contiguous}_<T>_<T>_<T>, min bound as a <T> scalar.
inline std::string clamp_min(Opnd& out, const Opnd& in, Scal mn) { return unary_iter("clamp_min_scalar", out, in, mn, in.dt); }

}  // namespace tk
