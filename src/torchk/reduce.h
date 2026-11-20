// Reductions, normalisation, pooling, scans and nonzero: exact replays of torch-MPS kernel dispatches.
//
// Mirrors (torch 2.15 @ ref/pytorch-src):
//   aten/src/ATen/native/mps/operations/ReduceOps.mm      (reduction framework, sum/mean/norm/std/min/max/any/all/arg*)
//   aten/src/ATen/native/mps/kernels/ReduceOps.h          (tuning constants, NormParams)
//   aten/src/ATen/native/mps/operations/Normalization.mm  (layer_norm_mps)
//   aten/src/ATen/native/mps/operations/Pooling.mm        (launch_max_pool_kernel)
//   aten/src/ATen/native/mps/operations/ScanKernel.mm     (cumsum)
//   aten/src/ATen/native/mps/operations/Indexing.mm       (nonzero_impl_mps)
//
// Conventions
//   * RTensor describes an operand: Buf (buf.off already includes storage_offset*itemsize), dtype, sizes and
//     strides in ELEMENTS (torch strides).
//   * Result buffers are allocated by the caller (contiguous results). Internal scratch ("partials", block sums)
//     is allocated through tk::reduce_alloc(), which defaults to a fresh shared MTLBuffer; a runtime may replace
//     the hook with its own pool. Scratch contents never need initialisation.
//   * All tensors are assumed to satisfy canUse32BitIndexMath (numel < 2^31); the 64-bit variants are not replayed.
#pragma once
#include "base.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace tk {

// ------------------------------------------------------------------------------------------------------------
// Operand description + small view helpers
// ------------------------------------------------------------------------------------------------------------
struct RTensor {
  Buf buf;
  DT dt = F32;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;  // elements
  int64_t dim() const { return (int64_t)sizes.size(); }
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : sizes) n *= s;
    return n;
  }
  // TensorImpl::compute_contiguous
  bool is_contiguous() const {
    if (numel() == 0) return true;
    int64_t expected = 1;
    for (int64_t d = dim() - 1; d >= 0; d--) {
      if (sizes[d] == 1) continue;
      if (strides[d] != expected) return false;
      expected *= sizes[d];
    }
    return true;
  }
  RTensor permute(const std::vector<int64_t>& perm) const {
    RTensor r = *this;
    for (size_t i = 0; i < perm.size(); i++) {
      r.sizes[i] = sizes[perm[i]];
      r.strides[i] = strides[perm[i]];
    }
    return r;
  }
  // Tensor::flatten(first, last) for a view-able (contiguous over those dims) tensor.
  RTensor flatten(int64_t first, int64_t last) const {
    RTensor r = *this;
    int64_t n = 1;
    for (int64_t d = first; d <= last; d++) n *= sizes[d];
    r.sizes.erase(r.sizes.begin() + first, r.sizes.begin() + last + 1);
    r.strides.erase(r.strides.begin() + first, r.strides.begin() + last + 1);
    r.sizes.insert(r.sizes.begin() + first, n);
    r.strides.insert(r.strides.begin() + first, strides[last]);
    return r;
  }
};

inline std::vector<int64_t> rd_contig_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t s = 1;
  for (int64_t d = (int64_t)sizes.size() - 1; d >= 0; d--) {
    st[d] = s;
    s *= std::max<int64_t>(sizes[d], 1);
  }
  return st;
}
inline RTensor rd_contig(Buf b, DT dt, std::vector<int64_t> sizes) {
  RTensor t{b, dt, std::move(sizes), {}};
  t.strides = rd_contig_strides(t.sizes);
  return t;
}

// Scratch allocator hook (see header comment).
inline std::function<Buf(size_t)>& reduce_alloc() {
  static std::function<Buf(size_t)> fn = [](size_t bytes) {
    Buf b;
    b.buf = [ctx().dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
    b.off = 0;
    return b;
  };
  return fn;
}

[[noreturn]] inline void rd_fail(const char* msg) {
  fprintf(stderr, "tk::reduce: %s\n", msg);
  abort();
}

// at::collapse_dims (aten/src/ATen/CollapseDims.h). Returns {remapped excluded dim, new ndim}.
inline std::pair<int64_t, int64_t> rd_collapse_dims(std::vector<int64_t>& sizes, std::vector<int64_t>& strides,
                                                    int64_t dims, int64_t excludeDim = -1) {
  int64_t stopDim = (excludeDim == -1) ? dims : excludeDim;
  int64_t newIndex = -1, oldIndex = 0, remapped = -1;
  while (oldIndex < dims) {
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) continue;
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      ++oldIndex;
      break;
    }
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) continue;
      if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
        sizes[newIndex] *= sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      } else {
        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      }
    }
    if (oldIndex != dims) {
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      remapped = newIndex;
      ++oldIndex;
      stopDim = dims;
    }
  }
  if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
    sizes[0] = 1;
    strides[0] = 1;
    return {0, 1};
  }
  return {remapped, newIndex + 1};
}

// ------------------------------------------------------------------------------------------------------------
// Reduction framework (ReduceOps.mm: select_reduction_plan / encode_reduction / reduction_dispatch_mps)
// ------------------------------------------------------------------------------------------------------------
namespace rd {
// kernels/ReduceOps.h
constexpr uint32_t MAX_THREADGROUP_SIZE = 1024, SUM_NCHAINS = 8, INNER_TG_SIZE = 256, SIMD = 32, MAX_NDIM = 16;
constexpr uint32_t CHUNK_MAX_ROW_LEN = 256, CHUNK_ELEMS_PER_LANE = 16, CHUNK_MIN_NUMEL = 65536;
constexpr uint32_t SPLIT_MIN_ROW_LEN = 2048, SPLIT_MIN_SEG_LEN = 64, SPLIT_MAX_SEGS = 2048, SPLIT_TARGET_PARTIALS = 8192,
                   SPLIT_MIN_TGS = 64;
constexpr uint32_t OUTER_TG_WIDTH = 32, OUTER_TG_HEIGHT = 32, NARROW_TG_SIZE = 256, OUTER_SMALL_DIM_MAX_SIZE = 256,
                   NARROW_BATCHED_MIN_DIM_SIZE = 128, NARROW_SPLIT_ELEMS_PER_TG = 8192;
constexpr uint32_t OUTER_SPLIT_MIN_DIM_SIZE = 4096, OUTER_SPLIT_MIN_TGS = 32, OUTER_SPLIT_MIN_SEG_LEN = 512,
                   OUTER_SPLIT_MAX_TGS = 2048, OUTER_SPLIT_STRIDED_TARGET_TGS = 256;
constexpr uint32_t ARG_SPLIT_MIN_TGS = 512, ARG_SPLIT_MIN_SEG_LEN = 512, ARG_SPLIT_TARGET_PARTIALS = 16384;

inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// kernels/ReduceOps.h NormParams<16>
struct NormParams {
  float p;
  uint32_t reduction_size;
  uint32_t ndim;
  uint32_t input_sizes[MAX_NDIM];
  uint32_t input_strides[MAX_NDIM];
  uint32_t output_sizes[MAX_NDIM];
  uint32_t output_strides[MAX_NDIM];
};
static_assert(sizeof(NormParams) == 268, "NormParams layout");

enum class Kernel { Generic, Innermost, InnermostChunk, Outer, OuterSmallDim, Narrow, Flat, FlatStrided, ArgCombine };

struct Layout {
  uint32_t outer_size, dim_size, inner_size;
  std::array<uint32_t, 4> strides;
  bool is_contiguous;
  static Layout contiguous(uint32_t o, uint32_t d, uint32_t i) { return {o, d, i, {i, 1, d * i, 0}, true}; }
};

struct Plan {
  Kernel kernel = Kernel::Generic;
  Layout layout{};
  uint32_t num_segments = 1;
  uint32_t lanes = SIMD;
  std::array<uint32_t, 4> flat_sizes{};
  std::array<uint32_t, 4> flat_strides{};
};

// ReduceOps.mm select_flat_strided
inline std::optional<Plan> select_flat_strided(const RTensor& input, uint32_t num_groups) {
  auto sizes = input.sizes;
  auto strides = input.strides;
  const auto ndim = rd_collapse_dims(sizes, strides, input.dim()).second;
  if (ndim < 1 || ndim > 2) return std::nullopt;
  const bool has_rows = ndim == 2;
  const uint32_t rows = has_rows ? (uint32_t)sizes[0] : 1u;
  const uint32_t run = (uint32_t)sizes[ndim - 1];
  constexpr uint32_t MIN_CHUNK = 1024;
  uint32_t chunks = 1;
  if (rows < num_groups * 4 && run >= 2 * MIN_CHUNK) {
    chunks = std::min(cdiv(num_groups * 4, rows), run / MIN_CHUNK);
    while (chunks > 1 && run % chunks != 0) chunks--;
  }
  if (rows * chunks < num_groups) return std::nullopt;
  Plan plan;
  plan.kernel = Kernel::FlatStrided;
  plan.layout = Layout::contiguous(1, (uint32_t)input.numel(), 1);
  plan.num_segments = num_groups;
  plan.flat_sizes = {rows, run / chunks, chunks, 0};
  plan.flat_strides = {has_rows ? (uint32_t)strides[0] : 0u, (uint32_t)strides[ndim - 1], 0, 0};
  return plan;
}

// ReduceOps.mm outer_reduction_layout
inline std::optional<Layout> outer_reduction_layout(const RTensor& input, int64_t reduced_dim) {
  auto sizes = input.sizes;
  auto strides = input.strides;
  const auto [cd, cn] = rd_collapse_dims(sizes, strides, input.dim(), reduced_dim);
  if (cn > 2 && !(cn == 3 && cd == 1)) return std::nullopt;
  const bool has_outer = cd > 0;
  const bool has_inner = cd < cn - 1;
  return Layout{has_outer ? (uint32_t)sizes[0] : 1u,
                (uint32_t)sizes[cd],
                has_inner ? (uint32_t)sizes[cd + 1] : 1u,
                {(uint32_t)strides[cd], has_inner ? (uint32_t)strides[cd + 1] : 0u, has_outer ? (uint32_t)strides[0] : 0u, 0},
                input.is_contiguous()};
}

// ReduceOps.mm select_outer_reduction
inline Plan select_outer_reduction(const Layout& layout, bool is_arg, int64_t numel) {
  const uint32_t outer_size = layout.outer_size, dim_size = layout.dim_size, inner_size = layout.inner_size;
  const bool is_contiguous = layout.is_contiguous;
  const uint32_t natural_tgs = outer_size * cdiv(inner_size, OUTER_TG_WIDTH);
  const bool split = outer_size == 1 && dim_size >= OUTER_SPLIT_MIN_DIM_SIZE && natural_tgs < OUTER_SPLIT_MIN_TGS;
  const bool small_dim = !is_arg && dim_size <= OUTER_SMALL_DIM_MAX_SIZE && natural_tgs >= SPLIT_MIN_TGS;
  const bool use_narrow =
      is_arg ? split : (numel >= CHUNK_MIN_NUMEL && !(small_dim && dim_size < NARROW_BATCHED_MIN_DIM_SIZE));
  Plan plan;
  plan.kernel = Kernel::Outer;
  plan.layout = layout;
  if (is_contiguous && inner_size < OUTER_TG_WIDTH && use_narrow) {
    plan.kernel = Kernel::Narrow;
    plan.num_segments =
        outer_size > 1 ? 1u : std::clamp(dim_size * inner_size / NARROW_SPLIT_ELEMS_PER_TG, is_arg ? 2u : 1u, SPLIT_MAX_SEGS);
  } else if (small_dim) {
    plan.kernel = Kernel::OuterSmallDim;
  } else if (split) {
    plan.num_segments = (is_contiguous && !is_arg)
        ? std::min(dim_size / OUTER_SPLIT_MIN_SEG_LEN, std::max(2u, OUTER_SPLIT_MAX_TGS / natural_tgs))
        : std::clamp(OUTER_SPLIT_STRIDED_TARGET_TGS / natural_tgs, 2u, std::min(dim_size, SPLIT_MAX_SEGS));
  }
  return plan;
}

// ReduceOps.mm select_inner_reduction
inline Plan select_inner_reduction(const Layout& layout, bool is_arg, int64_t numel) {
  const uint32_t num_rows = layout.outer_size, row_len = layout.dim_size;
  Plan plan;
  plan.kernel = Kernel::Innermost;
  plan.layout = layout;
  if (!is_arg && row_len <= CHUNK_MAX_ROW_LEN && numel >= CHUNK_MIN_NUMEL) {
    plan.kernel = Kernel::InnermostChunk;
    plan.lanes = std::min(SIMD, std::bit_ceil(cdiv(row_len, CHUNK_ELEMS_PER_LANE)));
    return plan;
  }
  const uint32_t num_tgs = cdiv(num_rows, INNER_TG_SIZE / SIMD);
  if (num_tgs < (is_arg ? ARG_SPLIT_MIN_TGS : SPLIT_MIN_TGS) && row_len >= SPLIT_MIN_ROW_LEN) {
    uint32_t segments;
    if (is_arg) {
      segments = std::clamp(ARG_SPLIT_TARGET_PARTIALS / num_rows, 2u, std::max(row_len / ARG_SPLIT_MIN_SEG_LEN, 2u));
    } else {
      plan.kernel = Kernel::InnermostChunk;
      const uint32_t max_segments = std::min(SPLIT_MAX_SEGS, cdiv(row_len, SPLIT_MIN_SEG_LEN));
      segments = std::clamp(cdiv(SPLIT_TARGET_PARTIALS, num_rows), 2u, std::max(max_segments, 2u));
    }
    plan.num_segments = cdiv(row_len, cdiv(row_len, segments));
  }
  return plan;
}

// ReduceOps.mm select_reduction_plan. `output` is the keepdim-shaped view (same ndim as input).
inline Plan select_reduction_plan(const RTensor& input, const RTensor& output, bool is_arg) {
  const uint32_t reduction_size = (uint32_t)(input.numel() / output.numel());
  const uint32_t num_outputs = (uint32_t)output.numel();
  Plan plan;
  plan.layout = Layout::contiguous(num_outputs, reduction_size, 1);
  if (num_outputs == 1 && !is_arg) {
    uint32_t num_groups = std::min(512u, cdiv(reduction_size, MAX_THREADGROUP_SIZE * SUM_NCHAINS));
    if (!input.is_contiguous() && num_groups > 1) {
      constexpr uint32_t STRIDED_ELEMS_PER_GROUP = 256 * SUM_NCHAINS;
      const uint32_t strided_groups = std::min(512u, cdiv(reduction_size, STRIDED_ELEMS_PER_GROUP));
      if (auto strided = select_flat_strided(input, strided_groups)) return *strided;
    }
    while (num_groups > 1 && reduction_size % num_groups != 0) num_groups--;
    if (num_groups > 1) {
      plan.kernel = Kernel::Flat;
      plan.num_segments = num_groups;
    }
    return plan;
  }
  if (output.is_contiguous()) {
    int num_reduced = 0;
    int64_t first_reduced = input.dim(), reduced_dim = -1;
    for (int64_t d = 0; d < input.dim(); d++) {
      if (input.sizes[d] != output.sizes[d]) {
        num_reduced++;
        first_reduced = std::min(first_reduced, d);
        reduced_dim = d;
      }
    }
    const bool adjacent = num_reduced == (reduced_dim - first_reduced + 1);
    if (num_reduced == 1 && reduced_dim < input.dim() - 1) {
      if (auto layout = outer_reduction_layout(input, reduced_dim)) {
        return select_outer_reduction(*layout, is_arg, input.numel());
      }
    } else if (num_reduced <= 1 && input.is_contiguous()) {
      return select_inner_reduction(plan.layout, is_arg, input.numel());
    } else if (input.is_contiguous() && adjacent) {
      return select_reduction_plan(input.flatten(first_reduced, reduced_dim), output.flatten(first_reduced, reduced_dim),
                                   is_arg);
    }
  }
  return plan;
}

// ReduceOps.mm reduction_combine_plan
inline Plan reduction_combine_plan(const Plan& plan, bool is_arg) {
  Plan c;
  if (is_arg) {
    c.kernel = Kernel::ArgCombine;
    c.layout = Layout::contiguous(plan.layout.outer_size * plan.layout.inner_size, plan.num_segments, 1);
    return c;
  }
  c.kernel = plan.kernel;
  if (c.kernel == Kernel::InnermostChunk) c.kernel = Kernel::Innermost;
  else if (c.kernel == Kernel::Flat || c.kernel == Kernel::FlatStrided) c.kernel = Kernel::Generic;
  c.layout = Layout::contiguous(plan.layout.outer_size, plan.num_segments, plan.layout.inner_size);
  return c;
}

inline const char* kernel_suffix(Kernel k) {
  switch (k) {
    case Kernel::Generic: return "";
    case Kernel::Innermost: return "_innermost";
    case Kernel::InnermostChunk: return "_innermost_chunk";
    case Kernel::Outer: return "_outer";
    case Kernel::OuterSmallDim: return "_outer_small_dim";
    case Kernel::Narrow: return "_narrow";
    case Kernel::Flat: return "_flat";
    case Kernel::FlatStrided: return "_flat_strided";
    case Kernel::ArgCombine: return "_combine";
  }
  return "";
}

struct MetalType {
  std::string name;
  size_t size;
  MetalType(DT d) : name(dtName(d)), size(dtSize(d)) {}
  MetalType(std::string n, size_t s) : name(std::move(n)), size(s) {}
};

struct Op {
  bool is_arg;
  std::string prefix;
  MetalType input_type;
  MetalType output_type;
  float param = 0;
};

struct Partials {
  Buf values;
  Buf indices;
};

// ReduceOps.mm encode_reduction. Everything is dispatchThreads.
inline void encode(const RTensor& input, const RTensor& output, const Plan& plan, const Op& op, const Partials& partials = {}) {
  const auto& layout = plan.layout;
  const bool split_arg = op.is_arg && plan.num_segments > 1;
  const std::string name = op.prefix + "reduction" + kernel_suffix(plan.kernel);
  std::string kernel_name;
  if (split_arg) kernel_name = name + "_p1_" + op.input_type.name;
  else if (plan.kernel == Kernel::ArgCombine) kernel_name = name + "_" + op.input_type.name;
  else kernel_name = name + "_" + op.input_type.name + "_" + op.output_type.name;

  auto e = enc();
  [e setComputePipelineState:pso(kernel_name)];
  if (split_arg && plan.kernel != Kernel::Narrow) {
    setBuf(e, partials.values, 4);
    setBuf(e, partials.indices, 5);
  } else if (plan.kernel == Kernel::ArgCombine) {
    setBuf(e, partials.indices, 3);
  }
  MTLSize grid, group;
  switch (plan.kernel) {
    case Kernel::Innermost:
    case Kernel::InnermostChunk:
    case Kernel::ArgCombine: {
      uint32_t num_simdgroups = layout.outer_size * plan.num_segments;
      setBuf(e, input.buf, 0);
      setBuf(e, output.buf, 1);
      if (op.is_arg) {
        const std::array<uint32_t, 4> sizes{num_simdgroups, cdiv(layout.dim_size, plan.num_segments), plan.num_segments,
                                            layout.dim_size};
        setVal(e, sizes, 2);
      } else if (plan.kernel == Kernel::InnermostChunk) {
        const std::array<uint32_t, 4> sizes{layout.outer_size, layout.dim_size, plan.lanes, plan.num_segments};
        setVal(e, sizes, 2);
        setVal(e, op.param, 3);
        num_simdgroups = cdiv(num_simdgroups, SIMD / plan.lanes);
      } else {
        const std::array<uint32_t, 2> sizes{layout.outer_size, layout.dim_size};
        setVal(e, sizes, 2);
        setVal(e, op.param, 3);
      }
      const uint32_t num_tgs = cdiv(num_simdgroups, INNER_TG_SIZE / SIMD);
      grid = MTLSizeMake((NSUInteger)num_tgs * INNER_TG_SIZE, 1, 1);
      group = MTLSizeMake(INNER_TG_SIZE, 1, 1);
      break;
    }
    case Kernel::Outer:
    case Kernel::OuterSmallDim:
    case Kernel::Narrow: {
      const bool narrow = plan.kernel == Kernel::Narrow;
      const std::array<uint32_t, 4> sizes{layout.dim_size, layout.inner_size, plan.num_segments, 0};
      setBuf(e, input.buf, 0);
      if (op.is_arg && narrow) {
        setBuf(e, partials.values, 1);
        setBuf(e, partials.indices, 2);
        setVal(e, sizes, 3);
      } else if (op.is_arg) {
        setBuf(e, output.buf, 1);
        setVal(e, sizes, 2);
        setVal(e, layout.strides, 3);
      } else {
        setBuf(e, output.buf, 1);
        setVal(e, sizes, 2);
        setVal(e, op.param, 3);
        if (!narrow) setVal(e, layout.strides, 4);
      }
      if (narrow) {
        const uint32_t active = (NARROW_TG_SIZE / layout.inner_size) * layout.inner_size;
        grid = MTLSizeMake(active, plan.num_segments, layout.outer_size);
        group = MTLSizeMake(active, 1, 1);
      } else {
        const uint32_t height = plan.kernel == Kernel::OuterSmallDim ? 1u : OUTER_TG_HEIGHT;
        const uint32_t num_tgs = cdiv(layout.inner_size, OUTER_TG_WIDTH);
        grid = MTLSizeMake(num_tgs * OUTER_TG_WIDTH, plan.num_segments * height, layout.outer_size);
        group = MTLSizeMake(OUTER_TG_WIDTH, height, 1);
      }
      break;
    }
    case Kernel::Flat: {
      const std::array<uint32_t, 2> sizes{plan.num_segments, layout.dim_size / plan.num_segments};
      setBuf(e, input.buf, 0);
      setBuf(e, output.buf, 1);
      setVal(e, sizes, 2);
      grid = MTLSizeMake(plan.num_segments * 256, 1, 1);
      group = MTLSizeMake(256, 1, 1);
      break;
    }
    case Kernel::FlatStrided: {
      setBuf(e, input.buf, 0);
      setBuf(e, output.buf, 1);
      setVal(e, plan.flat_sizes, 2);
      setVal(e, plan.flat_strides, 3);
      grid = MTLSizeMake(plan.num_segments * 256, 1, 1);
      group = MTLSizeMake(256, 1, 1);
      break;
    }
    case Kernel::Generic: {
      if (input.dim() > (int64_t)MAX_NDIM) rd_fail("generic reduction: rank > 16");
      NormParams params{};
      params.ndim = (uint32_t)input.dim();
      params.p = op.param;
      params.reduction_size = layout.dim_size;
      for (int64_t d = 0; d < input.dim(); d++) {
        params.input_sizes[d] = (uint32_t)input.sizes[d];
        params.input_strides[d] = (uint32_t)input.strides[d];
        params.output_sizes[d] = (uint32_t)output.sizes[d];
        params.output_strides[d] = (uint32_t)output.strides[d];
      }
      setBuf(e, input.buf, 0);
      setBuf(e, output.buf, 1);
      setVal(e, params, 2);
      const uint32_t threads = std::min(MAX_THREADGROUP_SIZE, (params.reduction_size + 31u) / 32u * 32u);
      grid = MTLSizeMake((NSUInteger)output.numel() * threads, 1, 1);
      group = MTLSizeMake(threads, 1, 1);
      break;
    }
  }
  [e dispatchThreads:grid threadsPerThreadgroup:group];
  ctx().ndispatch++;
}

// ReduceOps.mm reduction_dispatch_mps. `output` must be the keepdim-shaped view (output.dim()==input.dim()).
// Returns the pass-1 plan (for tests / tracing).
inline Plan dispatch(RTensor input, RTensor output, const Op& op, const MetalType& partial_type,
                     const std::string& combine_prefix) {
  if (input.numel() <= 0 || output.numel() <= 0) rd_fail("empty reduction (torch handles these without a kernel)");
  if (output.dim() != input.dim()) rd_fail("output view must be keepdim-shaped");
  if (!op.is_arg && !input.is_contiguous()) {
    std::vector<int64_t> perm(input.dim());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return input.strides[a] > input.strides[b]; });
    RTensor permuted = input.permute(perm);
    if (permuted.is_contiguous()) {
      input = std::move(permuted);
      output = output.permute(perm);
    }
  }
  const Plan plan = select_reduction_plan(input, output, op.is_arg);
  if (plan.num_segments == 1) {
    encode(input, output, plan, op);
    return plan;
  }
  const Plan combine_plan = reduction_combine_plan(plan, op.is_arg);
  if (plan.kernel == Kernel::Flat && !input.is_contiguous()) {
    // torch: input = input.contiguous() (a strided copy kernel) -- not replayed here.
    rd_fail("Flat reduction over a non-contiguous input: caller must materialise a contiguous copy first");
  }
  const int64_t num_partials = output.numel() * plan.num_segments;
  Partials partials;
  // at::empty({num_partials * partial_size}, kByte): a 1-D byte tensor.
  RTensor pv = rd_contig(reduce_alloc()((size_t)num_partials * partial_type.size), U8,
                         {num_partials * (int64_t)partial_type.size});
  partials.values = pv.buf;
  if (op.is_arg) partials.indices = reduce_alloc()((size_t)num_partials * 4);
  const Op first_op{op.is_arg, op.prefix, op.input_type, partial_type, 0.0f};
  const Op combine_op{op.is_arg, combine_prefix, partial_type, op.output_type, op.param};
  encode(input, op.is_arg ? output : pv, plan, first_op, partials);
  encode(pv, output, combine_plan, combine_op, partials);
  return plan;
}

// at::toOpMathType for the dtypes that occur here.
inline DT opmath(DT d) { return (d == F16 || d == BF16) ? F32 : d; }

// The keepdim-shaped output view the kernels see. `result` is the caller's contiguous result buffer.
// make_reduction/review_reduce_result (ReduceOpsUtils.h): keepdim=false views the result with as_strided,
// inserting size 1 / stride 0 at reduced dims; keepdim=true uses the result's own strides. (Strides of size-1
// dims are only ever copied into NormParams.output_strides; the kernels never multiply by them.)
inline RTensor out_view(const RTensor& in, Buf result, DT dt, const std::vector<bool>& reduced, bool keepdim) {
  RTensor o;
  o.buf = result;
  o.dt = dt;
  o.sizes = in.sizes;
  for (int64_t d = 0; d < in.dim(); d++)
    if (reduced[d]) o.sizes[d] = 1;
  o.strides = rd_contig_strides(o.sizes);
  if (!keepdim)
    for (int64_t d = 0; d < in.dim(); d++)
      if (reduced[d]) o.strides[d] = 0;
  return o;
}

// dims list -> mask; empty list == all dims (make_dim_mask).
inline std::vector<bool> dim_mask(const std::vector<int64_t>& dims, int64_t ndim) {
  std::vector<bool> m(ndim, dims.empty());
  for (auto d : dims) m[d < 0 ? d + ndim : d] = true;
  return m;
}
}  // namespace rd

// ------------------------------------------------------------------------------------------------------------
// aten-level reductions
// ------------------------------------------------------------------------------------------------------------

// aten::sum.dim_IntList -- ReduceOps.mm sum_nansum_kernel_mps / sum_kernel_mps.
// out_dt: torch result dtype (bool/int inputs -> I64, read directly from the narrow input: "sum_..._bool_long").
// dims empty == full reduction. `out` holds prod(non-reduced sizes) elements of out_dt.
inline rd::Plan sum_dim(const RTensor& in, Buf out, DT out_dt, const std::vector<int64_t>& dims, bool keepdim) {
  const auto mask = rd::dim_mask(dims, in.dim());
  const rd::Op op{false, "sum_", in.dt, out_dt, 0.0f};
  return rd::dispatch(in, rd::out_view(in, out, out_dt, mask, keepdim), op, rd::opmath(out_dt), "sum_");
}

// aten::mean.dim -- ReduceOps.mm mean_kernel_mps: the sum kernel with param = float(reduction_size); the divide
// is fused into the (final) pass in opmath precision. Float dtypes only (out dtype == in dtype).
inline rd::Plan mean_dim(const RTensor& in, Buf out, const std::vector<int64_t>& dims, bool keepdim) {
  const auto mask = rd::dim_mask(dims, in.dim());
  RTensor ov = rd::out_view(in, out, in.dt, mask, keepdim);
  const rd::Op op{false, "sum_", in.dt, in.dt, (float)(in.numel() / ov.numel())};
  return rd::dispatch(in, ov, op, rd::opmath(in.dt), "sum_");
}

// aten::norm.ScalarOpt_dim / linalg_vector_norm -- ReduceOps.mm norm_kernel_mps (p in {0,1,2,+-inf}).
// NOTE (LinearAlgebra.cpp linalg_vector_norm_out): when every reduced dim has size 1 torch computes abs() instead
// of calling the norm kernel; that case is rejected here.
inline rd::Plan norm_dim(const RTensor& in, Buf out, double p, const std::vector<int64_t>& dims, bool keepdim) {
  const auto mask = rd::dim_mask(dims, in.dim());
  RTensor ov = rd::out_view(in, out, in.dt, mask, keepdim);
  if (ov.numel() == in.numel()) rd_fail("norm over size-1 dims is abs() in torch, not a reduction kernel");
  const DT acc = rd::opmath(in.dt);
  if (std::isinf(p)) {
    const rd::Op op{false, p > 0 ? "norm_inf_" : "norm_neginf_", in.dt, in.dt, 0.0f};
    return rd::dispatch(in, ov, op, in.dt, p > 0 ? "max_" : "min_");
  }
  if (p == 0 || p == 1 || p == 2) {
    const char* prefix = p == 0 ? "norm_l0_" : p == 1 ? "norm_l1_" : "norm_l2_";
    const rd::Op op{false, prefix, in.dt, in.dt, p == 2 ? 1.0f : 0.0f};
    return rd::dispatch(in, ov, op, acc, p == 2 ? "norm_l2_combine_" : "sum_");
  }
  rd_fail("norm: general p uses abs/pow/sum/pow in torch; not replayed");
}

// x.norm(dim=-1, keepdim=True) on a contiguous tensor (the TRELLIS RMS-norm / q,k normalisation).
// Trace: [1,4096,12,128] -> norm_l2_reduction_innermost_chunk_float_float threads=393216 tg=256
//        2:{49152,128,8,1} 3:1.0f.  Small tensors (numel < 65536) use norm_l2_reduction_innermost_float_float
//        with 2:{rows,C} 3:1.0f.
inline rd::Plan norm_l2_lastdim(Buf in, Buf out, DT dt, const std::vector<int64_t>& sizes) {
  return norm_dim(rd_contig(in, dt, sizes), out, 2.0, {(int64_t)sizes.size() - 1}, true);
}

// aten::std.correction / aten::var.correction -- ReduceOps.mm std_var_mps. Partials are "float3" (16 bytes,
// Welford-style count/mean/M2 triples); the combine pass uses the same prefix ("std_reduction_float3_float").
inline rd::Plan std_var_dim(const RTensor& in, Buf out, const std::vector<int64_t>& dims, bool keepdim,
                            float correction = 1.0f, bool take_sqrt = true) {
  const auto mask = rd::dim_mask(dims, in.dim());
  const char* prefix = take_sqrt ? "std_" : "var_";
  const rd::Op op{false, prefix, in.dt, in.dt, correction};
  return rd::dispatch(in, rd::out_view(in, out, in.dt, mask, keepdim), op, rd::MetalType("float3", 16), prefix);
}

// x.std(dim=[1,2,3,...]) for a contiguous [B, ...] tensor. With B==1 this is a single-output reduction:
// trace [1,8,16,16,16]-like 32768 floats -> std_reduction_flat_float_float3 threads=1024 tg=256 2:{4,8192}
// then std_reduction_float3_float (generic) threads=32 tg=32, NormParams{p=1, rs=4, ndim=1, in_sizes[0]=64}.
inline rd::Plan std_all_but_batch(Buf in, Buf out, DT dt, const std::vector<int64_t>& sizes, bool keepdim = false,
                                  float correction = 1.0f) {
  std::vector<int64_t> dims;
  for (int64_t d = 1; d < (int64_t)sizes.size(); d++) dims.push_back(d);
  return std_var_dim(rd_contig(in, dt, sizes), out, dims, keepdim, correction, true);
}

// amax/amin and full-tensor aten::max / aten::min -- ReduceOps.mm value_reduction_kernel_mps
// (min_max_mps_impl for the dim-less overload: dims empty, keepdim=false). bool is remapped to char.
inline rd::Plan minmax_values(const RTensor& in, Buf out, const std::vector<int64_t>& dims, bool keepdim, bool is_max) {
  const auto mask = rd::dim_mask(dims, in.dim());
  const DT k = in.dt == BOOL ? I8 : in.dt;
  const char* prefix = is_max ? "max_" : "min_";
  const rd::Op op{false, prefix, k, k, 0.0f};
  return rd::dispatch(in, rd::out_view(in, out, in.dt, mask, keepdim), op, k, prefix);
}
inline rd::Plan max_all(const RTensor& in, Buf out) { return minmax_values(in, out, {}, false, true); }
inline rd::Plan min_all(const RTensor& in, Buf out) { return minmax_values(in, out, {}, false, false); }

// aten::any / aten::all (.dim, .dims and full) -- ReduceOps.mm value_reduction_kernel_mps with "any_"/"all_":
// output type uchar (the bool result buffer is written as 0/1 bytes), partials uchar, combine "max_"/"min_".
// ReduceOps.cpp allany_impl: a 1-element input is handled as a copy (ne 0) without any reduction kernel ->
// returns false here and encodes nothing.
inline bool any_all(const RTensor& in, Buf out, const std::vector<int64_t>& dims, bool keepdim, bool is_any) {
  if (in.numel() == 1) return false;
  const auto mask = rd::dim_mask(dims, in.dim());
  const rd::Op op{false, is_any ? "any_" : "all_", in.dt, U8, 0.0f};
  rd::dispatch(in, rd::out_view(in, out, BOOL, mask, keepdim), op, U8, is_any ? "max_" : "min_");
  return true;
}

// aten::argmax / aten::argmin -- ReduceOps.mm argmax_argmin_out_mps. Output int64.
// dim == nullopt: flattened, input must be contiguous (torch does .contiguous().view(-1)).
// Returns false (nothing encoded) when the reduced dim has size 1 / input is 0-dim: torch does fill_(0).
inline bool argminmax(const RTensor& in_t, Buf out, std::optional<int64_t> dim, bool keepdim, bool is_max) {
  if (in_t.dim() == 0) return false;
  RTensor input, ov;
  if (dim) {
    const int64_t d = *dim < 0 ? *dim + in_t.dim() : *dim;
    if (in_t.sizes[d] == 1) return false;
    std::vector<bool> mask(in_t.dim(), false);
    mask[d] = true;
    input = in_t;
    // keepdim ? output : output.unsqueeze(dim): both have contiguous strides over the keepdim shape.
    ov = rd::out_view(in_t, out, I64, mask, true);
    if (!input.is_contiguous()) {
      std::vector<int64_t> inner, outer{d};
      for (int64_t i = 0; i < in_t.dim(); i++)
        if (i != d) {
          inner.push_back(i);
          outer.push_back(i);
        }
      inner.push_back(d);
      if (RTensor m = in_t.permute(inner); m.is_contiguous()) {
        input = m;
        ov = ov.permute(inner);
      } else if (RTensor m2 = in_t.permute(outer); m2.is_contiguous()) {
        input = m2;
        ov = ov.permute(outer);
      }
    }
  } else {
    if (!in_t.is_contiguous()) rd_fail("argmax(dim=None) of a non-contiguous tensor: materialise a copy first");
    input = rd_contig(in_t.buf, in_t.dt, {in_t.numel()});
    ov = rd_contig(out, I64, {1});
  }
  (void)keepdim;
  const DT k = input.dt == BOOL ? I8 : input.dt;
  const char* prefix = is_max ? "argmax_" : "argmin_";
  const rd::Op op{true, prefix, k, I64, 0.0f};
  rd::dispatch(input, ov, op, k, prefix);
  return true;
}

// aten::max.dim / aten::min.dim -- ReduceOps.mm min_max_out_mps: FIRST the arg reduction into `indices` (int64),
// THEN an independent value reduction ("max_"/"min_") into `values`. Each pass selects its own plan.
// Trace (int32 [N,3] with strides (4,1), dim 0):
//   N=1636:  argmax_reduction_outer_int_long threads=32,32,1 2:{1636,3,1,0} 3:{4,1,0,0};
//            max_reduction_outer_int_int same grid 2:{1636,3,1,0} 3:0.0f 4:{4,1,0,0}
//   N>=4096: argmax_reduction_outer_p1_int threads=32,8192,1 2:{N,3,256,0} 3:{4,1,0,0} 4:vals 5:idx;
//            argmax_reduction_combine_int threads=256 2:{3,256,1,256} 3:idx;
//            max_reduction_outer_int_int threads=32,8192,1 -> partials; max_reduction_outer_int_int 2:{256,3,1,0} 4:{3,1,768,0}
inline void minmax_dim(const RTensor& in, Buf values, Buf indices, int64_t dim, bool keepdim, bool is_max) {
  const int64_t d = dim < 0 ? dim + in.dim() : dim;
  if (!argminmax(in, indices, d, keepdim, is_max)) rd_fail("max.dim over a size-1 dim: torch fills indices with 0 (no kernel)");
  minmax_values(in, values, {d}, keepdim, is_max);
}
inline void max_dim(const RTensor& in, Buf values, Buf indices, int64_t dim, bool keepdim = false) {
  minmax_dim(in, values, indices, dim, keepdim, true);
}
inline void min_dim(const RTensor& in, Buf values, Buf indices, int64_t dim, bool keepdim = false) {
  minmax_dim(in, values, indices, dim, keepdim, false);
}

// ------------------------------------------------------------------------------------------------------------
// aten::native_layer_norm -- Normalization.mm layer_norm_mps
// ------------------------------------------------------------------------------------------------------------
// in/out: contiguous [rows, C] of dtype dt (C = prod(normalized_shape), rows = everything else).
// mean, rstd: rows elements of dt each (always written by the kernel; torch allocates them even when unused).
// weight/bias: C elements of dt, or nullptr -> setBuffer:nil (exactly what mtl_setArgs does for nullopt).
// eps is narrowed to float (1e-5 -> 0x3727c5ac, 1e-6 -> 0x358637bd).
// Kernel: layer_norm_{single_row if C <= 4096 else looped}_i32_{type}; dispatchThreadgroups(rows) with
// threadsPerThreadgroup = min(ceil(C/4), 1024)  (this fixes the in-kernel summation tree).
// Trace: rows=1029 C=1024 -> K layer_norm_single_row_i32_float grid=1029 tg=256 4:x00040000 5:xacc52737.
inline void layer_norm(Buf in, Buf out, Buf mean, Buf rstd, uint32_t rows, uint32_t C, float eps, const Buf* weight,
                       const Buf* bias, DT dt = F32) {
  constexpr uint32_t N_READS = 4;
  if (dt == I64) rd_fail("layer_norm: not implemented for long on MPS");
  const std::string name =
      std::string("layer_norm_") + (C <= 1024 * N_READS ? "single_row" : "looped") + "_i32_" + dtName(dt);
  auto e = enc();
  [e setComputePipelineState:pso(name)];
  setBuf(e, in, 0);
  setBuf(e, out, 1);
  setBuf(e, mean, 2);
  setBuf(e, rstd, 3);
  setVal(e, (uint32_t)C, 4);
  setVal(e, eps, 5);
  if (weight) setBuf(e, *weight, 6);
  else [e setBuffer:nil offset:0 atIndex:6];
  if (bias) setBuf(e, *bias, 7);
  else [e setBuffer:nil offset:0 atIndex:7];
  const NSUInteger tpg = std::min<uint64_t>(((uint64_t)C + N_READS - 1) / N_READS, 1024);
  [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  ctx().ndispatch++;
}

// ------------------------------------------------------------------------------------------------------------
// aten::max_pool3d_with_indices (and 1d/2d) -- Pooling.mm max_pool_with_indices_out_mps_template /
// launch_max_pool_kernel; kernels/Pooling.h PoolingParams<5>
// ------------------------------------------------------------------------------------------------------------
struct PoolingParams5 {
  int32_t dims;
  int32_t pooling_dims;
  int32_t input_sizes[5], input_strides[5];
  int32_t output_sizes[5], output_strides[5];
  int32_t indices_sizes[5], indices_strides[5];
  int32_t kernel_size[3], stride[3], padding[3], dilation[3];
  bool return_indices;
  bool adaptive;
};
static_assert(sizeof(PoolingParams5) == 180, "PoolingParams layout");

// in_sizes: full input shape (pooling_dims+1 or +2 dims, contiguous unless in_strides given);
// out_sizes: full output shape (caller computes pooling_output_shape; floor mode: (L+2p-d(k-1)-1)/s+1).
// indices: int64, same shape as output, or nullptr (-> nil buffer, return_indices=false; only for the
// no-indices entry points). torch leaves unused array slots / struct padding uninitialised; they are zero here.
// Dispatch: mtl_dispatch1DJob(numel(output)) -> threads=(numel,1,1) tg=min(numel, maxTotalThreadsPerThreadgroup).
// Trace: [1,1,64,64,64] k=2 s=2 -> T max_pool_float threads=32768 tg=1024, params {5,3,{1,1,64,64,64},
//        {262144,262144,4096,64,1},{1,1,32,32,32},{32768,32768,1024,32,1}, idx same, k{2,2,2}, s{2,2,2}, p{0}, d{1}}.
inline void max_pool_nd(Buf in, Buf out, const Buf* indices, DT dt, int32_t pooling_dims,
                        const std::vector<int64_t>& in_sizes, const std::vector<int64_t>& out_sizes,
                        const std::vector<int32_t>& kernel, const std::vector<int32_t>& stride,
                        const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation,
                        const std::vector<int64_t>* in_strides = nullptr) {
  const int32_t dims = (int32_t)in_sizes.size();
  if (dims != pooling_dims + 1 && dims != pooling_dims + 2) rd_fail("max_pool: bad input rank");
  PoolingParams5 p;
  memset(&p, 0, sizeof(p));
  p.dims = dims;
  p.pooling_dims = pooling_dims;
  const auto ist = in_strides ? *in_strides : rd_contig_strides(in_sizes);
  const auto ost = rd_contig_strides(out_sizes);
  int64_t numel = 1;
  for (int32_t d = 0; d < dims; d++) {
    p.input_sizes[d] = (int32_t)in_sizes[d];
    p.input_strides[d] = (int32_t)ist[d];
    p.output_sizes[d] = (int32_t)out_sizes[d];
    p.output_strides[d] = (int32_t)ost[d];
    if (indices) {
      p.indices_sizes[d] = (int32_t)out_sizes[d];
      p.indices_strides[d] = (int32_t)ost[d];
    }
    numel *= out_sizes[d];
  }
  // copy_and_maybe_expand: a 1-element list is broadcast; empty stride -> kernel_size; empty dilation -> 1.
  auto pick = [&](const std::vector<int32_t>& v, int32_t d, int32_t dflt) {
    return v.empty() ? dflt : v[v.size() == 1 ? 0 : d];
  };
  for (int32_t d = 0; d < pooling_dims; d++) {
    p.kernel_size[d] = pick(kernel, d, 1);
    p.stride[d] = pick(stride, d, p.kernel_size[d]);
    p.padding[d] = pick(padding, d, 0);
    p.dilation[d] = pick(dilation, d, 1);
  }
  p.return_indices = indices != nullptr;
  p.adaptive = false;
  auto ps = pso(std::string("max_pool_") + dtName(dt));
  auto e = enc();
  [e setComputePipelineState:ps];
  setBuf(e, in, 0);
  setBuf(e, out, 1);
  if (indices) setBuf(e, *indices, 2);
  else [e setBuffer:nil offset:0 atIndex:2];
  setVal(e, p, 3);
  const NSUInteger tg = std::min<NSUInteger>(std::max<NSUInteger>((NSUInteger)numel, 1), ps.maxTotalThreadsPerThreadgroup);
  [e dispatchThreads:MTLSizeMake((NSUInteger)numel, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  ctx().ndispatch++;
}
inline void max_pool3d(Buf in, Buf out, Buf indices, DT dt, const std::vector<int64_t>& in_sizes,
                       const std::vector<int64_t>& out_sizes, const std::vector<int32_t>& kernel,
                       const std::vector<int32_t>& stride = {}, const std::vector<int32_t>& padding = {},
                       const std::vector<int32_t>& dilation = {}) {
  max_pool_nd(in, out, &indices, dt, 3, in_sizes, out_sizes, kernel, stride, padding, dilation);
}

// ------------------------------------------------------------------------------------------------------------
// aten::cumsum -- ScanKernel.mm scan_simple_mps_impl (contiguous input, scan over the LAST dim only)
// ------------------------------------------------------------------------------------------------------------
// in: contiguous [..., axis] of in_dt; out: contiguous, same shape, out_dt (torch promotes bool/int8..int32 to
// int64; int32->int64 uses the fused "int_long" kernels, int64 "long", float "float").
// Selection (is_custom_scan_case = in_dt in {F32,F16,BF16,I32,I64}):
//   1. axis <= 32 && n_scans >= 4096 (custom dtypes)         -> cumsum_tiny_innermost_{tag}
//   2. axis >= 65536 && n_scans <= 1024 (custom dtypes)      -> int: cumsum_block_reduce_ + cumsum_block_carry_
//                                                               float: single-pass "decoupled look-back" kernel on
//                                                               Apple9+ (timing dependent carry fold; NOT replayed)
//   3. otherwise                                             -> cumsum_innermost_{tag}
// Trace (int64 1-D): axis=1 -> threads=32 tg=32; 1636 -> 832; 7786, 32454 -> 1024 (2: int64 axis);
//   132731 -> cumsum_block_reduce_long threads=23296,1,1 tg=256 2:132731 3:1459 4:91 (block_sums 91*8 B) then
//   cumsum_block_carry_long (in, out, block_sums, 132731, 1459, 91).
// Semantics: plain inclusive prefix sum; integer results are order independent, so every variant agrees bitwise.
inline void cumsum_lastdim(Buf in, DT in_dt, Buf out, DT out_dt, const std::vector<int64_t>& sizes) {
  const int64_t axis_size = sizes.back();
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  if (numel == 0) return;
  const int64_t n_scans = numel / axis_size;
  const bool custom = in_dt == F32 || in_dt == F16 || in_dt == BF16 || in_dt == I32 || in_dt == I64;
  const bool float_accum = in_dt == F32 || in_dt == F16 || in_dt == BF16;
  const std::string tag = in_dt == out_dt ? std::string(dtName(in_dt)) : std::string(dtName(in_dt)) + "_" + dtName(out_dt);
  const DT acc = (out_dt == F16 || out_dt == BF16) ? F32 : out_dt;  // scan_accum_scalar_type
  const int64_t n_reads = dtSize(acc) <= 4 ? 4 : 2;                 // scan_n_reads
  auto e = enc();

  if (custom && axis_size >= 1 && axis_size <= 32 && n_scans >= 4096) {  // scan_tiny_innermost_mps_impl
    constexpr int64_t kTinyTile = 2048, tg = 256;
    const int64_t rows_per_tg = kTinyTile / axis_size;
    const int64_t num_tg = (n_scans + rows_per_tg - 1) / rows_per_tg;
    [e setComputePipelineState:pso("cumsum_tiny_innermost_" + tag)];
    setBuf(e, in, 0);
    setBuf(e, out, 1);
    setVal(e, (uint32_t)axis_size, 2);
    setVal(e, (uint32_t)n_scans, 3);
    setVal(e, (uint32_t)rows_per_tg, 4);
    [e dispatchThreads:MTLSizeMake(tg * num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    ctx().ndispatch++;
    return;
  }
  if (custom && axis_size >= (1 << 16) && n_scans <= 1024) {  // use_contig_multiblock_scan
    if (float_accum) rd_fail("cumsum: float long-axis scan uses torch's decoupled look-back kernel (not replayed)");
    // scan_multiblock_mps_impl
    constexpr int64_t tg = 256;
    const int64_t elems_per_iter = tg * n_reads;
    const int64_t max_blocks_by_size = std::max<int64_t>(1, axis_size / elems_per_iter);
    int64_t num_blocks = (int64_t)std::sqrt((double)axis_size) / 4;
    num_blocks = std::max<int64_t>(num_blocks, 2);
    num_blocks = std::min<int64_t>(num_blocks, std::min<int64_t>(max_blocks_by_size, 4096));
    const int64_t block_size = (axis_size + num_blocks - 1) / num_blocks;
    num_blocks = (axis_size + block_size - 1) / block_size;
    Buf block_sums = reduce_alloc()((size_t)(n_scans * num_blocks) * dtSize(acc));
    [e setComputePipelineState:pso("cumsum_block_reduce_" + tag)];
    setBuf(e, in, 0);
    setBuf(e, block_sums, 1);
    setVal(e, (int64_t)axis_size, 2);
    setVal(e, (int64_t)block_size, 3);
    setVal(e, (int64_t)num_blocks, 4);
    [e dispatchThreads:MTLSizeMake(tg * num_blocks, n_scans, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    ctx().ndispatch++;
    [e setComputePipelineState:pso("cumsum_block_carry_" + tag)];
    setBuf(e, in, 0);
    setBuf(e, out, 1);
    setBuf(e, block_sums, 2);
    setVal(e, (int64_t)axis_size, 3);
    setVal(e, (int64_t)block_size, 4);
    setVal(e, (int64_t)num_blocks, 5);
    [e dispatchThreads:MTLSizeMake(tg * num_blocks, n_scans, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    ctx().ndispatch++;
    return;
  }
  // generic innermost scan
  auto ps = pso("cumsum_innermost_" + tag);
  [e setComputePipelineState:ps];
  setBuf(e, in, 0);
  setBuf(e, out, 1);
  setVal(e, (int64_t)axis_size, 2);
  const int max_tg = (int)ps.maxTotalThreadsPerThreadgroup;
  const int elements_per_simd = (int)n_reads * 32;
  int tgs = max_tg;
  if (axis_size <= n_reads * 1024) tgs = (int)((axis_size + elements_per_simd - 1) / elements_per_simd) * 32;
  else if (axis_size <= n_reads * 2048) tgs = (int)((axis_size / 2 + elements_per_simd - 1) / elements_per_simd) * 32;
  tgs = std::min(tgs, max_tg);
  // get_2d_grid_dims over the leading dims: (n_scans, 1) as long as n_scans < UINT32_MAX.
  [e dispatchThreads:MTLSizeMake(tgs, (NSUInteger)n_scans, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
  ctx().ndispatch++;
}

// ------------------------------------------------------------------------------------------------------------
// aten::nonzero -- Indexing.mm nonzero_impl_mps (count -> block prefix sum -> [host readback] -> scatter)
// ------------------------------------------------------------------------------------------------------------
// Output ordering: rows of out[nnz, ndim] (int64) are the coordinates of the nonzero elements in ascending
// flat (row-major / C) index order of the contiguous input -- identical to CPU torch.nonzero; deterministic.
// Usage:  auto st = nonzero_count(in, dt, sizes);  tk::sync();  int64_t nnz = nonzero_total(st);
//         <allocate out: nnz*ndim int64>;  if (nnz) nonzero_scatter(st, out, nnz);
// (torch does exactly this: steps 1+2, .item() sync, resize output, step 3.)
struct NonzeroState {
  Buf in;
  DT dt;
  std::vector<int64_t> sizes;
  int64_t numel = 0;
  uint32_t tpg = 0;         // maxTotalThreadsPerThreadgroup of the count kernel (1024)
  uint32_t num_blocks = 0;  // ceil(numel / tpg)
  Buf block_sums;           // int32[num_blocks]
  Buf tmp64;                // int64[num_blocks + 1]: block offsets, then the total count
};

// Steps 1+2. Trace: bool[1636,8] -> T count_nonzero_prefix_sum_bool_uint threads=13088 tg=1024 2:u64 0 3:u32 0;
//   T prefix_sum_blocks_uint threads=32 tg=32 0:block_sums 1:tmp64+0 2:tmp64+104 3:u32 13.
inline NonzeroState nonzero_count(Buf in, DT dt, const std::vector<int64_t>& sizes) {
  NonzeroState st;
  st.in = in;
  st.dt = dt;
  st.sizes = sizes;
  st.numel = 1;
  for (auto s : sizes) st.numel *= s;
  auto ps1 = pso(std::string("count_nonzero_prefix_sum_") + dtName(dt) + "_uint");
  st.tpg = (uint32_t)ps1.maxTotalThreadsPerThreadgroup;
  st.num_blocks = (uint32_t)(((uint64_t)st.numel + st.tpg - 1) / st.tpg);
  st.block_sums = reduce_alloc()((size_t)st.num_blocks * 4);
  st.tmp64 = reduce_alloc()((size_t)(st.num_blocks + 1) * 8);
  if ((uint64_t)st.numel > (1ull << 31)) rd_fail("nonzero: chunked (>2^31 element) path not replayed");
  auto e = enc();
  [e setComputePipelineState:ps1];
  if (st.numel > 0) {  // the chunk loop runs zero times for an empty input
    setBuf(e, in, 0);
    setBuf(e, st.block_sums, 1);
    setVal(e, (uint64_t)0, 2);  // flat base
    setVal(e, (uint32_t)0, 3);  // block base
    const NSUInteger tg = std::min<NSUInteger>((NSUInteger)st.numel, st.tpg);
    [e dispatchThreads:MTLSizeMake((NSUInteger)st.numel, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    ctx().ndispatch++;
  }
  [e setComputePipelineState:pso("prefix_sum_blocks_uint")];
  setBuf(e, st.block_sums, 0);
  setBuf(e, st.tmp64, 1);
  Buf total = st.tmp64;
  total.off += (uint64_t)st.num_blocks * 8;
  setBuf(e, total, 2);
  setVal(e, (uint32_t)st.num_blocks, 3);
  const uint32_t tgb = std::min(1024u, (st.num_blocks + 31u) / 32u * 32u);
  [e dispatchThreads:MTLSizeMake(tgb, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgb, 1, 1)];
  ctx().ndispatch++;
  return st;
}

// Host readback of the total count; call after tk::sync().
inline int64_t nonzero_total(const NonzeroState& st) {
  int64_t v;
  memcpy(&v, (const char*)[st.tmp64.buf contents] + st.tmp64.off + (uint64_t)st.num_blocks * 8, 8);
  return v;
}

// Step 3. out: contiguous int64[max_entries, ndim]. Not dispatched by torch when out.numel()==0.
// Trace: T scatter_nonzero_indices_bool_uint threads=13088 tg=1024 0:in 1:out 2:i32 ndim 3:int64 sizes[ndim]
//        4:tmp64 5:i64 nnz 6:u64 0 7:u32 0.
inline void nonzero_scatter(const NonzeroState& st, Buf out, int64_t max_entries) {
  if (max_entries * (int64_t)st.sizes.size() == 0) return;
  auto ps3 = pso(std::string("scatter_nonzero_indices_") + dtName(st.dt) + "_uint");
  auto e = enc();
  [e setComputePipelineState:ps3];
  setBuf(e, st.in, 0);
  setBuf(e, out, 1);
  setVal(e, (int32_t)st.sizes.size(), 2);
  setRaw(e, st.sizes.data(), st.sizes.size() * sizeof(int64_t), 3);
  setBuf(e, st.tmp64, 4);
  setVal(e, (int64_t)max_entries, 5);
  setVal(e, (uint64_t)0, 6);
  setVal(e, (uint32_t)0, 7);
  const NSUInteger tg = std::min<NSUInteger>((NSUInteger)st.numel, st.tpg);
  [e dispatchThreads:MTLSizeMake((NSUInteger)st.numel, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  ctx().ndispatch++;
}

}  // namespace tk
