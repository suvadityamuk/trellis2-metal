#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm(Tensor! out, Tensor x, Tensor weight) -> ()");
  ops.def("gemm_bias_gelu(Tensor! out, Tensor x, Tensor weight, Tensor bias) -> ()");
  ops.def("bias_gelu(Tensor! x, Tensor bias) -> ()");
  ops.def("resid_ln(Tensor! x, Tensor branch, Tensor vec, Tensor! out, int flags, float eps) -> ()");
  ops.def("qprep(Tensor! out, Tensor src, Tensor bias, Tensor gamma, Tensor phases, int col_offset, int flags) -> ()");
  ops.def("lbvh_build(Tensor vertices, Tensor faces) -> (Tensor, Tensor)");
  ops.def("mesh_nearest(Tensor! dist, Tensor! face, Tensor! bary, Tensor nodes, Tensor tris, Tensor points, float max_distance) -> ()");
  ops.def("telea_levels(Tensor mask) -> (Tensor, Tensor, Tensor)");
  ops.def("telea_fill(Tensor! img, Tensor lv, Tensor list, int[] offsets, int radius) -> ()");
#if defined(METAL_KERNEL)
  ops.impl("gemm", torch::kMPS, gemm);
  ops.impl("gemm_bias_gelu", torch::kMPS, gemm_bias_gelu);
  ops.impl("bias_gelu", torch::kMPS, bias_gelu);
  ops.impl("resid_ln", torch::kMPS, resid_ln);
  ops.impl("qprep", torch::kMPS, qprep);
  ops.impl("lbvh_build", torch::kCPU, lbvh_build);
  ops.impl("mesh_nearest", torch::kMPS, mesh_nearest);
  ops.impl("telea_levels", torch::kCPU, telea_levels);
  ops.impl("telea_fill", torch::kMPS, telea_fill);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
