#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm(Tensor! out, Tensor x, Tensor weight) -> ()");
  ops.def("gemm_bias_gelu(Tensor! out, Tensor x, Tensor weight, Tensor bias) -> ()");
  ops.def("bias_gelu(Tensor! x, Tensor bias) -> ()");
  ops.def("resid_ln(Tensor! x, Tensor branch, Tensor vec, Tensor! out, int flags, float eps) -> ()");
  ops.def("qprep(Tensor! out, Tensor src, Tensor bias, Tensor gamma, Tensor phases, int col_offset, int flags) -> ()");
#if defined(METAL_KERNEL)
  ops.impl("gemm", torch::kMPS, gemm);
  ops.impl("gemm_bias_gelu", torch::kMPS, gemm_bias_gelu);
  ops.impl("bias_gelu", torch::kMPS, bias_gelu);
  ops.impl("resid_ln", torch::kMPS, resid_ln);
  ops.impl("qprep", torch::kMPS, qprep);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
