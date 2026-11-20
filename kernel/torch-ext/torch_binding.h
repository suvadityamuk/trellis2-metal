#pragma once

#include <torch/torch.h>

void gemm(torch::Tensor &out, torch::Tensor const &x, torch::Tensor const &weight);
void gemm_bias_gelu(torch::Tensor &out, torch::Tensor const &x, torch::Tensor const &weight, torch::Tensor const &bias);
void bias_gelu(torch::Tensor &x, torch::Tensor const &bias);
void resid_ln(torch::Tensor &x, torch::Tensor const &branch, torch::Tensor const &vec, torch::Tensor &out, int64_t flags, double eps);
void qprep(torch::Tensor &out, torch::Tensor const &src, torch::Tensor const &bias, torch::Tensor const &gamma,
           torch::Tensor const &phases, int64_t col_offset, int64_t flags);
