#pragma once

#include <torch/torch.h>

#include <tuple>
#include <vector>

void gemm(torch::Tensor &out, torch::Tensor const &x, torch::Tensor const &weight);
void gemm_bias_gelu(torch::Tensor &out, torch::Tensor const &x, torch::Tensor const &weight, torch::Tensor const &bias);
void bias_gelu(torch::Tensor &x, torch::Tensor const &bias);
void resid_ln(torch::Tensor &x, torch::Tensor const &branch, torch::Tensor const &vec, torch::Tensor &out, int64_t flags, double eps);
void qprep(torch::Tensor &out, torch::Tensor const &src, torch::Tensor const &bias, torch::Tensor const &gamma,
           torch::Tensor const &phases, int64_t col_offset, int64_t flags);

// mesh post-processing (to_glb)
std::tuple<torch::Tensor, torch::Tensor> lbvh_build(torch::Tensor const &vertices, torch::Tensor const &faces);
void mesh_nearest(torch::Tensor &dist, torch::Tensor &face, torch::Tensor &bary, torch::Tensor const &nodes, torch::Tensor const &tris,
                  torch::Tensor const &points, double max_distance);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> telea_levels(torch::Tensor const &mask);
void telea_fill(torch::Tensor &img, torch::Tensor const &lv, torch::Tensor const &list, std::vector<int64_t> offsets, int64_t radius);
