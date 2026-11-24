// SparseStructureDecoder (ref/TRELLIS.2/trellis2/models/sparse_structure_vae.py) + the coords post-processing of
// Trellis2ImageTo3DPipeline.sample_sparse_structure — parity build.
//   logits = decoder(z_s)                       conv3d network, torso in fp16, norms in fp32 (ChannelLayerNorm32)
//   decoded = logits > 0
//   if resolution != 64: decoded = max_pool3d(decoded.float(), r, r, 0) > 0.5
//   coords = argwhere(decoded)[:, [0, 2, 3, 4]].int()
#pragma once
#include "common.h"

namespace t2 {

struct Conv3dP {
  Tensor w, b;
  void load(const STFile& f, const std::string& p) { w = f.get(p + ".weight"); b = f.get(p + ".bias"); }
  Tensor operator()(const Tensor& x) const { return conv3d_333(x, w, &b); }
};

struct SSDecoder {
  struct Res { Tensor n1w, n1b, n2w, n2b; Conv3dP c1, c2; };
  struct Blk { bool up = false; Res r; Conv3dP conv; };
  DT dt = tk::F16;
  int out_channels = 1, latent_channels = 8;
  Conv3dP input_layer, out_conv; Tensor onw, onb;
  std::vector<Res> middle; std::vector<Blk> blocks;

  static void loadRes(Res& r, const STFile& f, const std::string& p) {
    r.n1w = f.get(p + ".norm1.weight"); r.n1b = f.get(p + ".norm1.bias");
    r.n2w = f.get(p + ".norm2.weight"); r.n2b = f.get(p + ".norm2.bias");
    r.c1.load(f, p + ".conv1"); r.c2.load(f, p + ".conv2");
  }
  void load(const std::string& json, const std::string& st) {
    Json cfg = loadJson(json); const Json& a = cfg["args"];
    out_channels = (int)a["out_channels"].i(); latent_channels = (int)a["latent_channels"].i();
    int nres = (int)a["num_res_blocks"].i(), nmid = a.has("num_res_blocks_middle") ? (int)a["num_res_blocks_middle"].i() : 2;
    size_t nch = a["channels"].size();
    dt = (a.has("use_fp16") && a["use_fp16"].b) ? tk::F16 : tk::F32;
    STFile f(st);
    if (f.ent("middle_block.0.conv1.weight").dt != dt) { fprintf(stderr, "SSDecoder: checkpoint torso dtype != config dtype\n"); abort(); }
    input_layer.load(f, "input_layer");
    middle.resize(nmid);
    for (int i = 0; i < nmid; i++) loadRes(middle[i], f, "middle_block." + std::to_string(i));
    int bi = 0;
    for (size_t i = 0; i < nch; i++) {
      for (int j = 0; j < nres; j++) { Blk b; loadRes(b.r, f, "blocks." + std::to_string(bi++)); blocks.push_back(b); }
      if (i + 1 < nch) { Blk b; b.up = true; b.conv.load(f, "blocks." + std::to_string(bi++) + ".conv"); blocks.push_back(b); }
    }
    onw = f.get("out_layer.0.weight"); onb = f.get("out_layer.0.bias"); out_conv.load(f, "out_layer.2");
  }

  // ChannelLayerNorm32: permute(0,2,3,4,1).contiguous() -> float32 layer_norm -> cast back -> permute(0,4,1,2,3).contiguous()
  static Tensor chanNorm(const Tensor& x, const Tensor& w, const Tensor& b) {
    Tensor h = contiguous(x.permute({0, 2, 3, 4, 1}));
    h = to(layer_norm(to(h, tk::F32), 1e-5f, &w, &b), x.dt);
    return contiguous(h.permute({0, 4, 1, 2, 3}));
  }
  static Tensor res(const Res& r, const Tensor& x) {
    Tensor h = chanNorm(x, r.n1w, r.n1b);
    h = silu(h);
    h = r.c1(h);
    h = chanNorm(h, r.n2w, r.n2b);
    h = silu(h);
    h = r.c2(h);
    return add(h, x);  // skip_connection is Identity (channels == out_channels everywhere in this decoder)
  }
  // pixel_shuffle_3d(x, 2)
  static Tensor pixelShuffle(const Tensor& x) {
    int64_t B = x.shape[0], C = x.shape[1] / 8, H = x.shape[2], W = x.shape[3], D = x.shape[4];
    Tensor v = x.view({B, C, 2, 2, 2, H, W, D}).permute({0, 1, 5, 2, 6, 3, 7, 4});
    return contiguous(v).view({B, C, H * 2, W * 2, D * 2});
  }

  // z f32 [B,8,16,16,16] -> logits f32 [B,1,64,64,64]
  Tensor operator()(const Tensor& z) const {
    Tensor h = input_layer(z);
    h = to(h, dt);
    for (auto& r : middle) h = res(r, h);
    for (auto& b : blocks) h = b.up ? pixelShuffle(b.conv(h)) : res(b.r, h);
    h = to(h, z.dt);
    h = chanNorm(h, onw, onb);
    h = silu(h);
    return out_conv(h);
  }

  // Post-processing of sample_sparse_structure. Only comparisons / max over {0,1} / index bookkeeping, so it is done on
  // the CPU (bit exact by construction): occupancy = logits > 0, OR-pooled by ratio = R / resolution, then argwhere in
  // row-major order (what torch.argwhere returns), columns [batch, x, y, z] as int32.
  static Tensor coordsFromLogits(const Tensor& logits_, int resolution) {
    tk::sync();
    Tensor logits = contiguous_cpu(logits_);
    int64_t B = logits.shape[0], C = logits.shape[1], R = logits.shape[2];
    if (C != 1 || logits.shape[3] != R || logits.shape[4] != R || logits.dt != tk::F32) { fprintf(stderr, "coordsFromLogits: bad logits\n"); abort(); }
    int64_t ratio = resolution != R ? R / resolution : 1, r = resolution != R ? R / ratio : R;  // max_pool3d floor mode
    const float* p = logits.ptr<float>();
    std::vector<int32_t> out;
    for (int64_t b = 0; b < B; b++)
      for (int64_t x = 0; x < r; x++)
        for (int64_t y = 0; y < r; y++)
          for (int64_t z = 0; z < r; z++) {
            bool occ = false;
            for (int64_t i = 0; i < ratio && !occ; i++)
              for (int64_t j = 0; j < ratio && !occ; j++)
                for (int64_t k = 0; k < ratio && !occ; k++)
                  occ = p[((b * R + x * ratio + i) * R + y * ratio + j) * R + z * ratio + k] > 0.0f;
            if (occ) { out.push_back((int32_t)b); out.push_back((int32_t)x); out.push_back((int32_t)y); out.push_back((int32_t)z); }
          }
    int64_t n = (int64_t)out.size() / 4;
    Tensor t(tk::I32, {n, 4});
    if (n) memcpy(t.raw(), out.data(), out.size() * 4);
    return t;
  }

  // decode + post-process: z_s -> coords int32 [N,4]; optionally returns the logits
  Tensor run(const Tensor& z, int resolution, Tensor* logits_out = nullptr) const {
    Tensor logits = (*this)(z);
    if (logits_out) *logits_out = logits;
    return coordsFromLogits(logits, resolution);
  }
};

}  // namespace t2
