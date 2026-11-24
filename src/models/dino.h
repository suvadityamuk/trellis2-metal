// DINOv3 ViT-L/16 image conditioning: Trellis2ImageTo3DPipeline.get_cond -> DinoV3FeatureExtractor.__call__
// (ref/TRELLIS.2/trellis2/modules/image_feature_extractor.py:59) over transformers' DINOv3ViTModel
// (models/dinov3_vit/modeling_dinov3_vit.py) — parity build, a line-by-line transcription into t2 ops.
//   CPU (as in the reference): PIL resize to (res, res) LANCZOS, uint8 -> float32 / 255 (numpy), HWC -> CHW, stack.
//   MPS: Normalize (sub_/div_ by [3,1,1] views), patch conv (MPSGraph), RoPE angles/cos/sin, 24 pre-norm layers
//        (sdpa prefill kernel), final F.layer_norm without affine (the checkpoint's `norm.*` is NOT applied).
//   neg_cond = zeros_like(cond).
#pragma once
#include "common.h"
#include "image.h"

namespace t2 {

// inv_freq = 1 / 100.0 ** torch.arange(0, 1, 4 / 64, dtype=float32) — computed on the CPU at model construction
// (DINOv3ViTRopePositionEmbedding.__init__); raw float32 bits dumped from torch (CPU pow is not libm's powf).
static const uint32_t kDinoInvFreqBits[16] = {
  0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77,
  0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba15, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2};

struct Dino {
  struct Layer { Tensor n1w, n1b, n2w, n2b, ls1, ls2; LinearP q, k, v, o, up, down; };
  int hidden = 1024, heads = 16, patch = 16, nreg = 4;
  float eps = 1e-5f;
  Tensor cls, reg, pw, pb, mean, stdv, inv_freq;
  std::vector<Layer> layers;

  void load(const std::string& json, const std::string& st) {
    Json cfg = loadJson(json);
    hidden = (int)cfg["hidden_size"].i(); heads = (int)cfg["num_attention_heads"].i(); patch = (int)cfg["patch_size"].i();
    nreg = (int)cfg["num_register_tokens"].i(); eps = (float)cfg["layer_norm_eps"].d();
    int nl = (int)cfg["num_hidden_layers"].i();
    if (hidden / heads != 64 || cfg["rope_theta"].d() != 100.0 || cfg["hidden_act"].str != "gelu" || (cfg.has("use_gated_mlp") && cfg["use_gated_mlp"].b)) {
      fprintf(stderr, "Dino: unsupported config (need head_dim 64, rope_theta 100, gelu MLP)\n"); abort();
    }
    STFile f(st);
    cls = f.get("embeddings.cls_token"); reg = f.get("embeddings.register_tokens");
    pw = f.get("embeddings.patch_embeddings.weight"); pb = f.get("embeddings.patch_embeddings.bias");
    layers.resize(nl);
    for (int i = 0; i < nl; i++) {
      Layer& L = layers[i]; std::string p = "layer." + std::to_string(i);
      L.n1w = f.get(p + ".norm1.weight"); L.n1b = f.get(p + ".norm1.bias");
      L.n2w = f.get(p + ".norm2.weight"); L.n2b = f.get(p + ".norm2.bias");
      L.ls1 = f.get(p + ".layer_scale1.lambda1"); L.ls2 = f.get(p + ".layer_scale2.lambda1");
      L.q.load(f, p + ".attention.q_proj", tk::F32); L.k.load(f, p + ".attention.k_proj", tk::F32);
      L.v.load(f, p + ".attention.v_proj", tk::F32); L.o.load(f, p + ".attention.o_proj", tk::F32);
      L.up.load(f, p + ".mlp.up_proj", tk::F32); L.down.load(f, p + ".mlp.down_proj", tk::F32);
    }
    // transforms.Normalize: torch.as_tensor(python list, dtype=float32)
    const float m[3] = {(float)0.485, (float)0.456, (float)0.406}, s[3] = {(float)0.229, (float)0.224, (float)0.225};
    mean = from_cpu(tk::F32, {3}, m); stdv = from_cpu(tk::F32, {3}, s);
    inv_freq = from_cpu(tk::F32, {16}, kDinoInvFreqBits);
  }

  // [i.resize((res, res), LANCZOS)] -> np.array(i.convert('RGB')).astype(float32) / 255 -> permute(2, 0, 1) -> stack
  // (all CPU in the reference) -> .cuda(): f32 [1, 3, res, res] on the GPU
  static Tensor imageToTensor(const Image8& img, int res) {
    if (img.c != 3) imageFail("Dino: RGB image expected (run preprocess_image first)");
    Image8 r = resize_lanczos(img, res, res);
    size_t n = (size_t)res * res;
    std::vector<float> chw(3 * n);
    for (size_t p = 0; p < n; p++) for (int c = 0; c < 3; c++) chw[c * n + p] = (float)r.d[p * 3 + c] / 255.0f;
    return from_cpu(tk::F32, {1, 3, res, res}, chw.data());
  }

  // DINOv3ViTRopePositionEmbedding.forward: cos, sin f32 [nh * nw, head_dim]
  void rope(int64_t nh, int64_t nw, Tensor& cosT, Tensor& sinT) const {
    // torch.arange(0.5, n, dtype=float32, device=mps): arange_float_i32 computes 0.5f + 1.0f * i, which is exact in
    // float32 for every representable i here, so the values are produced on the CPU (no torchk arange route exists).
    auto arange05 = [](int64_t n) { std::vector<float> v(n); for (int64_t i = 0; i < n; i++) v[i] = 0.5f + 1.0f * (float)i; return from_cpu(tk::F32, {n}, v.data()); };
    Tensor ch = arange05(nh), cw = arange05(nw);
    ch = div(ch, (int64_t)nh);
    cw = div(cw, (int64_t)nw);
    // torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
    Tensor gh = ch.view({nh, 1}), gw = cw.view({1, nw});
    gh.shape = {nh, nw}; gh.strides = {1, 0};
    gw.shape = {nh, nw}; gw.strides = {0, 1};
    Tensor coords = stack({gh, gw}, -1).view({nh * nw, 2});
    coords = sub(mul(coords, 2.0), 1.0);                          // 2.0 * coords - 1.0
    // angles = 2 * math.pi * patch_coords[:, :, None] * inv_freq[None, None, :]
    Tensor ang = mul(coords.unsqueeze(2), 2 * M_PI);
    ang = mul(ang, inv_freq.view({1, 1, 16}));                    // [N, 2, 16]
    ang = ang.view({nh * nw, 32});                                // flatten(1, 2)
    ang = cat({ang, ang}, -1);                                    // tile(2): repeat_mps, pure data movement
    cosT = un("cos", ang); sinT = un("sin", ang);
  }

  static Tensor rotateHalf(const Tensor& x) {
    int64_t h = x.shape.back() / 2;
    return cat({un("neg", x.slice(-1, h, 2 * h)), x.slice(-1, 0, h)}, -1);
  }
  // apply_rotary_pos_emb for one of q / k: [1, H, L, D] (transposed view); rope only on the patch tokens
  static Tensor applyRope(const Tensor& x, const Tensor& cosT, const Tensor& sinT) {
    int64_t L = x.shape[2], np = sinT.shape[0], npre = L - np;
    Tensor pre = x.slice(2, 0, npre), pat = x.slice(2, npre, L);
    pat = add(mul(pat, cosT), mul(rotateHalf(pat), sinT));
    return cat({pre, pat}, -2);
  }

  Tensor layer(const Layer& P, const Tensor& x, const Tensor& cosT, const Tensor& sinT) const {
    int64_t B = x.shape[0], L = x.shape[1], hd = hidden / heads;
    Tensor h = layer_norm(x, eps, &P.n1w, &P.n1b);
    Tensor q = P.q(h).view({B, L, heads, hd}).transpose(1, 2);
    Tensor k = P.k(h).view({B, L, heads, hd}).transpose(1, 2);
    Tensor v = P.v(h).view({B, L, heads, hd}).transpose(1, 2);
    q = applyRope(q, cosT, sinT);
    k = applyRope(k, cosT, sinT);
    Tensor a = sdpa(q, k, v);                                     // scale = head_dim ** -0.5, no mask
    a = contiguous(a.transpose(1, 2)).view({B, L, (int64_t)hidden});
    a = P.o(a);
    a = mul(a, P.ls1);                                            // layer_scale1
    Tensor y = add(a, x);                                         // drop_path(h) + residual
    h = layer_norm(y, eps, &P.n2w, &P.n2b);
    h = P.down(gelu(P.up(h), false));
    h = mul(h, P.ls2);
    return add(h, y);
  }

  // image: f32 [B, 3, H, W] in [0, 1] (H, W multiples of the patch size) -> features f32 [B, 1 + nreg + H/p * W/p, hidden]
  Tensor features(const Tensor& image) const {
    int64_t B = image.shape[0], H = image.shape[2], W = image.shape[3], nh = H / patch, nw = W / patch;
    if (image.shape[1] != 3 || H % patch || W % patch) { fprintf(stderr, "Dino: bad image shape\n"); abort(); }
    // transforms.Normalize: tensor.clone(); tensor.sub_(mean.view(-1, 1, 1)).div_(std.view(-1, 1, 1))
    Tensor x = clone(image);
    sub_(x, mean.view({3, 1, 1}));
    div_(x, stdv.view({3, 1, 1}));
    // embeddings: patch conv -> flatten(2).transpose(1, 2) -> cat([cls, register, patches], dim=1)
    Tensor pe(tk::F32, {B, (int64_t)hidden, nh, nw});
    tk::Buf bb = pb.buf();
    tk::conv2d_patch(x.buf(), pw.buf(), &bb, pe.buf(), tk::F32, B, 3, H, W, hidden, patch);
    pace();
    Tensor tok = pe.view({B, (int64_t)hidden, nh * nw}).transpose(1, 2);
    Tensor c = cls, r = reg;
    c.shape[0] = B; r.shape[0] = B; if (B != 1) { c.strides[0] = 0; r.strides[0] = 0; }
    Tensor hs = cat({c, r, tok}, 1);
    Tensor cosT, sinT; rope(nh, nw, cosT, sinT);
    for (auto& P : layers) hs = layer(P, hs, cosT, sinT);
    return layer_norm(hs, 1e-5f);                                 // F.layer_norm(hs, hs.shape[-1:])
  }

  // get_cond([image], resolution)['cond']; neg_cond is zeros_like(cond) (see negCond)
  Tensor operator()(const Image8& img, int resolution) const { return features(imageToTensor(img, resolution)); }
  static Tensor negCond(const Tensor& cond) { return zeros(cond.dt, cond.shape); }
};

}  // namespace t2
