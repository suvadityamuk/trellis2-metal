// SparseStructureFlowModel (dense 16^3 DiT) — parity build.
#pragma once
#include "common.h"

namespace t2 {

struct DenseFlow {
  int res = 16, C_in = 8, C_out = 8, ch = 1536, heads = 12, nblocks = 30;
  DT dt = tk::BF16;
  TimestepEmbedder temb; LinearP adaln, in_layer, out_layer;
  std::vector<BlockP> blocks; Tensor phases;  // C64 [L, 1, 64]

  void load(const std::string& json, const std::string& st) {
    Json cfg = loadJson(json); const Json& a = cfg["args"];
    res = (int)a["resolution"].i(); C_in = (int)a["in_channels"].i(); C_out = (int)a["out_channels"].i();
    ch = (int)a["model_channels"].i(); heads = (int)a["num_heads"].i(); nblocks = (int)a["num_blocks"].i();
    STFile f(st);
    temb.load(f, "t_embedder"); adaln.load(f, "adaLN_modulation.1", tk::F32);
    in_layer.load(f, "input_layer", tk::F32); out_layer.load(f, "out_layer", tk::F32);
    blocks.resize(nblocks);
    for (int i = 0; i < nblocks; i++) { blocks[i].load(f, "blocks." + std::to_string(i), dt); blocks[i].sa.heads = blocks[i].ca.heads = heads; }
    phases = dense_rope_phases(res, ch / heads).unsqueeze(-2);
  }

  Tensor selfAttn(const AttnP& p, const Tensor& x) const {
    int64_t B = x.shape[0], L = x.shape[1], H = p.heads;
    Tensor qkv = p.qkv(x).view({B, L, 3, H, -1});
    Tensor q = rms_norm(qkv.select(2, 0), p.qg), k = rms_norm(qkv.select(2, 1), p.kg), v = qkv.select(2, 2);
    q = apply_rope(q, phases); k = apply_rope(k, phases);
    Tensor h = sdpa(q.permute({0, 2, 1, 3}), k.permute({0, 2, 1, 3}), v.permute({0, 2, 1, 3})).permute({0, 2, 1, 3});
    return p.out(contiguous(h).view({B, L, -1}));
  }
  Tensor crossAttn(const AttnP& p, const Tensor& x, const Tensor& ctx) const {
    int64_t B = x.shape[0], L = x.shape[1], Lk = ctx.shape[1], H = p.heads;
    Tensor q = p.q(x).view({B, L, H, -1});
    Tensor kv = p.kv(ctx).view({B, Lk, 2, H, -1});
    q = rms_norm(q, p.qg);
    Tensor k = rms_norm(kv.select(2, 0), p.kg), v = kv.select(2, 1);
    Tensor h = sdpa(q.permute({0, 2, 1, 3}), k.permute({0, 2, 1, 3}), v.permute({0, 2, 1, 3})).permute({0, 2, 1, 3});
    return p.out(contiguous(h).view({B, L, -1}));
  }
  Tensor block(const BlockP& p, Tensor x, const Tensor& mod, const Tensor& ctx) const {
    auto m = to(add(p.modulation, mod), mod.dt).chunk(6, 1);
    Tensor h = layer_norm32(x, 1e-6f);
    h = add(mul(h, add(1, m[1].unsqueeze(1))), m[0].unsqueeze(1));
    h = selfAttn(p.sa, h);
    h = mul(h, m[2].unsqueeze(1));
    x = add(x, h);
    h = layer_norm32(x, 1e-6f, &p.n2w, &p.n2b);
    h = crossAttn(p.ca, h, ctx);
    x = add(x, h);
    h = layer_norm32(x, 1e-6f);
    h = add(mul(h, add(1, m[4].unsqueeze(1))), m[3].unsqueeze(1));
    h = p.mlp(h);
    h = mul(h, m[5].unsqueeze(1));
    x = add(x, h);
    return x;
  }
  // x f32 [B,C,R,R,R]; t f32 [B]; cond f32 [B,Lc,1024]
  Tensor operator()(const Tensor& x, const Tensor& t, const Tensor& cond) const {
    int64_t B = x.shape[0];
    Tensor h = contiguous(x.view({B, (int64_t)C_in, -1}).permute({0, 2, 1}));
    h = in_layer(h);
    Tensor te = adaln(silu(temb(t)));
    te = to(te, dt); h = to(h, dt);
    Tensor c = to(cond, dt);
    for (auto& b : blocks) h = block(b, h, te, c);
    h = to(h, tk::F32);
    h = layer_norm(h, 1e-5f);
    h = out_layer(h);
    Tensor o = h.permute({0, 2, 1});
    Tensor r(tk::F32, {B, (int64_t)C_out, (int64_t)res, (int64_t)res, (int64_t)res});
    copy_(r.view({B, (int64_t)C_out, -1}), o);
    return r;
  }
};

}  // namespace t2
