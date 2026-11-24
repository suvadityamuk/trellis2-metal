// Texture bake: UV-space rasterization (Metal render pass writing face ids), one fused compute kernel per texel
// (barycentric position -> nearest point on the input mesh -> trilinear sample of the sparse voxel attributes -> uint8),
// wavefront TELEA inpainting of the empty texels, and packing into the glTF base-colour / metallic-roughness layout.
#pragma once
#include "atlas.h"

namespace glb {

// ids U32 [H * W] (face index + 1, 0 = empty); texel row i covers v in [i / H, (i + 1) / H)
inline Tensor rasterize_uv(const Tensor& uv, const Tensor& faces, int64_t nf, int W, int H) {
  static id<MTLRenderPipelineState> rps = nil;
  id<MTLDevice> dev = tk::ctx().dev;
  if (!rps) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [kernlib().lib newFunctionWithName:@"uv_vs"];
    d.fragmentFunction = [kernlib().lib newFunctionWithName:@"uv_fs"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    NSError* err = nil;
    rps = [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!rps) { fprintf(stderr, "glb: uv raster pipeline failed: %s\n", err.localizedDescription.UTF8String); exit(1); }
  }
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:W height:H mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget; td.storageMode = MTLStorageModePrivate;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
  Tensor ids = gbuf<uint32_t>((int64_t)W * H);
  tk::endEncoder();
  id<MTLCommandBuffer> cb = tk::cmdbuf();
  MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = tex;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
  [re setRenderPipelineState:rps];
  [re setCullMode:MTLCullModeNone];
  [re setVertexBuffer:uv.buf().buf offset:uv.buf().off atIndex:0];
  if (nf > 0) [re drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:(NSUInteger)nf * 3 indexType:MTLIndexTypeUInt32 indexBuffer:faces.buf().buf indexBufferOffset:faces.buf().off];
  [re endEncoding];
  id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
  [be copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:ids.buf().buf
      destinationOffset:ids.buf().off destinationBytesPerRow:(NSUInteger)W * 4 destinationBytesPerImage:(NSUInteger)W * H * 4];
  [be endEncoding];
  return ids;
}

// CPU reference rasterizer (tests): texel centers, top-left fill rule (Metal's rule; framebuffer y grows downward and
// texel row i is v in [i / H, (i + 1) / H)), later faces overwrite earlier ones. cnt (optional) counts faces per texel.
inline void rasterize_uv_cpu(const float* uv, const int32_t* F, int64_t nf, int W, int H, uint32_t* ids, uint8_t* cnt = nullptr) {
  memset(ids, 0, (size_t)W * H * 4);
  if (cnt) memset(cnt, 0, (size_t)W * H);
  for (int64_t f = 0; f < nf; f++) {
    double x[3], y[3];
    for (int k = 0; k < 3; k++) { x[k] = (double)uv[2 * (int64_t)F[3 * f + k]] * W; y[k] = (double)uv[2 * (int64_t)F[3 * f + k] + 1] * H; }
    double a2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if (a2 == 0) continue;
    if (a2 < 0) { std::swap(x[1], x[2]); std::swap(y[1], y[2]); }  // clockwise on screen
    int x0 = std::max(0, (int)std::floor(std::min({x[0], x[1], x[2]}) - 0.5)), x1 = std::min(W - 1, (int)std::ceil(std::max({x[0], x[1], x[2]}) - 0.5));
    int y0 = std::max(0, (int)std::floor(std::min({y[0], y[1], y[2]}) - 0.5)), y1 = std::min(H - 1, (int)std::ceil(std::max({y[0], y[1], y[2]}) - 0.5));
    for (int py = y0; py <= y1; py++) for (int px = x0; px <= x1; px++) {
      double cx = px + 0.5, cy = py + 0.5; bool in = true;
      for (int k = 0; k < 3 && in; k++) {
        int j = (k + 1) % 3; double ex = x[j] - x[k], ey = y[j] - y[k];
        double w = ex * (cy - y[k]) - ey * (cx - x[k]);
        if (w < 0 || (w == 0 && !(ey < 0 || (ey == 0 && ex > 0)))) in = false;  // on an edge: only top or left edges own the texel
      }
      if (in) { ids[(size_t)py * W + px] = (uint32_t)f + 1; if (cnt && cnt[(size_t)py * W + px] < 255) cnt[(size_t)py * W + px]++; }
    }
  }
}

struct VoxGrid {
  BitGrid g; Tensor attrs; int res = 0;
  void build(const int32_t* coords, const float* a, int64_t n, int res_) {
    res = res_; const uint64_t R = (uint64_t)res;
    g.init(R * R * R);
    par_for(n, 1 << 16, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) g.set(((uint64_t)coords[3 * i + 2] * R + (uint64_t)coords[3 * i + 1]) * R + (uint64_t)coords[3 * i]); });
    g.finalize();
    attrs = gbuf<float>(std::max<int64_t>(g.count, 1) * 6);
    float* A = attrs.ptr<float>();
    par_for(n, 1 << 16, [&](int64_t lo, int64_t hi) {
      for (int64_t i = lo; i < hi; i++) { int64_t r = g.find(((uint64_t)coords[3 * i + 2] * R + (uint64_t)coords[3 * i + 1]) * R + (uint64_t)coords[3 * i]); memcpy(A + 6 * r, a + 6 * i, 24); }
    });
  }
};

struct BakeP { int32_t ntri, W, H, res; uint32_t base, p0, p1, p2; };

// img U8 planar [6, H * W], lv I32 [H * W] (0 baked, -1 empty)
inline void bake_texels(const LBVH& bvh, const Tensor& ids, const Tensor& faces, const Tensor& pos, const Tensor& uv, const VoxGrid& vg, int W, int H, Tensor& img, Tensor& lv) {
  const int64_t P = (int64_t)W * H;
  img = gbuf<uint8_t>(P * 6); lv = gbuf<int32_t>(P);
  const int64_t CH = 1 << 21;
  for (int64_t o = 0; o < P; o += CH) {
    BakeP p = {(int32_t)bvh.n, W, H, vg.res, (uint32_t)o, 0, 0, 0};
    run("k_bake", std::min(CH, P - o), {bvh.nodes.buf(), bvh.tris.buf(), ids.buf(), faces.buf(), pos.buf(), uv.buf(), vg.g.bits.buf(), vg.g.rank.buf(), vg.attrs.buf(), img.buf(), lv.buf()}, p);
    if ((o / CH) % 4 == 3) tk::flush();
  }
}

// ---- TELEA ---------------------------------------------------------------------------------------------------------------
struct TeleaStats { int levels = 0; int64_t unknown = 0; double t_levels = 0, t_fill = 0; };

// 4-connected level of every unknown texel (lv == -1 on entry) and the texels grouped by level
inline void telea_levels(int32_t* lv, int W, int H, std::vector<uint32_t>& list, std::vector<int64_t>& off) {
  const int64_t P = (int64_t)W * H;
  const int C = 64; std::vector<std::vector<uint32_t>> part(C);
  par_chunks(H, C, [&](int c, int64_t r0, int64_t r1) {
    for (int64_t y = r0; y < r1; y++) for (int x = 0; x < W; x++) {
      int64_t i = y * W + x;
      if (lv[i] != -1) continue;
      if ((x > 0 && lv[i - 1] == 0) || (x + 1 < W && lv[i + 1] == 0) || (y > 0 && lv[i - W] == 0) || (y + 1 < H && lv[i + W] == 0)) part[c].push_back((uint32_t)i);
    }
  });
  list.clear(); off.assign(1, 0);
  for (int c = 0; c < C; c++) list.insert(list.end(), part[c].begin(), part[c].end());
  for (uint32_t i : list) lv[i] = 1;
  off.push_back((int64_t)list.size());
  (void)P;
  for (int L = 2;; L++) {
    int64_t a = off[off.size() - 2], b = off.back();
    if (a == b) { off.pop_back(); break; }
    for (int64_t j = a; j < b; j++) {
      uint32_t i = list[j]; int x = (int)(i % (uint32_t)W), y = (int)(i / (uint32_t)W);
      if (x > 0 && lv[i - 1] == -1) { lv[i - 1] = L; list.push_back(i - 1); }
      if (x + 1 < W && lv[i + 1] == -1) { lv[i + 1] = L; list.push_back(i + 1); }
      if (y > 0 && lv[i - W] == -1) { lv[i - W] = L; list.push_back(i - W); }
      if (y + 1 < H && lv[i + W] == -1) { lv[i + W] = L; list.push_back(i + W); }
    }
    off.push_back((int64_t)list.size());
  }
}

struct TeleaP { int32_t H, W, L, pad; };

inline void telea(Tensor& lv, Tensor& img, int W, int H, TeleaStats& st) {
  double t0 = now();
  tk::sync();
  std::vector<uint32_t> list; std::vector<int64_t> off;
  telea_levels(lv.ptr<int32_t>(), W, H, list, off);
  st.levels = (int)off.size() - 1; st.unknown = (int64_t)list.size();
  st.t_levels = now() - t0; t0 = now();
  if (!list.empty()) {
    Tensor lt = gbuf<uint32_t>((int64_t)list.size());
    memcpy(lt.ptr<uint32_t>(), list.data(), list.size() * 4);
    for (int L = 1; L <= st.levels; L++) {
      tk::Buf lb = lt.buf(); lb.off += (uint64_t)off[L - 1] * 4;
      TeleaP p = {H, W, L, 0};
      run("k_telea_fill", (uint64_t)(off[L] - off[L - 1]), {lv.buf(), img.buf(), lb}, p);
    }
    tk::sync();
  }
  st.t_fill = now() - t0;
}

// CPU reference of the same wavefront TELEA (tests). img planar [6, P], lv from telea_levels.
inline void telea_cpu(const int32_t* lv, uint8_t* img, int W, int H, const std::vector<uint32_t>& list, const std::vector<int64_t>& off) {
  const int64_t P = (int64_t)W * H;
  for (int L = 1; L < (int)off.size(); L++) {
    auto kn = [&](int r, int c) { if (r < 0 || c < 0 || r >= H || c >= W) return false; int v = lv[(int64_t)r * W + c]; return v >= 0 && v < L; };
    par_for(off[L] - off[L - 1], 1 << 10, [&](int64_t a, int64_t b) {
      for (int64_t q = a; q < b; q++) {
        int i = (int)list[off[L - 1] + q]; int y = i / W, x = i % W; float fL = (float)L;
        bool xr = kn(y, x + 1), xl = kn(y, x - 1), yd = kn(y + 1, x), yu = kn(y - 1, x);
        float gx = xr ? (xl ? ((float)lv[i + 1] - (float)lv[i - 1]) * 0.5f : (float)lv[i + 1] - fL) : (xl ? fL - (float)lv[i - 1] : 0.f);
        float gy = yd ? (yu ? ((float)lv[i + W] - (float)lv[i - W]) * 0.5f : (float)lv[i + W] - fL) : (yu ? fL - (float)lv[i - W] : 0.f);
        float Ia[6] = {0}, Jx[6] = {0}, Jy[6] = {0}; float s3 = 1e-20f, s1 = 1e-20f;
        for (int k = y - 3; k <= y + 3; k++) for (int l = x - 3; l <= x + 3; l++) {
          int dd = (k - y) * (k - y) + (l - x) * (l - x);
          if (dd > 9 || !kn(k, l)) continue;
          int j = k * W + l;
          float ry = (float)(y - k), rx = (float)(x - l), vl = rx * rx + ry * ry;
          float dir = rx * gx + ry * gy; if (std::fabs(dir) <= 0.01f) dir = 1e-6f;
          float w = std::fabs(dir / (vl * std::sqrt(vl) * (1.f + std::fabs((float)lv[j] - fL))));
          bool r1 = kn(k, l + 1), l1 = kn(k, l - 1), d1 = kn(k + 1, l), u1 = kn(k - 1, l);
          int c0 = dd <= 1 ? 6 : 3;
          for (int c = 0; c < c0; c++) {
            int64_t o = (int64_t)c * P + j; float v = (float)img[o];
            float ix = r1 ? (l1 ? ((float)img[o + 1] - (float)img[o - 1]) * 2.f : (float)img[o + 1] - v) : (l1 ? v - (float)img[o - 1] : 0.f);
            float iy = d1 ? (u1 ? ((float)img[o + W] - (float)img[o - W]) * 2.f : (float)img[o + W] - v) : (u1 ? v - (float)img[o - W] : 0.f);
            Ia[c] += w * v; Jx[c] -= w * ix * rx; Jy[c] -= w * iy * ry;
          }
          s3 += w; if (dd <= 1) s1 += w;
        }
        for (int c = 0; c < 6; c++) {
          float s = c < 3 ? s3 : s1;
          float sat = Ia[c] / s + (Jx[c] + Jy[c]) / (std::sqrt(Jx[c] * Jx[c] + Jy[c] * Jy[c]) + 1e-20f);
          img[(int64_t)c * P + i] = (uint8_t)std::min(std::max(std::rint(sat + 0.5f), 0.f), 255.f);
        }
      }
    });
  }
}

// planar [6, P] -> baseColor RGBA [P, 4], metallicRoughness RGB [P, 3]
inline void pack_textures(const Tensor& img, int64_t P, Tensor& base, Tensor& mr) {
  base = gbuf<uint8_t>(P * 4); mr = gbuf<uint8_t>(P * 3);
  uint32_t p = (uint32_t)P;
  run("k_pack", (uint64_t)P, {img.buf(), base.buf(), mr.buf()}, p);
  tk::sync();
}

}  // namespace glb
