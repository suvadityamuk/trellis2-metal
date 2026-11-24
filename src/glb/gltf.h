// Direct binary glTF 2.0 writer: one mesh primitive (POSITION, NORMAL, TEXCOORD_0, uint32 indices), one
// metallic-roughness material with base-colour and metallic-roughness textures embedded as WebP (EXT_texture_webp,
// what the official trimesh export writes) or PNG. The two textures are encoded on separate threads.
#pragma once
#include "util.h"
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#include "../../third_party/libwebp/src/webp/encode.h"

namespace glb {

struct TexOpts { bool png = false; float quality = 80.f; int method = 0; int alpha_quality = 100; };

inline std::vector<uint8_t> encode_webp(const uint8_t* px, int w, int h, int ch, const TexOpts& o) {
  WebPConfig cfg; WebPPicture pic; WebPMemoryWriter wr;
  if (!WebPConfigInit(&cfg) || !WebPPictureInit(&pic)) { fprintf(stderr, "glb: libwebp version mismatch\n"); exit(1); }
  cfg.quality = o.quality; cfg.method = o.method; cfg.thread_level = 1; cfg.alpha_quality = o.alpha_quality;
  pic.width = w; pic.height = h; pic.use_argb = 0;
  int ok = ch == 4 ? WebPPictureImportRGBA(&pic, px, w * 4) : WebPPictureImportRGB(&pic, px, w * 3);
  WebPMemoryWriterInit(&wr); pic.writer = WebPMemoryWrite; pic.custom_ptr = &wr;
  ok = ok && WebPEncode(&cfg, &pic);
  if (!ok) { fprintf(stderr, "glb: WebP encode failed (%d)\n", (int)pic.error_code); exit(1); }
  std::vector<uint8_t> out(wr.mem, wr.mem + wr.size);
  WebPPictureFree(&pic); WebPMemoryWriterClear(&wr);
  return out;
}

inline std::vector<uint8_t> encode_png(const uint8_t* px, int w, int h, int ch) {
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGDataProviderRef dp = CGDataProviderCreateWithData(nullptr, px, (size_t)w * h * ch, nullptr);
  CGImageRef img = CGImageCreate(w, h, 8, 8 * ch, (size_t)w * ch, cs, ch == 4 ? (CGBitmapInfo)kCGImageAlphaLast : (CGBitmapInfo)kCGImageAlphaNone, dp, nullptr, false, kCGRenderingIntentDefault);
  CFMutableDataRef data = CFDataCreateMutable(nullptr, 0);
  CGImageDestinationRef d = CGImageDestinationCreateWithData(data, CFSTR("public.png"), 1, nullptr);
  CGImageDestinationAddImage(d, img, nullptr);
  bool ok = CGImageDestinationFinalize(d);
  std::vector<uint8_t> out;
  if (ok) out.assign(CFDataGetBytePtr(data), CFDataGetBytePtr(data) + CFDataGetLength(data));
  CFRelease(d); CFRelease(data); CGImageRelease(img); CGDataProviderRelease(dp); CGColorSpaceRelease(cs);
  if (!ok) { fprintf(stderr, "glb: PNG encode failed\n"); exit(1); }
  return out;
}

struct GlbStats { double t_encode = 0, t_write = 0; int64_t bytes = 0, base_bytes = 0, mr_bytes = 0; };

// pos / nrm [nv, 3] (already in glTF axes), uv [nv, 2], idx [nf, 3]; base RGBA [H, W, 4], mr RGB [H, W, 3]
inline void write_glb(const std::string& path, const float* pos, const float* nrm, const float* uv, int64_t nv, const int32_t* idx, int64_t nf,
                      const uint8_t* base, const uint8_t* mr, int W, int H, const TexOpts& to, GlbStats& st) {
  double t0 = now();
  std::vector<uint8_t> ib, im;
  {
    std::thread t([&] { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0); im = to.png ? encode_png(mr, W, H, 3) : encode_webp(mr, W, H, 3, to); });
    ib = to.png ? encode_png(base, W, H, 4) : encode_webp(base, W, H, 4, to);
    t.join();
  }
  st.t_encode = now() - t0; t0 = now();
  st.base_bytes = (int64_t)ib.size(); st.mr_bytes = (int64_t)im.size();
  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (int64_t i = 0; i < nv; i++) for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], pos[3 * i + k]); hi[k] = std::max(hi[k], pos[3 * i + k]); }
  auto pad4 = [](size_t n) { return (n + 3) & ~(size_t)3; };
  size_t o_idx = 0, n_idx = (size_t)nf * 12, o_pos = pad4(o_idx + n_idx), n_pos = (size_t)nv * 12, o_nrm = pad4(o_pos + n_pos), n_nrm = (size_t)nv * 12,
         o_uv = pad4(o_nrm + n_nrm), n_uv = (size_t)nv * 8, o_ib = pad4(o_uv + n_uv), n_ib = ib.size(), o_im = pad4(o_ib + n_ib), n_im = im.size(), total = pad4(o_im + n_im);
  const char* mime = to.png ? "image/png" : "image/webp";
  char buf[4096];
  std::string j = "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis2-metal\"},\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0,\"name\":\"mesh\"}],";
  if (!to.png) j += "\"extensionsUsed\":[\"EXT_texture_webp\"],\"extensionsRequired\":[\"EXT_texture_webp\"],";
  j += "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":1,\"NORMAL\":2,\"TEXCOORD_0\":3},\"indices\":0,\"material\":0,\"mode\":4}]}],";
  j += "\"materials\":[{\"name\":\"material\",\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],\"metallicFactor\":1,\"roughnessFactor\":1,"
       "\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1}},\"alphaMode\":\"OPAQUE\",\"doubleSided\":false}],";
  if (to.png) j += "\"textures\":[{\"sampler\":0,\"source\":0},{\"sampler\":0,\"source\":1}],";
  else j += "\"textures\":[{\"sampler\":0,\"extensions\":{\"EXT_texture_webp\":{\"source\":0}}},{\"sampler\":0,\"extensions\":{\"EXT_texture_webp\":{\"source\":1}}}],";
  j += "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}],";
  snprintf(buf, sizeof buf, "\"images\":[{\"bufferView\":4,\"mimeType\":\"%s\"},{\"bufferView\":5,\"mimeType\":\"%s\"}],", mime, mime); j += buf;
  snprintf(buf, sizeof buf,
           "\"accessors\":[{\"bufferView\":0,\"componentType\":5125,\"count\":%lld,\"type\":\"SCALAR\"},"
           "{\"bufferView\":1,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
           "{\"bufferView\":2,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC3\"},{\"bufferView\":3,\"componentType\":5126,\"count\":%lld,\"type\":\"VEC2\"}],",
           (long long)nf * 3, (long long)nv, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], (long long)nv, (long long)nv);
  j += buf;
  snprintf(buf, sizeof buf,
           "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34963},{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},"
           "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},"
           "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu},{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}],\"buffers\":[{\"byteLength\":%zu}]}",
           o_idx, n_idx, o_pos, n_pos, o_nrm, n_nrm, o_uv, n_uv, o_ib, n_ib, o_im, n_im, total);
  j += buf;
  while (j.size() % 4) j += ' ';
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "glb: cannot write %s\n", path.c_str()); exit(1); }
  uint32_t hdr[3] = {0x46546C67u, 2u, (uint32_t)(12 + 8 + j.size() + 8 + total)};
  uint32_t jh[2] = {(uint32_t)j.size(), 0x4E4F534Au}, bh[2] = {(uint32_t)total, 0x004E4942u};
  fwrite(hdr, 4, 3, f); fwrite(jh, 4, 2, f); fwrite(j.data(), 1, j.size(), f); fwrite(bh, 4, 2, f);
  static const uint8_t zero[4] = {0, 0, 0, 0};
  size_t at = 0;
  auto put = [&](size_t off, const void* p, size_t n) { if (off > at) fwrite(zero, 1, off - at, f); fwrite(p, 1, n, f); at = off + n; };
  put(o_idx, idx, n_idx); put(o_pos, pos, n_pos); put(o_nrm, nrm, n_nrm); put(o_uv, uv, n_uv); put(o_ib, ib.data(), n_ib); put(o_im, im.data(), n_im);
  if (total > at) fwrite(zero, 1, total - at, f);
  fclose(f);
  st.bytes = (int64_t)hdr[2];
  st.t_write = now() - t0;
}

}  // namespace glb
