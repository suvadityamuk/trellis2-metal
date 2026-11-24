// Image loading + Trellis2ImageTo3DPipeline.preprocess_image (ref/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127)
// — parity build. Everything here runs on the CPU in the reference too (PIL + numpy), so it is plain CPU code that
// mirrors PIL's integer pipelines and numpy's float32 op order exactly:
//   decode        ImageIO (CGImageSource). The undecoded-colour-space provider bytes of an RGBA PNG / WebP-with-alpha
//                 are straight (non-premultiplied) RGBA and byte-identical to PIL's decode (verified on the example
//                 images); anything ImageIO does not hand back as 8-bit straight RGBA is rejected, never converted.
//   resize        PIL Image.resize(LANCZOS): 8bpc fixed-point two-pass resampling (libImaging/Resample.c); RGBA goes
//                 through premultiplied RGBa and back (libImaging/Convert.c rgba2rgbA / rgbA2rgba).
//   crop          PIL Image.crop with a float box: each edge is round()-ed half-to-even; outside pixels are zero.
//   composite     (u8 -> float32) / 255 ; rgb * alpha ; * 255 ; astype(uint8) (truncation) — numpy float32.
// rembg (BiRefNet) is stubbed in the reference, so the input must carry a real alpha channel.
#pragma once
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace t2 {

struct Image8 {
  int w = 0, h = 0, c = 0;  // c = 4 (RGBA, straight alpha) or 3 (RGB); row-major HWC
  std::vector<uint8_t> d;
  const uint8_t* px(int x, int y) const { return &d[((size_t)y * w + x) * c]; }
};

[[noreturn]] inline void imageFail(const std::string& m) { fprintf(stderr, "t2 image: %s\n", m.c_str()); abort(); }

// PIL Image.open(path) for the inputs the reference pipeline accepts without background removal: mode RGBA.
inline Image8 load_image(const std::string& path) {
  Image8 im;
  @autoreleasepool {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!src) imageFail("cannot open " + path);
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    if (!img) imageFail("cannot decode " + path);
    size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img), rb = CGImageGetBytesPerRow(img);
    CGImageAlphaInfo ai = CGImageGetAlphaInfo(img);
    CGColorSpaceRef cs = CGImageGetColorSpace(img);
    bool rgb = cs && CGColorSpaceGetModel(cs) == kCGColorSpaceModelRGB;
    if (ai == kCGImageAlphaNone || ai == kCGImageAlphaNoneSkipLast || ai == kCGImageAlphaNoneSkipFirst || !rgb)
      imageFail(path + ": no RGBA alpha channel (opaque / grayscale / palette input needs background removal, which the reference stubs)");
    if (ai != kCGImageAlphaLast || CGImageGetBitsPerComponent(img) != 8 || CGImageGetBitsPerPixel(img) != 32 ||
        (CGImageGetBitmapInfo(img) & kCGBitmapByteOrderMask) != kCGBitmapByteOrderDefault)
      imageFail(path + ": decoder did not return 8-bit straight RGBA (premultiplied / 16-bit sources are not supported)");
    CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(img));
    if (!data || (size_t)CFDataGetLength(data) < rb * (h - 1) + w * 4) imageFail(path + ": short pixel data");
    const uint8_t* p = CFDataGetBytePtr(data);
    im.w = (int)w; im.h = (int)h; im.c = 4; im.d.resize(w * h * 4);
    for (size_t y = 0; y < h; y++) memcpy(&im.d[y * w * 4], p + y * rb, w * 4);
    CFRelease(data);
    CGImageRelease(img);
  }
  return im;
}

// ---- PIL libImaging/Resample.c, 8bpc path, LANCZOS filter (support 3.0) -------------------------------------------------
namespace pil {
static constexpr int kPrecisionBits = 32 - 8 - 2;
inline double sincFilter(double x) { if (x == 0.0) return 1.0; x = x * M_PI; return sin(x) / x; }
inline double lanczosFilter(double x) { return (-3.0 <= x && x < 3.0) ? sincFilter(x) * sincFilter(x / 3) : 0.0; }
// precompute_coeffs + normalize_coeffs_8bpc
inline int coeffs(int inSize, float in0, float in1, int outSize, std::vector<int>& bounds, std::vector<int>& kk) {
  double scale = (double)(in1 - in0) / outSize, filterscale = scale < 1.0 ? 1.0 : scale, support = 3.0 * filterscale;
  int ksize = (int)ceil(support) * 2 + 1;
  std::vector<double> k((size_t)outSize * ksize);
  bounds.resize((size_t)outSize * 2);
  for (int xx = 0; xx < outSize; xx++) {
    double center = in0 + (xx + 0.5) * scale, ww = 0.0, ss = 1.0 / filterscale;
    int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    double* kp = &k[(size_t)xx * ksize];
    int x;
    for (x = 0; x < xmax; x++) { double w = lanczosFilter((x + xmin - center + 0.5) * ss); kp[x] = w; ww += w; }
    for (x = 0; x < xmax; x++) if (ww != 0.0) kp[x] /= ww;
    for (; x < ksize; x++) kp[x] = 0;
    bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
  }
  kk.resize(k.size());
  for (size_t i = 0; i < k.size(); i++)
    kk[i] = k[i] < 0 ? (int)(-0.5 + k[i] * (1 << kPrecisionBits)) : (int)(0.5 + k[i] * (1 << kPrecisionBits));
  return ksize;
}
inline uint8_t clip8(int v) { v >>= kPrecisionBits; return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
}  // namespace pil

// ImagingResample(im, W, H, LANCZOS, box = whole image) on independent 8-bit channels (no alpha handling)
inline Image8 resize_lanczos(const Image8& im, int W, int H) {
  using namespace pil;
  const int C = im.c;
  if (im.w == W && im.h == H) return im;
  std::vector<int> bh, kh, bv, kv;
  int ksh = coeffs(im.w, 0, (float)im.w, W, bh, kh), ksv = coeffs(im.h, 0, (float)im.h, H, bv, kv);
  bool need_h = W != im.w, need_v = H != im.h;
  int yfirst = bv[0], ylast = bv[(size_t)H * 2 - 2] + bv[(size_t)H * 2 - 1];
  Image8 cur = im;
  if (need_h) {  // horizontal pass over the rows the vertical pass will read
    for (int i = 0; i < H; i++) bv[i * 2] -= yfirst;
    Image8 t; t.w = W; t.h = ylast - yfirst; t.c = C; t.d.resize((size_t)t.w * t.h * C);
    for (int y = 0; y < t.h; y++)
      for (int x = 0; x < W; x++) {
        int ss[4] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
        const int* k = &kh[(size_t)x * ksh];
        const uint8_t* s = &cur.d[((size_t)(y + yfirst) * cur.w + bh[x * 2]) * C];
        for (int j = 0; j < bh[x * 2 + 1]; j++) for (int c = 0; c < C; c++) ss[c] += s[j * C + c] * k[j];
        for (int c = 0; c < C; c++) t.d[((size_t)y * W + x) * C + c] = clip8(ss[c]);
      }
    cur = std::move(t);
  }
  if (need_v) {
    Image8 t; t.w = W; t.h = H; t.c = C; t.d.resize((size_t)W * H * C);
    for (int y = 0; y < H; y++) {
      const int* k = &kv[(size_t)y * ksv];
      for (int x = 0; x < W; x++) {
        int ss[4] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
        for (int j = 0; j < bv[y * 2 + 1]; j++) {
          const uint8_t* s = &cur.d[((size_t)(j + bv[y * 2]) * W + x) * C];
          for (int c = 0; c < C; c++) ss[c] += s[c] * k[j];
        }
        for (int c = 0; c < C; c++) t.d[((size_t)y * W + x) * C + c] = clip8(ss[c]);
      }
    }
    cur = std::move(t);
  }
  return cur;
}

// PIL Image.resize((W, H), LANCZOS): RGBA is resampled as premultiplied RGBa (Image.py resize), RGB directly.
inline Image8 pil_resize(const Image8& im, int W, int H) {
  if (im.c != 4) return resize_lanczos(im, W, H);
  if (im.w == W && im.h == H) return im;  // Image.resize returns self.copy() before any conversion
  Image8 p = im;
  for (size_t i = 0; i < p.d.size(); i += 4)  // Convert.c rgba2rgbA: MULDIV255
    for (int c = 0; c < 3; c++) { unsigned t = (unsigned)p.d[i + c] * p.d[i + 3] + 128; p.d[i + c] = (uint8_t)(((t >> 8) + t) >> 8); }
  Image8 o = resize_lanczos(p, W, H);
  for (size_t i = 0; i < o.d.size(); i += 4) {  // Convert.c rgbA2rgba: CLIP8(255 * c / a) unless a is 0 or 255
    unsigned a = o.d[i + 3];
    if (a != 255 && a != 0) for (int c = 0; c < 3; c++) o.d[i + c] = (uint8_t)std::min(255u, 255u * o.d[i + c] / a);
  }
  return o;
}

// preprocess_image (alpha branch). in: RGBA straight alpha. out: RGB, premultiplied on black, cropped to the object.
inline Image8 preprocess_image(const Image8& in0) {
  if (in0.c != 4) imageFail("preprocess_image: RGBA input required");
  bool has_alpha = false;
  for (size_t i = 3; i < in0.d.size(); i += 4) if (in0.d[i] != 255) { has_alpha = true; break; }
  if (!has_alpha) imageFail("preprocess_image: fully opaque input needs background removal (stubbed in the reference)");
  int max_size = std::max(in0.w, in0.h);
  double scale = std::min(1.0, 1024.0 / max_size);
  const Image8 in = scale < 1 ? pil_resize(in0, (int)(in0.w * scale), (int)(in0.h * scale)) : in0;
  // bbox = argwhere(alpha > 0.8 * 255)  (0.8 * 255 = 204.0 in float64)
  int x0 = in.w, y0 = in.h, x1 = -1, y1 = -1;
  for (int y = 0; y < in.h; y++)
    for (int x = 0; x < in.w; x++)
      if ((double)in.px(x, y)[3] > 0.8 * 255) { x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
  if (x1 < 0) imageFail("preprocess_image: no pixel with alpha > 0.8");
  double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;
  int size = std::max(x1 - x0, y1 - y0), hs = size / 2;
  // Image.crop: box edges go through round() (half to even); pixels outside the image are zero
  int bx0 = (int)std::nearbyint(cx - hs), by0 = (int)std::nearbyint(cy - hs), bx1 = (int)std::nearbyint(cx + hs), by1 = (int)std::nearbyint(cy + hs);
  Image8 o; o.w = bx1 - bx0; o.h = by1 - by0; o.c = 3; o.d.assign((size_t)o.w * o.h * 3, 0);
  for (int y = 0; y < o.h; y++)
    for (int x = 0; x < o.w; x++) {
      int sx = x + bx0, sy = y + by0;
      if (sx < 0 || sy < 0 || sx >= in.w || sy >= in.h) continue;
      const uint8_t* s = in.px(sx, sy);
      float a = (float)s[3] / 255.f;
      for (int c = 0; c < 3; c++) o.d[((size_t)y * o.w + x) * 3 + c] = (uint8_t)(((float)s[c] / 255.f * a) * 255.f);
    }
  return o;
}

}  // namespace t2
