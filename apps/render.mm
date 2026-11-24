// Offscreen Metal rasterizer for pipeline outputs (MeshWithVoxel safetensors written by build/t2 --out).
//   build/render --mesh x.safetensors --out prefix [--views 8] [--size 768] [--elev 20] [--mode color|clay|normal] [--up z|y] [--az0 deg]
//   build/render --glb x.glb --out prefix [...]      textured GLB (build/t2 --glb, or any single-primitive GLB with WebP/PNG textures)
// Writes prefix_000.png ... one per azimuth. Vertex i carries voxel attrs i (base color rgb, metallic, roughness, alpha);
// a GLB is drawn with its base-colour and metallic-roughness textures, same camera and shading.
#include "../src/rt/st.h"
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <simd/simd.h>
using namespace t2;

static const char* kShader = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct U { float4x4 mvp; float4x4 model; float3 eye; uint mode; };
struct VIn { packed_float3 p; };
struct VOut { float4 pos [[position]]; float3 w; float3 col; float2 mr; };
vertex VOut vs(uint vid [[vertex_id]], const device VIn* P [[buffer(0)]], const device float* A [[buffer(1)]], constant U& u [[buffer(2)]]) {
  VOut o; float4 w = u.model * float4(float3(P[vid].p), 1.0);
  o.pos = u.mvp * float4(float3(P[vid].p), 1.0); o.w = w.xyz;
  if (u.mode >= 4) { o.col = float3(A[vid * 2], A[vid * 2 + 1], 0.0); o.mr = float2(0.0); }  // GLB: texture coordinates
  else { o.col = float3(A[vid * 6], A[vid * 6 + 1], A[vid * 6 + 2]); o.mr = float2(A[vid * 6 + 3], A[vid * 6 + 4]); }
  return o;
}
fragment float4 fs(VOut i [[stage_in]], constant U& u [[buffer(2)]], texture2d<float> tb [[texture(0)]], texture2d<float> tm [[texture(1)]], sampler sm [[sampler(0)]]) {
  if (u.mode >= 4) {
    float2 uv = i.col.xy;
    i.col = tb.sample(sm, uv).rgb; float3 m = tm.sample(sm, uv).rgb; i.mr = float2(m.b, m.g);  // glTF: G roughness, B metallic
  }
  uint mode = u.mode & 3;
  float3 n = normalize(cross(dfdx(i.w), dfdy(i.w)));
  float3 v = normalize(u.eye - i.w);
  if (dot(n, v) < 0) n = -n;
  if (mode == 2) return float4(n * 0.5 + 0.5, 1.0);
  float3 base = mode == 1 ? float3(0.62) : pow(clamp(i.col, 0.0, 1.0), float3(2.2));
  float rough = mode == 1 ? 0.6 : clamp(i.mr.y, 0.05, 1.0), metal = mode == 1 ? 0.0 : clamp(i.mr.x, 0.0, 1.0);
  float3 up = float3(0, 1, 0), right = normalize(cross(v, up));
  float3 L1 = normalize(v + 0.6 * up + 0.5 * right), L2 = normalize(v * 0.3 - 0.8 * right + 0.3 * up);
  float3 c = base * 0.28;
  float3 Ls[2] = {L1, L2}; float I[2] = {0.85, 0.35};
  for (int k = 0; k < 2; k++) {
    float nl = max(dot(n, Ls[k]), 0.0);
    float3 h = normalize(Ls[k] + v);
    float sp = pow(max(dot(n, h), 0.0), mix(64.0, 4.0, rough)) * (1.0 - rough) * 0.35;
    c += I[k] * (base * (1.0 - 0.5 * metal) * nl + mix(float3(sp), base * sp * 3.0, metal));
  }
  return float4(pow(clamp(c, 0.0, 1.0), float3(1.0 / 2.2)), 1.0);
}
)MSL";

static void writePNG(const std::string& path, const uint8_t* rgba, int w, int h) {
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGDataProviderRef dp = CGDataProviderCreateWithData(nullptr, rgba, (size_t)w * h * 4, nullptr);
  CGImageRef img = CGImageCreate(w, h, 8, 32, w * 4, cs, (CGBitmapInfo)kCGImageAlphaNoneSkipLast, dp, nullptr, false, kCGRenderingIntentDefault);
  CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, (const UInt8*)path.c_str(), path.size(), false);
  CGImageDestinationRef d = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr);
  CGImageDestinationAddImage(d, img, nullptr); CGImageDestinationFinalize(d);
  CFRelease(d); CFRelease(url); CGImageRelease(img); CGDataProviderRelease(dp); CGColorSpaceRelease(cs);
}
static std::vector<uint8_t> decodeRGBA(const uint8_t* data, size_t n, int& w, int& h) {
  CFDataRef d = CFDataCreate(nullptr, data, (CFIndex)n);
  CGImageSourceRef src = CGImageSourceCreateWithData(d, nullptr);
  CGImageRef img = src ? CGImageSourceCreateImageAtIndex(src, 0, nullptr) : nullptr;
  if (!img) { fprintf(stderr, "render: cannot decode a GLB texture\n"); exit(1); }
  w = (int)CGImageGetWidth(img); h = (int)CGImageGetHeight(img);
  // raw decoded pixels (drawing into a bitmap context would premultiply by the alpha channel, which the OPAQUE material ignores)
  CFDataRef raw = CGDataProviderCopyData(CGImageGetDataProvider(img));
  const uint8_t* p = CFDataGetBytePtr(raw); size_t bpr = CGImageGetBytesPerRow(img), bpp = CGImageGetBitsPerPixel(img) / 8;
  CGImageAlphaInfo ai = CGImageGetAlphaInfo(img);
  if (CGImageGetBitsPerComponent(img) != 8 || (bpp != 3 && bpp != 4)) { fprintf(stderr, "render: unsupported texture pixel format\n"); exit(1); }
  const bool first = ai == kCGImageAlphaFirst || ai == kCGImageAlphaPremultipliedFirst || ai == kCGImageAlphaNoneSkipFirst;
  const bool premul = ai == kCGImageAlphaPremultipliedFirst || ai == kCGImageAlphaPremultipliedLast;
  std::vector<uint8_t> out((size_t)w * h * 4);
  for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
    const uint8_t* q = p + (size_t)y * bpr + (size_t)x * bpp; uint8_t* o = &out[((size_t)y * w + x) * 4];
    int c0 = bpp == 4 && first ? 1 : 0; int a = bpp == 4 ? q[first ? 0 : 3] : 255;
    for (int k = 0; k < 3; k++) o[k] = premul && a > 0 ? (uint8_t)std::min(255, (q[c0 + k] * 255 + a / 2) / a) : q[c0 + k];
    o[3] = (uint8_t)a;
  }
  if (premul) fprintf(stderr, "render: texture decoded with premultiplied alpha, colours un-premultiplied\n");
  CFRelease(raw); CGImageRelease(img); CFRelease(src); CFRelease(d);
  return out;
}
static simd_float4x4 lookAt(simd_float3 eye, simd_float3 at, simd_float3 up) {
  simd_float3 f = simd_normalize(at - eye), s = simd_normalize(simd_cross(f, up)), u = simd_cross(s, f);
  return simd_matrix(simd_make_float4(s.x, u.x, -f.x, 0), simd_make_float4(s.y, u.y, -f.y, 0), simd_make_float4(s.z, u.z, -f.z, 0),
                     simd_make_float4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1));
}
static simd_float4x4 persp(float fovy, float aspect, float n, float f) {
  float t = 1.0f / tanf(fovy * 0.5f);
  return simd_matrix(simd_make_float4(t / aspect, 0, 0, 0), simd_make_float4(0, t, 0, 0), simd_make_float4(0, 0, f / (n - f), -1),
                     simd_make_float4(0, 0, n * f / (n - f), 0));
}

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string mesh, glb, out = "render", mode = "color", up = "z"; int views = 8, size = 768; float elev = 20, az0 = 30;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
      if (a == "--mesh") mesh = nx(); else if (a == "--glb") { glb = nx(); up = "y"; } else if (a == "--out") out = nx(); else if (a == "--views") views = atoi(nx().c_str());
      else if (a == "--size") size = atoi(nx().c_str()); else if (a == "--elev") elev = atof(nx().c_str()); else if (a == "--mode") mode = nx();
      else if (a == "--up") up = nx(); else if (a == "--az0") az0 = atof(nx().c_str());
    }
    if (mesh.empty() && glb.empty()) { fprintf(stderr, "usage: render --mesh x.safetensors | --glb x.glb --out prefix [--views N] [--size S] [--mode color|clay|normal]\n"); return 2; }
    tk::init(nullptr);
    id<MTLDevice> dev = tk::ctx().dev;
    Tensor V, F, A; id<MTLTexture> texBase = nil, texMR = nil;
    if (glb.empty()) {
      STFile f(mesh);
      V = f.get("vertices", tk::F32); F = f.get("faces", tk::I32); A = f.get("attrs", tk::F32);
    } else {
      mesh = glb;
      FILE* fp = fopen(glb.c_str(), "rb"); if (!fp) { fprintf(stderr, "cannot open %s\n", glb.c_str()); return 1; }
      fseek(fp, 0, SEEK_END); size_t sz = (size_t)ftell(fp); fseek(fp, 0, SEEK_SET);
      std::vector<uint8_t> raw(sz); if (fread(raw.data(), 1, sz, fp) != sz) return 1; fclose(fp);
      uint32_t hd[5]; memcpy(hd, raw.data(), 20);
      if (hd[0] != 0x46546C67u || hd[4] != 0x4E4F534Au) { fprintf(stderr, "%s is not a binary glTF\n", glb.c_str()); return 1; }
      Json js = parseJson((const char*)raw.data() + 20, hd[3]);
      const uint8_t* bin = raw.data() + 20 + hd[3] + 8;
      auto view = [&](int64_t bv, size_t& len, size_t& stride) { const Json& b = js["bufferViews"][(size_t)bv]; len = (size_t)b["byteLength"].i(); stride = b.has("byteStride") ? (size_t)b["byteStride"].i() : 0;
        return bin + (b.has("byteOffset") ? b["byteOffset"].i() : 0); };
      auto acc = [&](int64_t ai, int comps, int64_t& count, int& ctype, size_t& stride) { const Json& a = js["accessors"][(size_t)ai]; size_t len; count = a["count"].i(); ctype = (int)a["componentType"].i();
        const uint8_t* p = view(a["bufferView"].i(), len, stride) + (a.has("byteOffset") ? a["byteOffset"].i() : 0);
        size_t es = (ctype == 5126 || ctype == 5125 ? 4 : ctype == 5123 ? 2 : 1) * (size_t)comps; if (!stride) stride = es; return p; };
      const Json& pr = js["meshes"][(size_t)0]["primitives"][(size_t)0];
      int64_t nvv, nuv, ni; int ct; size_t st;
      const uint8_t* pp = acc(pr["attributes"]["POSITION"].i(), 3, nvv, ct, st);
      V = Tensor(tk::F32, {nvv, 3}); A = Tensor(tk::F32, {nvv, 2}); tk::sync();
      for (int64_t i = 0; i < nvv; i++) memcpy(V.ptr<float>() + 3 * i, pp + i * st, 12);
      const uint8_t* pu = acc(pr["attributes"]["TEXCOORD_0"].i(), 2, nuv, ct, st);
      if (ct != 5126 || nuv != nvv) { fprintf(stderr, "render: TEXCOORD_0 must be float\n"); return 1; }
      for (int64_t i = 0; i < nvv; i++) memcpy(A.ptr<float>() + 2 * i, pu + i * st, 8);
      const uint8_t* pi = acc(pr["indices"].i(), 1, ni, ct, st);
      F = Tensor(tk::I32, {ni / 3, 3});tk::sync();
      for (int64_t i = 0; i < ni; i++) F.ptr<int32_t>()[i] = ct == 5125 ? (int32_t)*(const uint32_t*)(pi + i * st) : ct == 5123 ? *(const uint16_t*)(pi + i * st) : pi[i * st];
      const Json& pbr = js["materials"][(size_t)(pr.has("material") ? pr["material"].i() : 0)]["pbrMetallicRoughness"];
      auto loadTex = [&](const char* key) -> id<MTLTexture> {
        int w = 1, h = 1; std::vector<uint8_t> px = {255, 255, 255, 255};
        if (pbr.has(key)) {
          const Json& t = js["textures"][(size_t)pbr[key]["index"].i()];
          int64_t src = t.has("extensions") && t["extensions"].has("EXT_texture_webp") ? t["extensions"]["EXT_texture_webp"]["source"].i() : t["source"].i();
          size_t len, s2; const uint8_t* d = view(js["images"][(size_t)src]["bufferView"].i(), len, s2);
          px = decodeRGBA(d, len, w, h);
        }
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:YES];
        id<MTLTexture> t = [dev newTextureWithDescriptor:td];
        [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:px.data() bytesPerRow:(NSUInteger)w * 4];
        id<MTLCommandBuffer> cb = [tk::ctx().queue commandBuffer]; id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be generateMipmapsForTexture:t]; [be endEncoding]; [cb commit]; [cb waitUntilCompleted];
        return t;
      };
      texBase = loadTex("baseColorTexture"); texMR = loadTex("metallicRoughnessTexture");
    }
    tk::sync();
    const int64_t nv = V.shape[0], nf = F.shape[0];
    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear; sd.magFilter = MTLSamplerMinMagFilterLinear; sd.mipFilter = MTLSamplerMipFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeRepeat; sd.tAddressMode = MTLSamplerAddressModeRepeat; sd.maxAnisotropy = 8;
    id<MTLSamplerState> samp = [dev newSamplerStateWithDescriptor:sd];
    // model matrix: recentre on the bbox, map the data up-axis to +Y
    simd_float3 lo = simd_make_float3(1e9, 1e9, 1e9), hi = -lo; const float* vp = V.ptr<float>();
    for (int64_t i = 0; i < nv; i++) { simd_float3 p = simd_make_float3(vp[i * 3], vp[i * 3 + 1], vp[i * 3 + 2]); lo = simd_min(lo, p); hi = simd_max(hi, p); }
    simd_float3 ctr = (lo + hi) * 0.5f; float sc = 1.0f / simd_length(hi - lo);
    simd_float4x4 M = matrix_identity_float4x4;
    if (up == "z") M = simd_matrix(simd_make_float4(sc, 0, 0, 0), simd_make_float4(0, 0, -sc, 0), simd_make_float4(0, sc, 0, 0), simd_make_float4(0, 0, 0, 1));
    else M = simd_matrix(simd_make_float4(sc, 0, 0, 0), simd_make_float4(0, sc, 0, 0), simd_make_float4(0, 0, sc, 0), simd_make_float4(0, 0, 0, 1));
    simd_float4 tc = simd_mul(M, simd_make_float4(ctr.x, ctr.y, ctr.z, 0));
    M.columns[3] = simd_make_float4(-tc.x, -tc.y, -tc.z, 1);

    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kShader] options:nil error:&err];
    if (!lib) { fprintf(stderr, "shader: %s\n", err.localizedDescription.UTF8String); return 1; }
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm; pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float; pd.rasterSampleCount = 4;
    id<MTLRenderPipelineState> rps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!rps) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionLess; dd.depthWriteEnabled = YES;
    id<MTLDepthStencilState> dss = [dev newDepthStencilStateWithDescriptor:dd];
    auto tex = [&](MTLPixelFormat pf, int samples) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:size height:size mipmapped:NO];
      td.usage = MTLTextureUsageRenderTarget; td.storageMode = samples > 1 ? MTLStorageModePrivate : MTLStorageModeShared;
      if (samples > 1) { td.textureType = MTLTextureType2DMultisample; td.sampleCount = samples; }
      return [dev newTextureWithDescriptor:td];
    };
    id<MTLTexture> msaa = tex(MTLPixelFormatRGBA8Unorm, 4), depth = tex(MTLPixelFormatDepth32Float, 4), resolve = tex(MTLPixelFormatRGBA8Unorm, 1);
    std::vector<uint8_t> px((size_t)size * size * 4);
    const uint32_t md = (mode == "clay" ? 1 : mode == "normal" ? 2 : 0) | (texBase ? 4 : 0);
    for (int k = 0; k < views; k++) {
      float az = (az0 + 360.0f * k / views) * (float)M_PI / 180, el = elev * (float)M_PI / 180, dist = 1.75f;
      simd_float3 eye = simd_make_float3(dist * cosf(el) * sinf(az), dist * sinf(el), dist * cosf(el) * cosf(az));
      struct { simd_float4x4 mvp, model; simd_float3 eye; uint32_t mode; } u;
      u.model = M; u.mvp = simd_mul(persp(34.0f * (float)M_PI / 180, 1.0f, 0.1f, 10.0f), simd_mul(lookAt(eye, simd_make_float3(0, 0, 0), simd_make_float3(0, 1, 0)), M));
      u.eye = eye; u.mode = md;
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor new];
      rp.colorAttachments[0].texture = msaa; rp.colorAttachments[0].resolveTexture = resolve;
      rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      rp.colorAttachments[0].clearColor = MTLClearColorMake(1, 1, 1, 1);
      rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = MTLLoadActionClear; rp.depthAttachment.clearDepth = 1.0;
      rp.depthAttachment.storeAction = MTLStoreActionDontCare;
      id<MTLCommandBuffer> cb = [tk::ctx().queue commandBuffer];
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      [e setRenderPipelineState:rps]; [e setDepthStencilState:dss]; [e setCullMode:MTLCullModeNone];
      [e setVertexBuffer:V.st->buf offset:V.off atIndex:0]; [e setVertexBuffer:A.st->buf offset:A.off atIndex:1];
      if (texBase) { [e setFragmentTexture:texBase atIndex:0]; [e setFragmentTexture:texMR atIndex:1]; [e setFragmentSamplerState:samp atIndex:0]; }
      [e setVertexBytes:&u length:sizeof u atIndex:2]; [e setFragmentBytes:&u length:sizeof u atIndex:2];
      [e drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:(NSUInteger)nf * 3 indexType:MTLIndexTypeUInt32 indexBuffer:F.st->buf indexBufferOffset:F.off];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      [resolve getBytes:px.data() bytesPerRow:size * 4 fromRegion:MTLRegionMake2D(0, 0, size, size) mipmapLevel:0];
      char name[512]; snprintf(name, sizeof name, "%s_%03d.png", out.c_str(), k);
      writePNG(name, px.data(), size, size);
    }
    printf("rendered %d views of %s (%lld verts, %lld faces)\n", views, mesh.c_str(), (long long)nv, (long long)nf);
  }
}
