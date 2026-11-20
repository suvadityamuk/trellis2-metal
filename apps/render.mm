// Offscreen Metal rasterizer for pipeline outputs (MeshWithVoxel safetensors written by build/t2 --out).
//   build/render --mesh x.safetensors --out prefix [--views 8] [--size 768] [--elev 20] [--mode color|clay|normal] [--up z|y] [--az0 deg]
// Writes prefix_000.png ... one per azimuth. Vertex i carries voxel attrs i (base color rgb, metallic, roughness, alpha).
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
  o.col = float3(A[vid * 6], A[vid * 6 + 1], A[vid * 6 + 2]); o.mr = float2(A[vid * 6 + 3], A[vid * 6 + 4]);
  return o;
}
fragment float4 fs(VOut i [[stage_in]], constant U& u [[buffer(2)]]) {
  float3 n = normalize(cross(dfdx(i.w), dfdy(i.w)));
  float3 v = normalize(u.eye - i.w);
  if (dot(n, v) < 0) n = -n;
  if (u.mode == 2) return float4(n * 0.5 + 0.5, 1.0);
  float3 base = u.mode == 1 ? float3(0.62) : pow(clamp(i.col, 0.0, 1.0), float3(2.2));
  float rough = u.mode == 1 ? 0.6 : clamp(i.mr.y, 0.05, 1.0), metal = u.mode == 1 ? 0.0 : clamp(i.mr.x, 0.0, 1.0);
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
    std::string mesh, out = "render", mode = "color", up = "z"; int views = 8, size = 768; float elev = 20, az0 = 30;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
      if (a == "--mesh") mesh = nx(); else if (a == "--out") out = nx(); else if (a == "--views") views = atoi(nx().c_str());
      else if (a == "--size") size = atoi(nx().c_str()); else if (a == "--elev") elev = atof(nx().c_str()); else if (a == "--mode") mode = nx();
      else if (a == "--up") up = nx(); else if (a == "--az0") az0 = atof(nx().c_str());
    }
    if (mesh.empty()) { fprintf(stderr, "usage: render --mesh x.safetensors --out prefix [--views N] [--size S] [--mode color|clay|normal]\n"); return 2; }
    tk::init(nullptr);
    id<MTLDevice> dev = tk::ctx().dev;
    STFile f(mesh);
    Tensor V = f.get("vertices", tk::F32), F = f.get("faces", tk::I32), A = f.get("attrs", tk::F32);
    tk::sync();
    const int64_t nv = V.shape[0], nf = F.shape[0];
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
    const uint32_t md = mode == "clay" ? 1 : mode == "normal" ? 2 : 0;
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
