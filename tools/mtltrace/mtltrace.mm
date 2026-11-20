// In-process Metal dispatch tracer (offline tool). Loaded into the reference Python process with ctypes; swizzles the
// concrete compute-encoder / device / MPS classes and logs every kernel dispatch torch-MPS issues: function name,
// bound buffers (id, offset), inline constants (hex), grid and threadgroup sizes. Used to learn exactly which bundled
// kernels the official pipeline runs so the parity build can issue the same dispatches.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <map>
#include <mutex>

static FILE* g_out = nullptr; static bool g_on = false; static std::mutex g_mu;
static std::unordered_map<void*, std::string> g_pso;       // pso -> function name
static std::unordered_map<void*, int> g_buf; static int g_nbuf = 0;
struct EncState { std::string fn; std::map<int, std::string> args; };
static std::unordered_map<void*, EncState> g_enc;

extern "C" void mtltrace_start(const char* path) { std::lock_guard<std::mutex> l(g_mu); if (g_out) fclose(g_out); g_out = fopen(path, "w"); g_on = true; }
extern "C" void mtltrace_stop() { std::lock_guard<std::mutex> l(g_mu); g_on = false; if (g_out) { fclose(g_out); g_out = nullptr; } }
extern "C" void mtltrace_mark(const char* s) { std::lock_guard<std::mutex> l(g_mu); if (g_on && g_out) fprintf(g_out, "# %s\n", s); }
static int bufid(id b) { auto it = g_buf.find((__bridge void*)b); if (it != g_buf.end()) return it->second; return g_buf[(__bridge void*)b] = g_nbuf++; }

template <class F> static IMP swap(Class c, SEL s, F blk, IMP* orig) {
  Method m = class_getInstanceMethod(c, s); if (!m) { fprintf(stderr, "mtltrace: no method %s on %s\n", sel_getName(s), class_getName(c)); return nullptr; }
  IMP n = imp_implementationWithBlock(blk); *orig = method_getImplementation(m);
  if (!class_addMethod(c, s, n, method_getTypeEncoding(m))) method_setImplementation(m, n);
  return n;
}
static IMP o_newpso, o_newpso2, o_setpso, o_setbuf, o_setbufs, o_setbytes, o_disp, o_dispT, o_mm, o_grun, o_genc;

static void hook_encoder(Class c) {
  static std::unordered_map<void*, bool> done; if (done[(__bridge void*)c]) return; done[(__bridge void*)c] = true;
  swap(c, @selector(setComputePipelineState:), ^(id self, id pso) {
    { std::lock_guard<std::mutex> l(g_mu); auto& e = g_enc[(__bridge void*)self]; auto it = g_pso.find((__bridge void*)pso); e.fn = it == g_pso.end() ? "?" : it->second; e.args.clear(); }
    ((void (*)(id, SEL, id))o_setpso)(self, @selector(setComputePipelineState:), pso); }, &o_setpso);
  swap(c, @selector(setBuffer:offset:atIndex:), ^(id self, id buf, NSUInteger off, NSUInteger idx) {
    { std::lock_guard<std::mutex> l(g_mu); char t[96]; if (buf) snprintf(t, 96, "b%d+%lu(%lu)", bufid(buf), (unsigned long)off, (unsigned long)[(id<MTLBuffer>)buf length]); else snprintf(t, 96, "nil");
      g_enc[(__bridge void*)self].args[(int)idx] = t; }
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))o_setbuf)(self, @selector(setBuffer:offset:atIndex:), buf, off, idx); }, &o_setbuf);
  swap(c, @selector(setBytes:length:atIndex:), ^(id self, const void* p, NSUInteger len, NSUInteger idx) {
    { std::lock_guard<std::mutex> l(g_mu); std::string t = "x"; char h[4]; for (NSUInteger i = 0; i < len && i < 160; i++) { snprintf(h, 4, "%02x", ((const uint8_t*)p)[i]); t += h; }
      g_enc[(__bridge void*)self].args[(int)idx] = t; }
    ((void (*)(id, SEL, const void*, NSUInteger, NSUInteger))o_setbytes)(self, @selector(setBytes:length:atIndex:), p, len, idx); }, &o_setbytes);
  swap(c, @selector(dispatchThreadgroups:threadsPerThreadgroup:), ^(id self, MTLSize g, MTLSize t) {
    { std::lock_guard<std::mutex> l(g_mu); if (g_on && g_out) { auto& e = g_enc[(__bridge void*)self]; fprintf(g_out, "K %s grid=%lu,%lu,%lu tg=%lu,%lu,%lu", e.fn.c_str(), (unsigned long)g.width, (unsigned long)g.height, (unsigned long)g.depth, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth);
        for (auto& [i, s] : e.args) fprintf(g_out, " %d:%s", i, s.c_str()); fprintf(g_out, "\n"); } }
    ((void (*)(id, SEL, MTLSize, MTLSize))o_disp)(self, @selector(dispatchThreadgroups:threadsPerThreadgroup:), g, t); }, &o_disp);
  swap(c, @selector(dispatchThreads:threadsPerThreadgroup:), ^(id self, MTLSize g, MTLSize t) {
    { std::lock_guard<std::mutex> l(g_mu); if (g_on && g_out) { auto& e = g_enc[(__bridge void*)self]; fprintf(g_out, "T %s threads=%lu,%lu,%lu tg=%lu,%lu,%lu", e.fn.c_str(), (unsigned long)g.width, (unsigned long)g.height, (unsigned long)g.depth, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth);
        for (auto& [i, s] : e.args) fprintf(g_out, " %d:%s", i, s.c_str()); fprintf(g_out, "\n"); } }
    ((void (*)(id, SEL, MTLSize, MTLSize))o_dispT)(self, @selector(dispatchThreads:threadsPerThreadgroup:), g, t); }, &o_dispT);
}

static std::string ndshape(MPSNDArray* a) { std::string s = "["; for (NSUInteger i = 0; i < a.numberOfDimensions; i++) s += std::to_string([a lengthOfDimension:i]) + (i + 1 < a.numberOfDimensions ? "," : ""); return s + "]dt" + std::to_string((unsigned)a.dataType); }

extern "C" void mtltrace_install() {
  static bool done = false; if (done) return; done = true;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  Class dc = object_getClass(dev);
  swap(dc, @selector(newComputePipelineStateWithFunction:error:), ^id(id self, id<MTLFunction> f, NSError** err) {
    id p = ((id (*)(id, SEL, id, NSError**))o_newpso)(self, @selector(newComputePipelineStateWithFunction:error:), f, err);
    std::lock_guard<std::mutex> l(g_mu); if (p) g_pso[(__bridge void*)p] = [[f name] UTF8String]; return p; }, &o_newpso);
  id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; hook_encoder(object_getClass(e)); [e endEncoding]; [cb commit];
  swap([MPSNDArrayMatrixMultiplication class], @selector(encodeToCommandEncoder:commandBuffer:sourceArrays:destinationArray:), ^(id self, id enc, id cb2, NSArray<MPSNDArray*>* src, MPSNDArray* dst) {
    { std::lock_guard<std::mutex> l(g_mu); if (g_on && g_out) { fprintf(g_out, "M MPSNDArrayMatrixMultiplication"); for (MPSNDArray* a in src) fprintf(g_out, " %s", ndshape(a).c_str()); fprintf(g_out, " -> %s\n", ndshape(dst).c_str()); } }
    ((void (*)(id, SEL, id, id, id, id))o_mm)(self, @selector(encodeToCommandEncoder:commandBuffer:sourceArrays:destinationArray:), enc, cb2, src, dst); }, &o_mm);
  swap([MPSGraph class], @selector(encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:), ^(id self, id cb2, NSDictionary* feeds, id ops, NSDictionary* res, id ed) {
    { std::lock_guard<std::mutex> l(g_mu); if (g_on && g_out) { fprintf(g_out, "G MPSGraph feeds=%lu results=%lu ops:", (unsigned long)feeds.count, (unsigned long)res.count);
        for (MPSGraphTensor* t in res) fprintf(g_out, " %s", [[[t operation] name] UTF8String]); fprintf(g_out, "\n"); } }
    ((void (*)(id, SEL, id, id, id, id, id))o_genc)(self, @selector(encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:), cb2, feeds, ops, res, ed); }, &o_genc);
}
