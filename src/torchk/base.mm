#include "base.h"

namespace tk {

static Ctx g;
static bool g_unwaited = false;
Ctx& ctx() { return g; }

void init(const char* path) {
  g.dev = MTLCreateSystemDefaultDevice();
  g.queue = [g.dev newCommandQueue];
  if (path) {
    NSData* d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!d) { fprintf(stderr, "tk: cannot read %s\n", path); exit(1); }
    dispatch_data_t dd = dispatch_data_create(d.bytes, d.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* err = nil;
    g.lib = [g.dev newLibraryWithData:dd error:&err];
    if (!g.lib) { fprintf(stderr, "tk: newLibraryWithData failed: %s\n", err.localizedDescription.UTF8String); exit(1); }
  }
}

id<MTLCommandBuffer> cmdbuf() {
  if (!g.cb) g.cb = [MPSCommandBuffer commandBufferFromCommandQueue:g.queue];  // like torch's MPSStream
  return g.cb;
}
id<MTLComputeCommandEncoder> enc() {
  if (!g.ce) g.ce = [cmdbuf() computeCommandEncoder];
  return g.ce;
}
void endEncoder() {
  if (g.ce) { [g.ce endEncoding]; g.ce = nil; }
}
void flush() {
  endEncoder();
  if (g.cb) { [g.cb commit]; g.cb = nil; g_unwaited = true; }
}
void sync() {
  endEncoder();
  if (g.cb) {
    id<MTLCommandBuffer> cb = g.cb;
    g.cb = nil;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "tk: command buffer error: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
  } else if (g_unwaited) {
    // make sure previously flushed work is done
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
  }
  g_unwaited = false;
}
bool hasFn(const std::string& name) {
  static NSSet<NSString*>* names = nil;
  if (!names) names = [NSSet setWithArray:g.lib.functionNames];
  return [names containsObject:[NSString stringWithUTF8String:name.c_str()]];
}
id<MTLComputePipelineState> pso(const std::string& name) {
  auto it = g.psos.find(name);
  if (it != g.psos.end()) return it->second;
  id<MTLFunction> fn = [g.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!fn) { fprintf(stderr, "tk: kernel '%s' not found in torch metallib\n", name.c_str()); abort(); }
  NSError* err = nil;
  id<MTLComputePipelineState> p = [g.dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) { fprintf(stderr, "tk: pso '%s' failed: %s\n", name.c_str(), err.localizedDescription.UTF8String); abort(); }
  g.psos[name] = p;
  return p;
}

}  // namespace tk
