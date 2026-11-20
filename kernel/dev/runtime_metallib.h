// Development stand-in for the header kernel-builder generates from the compiled metallib:
// compiles kernels.metal at runtime with the options the builder uses (Metal 4.0, default math mode).
#pragma once
#include <Metal/Metal.h>

namespace EMBEDDED_METALLIB_NAMESPACE {
inline id<MTLLibrary> createLibrary(id<MTLDevice> device, NSError** error = nullptr) {
  NSString* src = [NSString stringWithContentsOfFile:@T2_METAL_SOURCE encoding:NSUTF8StringEncoding error:error];
  if (!src) return nil;
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_0;
  opts.mathMode = T2_FAST_MATH ? MTLMathModeFast : MTLMathModeSafe;
  return [device newLibraryWithSource:src options:opts error:error];
}
}  // namespace EMBEDDED_METALLIB_NAMESPACE
