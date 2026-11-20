// Minimal Metal context for replaying torch-MPS kernel dispatches (parity build).
// The kernels themselves come from the metallib that ships inside the reference
// torch build (build/torchlibs/lib_01.metallib == section metal_40), so the GPU
// code that runs is byte-identical to what torch runs.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace tk {

// c10::ScalarType numbering (what torch kernels expect in "type" arguments).
enum DT : uint32_t { U8 = 0, I8 = 1, I16 = 2, I32 = 3, I64 = 4, F16 = 5, F32 = 6, F64 = 7, C64 = 9, BOOL = 11, BF16 = 15 };
inline uint32_t dtSize(DT d) {
  switch (d) {
    case U8: case I8: case BOOL: return 1;
    case I16: case F16: case BF16: return 2;
    case I32: case F32: return 4;
    default: return 8;
  }
}
// scalarToMetalTypeString
inline const char* dtName(DT d) {
  switch (d) {
    case U8: return "uchar"; case I8: return "char"; case I16: return "short"; case I32: return "int";
    case I64: return "long"; case F16: return "half"; case F32: return "float"; case C64: return "float2";
    case BOOL: return "bool"; case BF16: return "bfloat"; default: return "?";
  }
}

struct Buf {
  id<MTLBuffer> buf = nil;
  uint64_t off = 0;  // bytes
};

struct Ctx {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;  // torch's precompiled kernel library
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> ce = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos;
  uint64_t ndispatch = 0;
};
Ctx& ctx();
void init(const char* torch_metallib_path);
// Current compute encoder on the current command buffer (created lazily). All torch kernels
// and MPS kernels are encoded onto this one encoder, like torch's MPSStream does.
id<MTLComputeCommandEncoder> enc();
id<MTLCommandBuffer> cmdbuf();  // current command buffer (created lazily; does not end the encoder)
void endEncoder();              // end compute encoder (keeps command buffer open)
void flush();                   // commit without waiting
void sync();                    // commit and wait until completed
id<MTLComputePipelineState> pso(const std::string& name);
bool hasFn(const std::string& name);

inline void setBuf(id<MTLComputeCommandEncoder> e, const Buf& b, unsigned idx) { [e setBuffer:b.buf offset:b.off atIndex:idx]; }
template <class T> inline void setVal(id<MTLComputeCommandEncoder> e, const T& v, unsigned idx) { [e setBytes:&v length:sizeof(T) atIndex:idx]; }
inline void setRaw(id<MTLComputeCommandEncoder> e, const void* p, size_t n, unsigned idx) { [e setBytes:p length:n atIndex:idx]; }

}  // namespace tk
