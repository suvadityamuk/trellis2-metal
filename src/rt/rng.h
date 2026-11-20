// Bit-exact torch CPU generator: torch.manual_seed(seed) then successive torch.randn(n) calls (n >= 16; aarch64 scalar
// normal_fill path). Must be compiled with -ffp-contract=off.
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace t2 {
struct TorchRng {
  std::mt19937 g;
  explicit TorchRng(uint64_t seed = 0) : g((uint32_t)seed) {}
  void seed(uint64_t s) { g.seed((uint32_t)s); }
  float uni() { return (float)(g() & ((1u << 24) - 1)) * (1.0f / 16777216.0f); }
  static void fill16(float* p) {
    for (int j = 0; j < 8; j++) {
      float u1 = 1 - p[j], u2 = p[j + 8];
      float r = std::sqrt(-2.0f * std::log(u1)), th = (float)(2.0f * M_PI * u2);
      p[j] = r * std::cos(th); p[j + 8] = r * std::sin(th);
    }
  }
  void randn(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = uni();
    for (size_t i = 0; i + 16 <= n; i += 16) fill16(&d[i]);
    if (n % 16) { float* p = &d[n - 16]; for (int i = 0; i < 16; i++) p[i] = uni(); fill16(p); }
  }
};
}  // namespace t2
