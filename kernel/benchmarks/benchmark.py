import math

import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark

BF = torch.bfloat16


class Trellis2MetalBenchmark(Benchmark):
    """Shapes of the TRELLIS.2 1024 flow transformer: 16384 tokens, 1536 channels, MLP ratio 4."""

    seed = 0
    tokens, channels = 16384, 1536

    def _linear(self, out_features: int):
        n, c = self.tokens, self.channels
        self.x = torch.randn(n, c, device=self.device).to(BF)
        # small weights keep the outputs well inside the runner's absolute tolerance at bfloat16 precision
        self.weight = (torch.randn(out_features, c, device=self.device) * (0.1 / math.sqrt(c))).to(BF)
        self.bias = torch.randn(out_features, device=self.device) * 0.1
        self.out = torch.empty(n, out_features, device=self.device, dtype=BF)

    def setup_gemm_qkv(self):
        self._linear(3 * self.channels)

    def benchmark_gemm_qkv(self):
        self.kernel.gemm(self.x, self.weight, self.out)

    def verify_gemm_qkv(self) -> torch.Tensor:
        return self.x @ self.weight.T

    def setup_gemm_bias_gelu_mlp(self):
        self._linear(4 * self.channels)

    def benchmark_gemm_bias_gelu_mlp(self):
        self.kernel.gemm_bias_gelu(self.x, self.weight, self.bias, self.out)

    def verify_gemm_bias_gelu_mlp(self) -> torch.Tensor:
        return F.gelu((self.x @ self.weight.T).float() + self.bias, approximate="tanh").to(BF)

    def setup_resid_ln(self):
        n, c = self.tokens, self.channels
        self.stream = torch.randn(n, c, device=self.device)
        self.branch = torch.randn(n, c, device=self.device).to(BF)
        # a small layer-norm weight keeps the outputs inside the runner's absolute tolerance at bfloat16 precision
        self.vec = torch.stack(
            [torch.zeros(c), torch.ones(c), torch.full((c,), 0.25), torch.zeros(c)]
        ).to(self.device)
        # benchmark without the in-place residual update so that every iteration sees the same input
        self.out = torch.empty(n, c, device=self.device, dtype=BF)

    def benchmark_resid_ln(self):
        self.kernel.resid_ln(self.stream, self.vec, out=self.out, residual=False)

    def verify_resid_ln(self) -> torch.Tensor:
        return (F.layer_norm(self.stream, (self.channels,), eps=1e-6) * 0.25).to(BF)
