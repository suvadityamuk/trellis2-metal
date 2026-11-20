"""cumesh is CUDA-only; the Mac reference skips hole filling / simplification (documented in README)."""
class CuMesh:
    def __init__(self, *a, **k): raise NotImplementedError("cumesh is CUDA-only")
