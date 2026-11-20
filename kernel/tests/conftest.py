import os
import sys

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    return torch.device("mps")


@pytest.fixture(scope="session")
def kernel():
    # T2_KERNEL_DEV_PATH points at a directory holding a locally built `trellis2_metal` package.
    dev = os.environ.get("T2_KERNEL_DEV_PATH")
    if dev:
        sys.path.insert(0, dev)
        import trellis2_metal

        return trellis2_metal
    import kernels

    # T2_KERNEL_LOCAL_PATH points at a kernel-builder output directory (the one holding the variant directories).
    local = os.environ.get("T2_KERNEL_LOCAL_PATH")
    if local:
        from pathlib import Path

        return kernels.get_local_kernel(Path(local), trust_remote_code=True)
    return kernels.get_kernel("suvadityamuk/trellis2-metal", version=0, trust_remote_code=True)
