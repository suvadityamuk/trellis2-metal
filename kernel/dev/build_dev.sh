#!/bin/sh
# Development-only build of the extension without Nix/kernel-builder (never publish its output):
# compiles the Metal source at runtime instead of embedding a metallib.
#   PYTHON=/path/to/python-with-torch dev/build_dev.sh [outdir]   ->   <outdir>/trellis2_metal/
# then: T2_KERNEL_DEV_PATH=<outdir> python -m pytest tests
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
OUT=${1:-build-dev}
PKG="$OUT/trellis2_metal"
mkdir -p "$PKG"
TORCH=$($PY -c 'import torch, os; print(os.path.dirname(torch.__file__))')
PYINC=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
BUILDER_REG=${REGISTRATION_H:-dev/registration.h}
FLAGS="-std=c++20 -w -O2 -fPIC -fobjc-arc -mmacosx-version-min=26.0 -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$PYINC -Idev -Itorch-ext
  -DTORCH_EXTENSION_NAME=_trellis2_metal_dev -DMETAL_KERNEL -DEMBEDDED_METALLIB_NAMESPACE=trellis2_metal_dev_metal
  -DT2_METAL_SOURCE=\"$(pwd)/trellis2_metal/kernels.metal\" -DT2_FAST_MATH=${T2_FAST_MATH:-1}"
clang++ $FLAGS -DEMBEDDED_METALLIB_HEADER='"runtime_metallib.h"' -c trellis2_metal/ops.mm -o "$OUT/ops.o"
clang++ $FLAGS -c torch-ext/torch_binding.cpp -o "$OUT/torch_binding.o"
clang++ -shared -undefined dynamic_lookup "$OUT/ops.o" "$OUT/torch_binding.o" -L"$TORCH/lib" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH/lib" \
  -framework Metal -framework Foundation -o "$PKG/_trellis2_metal_dev.abi3.so"
cp torch-ext/trellis2_metal/__init__.py "$PKG/"
printf 'import torch\nfrom . import _trellis2_metal_dev\nops = torch.ops._trellis2_metal_dev\n' > "$PKG/_ops.py"
echo "built $PKG"
