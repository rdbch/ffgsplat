#!/usr/bin/env bash
set -euo pipefail

export_cuda128() {
    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
}

if command -v module >/dev/null 2>&1; then
    module load miniforge
    module load arch/h100
    module load cuda/12.8.0
elif command -v export_cuda128 >/dev/null 2>&1; then
    export_cuda128
else
    echo "No CUDA environment setup command found" >&2
    return 1 2>/dev/null || exit 1
fi

export HF_HOME=$WORK/cache_dir 
export TORCH_HOME=$WORK/cache_dir/torch 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 
export MAX_JOBS=6
export NVCC_THREADS=4
# Build rules for the flash-attention submodule.
#
# Builds from this local clone (not from PyPI or a remote git URL), so the
# exact checked-out source is what gets installed. Called by the repo root's
# ./install.sh, or run standalone from this directory:
#
#   cd submodules/flash-attention
#   ./setup.sh
#
# Assumes an already-activated conda env with a matching PyTorch + CUDA
# toolkit already installed. This build compiles CUDA kernels from source
# and can take a long time.

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---no-cache-dir --user --no-build-isolation}"

cd "${MODULE_DIR}"

echo "==> [flash-attention] Building from local source (this can take a while)"
pip install ${PIP_INSTALL_FLAGS} .

echo "==> [flash-attention] Done."
