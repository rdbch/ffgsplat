#!/usr/bin/env bash
set -euo pipefail

module load miniforge
module load arch/h100
module load cuda/12.8.0
conda activate mgs_311

export HF_HOME=$WORK/cache_dir 
export TORCH_HOME=$WORK/cache_dir/torch 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 
export MAX_JOBS=4
export NVCC_THREADS=2
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
