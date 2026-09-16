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


pip install --no-build-isolation --user --no-cache-dir -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation --user --no-cache-dir -e submodules/gaussian-splatting/submodules/simple-knn
pip install --no-build-isolation --user --no-cache-dir -e submodules/RoMa
