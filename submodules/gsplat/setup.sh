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

# Build rules for the gsplat submodule.
#
# Builds from this local clone (not from PyPI), so the exact checked-out
# source -- including any local kernel changes -- is what gets installed.
# Called by the repo root's ./install.sh, or run standalone from this
# directory:
#
#   cd submodules/gsplat
#   ./setup.sh [--cuda-arch=ARCH] [--editable]
#
# Options:
#   --cuda-arch=ARCH   arch to compile the CUDA kernels for, e.g. 80 (A100),
#                       86 (RTX 30xx), 90 (H100). Default: 90. Passed to the
#                       build as TORCH_CUDA_ARCH_LIST (9.0 for 90); without
#                       it torch would probe the *login* node, which usually
#                       has no GPU. See https://developer.nvidia.com/cuda-gpus
#   --editable         install with `pip install -e .` so Python-side edits
#                       are picked up without reinstalling (the compiled
#                       extension still needs a rebuild after CUDA changes).
#
# Assumes an already-activated conda env with a matching PyTorch + CUDA
# toolkit, and gsplat's Python dependencies (ninja, jaxtyping, rich, numpy)
# already installed -- the latter live in the repo root's pyproject.toml and
# are installed by ./install.sh before this script runs. This build compiles
# CUDA kernels from source and takes a while.

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On a GPU cluster, home/project quotas are often small and shared envs are
# frequently not writable by the user -- so every pip call disables the
# build cache and installs to the user site-packages.
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---no-cache-dir --user --no-build-isolation}"

CUDA_ARCH="90"
EDITABLE=""

for arg in "$@"; do
  case "${arg}" in
    --cuda-arch=*) CUDA_ARCH="${arg#*=}" ;;
    --editable) EDITABLE="-e" ;;
    *) ;; # ignore unknown flags so the root installer can forward its own
  esac
done

cd "${MODULE_DIR}"

# gsplat vendors glm as a nested git submodule; its headers are a hard
# requirement of the CUDA build (setup.py adds them to include_dirs).
GLM_DIR="gsplat/cuda/csrc/third_party/glm"
if [ ! -f "${GLM_DIR}/glm/glm.hpp" ]; then
  echo "error: ${GLM_DIR} is empty -- fetch it with:" >&2
  echo "         (cd ${MODULE_DIR} && git submodule update --init --recursive)" >&2
  exit 1
fi

# Keep nvcc from saturating a shared login/compute node.
export MAX_JOBS=8
export NVCC_THREADS=2
# torch wants a dotted arch list ("9.0"), the other setup.sh scripts take a
# plain gencode number ("90") -- convert, unless the caller set it already.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${CUDA_ARCH%?}.${CUDA_ARCH: -1}}"

echo "==> [gsplat] Building CUDA extension from local source (arch sm_${CUDA_ARCH}, this can take a while)"
pip install ${PIP_INSTALL_FLAGS} ${EDITABLE} .

echo "==> [gsplat] Done."