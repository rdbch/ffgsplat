#!/usr/bin/env bash
set -euo pipefail

module load miniforge
module load arch/h100
module load cuda/12.8.0
conda activate ffgs_311

export HF_HOME=$WORK/cache_dir 
export TORCH_HOME=$WORK/cache_dir/torch 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 

# Build rules for the LitePT submodule.
#
# Assumes an already-activated conda env with PyTorch + CUDA toolkit and
# this submodule's Python dependencies already installed (both handled by
# the repo root's ./install.sh, from ../../pyproject.toml). Called by the
# root installer with no args, or run standalone from this directory:
#
#   cd submodules/LitePT
#   ./setup.sh [--with-extras] [--cuda-arch=ARCH]
#
# Options:
#   --with-extras      also build pointops (evaluator) and pointgroup_ops
#                       (PointGroup instance segmentation). Off by default,
#                       they're only needed for those two use cases.
#   --cuda-arch=ARCH   gencode arch for pointrope's CUDA kernels, e.g.
#                       80 (A100), 86 (RTX 30xx), 90 (H100). Default: 90.
#                       See https://developer.nvidia.com/cuda-gpus
#
# pointgroup_ops additionally needs google-sparsehash from conda, which pip
# cannot install:
#   conda install -c bioconda google-sparsehash
#
# flash-attention is built by submodules/flash-attention/setup.sh, called
# by the root install.sh -- not by this script.

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On a GPU cluster, home/project quotas are often small and shared envs are
# frequently not writable by the user -- so every pip call disables the
# build cache and installs to the user site-packages.
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---no-cache-dir --user --no-build-isolation}"

WITH_EXTRAS=0
CUDA_ARCH="90"

for arg in "$@"; do
  case "${arg}" in
    --with-extras) WITH_EXTRAS=1 ;;
    --cuda-arch=*) CUDA_ARCH="${arg#*=}" ;;
    *) ;; # ignore unknown flags so the root installer can forward its own
  esac
done

cd "${MODULE_DIR}"

# Python dependencies (requirements.txt, spconv) are installed from the repo
# root's pyproject.toml by install.sh, not here. Run this script standalone
# only after those are installed (see repo root readme.md).

echo "==> [LitePT] Building pointrope CUDA extension (arch sm_${CUDA_ARCH})"
sed -i.bak -E "s/arch=compute_[0-9]+,code=sm_[0-9]+/arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}/" libs/pointrope/setup.py
rm -f libs/pointrope/setup.py.bak
( cd libs/pointrope && pip install ${PIP_INSTALL_FLAGS} . )


echo "==> [LitePT] Building pointops CUDA extension (evaluator)"
( cd libs/pointops && pip install ${PIP_INSTALL_FLAGS} . )

echo "==> [LitePT] Building pointgroup_ops CUDA extension (PointGroup instance segmentation)"
echo "    Requires google-sparsehash: conda install -c bioconda google-sparsehash"
( cd libs/pointgroup_ops && pip install ${PIP_INSTALL_FLAGS} . )


echo "==> [LitePT] Done."
