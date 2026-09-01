#!/usr/bin/env bash
set -euo pipefail

module load miniforge
module load arch/h100
module load cuda/12.8.0
conda activate ffs_311

export HF_HOME=$WORK/cache_dir 
export TORCH_HOME=$WORK/cache_dir/torch 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 



# Root installer for ffgsplat and its submodules.
#
# Run inside an already-activated conda environment:
#   conda create -n ffgs_311 python=3.11
#   conda activate ffgs_311
#   ./install.sh [options]
#
# Any options are forwarded, unchanged, to every submodule's setup.sh --
# see each submodule's setup.sh / README for the flags it understands
# (e.g. submodules/LitePT/setup.sh --with-extras --cuda-arch=80).
#
# All pip installs use --no-cache-dir --user, since GPU cluster home/scratch
# quotas are tight and shared conda envs are frequently read-only.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULES_DIR="${REPO_ROOT}/submodules"

export PIP_INSTALL_FLAGS="--no-cache-dir --user --no-build-isolation"

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "error: no active conda environment detected. Run 'conda activate <env>' first." >&2
  exit 1
fi

echo "==> Installing ffgsplat into conda env '${CONDA_DEFAULT_ENV}'"

if [ -f "${REPO_ROOT}/pyproject.toml" ]; then
  echo "==> Installing root project dependencies (pyproject.toml)"
  pip install ${PIP_INSTALL_FLAGS} -e "${REPO_ROOT}"
fi

# Each submodule owns its own build rules via <submodule>/setup.sh. To add a
# new submodule to the install, just drop a setup.sh next to it -- no changes
# needed here.
for module_dir in "${SUBMODULES_DIR}"/*/; do
  module_name="$(basename "${module_dir}")"
  setup_script="${module_dir}setup.sh"
  if [ -f "${setup_script}" ]; then
    echo "==> Building submodule: ${module_name}"
    ( cd "${module_dir}" && bash setup.sh "$@" )
  else
    echo "==> Skipping ${module_name} (no setup.sh found)"
  fi
done

echo "==> Done."
