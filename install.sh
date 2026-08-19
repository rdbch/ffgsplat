#!/usr/bin/env bash
set -euo pipefail

# Root installer for ffgsplat and its submodules.
#
# Run inside an already-activated conda environment:
#   conda create -n ffgsplat python=3.10
#   conda activate ffgsplat
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

export PIP_INSTALL_FLAGS="--no-cache-dir --user"

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
