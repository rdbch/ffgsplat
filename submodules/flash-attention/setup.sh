#!/usr/bin/env bash
set -euo pipefail

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
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---no-cache-dir --user}"

cd "${MODULE_DIR}"

echo "==> [flash-attention] Building from local source (this can take a while)"
pip install ${PIP_INSTALL_FLAGS} .

echo "==> [flash-attention] Done."
