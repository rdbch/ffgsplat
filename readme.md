# Very Fast GS
===

## Installation

This repo is organized as a root project plus a set of vendored submodules under `submodules/`, each with its own build requirements (some ship custom CUDA extensions). A single script drives the whole install.

Intended for a conda environment on a GPU cluster; every `pip install` uses `--no-cache-dir --user` (tight home/scratch quotas, shared envs are often read-only).

```shell
conda create -n ffgs_311 python=3.11
conda activate ffgs_311
# install pytorch first, matching your cluster's CUDA version
pip install --no-cache-dir --user torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

./install.sh
```

Root project dependencies live in [`pyproject.toml`](./pyproject.toml) — add packages to `[project.dependencies]` there. `install.sh` installs them with `pip install --no-cache-dir --user -e .`, then walks every directory under `submodules/` and runs its `setup.sh` if one exists, so each submodule owns its own build rules. Extra flags are forwarded to every submodule's `setup.sh`, e.g.:

```shell
./install.sh --with-extras --cuda-arch=80   # A100; see submodules/LitePT/setup.sh for its flags
```

To install or rebuild a single submodule, run its `setup.sh` directly from its own directory (see that submodule's README).
