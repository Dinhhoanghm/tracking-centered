#!/usr/bin/env bash
set -euo pipefail

# Activate python venv
if [ ! -d .venv ]; then
  echo "No .venv found. Run scripts/setup.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Ensure local cache directories exist
mkdir -p .cache/pip .cache/uv .cache/npm .cache/torch .cache/huggingface .cache/ultralytics

# Local caches
export PIP_CACHE_DIR="$(pwd)/.cache/pip"
export UV_CACHE_DIR="$(pwd)/.cache/uv"
export npm_config_cache="$(pwd)/.cache/npm"
export TORCH_HOME="$(pwd)/.cache/torch"
export HF_HOME="$(pwd)/.cache/huggingface"
export ULTRALYTICS_CACHE_DIR="$(pwd)/.cache/ultralytics"
export XDG_CACHE_HOME="$(pwd)/.cache"

# Ensure Node CLI uses project venv python
export VENV_PY="$(pwd)/.venv/bin/python"

echo "Environment ready. Python venv active. npm cache: $npm_config_cache, torch cache: $TORCH_HOME" 