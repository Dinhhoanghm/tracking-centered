#!/usr/bin/env bash
set -euo pipefail

# Create local cache directories
mkdir -p .cache/npm .cache/pip .cache/uv

# Python: create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip and install packages
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  PIP_CACHE_DIR="$(pwd)/.cache/pip" PIP_DEFAULT_TIMEOUT=100 python -m pip install -r requirements.txt
fi

# Node: install deps locally with cache
if [ -f package.json ]; then
  npm ci --prefer-offline --cache ./.cache/npm || npm install --prefer-offline --cache ./.cache/npm
fi

echo "Setup complete. Activate venv with: source .venv/bin/activate" 