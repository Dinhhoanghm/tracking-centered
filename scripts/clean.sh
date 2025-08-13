#!/usr/bin/env bash
set -euo pipefail

rm -rf .venv node_modules .cache .npm .python-version __pycache__ .pytest_cache .mypy_cache .ruff_cache
rm -rf .data logs output dist build coverage .parcel-cache .next .turbo

find . -type d -name "__pycache__" -prune -exec rm -rf {} +

echo "Cleaned local environment and caches." 