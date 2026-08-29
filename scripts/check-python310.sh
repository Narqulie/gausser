#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${PYTHON310_IMAGE:-python:3.10-slim}

docker run --rm \
  --volume "$repo_root:/source:ro" \
  "$image" \
  sh -c '
    set -eu
    mkdir /workspace
    tar -C /source \
      --exclude=.git \
      --exclude=.venv \
      --exclude=build \
      --exclude=dist \
      --exclude=.pytest_cache \
      --exclude=macos/GausserStudio/.build \
      -cf - . | tar -C /workspace -xf -
    cd /workspace
    python -m pip install --quiet --disable-pip-version-check ".[test,quality]" build
    python -m pytest -q -p no:cacheprovider
    ruff check src tests monitor_and_run.py
    mypy src
    python -m build --sdist --wheel
    python -c "import gausser.scene, gausser.manifest"
  '
