#!/usr/bin/env bash
# Install gausser and every tool it drives on an Apple Silicon Mac.
# Safe to rerun: finished steps are skipped or are no-ops.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAM3_REV=f6e51f59500a87c576c2df2323ce56b9fd7a12de
BRUSH_REV=1388f74c6fe0236f68ee4915564bf00e9d2e3747
SUPERSPLAT_DIR="${SUPERSPLAT_DIR:-$HOME/Developer/supersplat}"

step() { printf '\n== %s\n' "$*"; }

step "Homebrew: COLMAP, ffmpeg, Python 3.13, Node, Rust"
brew install colmap ffmpeg python@3.13 node rust

step "splat-transform"
npm install -g @playcanvas/splat-transform

step "Python environment ($ROOT/.venv)"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$(brew --prefix python@3.13)/bin/python3.13" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" install -e "$ROOT[sam,msplat,dev]" sharp-frames \
  "git+https://github.com/facebookresearch/sam3.git@$SAM3_REV"

step "SAM3 macOS patch"
SAM3_DIR="$("$ROOT/.venv/bin/python" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("sam3").origin))')"
if patch -p1 -R --dry-run -s -f -d "$SAM3_DIR" < "$ROOT/patches/sam3-macos.patch" > /dev/null; then
  echo "already applied"
else
  patch -p1 -N -d "$SAM3_DIR" < "$ROOT/patches/sam3-macos.patch"
fi

step "Brush @ ${BRUSH_REV:0:8} (first build takes a few minutes)"
cargo install --locked --git https://github.com/ArthurBrussee/brush --rev "$BRUSH_REV" brush-app --bin brush

step "SuperSplat viewer ($SUPERSPLAT_DIR)"
if [ ! -f "$SUPERSPLAT_DIR/dist/index.html" ]; then
  [ -d "$SUPERSPLAT_DIR/.git" ] || git clone https://github.com/playcanvas/supersplat "$SUPERSPLAT_DIR"
  (cd "$SUPERSPLAT_DIR" && npm ci && npm run build)
else
  echo "already built"
fi

step "Commands in ~/.local/bin"
mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/.venv/bin/gausser" "$HOME/.local/bin/gausser"
ln -sf "$ROOT/.venv/bin/sharp-frames" "$HOME/.local/bin/sharp-frames"

step "SAM3 checkpoint"
if [ -f "$HOME/.cache/gausser/sam3.pt" ]; then
  echo "found ~/.cache/gausser/sam3.pt"
else
  echo "Missing. Copy sam3.pt to ~/.cache/gausser/sam3.pt, or run: $ROOT/.venv/bin/hf auth login"
fi

step "Check"
"$ROOT/.venv/bin/gausser" status -s "$ROOT" | sed -n '/^Tools/,$p'
