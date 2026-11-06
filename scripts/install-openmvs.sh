#!/usr/bin/env bash
# Build OpenMVS (dense point clouds and meshes, Metal patch match on Apple Silicon) for `gausser dense` and `gausser mesh`.
# Safe to rerun: skips the build when the pinned commit is already built. Called by install-macos.sh.
set -euo pipefail

OPENMVS_REV=14ad2571aac650bfabf54296d31aabed7a19fdde  # develop, 2026-11-06; Metal backend merged in PR #1279
VCPKG_REV=9eec002df1cd109f8f104cd966a763daf9751489    # the vcpkg commit OpenMVS's CI pins for this revision
DIR="${GAUSSER_OPENMVS_DIR:-$HOME/.local/share/gausser/openmvs}"
TOOLS=(InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh)

checkout() {  # checkout URL DIR REV
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" cat-file -e "$3^{commit}" 2>/dev/null || git -C "$2" fetch -q origin
  git -C "$2" checkout -q "$3"
}

if [ "$(cat "$DIR/built-rev" 2>/dev/null)" = "$OPENMVS_REV" ]; then
  echo "OpenMVS ${OPENMVS_REV:0:7} already built"
else
  brew install cmake ninja automake autoconf autoconf-archive libtool pkgconf
  mkdir -p "$DIR"
  checkout https://github.com/microsoft/vcpkg "$DIR/vcpkg" "$VCPKG_REV"
  [ -x "$DIR/vcpkg/vcpkg" ] || "$DIR/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
  checkout https://github.com/cdcseacave/openMVS "$DIR/src" "$OPENMVS_REV"
  rm -f "$DIR/build/CMakeCache.txt"  # a failed configure caches its dependency misses
  # The vcpkg toolchain builds OpenMVS's dependencies (OpenCV, CGAL, Ceres, ...) from its manifest.
  cmake -S "$DIR/src" -B "$DIR/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DVCPKG_ROOT="$DIR/vcpkg" -DCMAKE_TOOLCHAIN_FILE="$DIR/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_USE_BREAKPAD=OFF -DOpenMVS_BUILD_VIEWER=OFF
  cmake --build "$DIR/build" --target "${TOOLS[@]}"
  echo "$OPENMVS_REV" > "$DIR/built-rev"
fi

mkdir -p "$HOME/.local/bin"
for tool in "${TOOLS[@]}"; do
  ln -sf "$DIR/build/bin/$tool" "$HOME/.local/bin/$tool"
done
"$HOME/.local/bin/DensifyPointCloud" --help 2>&1 | grep -m1 "OpenMVS"
