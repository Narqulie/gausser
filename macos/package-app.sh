#!/usr/bin/env bash
# Create a locally runnable, ad-hoc-signed Gausser Studio developer app.
# This is deliberately not a distributable/notarized bundle. It captures the
# currently active Python environment's gausser-engine entrypoint.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
studio_root="$repo_root/macos/GausserStudio"
output_root=${1:-"$repo_root/macos/build"}
app="$output_root/Gausser Studio.app"

engine=$(python - <<'PY'
from pathlib import Path
import sys
candidate = Path(sys.executable).with_name("gausser-engine")
if not candidate.is_file():
    raise SystemExit(f"gausser-engine is not installed beside {sys.executable}")
print(candidate)
PY
)

bin_path=$(cd "$studio_root" && swift build -c release --show-bin-path)
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin_path/GausserStudio" "$app/Contents/MacOS/GausserStudio"
cp "$engine" "$app/Contents/Resources/gausser-engine"
find "$bin_path" -maxdepth 1 -name '*.bundle' -exec cp -R {} "$app/Contents/Resources/" \;

cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>GausserStudio</string>
  <key>CFBundleIdentifier</key><string>com.narqulie.gausserstudio</string>
  <key>CFBundleName</key><string>Gausser Studio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$app" >/dev/null
printf 'Created developer app: %s\n' "$app"
printf 'Bundled engine launcher: %s\n' "$engine"
