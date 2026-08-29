#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}

rm -rf "$repo_root/build" "$repo_root/dist"
cd "$repo_root"
"$python_bin" -m build --sdist --wheel

wheels=(dist/*.whl)
sdists=(dist/*.tar.gz)
if [[ ${#wheels[@]} -ne 1 || ! -f ${wheels[0]} ]]; then
  echo "expected exactly one wheel in dist/" >&2
  exit 1
fi
if [[ ${#sdists[@]} -ne 1 || ! -f ${sdists[0]} ]]; then
  echo "expected exactly one source distribution in dist/" >&2
  exit 1
fi

"$python_bin" - "${wheels[0]}" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

wheel = Path(sys.argv[1])
with ZipFile(wheel) as archive:
    names = set(archive.namelist())
    if "gausser/launcher.py" not in names:
        raise SystemExit(f"{wheel} does not contain gausser/launcher.py")
    if "gausser/interactive.py" not in names:
        raise SystemExit(f"{wheel} does not contain gausser/interactive.py")
    if "gausser/tui.py" in names:
        raise SystemExit(f"{wheel} contains retired gausser/tui.py")
    entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
    entry_point_text = archive.read(entry_points).decode("utf-8")
    if "gausser = gausser.launcher:main" not in entry_point_text:
        raise SystemExit(f"{wheel} does not expose the launcher entry point")
print(f"Verified package contents: {wheel}")
PY

"$python_bin" - "${sdists[0]}" <<'PY'
from pathlib import Path
import sys
import tarfile

sdist = Path(sys.argv[1])
with tarfile.open(sdist, "r:gz") as archive:
    names = set(archive.getnames())
    if not any(name.endswith("/src/gausser/launcher.py") for name in names):
        raise SystemExit(f"{sdist} does not contain gausser/launcher.py")
    if not any(name.endswith("/src/gausser/interactive.py") for name in names):
        raise SystemExit(f"{sdist} does not contain gausser/interactive.py")
    if any(name.endswith("/src/gausser/tui.py") for name in names):
        raise SystemExit(f"{sdist} contains retired gausser/tui.py")
print(f"Verified package contents: {sdist}")
PY

smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/gausser-package-smoke.XXXXXX")"
trap 'rm -rf "$smoke_root"' EXIT
commands=(gausser sharpframes sammer colmapper brusher msplatter lodder gausser-engine)
interactive_commands=(gausser sharpframes sammer colmapper brusher msplatter lodder)

smoke_install() {
  local artifact="$1"
  local label="$2"
  local environment="$smoke_root/$label"
  local output="$smoke_root/$label.out"

  "$python_bin" -m venv "$environment"
  "$environment/bin/python" -m pip install --quiet "$artifact"

  for command in "${commands[@]}"; do
    "$environment/bin/$command" --help > /dev/null
  done
  for command in "${interactive_commands[@]}"; do
    if "$environment/bin/$command" < /dev/null > "$output" 2>&1; then
      echo "$label $command unexpectedly accepted bare non-TTY input" >&2
      exit 1
    fi
    if ! grep -q "requires a terminal" "$output"; then
      echo "$label $command did not report its non-TTY contract" >&2
      cat "$output" >&2
      exit 1
    fi
  done
  "$environment/bin/gausser-engine" < /dev/null > /dev/null
  echo "Verified installed $label: $artifact"
}

smoke_install "${wheels[0]}" wheel
smoke_install "${sdists[0]}" sdist
