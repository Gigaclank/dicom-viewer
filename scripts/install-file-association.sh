#!/usr/bin/env bash
# Register .dcmproj files to open in DICOM Viewer (Linux user-local install).
#
# Run once from the project root:
#   bash scripts/install-file-association.sh
#
# Uninstall later with:
#   xdg-mime uninstall --mode user scripts/dicom-viewer.xml
#   rm ~/.local/share/applications/dicom-viewer.desktop
#   update-desktop-database ~/.local/share/applications

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "error: $PY not found. Create the venv with 'python3 -m venv .venv' first." >&2
    exit 1
fi

# Substitute the venv python path into the .desktop file template so the
# installed desktop entry can be double-clicked.
DESKTOP_SRC="$ROOT/scripts/dicom-viewer.desktop"
DESKTOP_DST="$HOME/.local/share/applications/dicom-viewer.desktop"
mkdir -p "$(dirname "$DESKTOP_DST")"
sed "s|__EXEC__|$PY -m dicom_viewer|g" "$DESKTOP_SRC" > "$DESKTOP_DST"
chmod 644 "$DESKTOP_DST"

# Register the .dcmproj MIME type, then bind .dcmproj files to the desktop entry.
xdg-mime install --mode user --novendor "$ROOT/scripts/dicom-viewer.xml"
xdg-mime default dicom-viewer.desktop application/x-dicom-viewer-project
update-desktop-database "$HOME/.local/share/applications" || true

echo "Installed:"
echo "  $DESKTOP_DST"
echo "  MIME type application/x-dicom-viewer-project (from scripts/dicom-viewer.xml)"
echo
echo "Test:  xdg-open /path/to/some.dcmproj"
