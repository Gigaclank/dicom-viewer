"""dmgbuild settings for the DICOM Viewer release.

Usage:
    dmgbuild -s scripts/dmg-settings.py "DICOM Viewer" dist/DICOM-Viewer.dmg
"""
import os

application = os.environ.get(
    "DMG_APP_PATH", "dist/DICOM Viewer.app"
)
appname = "DICOM Viewer"

# Layout — drag-the-app-onto-Applications pattern.
format = "UDBZ"
size = None  # auto-size to contents
files = [application]
symlinks = {"Applications": "/Applications"}
icon_locations = {
    f"{appname}.app": (140, 120),
    "Applications": (380, 120),
}
window_rect = ((100, 100), (520, 320))
icon_size = 96
text_size = 12
background = None
