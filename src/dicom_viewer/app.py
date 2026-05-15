"""Application entry point.

CLI usage:
  python -m dicom_viewer                       # opens the last project (if any)
  python -m dicom_viewer path/to/file.dcmproj  # opens a project
  python -m dicom_viewer path/to/dicom-folder  # opens a folder
  python -m dicom_viewer path/to/scan.dcm      # opens a single file
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from dicom_viewer.io.config import load_last_project
from dicom_viewer.io.project import PROJECT_EXTENSION
from dicom_viewer.ui.main_window import MainWindow

APP_NAME = "DICOM Viewer"


def _icon_path() -> Path | None:
    """Locate the app icon. Works both for source runs and PyInstaller bundles."""
    candidates: list[Path] = []
    # PyInstaller bundle: sys._MEIPASS is the temp extract dir; assets live
    # in dist/<name>/assets/ relative to the executable.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "assets" / "icon.png")
    # Source run: assets/ sits at the repo root, two levels above this file
    # (src/dicom_viewer/app.py -> repo / assets / icon.png).
    candidates.append(Path(__file__).resolve().parent.parent.parent / "assets" / "icon.png")
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    icon_path = _icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    # Start maximised so the four-panel quad has room to breathe. We use
    # showMaximized rather than showFullScreen so the OS title bar stays
    # available — users can still resize/move if they want.
    window.showMaximized()
    _apply_startup_argument(window, sys.argv[1:])
    return app.exec()


def _apply_startup_argument(window: MainWindow, args: list[str]) -> None:
    if args:
        path = Path(args[0]).expanduser()
        if not path.exists():
            return
        if path.suffix.lower() == PROJECT_EXTENSION:
            window.load_project_from_path(path)
        elif path.is_dir():
            window.open_folder_path(path)
        elif path.is_file():
            window.open_file_path(path)
        return
    # No CLI argument — try to restore the last project the user worked on.
    last = load_last_project()
    if last is not None:
        window.load_project_from_path(last)


if __name__ == "__main__":
    raise SystemExit(main())
