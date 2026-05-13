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

from PyQt6.QtWidgets import QApplication

from dicom_viewer.io.config import load_last_project
from dicom_viewer.io.project import PROJECT_EXTENSION
from dicom_viewer.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
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
