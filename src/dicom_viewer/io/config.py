"""User config directory helpers — currently just the last-project pointer."""
from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    """Return ~/.config/dicom-viewer (or $XDG_CONFIG_HOME/dicom-viewer), creating it."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    d = Path(base) / "dicom-viewer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def last_project_pointer() -> Path:
    return config_dir() / "last_project.txt"


def save_last_project(path: Path) -> None:
    last_project_pointer().write_text(str(path), encoding="utf-8")


def load_last_project() -> Path | None:
    pointer = last_project_pointer()
    if not pointer.exists():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    p = Path(text)
    return p if p.is_file() else None
