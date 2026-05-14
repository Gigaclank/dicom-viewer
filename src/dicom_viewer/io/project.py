"""Project file format — save and load DICOM viewer settings as JSON.

A .dcmproj file is a small JSON document describing:
  * what to load (folder or single file)
  * windowing settings
  * segmentation parameters and refinement flags
  * the active region (axis-aligned crop)
  * export options (smoothing, decimation, manifold)

Projects are versioned so future formats can be migrated cleanly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_VERSION = 1
PROJECT_EXTENSION = ".dcmproj"


class ProjectError(Exception):
    """Raised when a project file cannot be read or is malformed."""


@dataclass
class SourceSettings:
    kind: str = ""           # "folder" | "file" | "" (empty)
    path: str = ""
    # For folder sources with multiple series, identifies which one the user
    # was viewing. Empty string means "load whatever was first".
    series_uid: str = ""


@dataclass
class WindowingSettings:
    center: float = 40.0
    width: float = 400.0


@dataclass
class SegmentationSettings:
    method: str = "Threshold"      # matches QComboBox label
    low: int = 300
    high: int = 3000
    seed_z: int = 0
    seed_y: int = 0
    seed_x: int = 0
    tolerance: int = 100
    keep_largest_component: bool = True
    smooth: bool = False
    live_preview: bool = True


@dataclass
class RegionSettings:
    z: tuple[int, int] = (0, 0)
    y: tuple[int, int] = (0, 0)
    x: tuple[int, int] = (0, 0)


@dataclass
class ExportSettings:
    smoothing_iterations: int = 15
    decimation_reduction: float = 0.5
    ensure_manifold: bool = True


@dataclass
class MaskEntry:
    """A named mask in the project's mask library.

    `path` is RELATIVE to the project file (so the project directory can be
    moved / copied as a unit). The file at that path is the NIfTI-encoded
    mask saved by save_segmentation_to_nifti().
    """

    name: str = ""
    path: str = ""


@dataclass
class Project:
    source: SourceSettings = field(default_factory=SourceSettings)
    windowing: WindowingSettings = field(default_factory=WindowingSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    region: RegionSettings = field(default_factory=RegionSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    # Mask library — each entry points to a companion NIfTI file alongside
    # the project. active_mask names the one currently loaded as the
    # document's segmentation.
    masks: list[MaskEntry] = field(default_factory=list)
    active_mask: str = ""
    version: int = PROJECT_VERSION


def save_project(path: Path, project: Project) -> None:
    """Write `project` to `path` as JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(project)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_project(path: Path) -> Project:
    """Read a project from disk. Raises ProjectError on failure."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ProjectError(f"project file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ProjectError(f"project file is not valid JSON ({path}): {e}") from e
    if not isinstance(raw, dict):
        raise ProjectError(f"project file must be a JSON object: {path}")
    version = raw.get("version", 0)
    if version != PROJECT_VERSION:
        raise ProjectError(
            f"unsupported project version {version} in {path} "
            f"(this build expects v{PROJECT_VERSION})"
        )
    masks_raw = raw.get("masks", []) or []
    masks: list[MaskEntry] = []
    if isinstance(masks_raw, list):
        for m in masks_raw:
            if isinstance(m, dict):
                masks.append(_take(MaskEntry, m))

    return Project(
        source=_take(SourceSettings, raw.get("source", {})),
        windowing=_take(WindowingSettings, raw.get("windowing", {})),
        segmentation=_take(SegmentationSettings, raw.get("segmentation", {})),
        region=_region_from(raw.get("region", {})),
        export=_take(ExportSettings, raw.get("export", {})),
        masks=masks,
        active_mask=str(raw.get("active_mask", "") or ""),
    )


def _take(cls: type, data: dict[str, Any]) -> Any:
    """Construct `cls` using only its declared fields, ignoring extras."""
    declared = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in declared})


def _region_from(data: dict[str, Any]) -> RegionSettings:
    return RegionSettings(
        z=tuple(data.get("z", [0, 0])),  # type: ignore[arg-type]
        y=tuple(data.get("y", [0, 0])),  # type: ignore[arg-type]
        x=tuple(data.get("x", [0, 0])),  # type: ignore[arg-type]
    )
