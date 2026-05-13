import json

import pytest

from dicom_viewer.io.project import (
    ExportSettings,
    Project,
    ProjectError,
    RegionSettings,
    SegmentationSettings,
    SourceSettings,
    WindowingSettings,
    load_project,
    save_project,
)


def test_round_trip_default_project(tmp_path):
    p = tmp_path / "default.dcmproj"
    save_project(p, Project())
    loaded = load_project(p)
    assert loaded == Project()


def test_round_trip_custom_project(tmp_path):
    p = tmp_path / "custom.dcmproj"
    project = Project(
        source=SourceSettings(kind="folder", path="/data/scans/foo"),
        windowing=WindowingSettings(center=400, width=1500),
        segmentation=SegmentationSettings(
            method="Threshold",
            low=300,
            high=2000,
            keep_largest_component=True,
            smooth=True,
        ),
        region=RegionSettings(z=(10, 80), y=(20, 480), x=(20, 480)),
        export=ExportSettings(
            smoothing_iterations=20, decimation_reduction=0.7, ensure_manifold=True
        ),
    )
    save_project(p, project)
    loaded = load_project(p)
    assert loaded == project


def test_load_missing_raises(tmp_path):
    with pytest.raises(ProjectError):
        load_project(tmp_path / "nope.dcmproj")


def test_load_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.dcmproj"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(ProjectError):
        load_project(p)


def test_load_wrong_version_raises(tmp_path):
    p = tmp_path / "old.dcmproj"
    p.write_text(json.dumps({"version": 999}), encoding="utf-8")
    with pytest.raises(ProjectError):
        load_project(p)


def test_load_ignores_unknown_keys(tmp_path):
    p = tmp_path / "extras.dcmproj"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "windowing": {"center": 50, "width": 600, "future_field": "ignored"},
                "future_section": {"x": 1},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_project(p)
    assert loaded.windowing.center == 50
    assert loaded.windowing.width == 600
