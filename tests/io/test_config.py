from pathlib import Path

from dicom_viewer.io import config


def test_save_and_load_last_project(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    target = tmp_path / "fake.dcmproj"
    target.write_text("{}", encoding="utf-8")
    config.save_last_project(target)
    loaded = config.load_last_project()
    assert loaded == target


def test_load_last_project_returns_none_when_no_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.load_last_project() is None


def test_load_last_project_returns_none_when_target_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config.save_last_project(Path("/nonexistent/path/proj.dcmproj"))
    assert config.load_last_project() is None
