import pytest

from dicom_viewer.core.region import Region


def test_region_from_bounds_basic():
    r = Region(z=(0, 5), y=(0, 10), x=(0, 20))
    assert r.shape == (5, 10, 20)
    assert r.is_empty is False


def test_region_validates_ordering():
    with pytest.raises(ValueError):
        Region(z=(5, 0), y=(0, 10), x=(0, 20))


def test_region_intersect_overlap():
    a = Region(z=(0, 10), y=(0, 10), x=(0, 10))
    b = Region(z=(5, 15), y=(2, 8), x=(0, 20))
    c = a.intersect(b)
    assert c == Region(z=(5, 10), y=(2, 8), x=(0, 10))


def test_region_intersect_disjoint_is_empty():
    a = Region(z=(0, 5), y=(0, 5), x=(0, 5))
    b = Region(z=(10, 15), y=(0, 5), x=(0, 5))
    assert a.intersect(b).is_empty


def test_region_clamp_to():
    r = Region(z=(-1, 12), y=(-3, 6), x=(0, 100))
    clamped = r.clamp_to(Region(z=(0, 10), y=(0, 5), x=(0, 50)))
    assert clamped == Region(z=(0, 10), y=(0, 5), x=(0, 50))


def test_region_size_mm():
    r = Region(z=(0, 5), y=(0, 10), x=(0, 20))
    assert r.size_mm(spacing_mm=(2.0, 1.0, 0.5)) == (10.0, 10.0, 10.0)
