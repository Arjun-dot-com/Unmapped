import numpy as np
import pytest

from phase3_reconstruction.io.ply import read_ply, write_ply, write_point_cloud, read_point_cloud


def test_binary_roundtrip_with_extra_columns(tmp_path):
    n = 500
    rng = np.random.default_rng(0)
    cols = {
        "x": rng.normal(size=n).astype(np.float32),
        "y": rng.normal(size=n).astype(np.float32),
        "z": rng.normal(size=n).astype(np.float32),
        "red": (rng.integers(0, 256, n)).astype(np.uint8),
        "green": (rng.integers(0, 256, n)).astype(np.uint8),
        "blue": (rng.integers(0, 256, n)).astype(np.uint8),
        "opacity": rng.random(n).astype(np.float32),
        "observation_count": rng.integers(0, 20, n).astype(np.uint16),
        "confidence": rng.random(n).astype(np.float32),
    }
    p = tmp_path / "c.ply"
    write_ply(str(p), cols, comments=["hello", "multi\nline"])
    back = read_ply(str(p))
    assert back.count == n
    assert back.has("observation_count", "confidence")
    np.testing.assert_allclose(back.columns["x"], cols["x"], rtol=1e-6)
    np.testing.assert_array_equal(back.columns["red"].astype(np.uint8), cols["red"])
    np.testing.assert_array_equal(back.columns["observation_count"].astype(np.uint16),
                                  cols["observation_count"])


def test_ascii_ply_is_readable(tmp_path):
    p = tmp_path / "a.ply"
    p.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "0 0 0 255 0 0\n1 2 3 0 255 0\n-1 -2 -3 0 0 255\n"
    )
    ply = read_ply(str(p))
    assert ply.count == 3
    np.testing.assert_allclose(ply.xyz()[1], [1, 2, 3])
    np.testing.assert_array_equal(ply.rgb()[0], [255, 0, 0])


def test_write_point_cloud_helper(tmp_path):
    xyz = np.random.default_rng(1).normal(size=(100, 3))
    rgb = np.random.default_rng(2).integers(0, 256, (100, 3))
    extra = {"confidence": np.linspace(0, 1, 100).astype(np.float32)}
    p = tmp_path / "pc.ply"
    write_point_cloud(str(p), xyz, rgb, extra=extra)
    x2, c2 = read_point_cloud(str(p))
    assert x2.shape == (100, 3)
    np.testing.assert_allclose(x2, xyz, atol=1e-4)


def test_bad_header_raises(tmp_path):
    p = tmp_path / "bad.ply"
    p.write_text("not a ply file\n")
    with pytest.raises(ValueError):
        read_ply(str(p))
