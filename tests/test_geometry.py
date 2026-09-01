import numpy as np
import pytest

from phase3_reconstruction.geometry import (Camera, Intrinsics, SimilarityTransform,
                                            look_at, orthonormalize)


def _cam(seed=0):
    rng = np.random.default_rng(seed)
    intr = Intrinsics(fx=600, fy=610, cx=320, cy=240, width=640, height=480)
    eye = rng.normal(0, 5, 3) + np.array([0, 0, 20.0])
    R = look_at(eye, np.zeros(3))
    return Camera.from_center("f", eye, R, intr)


def test_project_backproject_roundtrip():
    cam = _cam(1)
    rng = np.random.default_rng(2)
    pts = rng.normal(0, 3, (400, 3))
    uv, z, valid = cam.project(pts, clip=False)
    rec = cam.backproject(uv[valid], z[valid])
    np.testing.assert_allclose(rec, pts[valid], atol=1e-6)


def test_camera_center_and_forward():
    cam = _cam(3)
    # a point straight ahead of the camera projects near principal point
    ahead = cam.center + cam.forward * 10.0
    uv, z, valid = cam.project(ahead[None], clip=False)
    assert z[0] > 0
    assert abs(uv[0, 0] - cam.intr.cx) < 1.0 and abs(uv[0, 1] - cam.intr.cy) < 1.0


def test_orthonormalize_fixes_noisy_rotation():
    rng = np.random.default_rng(4)
    R = look_at(rng.normal(size=3) + 10, np.zeros(3))
    Rn = orthonormalize(R + rng.normal(0, 1e-2, (3, 3)))
    assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(Rn) - 1.0) < 1e-6


def test_similarity_transform_inverse():
    rng = np.random.default_rng(5)
    pts = rng.normal(0, 4, (200, 3))
    sim = SimilarityTransform.normalizing(pts, target_radius=1.0)
    back = sim.apply_inverse(sim.apply(pts))
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_similarity_transform_preserves_projection():
    """A transformed point seen by the transformed camera lands on the same pixel."""
    cam = _cam(6)
    rng = np.random.default_rng(7)
    pts = rng.normal(0, 3, (300, 3))
    sim = SimilarityTransform.normalizing(pts, target_radius=1.0)

    cam2 = sim.transform_camera(cam)
    pts2 = sim.apply(pts)

    uv1, z1, v1 = cam.project(pts, clip=False)
    uv2, z2, v2 = cam2.project(pts2, clip=False)
    m = v1 & v2
    np.testing.assert_allclose(uv1[m], uv2[m], atol=1e-4)
    # depths scale by sim.scale
    np.testing.assert_allclose(z2[m], z1[m] * sim.scale, rtol=1e-5)
