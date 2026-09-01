"""Unit tests for the core module: depth <-> metric-scale fusion.

We build a slanted plane with a KNOWN metric depth map, synthesise a monocular
prediction as a known affine transform of the inverse depth (that is exactly the
DA-V2 model: affine-invariant inverse depth), then check the fusion module
recovers metres from a handful of sparse anchors.
"""

import numpy as np
import pytest

from phase3_reconstruction.config import FusionConfig
from phase3_reconstruction.fusion.depth_scale_fusion import (
    align_depth_to_metric, solve_global_scale_shift, fuse_scene)
from phase3_reconstruction.geometry import Camera, Intrinsics


class _Pred:
    """Minimal duck-typed stand-in for depth.DepthPrediction."""
    def __init__(self, relative, kind="inverse_depth", frame_id="t"):
        self.relative = np.asarray(relative, np.float32)
        self.kind = kind
        self.frame_id = frame_id
        self.valid = np.isfinite(self.relative)


def _slanted_plane_scene(H=200, W=260, seed=0):
    rng = np.random.default_rng(seed)
    intr = Intrinsics(fx=520.0, fy=520.0, cx=W / 2, cy=H / 2, width=W, height=H)
    cam = Camera(frame_id="t", R=np.eye(3), t=np.zeros(3), intr=intr)

    # plane in camera space: n . X = c  ->  depth per pixel
    n = np.array([0.25, 0.1, 1.0]); n /= np.linalg.norm(n)
    c = 11.0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - intr.cx) / intr.fx
    y = (vs - intr.cy) / intr.fy
    denom = n[0] * x + n[1] * y + n[2]
    depth = c / denom                       # metric GT depth (H,W)
    assert depth.min() > 3 and depth.max() < 40

    # sparse anchors: random pixels -> world (== camera) coords
    k = 250
    pu = rng.integers(0, W, k)
    pv = rng.integers(0, H, k)
    pd = depth[pv, pu]
    uv = np.stack([pu, pv], 1).astype(float)
    pts_world = cam.backproject(uv, pd)
    return cam, depth, pts_world


@pytest.mark.parametrize("space", ["disparity", "depth"])
def test_recovers_metric_scale(space):
    cam, gt_depth, pts = _slanted_plane_scene()
    rng = np.random.default_rng(1)

    a, b = 4.0, 0.03
    if space == "disparity":
        mono = a * (1.0 / gt_depth) + b
        kind = "inverse_depth"
    else:
        mono = a * gt_depth + b            # relative depth
        kind = "depth"
    mono = mono + rng.normal(0, 0.002 * mono.std(), mono.shape)

    cfg = FusionConfig(space=space, solver="lstsq", min_points=10)
    res = align_depth_to_metric(_Pred(mono, kind), pts, cam, cfg=cfg)

    assert res.ok
    good = np.isfinite(res.metric_depth)
    rel_err = np.abs(res.metric_depth[good] - gt_depth[good]) / gt_depth[good]
    assert np.median(rel_err) < 0.02, np.median(rel_err)
    assert res.rmse_m < 0.15


def test_ransac_is_robust_to_outlier_anchors():
    cam, gt_depth, pts = _slanted_plane_scene(seed=3)
    rng = np.random.default_rng(4)
    mono = 3.5 * (1.0 / gt_depth) + 0.05

    # corrupt 30% of the anchors by pushing them far off the plane
    n_bad = len(pts) // 3
    bad = rng.choice(len(pts), n_bad, replace=False)
    pts = pts.copy()
    pts[bad] += rng.normal(0, 5.0, (n_bad, 3))

    cfg = FusionConfig(space="disparity", solver="ransac", min_points=10,
                       ransac_iters=300, ransac_thresh_m=0.25)
    res = align_depth_to_metric(_Pred(mono), pts, cam, cfg=cfg,
                                rng=np.random.default_rng(0))
    assert res.ok
    assert res.n_inliers >= 0.5 * res.n_anchors
    good = np.isfinite(res.metric_depth)
    rel_err = np.abs(res.metric_depth[good] - gt_depth[good]) / gt_depth[good]
    assert np.median(rel_err) < 0.05


def test_mask_excludes_dynamic_anchors():
    cam, gt_depth, pts = _slanted_plane_scene(seed=5)
    mono = 3.0 * (1.0 / gt_depth) + 0.02
    # a mask covering the left half - anchors there must be ignored
    mask = np.zeros(gt_depth.shape, bool)
    mask[:, : gt_depth.shape[1] // 2] = True
    cfg = FusionConfig(space="disparity", solver="lstsq", min_points=5)
    res = align_depth_to_metric(_Pred(mono), pts, cam, mask=mask, cfg=cfg)
    assert res.ok
    # every used anchor must be on the right half
    uv, z, val = cam.project(pts, clip=True)
    on_right = uv[val][:, 0] >= gt_depth.shape[1] // 2
    assert res.n_anchors <= on_right.sum() + 2


def test_global_fallback_when_too_few_anchors():
    cam, gt_depth, pts = _slanted_plane_scene(seed=6)
    mono = 3.0 * (1.0 / gt_depth) + 0.02
    few = pts[:3]
    cfg = FusionConfig(space="disparity", solver="lstsq", min_points=30,
                       fallback_to_global=True)

    # no global -> not ok
    r0 = align_depth_to_metric(_Pred(mono), few, cam, cfg=cfg)
    assert not r0.ok

    # with a (correct) global scale/shift -> ok and metric
    r1 = align_depth_to_metric(_Pred(mono), few, cam, cfg=cfg,
                               global_scale_shift=(1.0 / 3.0, -0.02 / 3.0))
    assert r1.ok and r1.used_global
    good = np.isfinite(r1.metric_depth)
    rel_err = np.abs(r1.metric_depth[good] - gt_depth[good]) / gt_depth[good]
    assert np.median(rel_err) < 0.05


def test_fuse_scene_two_pass_and_global_solver():
    cam, gt_depth, pts = _slanted_plane_scene(seed=7)
    mono = 3.0 * (1.0 / gt_depth) + 0.02
    preds = {"a": _Pred(mono, frame_id="a"), "b": _Pred(mono, frame_id="b")}
    cams = {"a": cam, "b": cam}
    cfg = FusionConfig(space="disparity", solver="lstsq", min_points=10)
    out = fuse_scene(preds, cams, pts, cfg=cfg)
    assert set(out) == {"a", "b"}
    assert all(r.ok for r in out.values())
    g = solve_global_scale_shift(list(out.values()), cfg)
    assert g is not None and g[0] > 0
