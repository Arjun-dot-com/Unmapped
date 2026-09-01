"""Step 5 - depth-informed Gaussian initialisation.

Vanilla 3DGS initialises from the SfM sparse cloud (a few thousand points) or
random noise.  On a **single drone pass** with limited viewing angles that is a
bad start: densification struggles to grow geometry it has never seen a hint of,
and training is slow to converge.

Instead we back-project the *fused metric* depth maps (Step 3) from every usable
frame into world space, colour each point from the source image, drop points that
sit on dynamic-object masks or on depth discontinuities, voxel-downsample to a
manageable count, and merge in the SfM sparse cloud.  The result is a dense,
metrically-correct point set that gives training a strong geometric prior.

Output :class:`InitCloud` -> ``GaussianScene`` (via :func:`build_initial_gaussians`),
with per-Gaussian initial scale sized from local point spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from ..config import InitConfig
from ..gaussian_scene import GaussianScene
from ..geometry import Camera
from ..logging_utils import get_logger
from ..pointcloud import nn_spacing, statistical_outlier_mask, voxel_downsample

log = get_logger(__name__)


@dataclass
class InitCloud:
    xyz: np.ndarray            # (N,3) world / metric
    rgb01: np.ndarray          # (N,3) float [0,1]
    source: np.ndarray         # (N,) uint8: 0 = depth back-projection, 1 = SfM sparse
    per_frame_counts: Dict[str, int]

    def __len__(self):
        return len(self.xyz)


def _depth_edge_mask(depth: np.ndarray, rel_thresh: float) -> np.ndarray:
    """True where the depth map has a strong relative discontinuity (silhouette /
    flying-pixel candidate)."""
    d = depth.copy()
    d[~np.isfinite(d)] = 0.0
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    denom = np.maximum(np.abs(d), 1e-3)
    return (grad / denom) > rel_thresh


def build_initial_gaussians(fusion_results: Dict[str, "object"],
                            images: Dict[str, np.ndarray],
                            cameras: Dict[str, Camera],
                            masks: Optional[Dict[str, np.ndarray]],
                            sparse_xyz: np.ndarray,
                            sparse_rgb: np.ndarray,
                            cfg: InitConfig,
                            default_opacity: float = 0.1) -> "tuple[GaussianScene, InitCloud]":
    """Assemble the dense init cloud and wrap it as a :class:`GaussianScene`.

    ``fusion_results[fid]`` must expose ``.metric_depth`` (H,W float32, NaN =
    invalid) and ``.ok`` / ``.frame_id``.
    """
    masks = masks or {}
    all_xyz = []
    all_rgb = []
    all_src = []
    per_frame_counts: Dict[str, int] = {}

    for fid, fr in fusion_results.items():
        if not getattr(fr, "ok", True):
            per_frame_counts[fid] = 0
            continue
        cam = cameras[fid]
        depth = np.asarray(fr.metric_depth, dtype=np.float32)
        img = images[fid]
        if img.shape[:2] != depth.shape[:2]:
            img = cv2.resize(img, (depth.shape[1], depth.shape[0]),
                             interpolation=cv2.INTER_AREA)

        invalid = ~np.isfinite(depth)
        m = masks.get(fid)
        if m is not None:
            if m.shape != depth.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            invalid = invalid | m
        if cfg.depth_edge_filter:
            invalid = invalid | _depth_edge_mask(depth, cfg.edge_rel_thresh)

        pts, pix = cam.backproject_depth_map(depth, stride=cfg.pixel_stride,
                                             mask=invalid)
        if len(pts) == 0:
            per_frame_counts[fid] = 0
            continue
        cols = img[pix[:, 1], pix[:, 0]].astype(np.float32) / 255.0
        all_xyz.append(pts)
        all_rgb.append(cols)
        all_src.append(np.zeros(len(pts), dtype=np.uint8))
        per_frame_counts[fid] = len(pts)

    if all_xyz:
        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        src = np.concatenate(all_src, axis=0)
    else:
        xyz = np.zeros((0, 3)); rgb = np.zeros((0, 3), np.float32)
        src = np.zeros((0,), np.uint8)
        log.warning("no depth back-projected points - init will rely on the SfM "
                    "sparse cloud only")

    n_raw = len(xyz)
    # ---- voxel downsample the dense depth points -------------------- #
    if len(xyz) and cfg.voxel_size_m > 0:
        xyz, ds = voxel_downsample(xyz, cfg.voxel_size_m,
                                   attrs={"rgb": rgb, "src": src}, reduce="mean")
        rgb = np.clip(ds["rgb"], 0, 1)
        src = np.zeros(len(xyz), dtype=np.uint8)
    n_voxel = len(xyz)

    # ---- merge SfM sparse cloud ---------------------------------- #
    if cfg.merge_sparse_cloud and sparse_xyz is not None and len(sparse_xyz):
        sp_xyz = np.asarray(sparse_xyz, dtype=np.float64).reshape(-1, 3)
        sp_rgb = (np.asarray(sparse_rgb, dtype=np.float32).reshape(-1, 3) / 255.0
                  if sparse_rgb is not None and len(sparse_rgb) == len(sp_xyz)
                  else np.full((len(sp_xyz), 3), 0.5, np.float32))
        xyz = np.concatenate([xyz, sp_xyz], axis=0)
        rgb = np.concatenate([rgb, sp_rgb], axis=0)
        src = np.concatenate([src, np.ones(len(sp_xyz), dtype=np.uint8)], axis=0)

    # ---- cap total count --------------------------------------- #
    if len(xyz) > cfg.max_points:
        keep = np.random.default_rng(0).choice(len(xyz), cfg.max_points, replace=False)
        keep.sort()
        xyz, rgb, src = xyz[keep], rgb[keep], src[keep]

    # ---- light outlier removal on the whole thing ------------- #
    if len(xyz) > 50:
        keep = statistical_outlier_mask(xyz, k=10, std_ratio=3.0)
        n_out = int((~keep).sum())
        if n_out:
            xyz, rgb, src = xyz[keep], rgb[keep], src[keep]
            log.info("init: removed %d statistical outliers", n_out)

    init_cloud = InitCloud(xyz=xyz, rgb01=rgb, source=src,
                           per_frame_counts=per_frame_counts)

    # ---- size initial Gaussians from local spacing ----------- #
    if len(xyz):
        spacing = nn_spacing(xyz, k=4)
        spacing = np.clip(spacing, 1e-3, np.percentile(spacing, 98))
        scales = np.repeat(spacing[:, None].astype(np.float32), 3, axis=1)
    else:
        scales = np.zeros((0, 3), np.float32)

    scene = GaussianScene.from_points(xyz, rgb, scales, opacity=default_opacity)
    scene.meta.update({
        "init_points_raw": int(n_raw),
        "init_points_after_voxel": int(n_voxel),
        "init_points_final": int(len(xyz)),
        "init_from_depth": int((src == 0).sum()),
        "init_from_sfm": int((src == 1).sum()),
        "voxel_size_m": cfg.voxel_size_m,
        "pixel_stride": cfg.pixel_stride,
    })
    log.info("init cloud: %d pts (%d depth-backproj + %d SfM), voxel=%.2fm, "
             "raw depth pts=%d", len(xyz), int((src == 0).sum()),
             int((src == 1).sum()), cfg.voxel_size_m, n_raw)
    return scene, init_cloud
