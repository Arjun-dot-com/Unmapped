"""NumPy fallback "trainer" - runs when gsplat / CUDA is unavailable.

**It does NOT optimise a radiance field.**  It consolidates the depth-informed
init cloud into a clean Gaussian set:

1. voxel-average points (merge duplicate observations of the same surface),
2. statistical outlier removal (kill flying pixels / depth-fusion noise),
3. re-size each Gaussian from local point spacing,
4. set opacity from local density (sparse islands -> lower opacity),
5. measure a PSNR *proxy* by z-buffer-splatting the cloud into every training
   camera and comparing to the (mask-excluded) ground-truth image.

That proxy is reported honestly as ``final_psnr_db`` with a note that no
optimisation occurred.  It is still a useful "does the geometry explain the
footage" signal for the team demo.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import TrainingConfig
from ..gaussian_scene import GaussianScene
from ..logging_utils import get_logger
from ..pointcloud import nn_spacing, statistical_outlier_mask, voxel_downsample
from ..render import psnr, rasterize_points
from .result import TrainResult

log = get_logger(__name__)


def _density_opacity(xyz: np.ndarray, radius: float) -> np.ndarray:
    from ..pointcloud import cKDTree
    n = len(xyz)
    if n == 0:
        return np.zeros(0, np.float32)
    if cKDTree is None or n < 8:
        return np.full(n, 0.6, np.float32)
    tree = cKDTree(xyz)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    c = np.asarray(counts, dtype=np.float64)
    target = np.percentile(c, 75) + 1e-6
    return np.clip(0.15 + 0.8 * (c / target), 0.05, 0.97).astype(np.float32)


def fallback_consolidate(init_scene: GaussianScene, dataset, masks,
                         cfg: TrainingConfig, seed: int = 0) -> TrainResult:
    t0 = time.perf_counter()
    xyz = init_scene.means.copy()
    rgb = init_scene.colors.copy()

    from ..geometry import SimilarityTransform
    if len(xyz) == 0:
        log.error("fallback trainer received an empty init cloud")
        return TrainResult(scene=init_scene, backend="fallback",
                           train_time_seconds=time.perf_counter() - t0,
                           sim_used=SimilarityTransform.identity(),
                           extra={"warning": "empty init cloud"})

    # 1. voxel average
    if cfg.fallback_voxel_size_m > 0:
        xyz, ds = voxel_downsample(xyz, cfg.fallback_voxel_size_m,
                                   attrs={"rgb": rgb}, reduce="mean")
        rgb = np.clip(ds["rgb"], 0, 1).astype(np.float32)
    n_after_voxel = len(xyz)

    # 2. outlier removal
    keep = statistical_outlier_mask(xyz, k=cfg.fallback_outlier_nb,
                                    std_ratio=cfg.fallback_outlier_std)
    xyz, rgb = xyz[keep], rgb[keep]
    n_after_outlier = len(xyz)

    # 3. scales from spacing
    spacing = nn_spacing(xyz, k=4)
    spacing = np.clip(spacing, 1e-3, np.percentile(spacing, 98) if len(spacing) else 0.1)
    scales = np.repeat((0.6 * spacing)[:, None].astype(np.float32), 3, axis=1)

    # 4. density-based opacity
    opacity = _density_opacity(xyz, radius=max(2 * cfg.fallback_voxel_size_m, 0.2))

    scene = GaussianScene(
        means=xyz, colors=rgb, opacity=opacity, scales=scales,
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (len(xyz), 1)),
        meta=dict(init_scene.meta),
    )
    scene.meta.update({
        "trainer": "fallback_consolidate",
        "radiance_optimised": False,
        "points_after_voxel": int(n_after_voxel),
        "points_after_outlier": int(n_after_outlier),
    })

    # 5. PSNR proxy against training views.
    #    Measured ONLY where the consolidated cloud actually splats a point AND
    #    the pixel is not a dynamic-object pixel -> "where we have geometry, is the
    #    colour right?".  (Comparing empty sky to a black background would make the
    #    number meaningless.)
    psnrs = []
    n_eval = min(len(dataset.frames), 12)
    stride_eval = max(1, len(dataset.frames) // n_eval)
    for fr in dataset.frames[::stride_eval]:
        gt = fr.load_image()
        cam = fr.camera
        rendered, _, hits = rasterize_points(xyz, rgb, cam, point_radius_px=2,
                                             background=(0, 0, 0), max_points=500_000)
        cover = hits > 0
        if masks is not None:
            dm = masks.get(fr.frame_id, np.zeros(gt.shape[:2], bool))
            if dm.shape == gt.shape[:2]:
                cover &= ~dm
        if cover.sum() < 100:
            continue
        psnrs.append(psnr(rendered, gt, mask=cover))
    final_psnr = float(np.nanmean(psnrs)) if psnrs else float("nan")

    dt = time.perf_counter() - t0
    log.info("fallback consolidation: %d -> %d Gaussians in %.2fs | PSNR proxy %.2f dB",
             len(init_scene), len(scene), dt, final_psnr)

    return TrainResult(
        scene=scene, backend="fallback", iterations_run=0,
        train_time_seconds=dt, final_loss=float("nan"),
        final_psnr_db=final_psnr, psnr_history=[float(x) for x in psnrs],
        sim_used=SimilarityTransform.identity(),
        extra={"psnr_is_proxy": True,
               "note": "NumPy consolidation only; no radiance-field optimisation"},
    )
