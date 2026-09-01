"""Step 7 - occlusion / low-observation flagging.

Policy (from the brief, verbatim intent): **for v1 we do NOT hallucinate or
inpaint unseen geometry.**  Instead we measure, per Gaussian:

* ``observation_count`` - how many cameras actually see it: reprojects inside the
  image, in front of the camera, NOT on a dynamic-object mask, and with a depth
  that agrees with that frame's fused metric depth to within
  ``confidence.depth_consistency_m`` (so we don't count a camera that is really
  looking at a surface in front of this Gaussian).
* ``view_angle_spread_deg`` - angular spread of the viewing rays.  A point seen
  by 5 cameras all from nearly the same direction (single drone pass!) is far
  less constrained than one seen from a 40 deg spread.

``confidence`` in [0,1] combines the two.  Gaussians with
``observation_count <= confidence.low_obs_threshold`` are tagged low-confidence.
These attributes are written per-point into ``splat_scene.ply`` so Phase 4 / the
UI can grey them out and the team can tell judges *"flagged, not invented."*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from ..config import ConfidenceConfig
from ..gaussian_scene import GaussianScene
from ..geometry import Camera
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ConfidenceSummary:
    n_gaussians: int
    mean_observation_count: float
    median_observation_count: float
    frac_low_confidence: float
    frac_single_view: float
    mean_view_angle_spread_deg: float
    low_obs_threshold: int
    hist_observation_count: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_gaussians": self.n_gaussians,
            "mean_observation_count": round(self.mean_observation_count, 3),
            "median_observation_count": self.median_observation_count,
            "frac_low_confidence": round(self.frac_low_confidence, 4),
            "frac_single_view": round(self.frac_single_view, 4),
            "mean_view_angle_spread_deg": round(self.mean_view_angle_spread_deg, 2),
            "low_obs_threshold": self.low_obs_threshold,
            "observation_count_histogram": self.hist_observation_count,
        }


def compute_observation_confidence(scene: GaussianScene,
                                   cameras: Dict[str, Camera],
                                   fused_depths: Dict[str, np.ndarray],
                                   masks,
                                   cfg: ConfidenceConfig
                                   ) -> "tuple[GaussianScene, ConfidenceSummary]":
    """Fill ``scene.observation_count`` and ``scene.confidence`` in place-ish
    (returns the same object) and a :class:`ConfidenceSummary`."""
    N = len(scene)
    means = scene.means
    obs = np.zeros(N, dtype=np.int32)
    # running accumulation of viewing-ray directions for angular spread
    dir_sum = np.zeros((N, 3), dtype=np.float64)
    dir_sqsum = np.zeros(N, dtype=np.float64)          # for a cheap spread proxy
    ray_min = np.ones((N, 3)) * np.nan
    # We approximate angular spread via 1 - |mean_unit_dir| (0 = all same dir,
    # ->1 = very spread).  Convert to an equivalent cone half-angle at the end.

    for fid, cam in cameras.items():
        uv, z, valid = cam.project(means, clip=True)
        if not valid.any():
            continue
        depth_map = fused_depths.get(fid)
        vi = np.where(valid)[0]
        u = np.clip(uv[vi, 0].astype(int), 0, cam.intr.width - 1)
        v = np.clip(uv[vi, 1].astype(int), 0, cam.intr.height - 1)

        good = np.ones(len(vi), dtype=bool)
        if depth_map is not None:
            dm = depth_map
            if dm.shape != (cam.intr.height, cam.intr.width):
                dm = cv2.resize(dm, (cam.intr.width, cam.intr.height),
                                interpolation=cv2.INTER_NEAREST)
            dref = dm[v, u]
            with np.errstate(invalid="ignore"):
                consistent = np.isfinite(dref) & (np.abs(z[vi] - dref) <= cfg.depth_consistency_m)
            # if the frame has no valid fused depth at that pixel, fall back to
            # "in front of camera" only (don't over-penalise).
            good &= np.where(np.isfinite(dref), consistent, True)

        if masks is not None:
            dmask = masks.get(fid)
            if dmask is not None:
                if dmask.shape != (cam.intr.height, cam.intr.width):
                    dmask = cv2.resize(dmask.astype(np.uint8),
                                       (cam.intr.width, cam.intr.height),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                good &= ~dmask[v, u]

        seen_idx = vi[good]
        obs[seen_idx] += 1
        d = means[seen_idx] - cam.center                # ray from cam to gaussian
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        dir_sum[seen_idx] += d

    # angular spread
    with np.errstate(invalid="ignore"):
        mean_len = np.linalg.norm(dir_sum, axis=1) / np.maximum(obs, 1)
    spread = np.clip(1.0 - mean_len, 0.0, 1.0)          # 0 = colinear, 1 = spread
    spread_deg = np.degrees(np.arccos(np.clip(mean_len, -1, 1)))  # cone half-angle proxy
    spread_deg[obs <= 1] = 0.0

    # confidence score
    n_frames = max(len(cameras), 1)
    target_obs = float(min(6, max(2, 0.4 * n_frames)))
    obs_score = np.clip(obs / target_obs, 0.0, 1.0)
    ang_score = np.clip(spread_deg / max(cfg.angle_diversity_deg * 3.0, 1.0), 0.0, 1.0)
    conf = (0.65 * obs_score + 0.35 * ang_score).astype(np.float32)
    conf[obs == 0] = 0.0

    scene.observation_count = obs
    scene.confidence = conf

    low = obs <= cfg.low_obs_threshold
    hist = {int(k): int(v) for k, v in
            zip(*np.unique(np.clip(obs, 0, 15), return_counts=True))}
    summary = ConfidenceSummary(
        n_gaussians=N,
        mean_observation_count=float(obs.mean()) if N else 0.0,
        median_observation_count=int(np.median(obs)) if N else 0,
        frac_low_confidence=float(low.mean()) if N else 0.0,
        frac_single_view=float((obs <= 1).mean()) if N else 0.0,
        mean_view_angle_spread_deg=float(spread_deg[obs > 1].mean()) if (obs > 1).any() else 0.0,
        low_obs_threshold=cfg.low_obs_threshold,
        hist_observation_count=hist,
    )
    log.info("confidence: median obs=%d, %.1f%% flagged low-confidence (<=%d views), "
             "%.1f%% single-view, mean view-angle spread %.1f deg",
             summary.median_observation_count, 100 * summary.frac_low_confidence,
             cfg.low_obs_threshold, 100 * summary.frac_single_view,
             summary.mean_view_angle_spread_deg)
    return scene, summary
