"""Step 4 - dynamic-object mask integration.

Phase 1 hands us a binary mask per frame (255 = vehicle / person / animal, i.e. a
moving thing that must NOT end up in the static 3D model).  This module loads,
dilates and caches those masks once, and every downstream stage consults the same
:class:`MaskSet`:

* fusion  - dynamic pixels are excluded from the metric-scale anchor set
* init    - dynamic pixels are not back-projected into the init cloud
* training- dynamic pixels are excluded from the photometric loss / densification
* confidence - a Gaussian seen only through dynamic pixels gets no observation credit

Dilation matters: segmentation edges are rarely tight, and a thin halo of
car-coloured pixels around a vehicle is enough to spawn floater Gaussians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..config import MaskConfig
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class MaskSet:
    """``masks[frame_id] -> (H,W) bool``  (True = dynamic pixel, exclude)."""

    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    enabled: bool = True
    dynamic_fraction: Dict[str, float] = field(default_factory=dict)

    def get(self, frame_id: str, shape_hw) -> np.ndarray:
        m = self.masks.get(frame_id)
        if m is None or not self.enabled:
            return np.zeros(shape_hw, dtype=bool)
        if m.shape != tuple(shape_hw):
            import cv2
            m = cv2.resize(m.astype(np.uint8), (shape_hw[1], shape_hw[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        return m

    @property
    def mean_dynamic_fraction(self) -> float:
        v = list(self.dynamic_fraction.values())
        return float(np.mean(v)) if v else 0.0


def build_mask_set(dataset, cfg: MaskConfig) -> MaskSet:
    """Load + dilate every frame's dynamic mask according to ``cfg``."""
    ms = MaskSet(enabled=cfg.enabled)
    if not cfg.enabled:
        log.info("dynamic masking DISABLED via config - moving objects will leak "
                 "into the reconstruction")
        return ms

    n_with = 0
    for fr in dataset.frames:
        h, w = dataset.intrinsics.height, dataset.intrinsics.width
        try:
            m = fr.load_mask(shape_hw=(h, w), dilate_px=cfg.dilate_px)
        except Exception as e:                       # noqa: BLE001
            if cfg.treat_missing_as_static:
                log.warning("frame %s: mask unreadable (%s) - treating as fully static",
                            fr.frame_id, e)
                m = np.zeros((h, w), dtype=bool)
            else:
                raise
        if fr.mask_path is not None:
            n_with += 1
        ms.masks[fr.frame_id] = m
        ms.dynamic_fraction[fr.frame_id] = float(m.mean())

    log.info("masks: %d/%d frames have a Phase-1 mask; mean dynamic-pixel fraction "
             "%.1f%% (dilate=%dpx)", n_with, len(dataset.frames),
             100.0 * ms.mean_dynamic_fraction, cfg.dilate_px)
    if n_with == 0:
        log.warning("no dynamic masks found - if the scene has moving vehicles/people "
                    "they WILL corrupt the static reconstruction")
    return ms


def apply_mask_to_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``depth`` with dynamic pixels set to NaN."""
    if mask is None:
        return depth
    out = depth.astype(np.float32, copy=True)
    out[mask.astype(bool)] = np.nan
    return out
