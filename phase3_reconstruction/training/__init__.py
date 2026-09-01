"""Step 6 - radiance-field optimisation.

``train_scene`` dispatches to:
* :mod:`.gsplat_trainer`  - real 3D Gaussian Splatting (needs CUDA + ``gsplat``)
* :mod:`.fallback_trainer`- NumPy point-cloud consolidation (no GPU, NO radiance
  optimisation) so the pipeline still produces a valid, honest output anywhere.
"""

from __future__ import annotations

from ..config import TrainingConfig
from ..hardware import gsplat_trainable
from ..logging_utils import get_logger
from .fallback_trainer import fallback_consolidate
from .result import TrainResult

log = get_logger(__name__)


def _gsplat_available() -> bool:
    return gsplat_trainable()


def train_scene(init_scene, dataset, masks, cfg: TrainingConfig,
                sim_transform, out_dir, seed: int = 0) -> TrainResult:
    """Optimise (or consolidate) ``init_scene`` -> final :class:`GaussianScene`.

    Returns a :class:`TrainResult` (scene + metrics + which backend ran).
    """
    backend = cfg.backend
    if backend == "auto":
        backend = "gsplat" if _gsplat_available() else "fallback"

    if backend == "gsplat":
        if not _gsplat_available():
            raise RuntimeError("training.backend=gsplat but gsplat/CUDA unavailable")
        from .gsplat_trainer import train_gsplat
        log.info("training backend: gsplat (CUDA)")
        return train_gsplat(init_scene, dataset, masks, cfg, sim_transform, out_dir, seed)

    log.warning("training backend: FALLBACK (NumPy consolidation). No radiance-field "
                "optimisation is performed - install gsplat + CUDA for real training. "
                "Output geometry/colour come straight from fused depth + SfM.")
    return fallback_consolidate(init_scene, dataset, masks, cfg, seed)
