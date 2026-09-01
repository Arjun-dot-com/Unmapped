"""Phase 3 - 3D Reconstruction Engine (project "Unmapped", SIH26158).

Turns posed drone-video frames into a dense, metrically-scaled, explorable
3D Gaussian scene.

Pipeline (see ``phase3_reconstruction.pipeline``):

    Phase 1 frames + masks + manifest ─┐
                                       ├─> [1] load & join on frame_id
    Phase 2 poses + sparse cloud ──────┘
        -> [2] monocular depth (Depth Anything V2)
        -> [3] depth<->metric fusion  (THE generalization step)
        -> [4] dynamic-object mask integration
        -> [5] depth-informed Gaussian initialization
        -> [6] radiance-field optimization (gsplat)  |  numpy fallback
        -> [7] observation-count / confidence flagging
        -> [8] export (splat_scene.ply + native) + preview render
        -> [9] training_meta.json + console metrics
"""

__version__ = "0.1.0"

from .exceptions import (
    Phase3Error,
    Phase3InputError,
    Phase3ConfigError,
    Phase3DependencyError,
)

__all__ = [
    "__version__",
    "Phase3Error",
    "Phase3InputError",
    "Phase3ConfigError",
    "Phase3DependencyError",
]
