"""Step 8 (part 1) - export in the exact Phase 4 output contract.

Produces:
* ``splat_scene.ply``      - dense point cloud: x y z, red green blue (uchar),
  opacity (float), scale_x/y/z (float), observation_count (uint16),
  confidence (float).  Coordinates are in the **Phase-2 metric world frame**
  (any training-time similarity normalisation has already been inverted by the
  trainer; we assert the centroid is sane and record the transform in a comment).
* ``splat_scene_native/``  - the raw trained Gaussians:
    - gsplat backend  -> ``checkpoint.pt`` (means, log-scales, quats,
      logit-opacities, sh0, shN) + ``meta.json``
    - fallback backend -> ``gaussians.npz`` + ``meta.json``
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ExportConfig
from ..gaussian_scene import GaussianScene
from ..geometry import SimilarityTransform
from ..io.ply import write_ply
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ExportPaths:
    ply: str
    native_dir: str
    n_points_exported: int
    n_low_confidence: int


def _subsample(scene: GaussianScene, max_points: int, seed: int = 0) -> GaussianScene:
    if max_points <= 0 or len(scene) <= max_points:
        return scene
    rng = np.random.default_rng(seed)
    # keep ALL low-confidence points (they're the interesting flag) + sample the rest
    low = scene.observation_count <= 2
    hi_idx = np.where(~low)[0]
    lo_idx = np.where(low)[0]
    budget = max(0, max_points - len(lo_idx))
    if budget < len(hi_idx):
        hi_idx = rng.choice(hi_idx, budget, replace=False)
    keep = np.sort(np.concatenate([hi_idx, lo_idx]))
    log.info("export: subsampling %d -> %d points (all %d low-confidence kept)",
             len(scene), len(keep), len(lo_idx))
    return scene.subset(keep)


def export_scene(scene: GaussianScene, out_dir: str, cfg: ExportConfig,
                 sim_used: Optional[SimilarityTransform] = None,
                 train_result=None, extra_comments: Optional[list] = None,
                 seed: int = 0) -> ExportPaths:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exp = scene
    if cfg.drop_low_confidence:
        keep = scene.observation_count > 2
        log.warning("export.drop_low_confidence=true -> removing %d low-confidence "
                    "Gaussians. (Default policy is to FLAG, not drop.)",
                    int((~keep).sum()))
        exp = scene.subset(np.where(keep)[0])
    exp = _subsample(exp, cfg.max_export_points, seed=seed)

    # ---- splat_scene.ply -------------------------------------- #
    centroid = exp.centroid()
    cols = exp.to_ply_columns(include_confidence=cfg.write_confidence)
    n_low = int((exp.observation_count <= 2).sum())
    comments = [
        "phase3_reconstruction splat_scene.ply",
        "coordinate_frame: Phase-2 metric world (metres); NOT re-centred/re-scaled",
        f"similarity_normalisation_applied_during_training_then_inverted: "
        f"{'yes' if (sim_used and not sim_used.is_identity) else 'no'}",
        f"exported_points: {len(exp)}",
        f"low_confidence_points(observation_count<=2): {n_low} "
        f"({100.0 * n_low / max(len(exp),1):.1f}%)",
        f"centroid_xyz: {centroid[0]:.3f} {centroid[1]:.3f} {centroid[2]:.3f}",
        "fields: x y z | red green blue (uchar) | opacity (float) | "
        "scale_x scale_y scale_z (float, metres) | observation_count (uint16) | "
        "confidence (float 0..1; low => under-observed / near-occluded, FLAGGED not inpainted)",
    ]
    if sim_used and not sim_used.is_identity:
        comments.append("sim_transform_json: " + json.dumps(sim_used.to_dict()))
    comments += (extra_comments or [])

    ply_path = out / cfg.ply_name
    write_ply(str(ply_path), cols, comments=comments)
    log.info("wrote %s  (%d points, %d low-confidence)", ply_path, len(exp), n_low)

    # ---- splat_scene_native/ -------------------------------- #
    native_dir = out / cfg.native_dir
    native_dir.mkdir(parents=True, exist_ok=True)
    backend = getattr(train_result, "backend", "unknown")
    native_meta = {
        "backend": backend,
        "n_gaussians": len(scene),
        "sh_degree": scene.meta.get("sh_degree", 0),
        "radiance_optimised": bool(scene.meta.get("radiance_optimised", False)),
        "coordinate_frame": "Phase-2 metric world (metres)",
        "sim_transform": sim_used.to_dict() if sim_used else SimilarityTransform.identity().to_dict(),
    }

    if getattr(train_result, "native_export_path", None) and \
            Path(train_result.native_export_path).exists():
        src = Path(train_result.native_export_path)
        dst = native_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        native_meta["format"] = "gsplat_checkpoint_pt"
        native_meta["file"] = dst.name
        native_meta["param_keys"] = ["means", "scales(log)", "quats",
                                     "opacities(logit)", "sh0", "shN"]
    else:
        npz_path = native_dir / "gaussians.npz"
        scene.save_npz(str(npz_path))
        native_meta["format"] = "numpy_npz"
        native_meta["file"] = npz_path.name
        native_meta["arrays"] = ["means(N,3)", "colors(N,3 in 0..1)", "opacity(N)",
                                 "scales(N,3 metres)", "quats(N,4 wxyz)",
                                 "observation_count(N)", "confidence(N)"]

    (native_dir / "meta.json").write_text(json.dumps(native_meta, indent=2),
                                          encoding="utf-8")
    log.info("wrote %s/ (%s)", native_dir, native_meta["format"])

    return ExportPaths(ply=str(ply_path), native_dir=str(native_dir),
                       n_points_exported=len(exp), n_low_confidence=n_low)
