"""End-to-end Phase 3 orchestration: posed frames -> trained Gaussian scene.

``run_pipeline`` wires Steps 1-9 together.  Each stage is its own module; this
file only moves data between them and enforces ordering.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from .config import Phase3Config
from .confidence import compute_observation_confidence
from .data import load_scene
from .depth import create_depth_predictor
from .export import export_scene, render_preview
from .fusion import fuse_scene
from .fusion.depth_scale_fusion import FusionResult
from .gaussian_scene import GaussianScene
from .geometry import SimilarityTransform
from .init import build_initial_gaussians
from .logging_utils import get_logger, log_stage
from .masking import apply_mask_to_depth, build_mask_set
from .metrics import build_training_meta, write_training_meta
from .training import train_scene

log = get_logger(__name__)


def run_pipeline(frames_dir: str, poses_dir: str, out_dir: str,
                 cfg: Phase3Config, mock: bool = False) -> dict:
    t_start = time.perf_counter()
    np.random.seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load + validate + join --------------------------- #
    with log_stage(log, "Step 1/9  load Phase-1 + Phase-2 output"):
        ds = load_scene(frames_dir, poses_dir, max_frames=cfg.max_frames,
                        min_blur_score=cfg.min_blur_score,
                        min_pose_confidence=cfg.min_pose_confidence)

    cameras = {fr.frame_id: fr.camera for fr in ds.frames}
    frames_root = ds.source_paths.get("frames_root")

    # ---- Step 4 (built first, consumed by 3/5/7): dynamic masks --- #
    with log_stage(log, "Step 4/9  dynamic-object masks"):
        masks = build_mask_set(ds, cfg.mask)
    mask_dict = {fid: masks.get(fid, (ds.intrinsics.height, ds.intrinsics.width))
                 for fid in cameras}

    # ---- Step 2: monocular depth --------------------------------- #
    gt_depth_dir = None
    if frames_root:
        cand = Path(frames_root) / "gt_depth"
        if cand.is_dir():
            gt_depth_dir = str(cand)
    with log_stage(log, "Step 2/9  monocular depth (Depth Anything V2)"):
        predictor = create_depth_predictor(cfg.depth, gt_depth_dir=gt_depth_dir,
                                           mock=mock)
        depth_backend = predictor.backend
        images: Dict[str, np.ndarray] = {}
        predictions: Dict[str, object] = {}
        for fr in tqdm(ds.frames, desc="depth", unit="frame"):
            img = fr.load_image()
            images[fr.frame_id] = img
            predictions[fr.frame_id] = predictor.predict(img, frame_id=fr.frame_id)
        log.info("depth backend: %s | predicted %d frames", depth_backend, len(predictions))

    # ---- Step 3: depth <-> metric fusion ------------------------ #
    with log_stage(log, "Step 3/9  depth <-> metric-scale fusion  [core]"):
        fusion_results: Dict[str, FusionResult] = fuse_scene(
            predictions, cameras, ds.sparse_xyz, masks=mask_dict,
            cfg=cfg.fusion, seed=cfg.seed)
        # mask dynamic pixels out of the dense fused depth (Step 4 integration)
        fused_depths: Dict[str, np.ndarray] = {}
        for fid, r in fusion_results.items():
            fused_depths[fid] = apply_mask_to_depth(r.metric_depth, mask_dict.get(fid))
            r.metric_depth = fused_depths[fid]

    # ---- Step 5: depth-informed Gaussian init ----------------- #
    with log_stage(log, "Step 5/9  Gaussian initialisation"):
        init_scene, init_cloud = build_initial_gaussians(
            fusion_results, images, cameras, mask_dict,
            ds.sparse_xyz, ds.sparse_rgb, cfg.init)
        if len(init_scene) == 0:
            raise RuntimeError("initialisation produced 0 points - check fusion / "
                               "sparse cloud / masks")

    # ---- similarity normalisation candidate (a trainer may or may not use it;
    #      whichever transform is actually applied is reported back on the
    #      TrainResult and used for export / metadata) --------------------- #
    if cfg.training.normalize_scene:
        sim_candidate = SimilarityTransform.normalizing(
            init_scene.means, target_radius=cfg.training.normalize_radius)
    else:
        sim_candidate = SimilarityTransform.identity()

    # ---- Step 6: radiance-field optimisation (or fallback) --- #
    with log_stage(log, "Step 6/9  radiance-field optimisation (gsplat)"):
        train_result = train_scene(init_scene, ds, mask_dict, cfg.training,
                                   sim_candidate, str(out / cfg.export.native_dir),
                                   seed=cfg.seed)
    scene: GaussianScene = train_result.scene
    sim = train_result.sim_used
    if not sim.is_identity:
        log.info("scene was similarity-normalised for training (scale=%.5g) and "
                 "inverted before export", sim.scale)

    # ---- Step 7: occlusion / confidence flagging ------------ #
    with log_stage(log, "Step 7/9  occlusion / observation-count flagging"):
        scene, conf_summary = compute_observation_confidence(
            scene, cameras, fused_depths, mask_dict, cfg.confidence)

    # ---- Step 8: export + preview -------------------------- #
    with log_stage(log, "Step 8/9  export (.ply + native) + preview render"):
        exp = export_scene(scene, str(out), cfg.export, sim_used=sim,
                           train_result=train_result, seed=cfg.seed)
        prev = render_preview(scene, str(out), cfg.preview, seed=cfg.seed)

    # ---- Step 9: metrics --------------------------------- #
    with log_stage(log, "Step 9/9  training metrics"):
        meta = build_training_meta(
            train_result=train_result, confidence_summary=conf_summary, dataset=ds,
            n_frames_used=len(ds.frames),
            n_frames_dropped_blur=ds.drop_stats.get("dropped_blur", 0),
            n_frames_dropped_conf=ds.drop_stats.get("dropped_pose_confidence", 0),
            fusion_results=fusion_results, depth_backend=depth_backend,
            fusion_cfg=cfg.fusion, mask_frac=masks.mean_dynamic_fraction,
            wall_time_s=time.perf_counter() - t_start, sim_used=sim)
        meta_path = write_training_meta(meta, str(out))

    return {
        "output_dir": str(out),
        "splat_scene_ply": exp.ply,
        "splat_scene_native": exp.native_dir,
        "preview_render": prev.still,
        "preview_orbit": prev.video,
        "preview_confidence": prev.confidence_still,
        "training_meta": meta_path,
        "num_frames_used": len(ds.frames),
        "num_gaussians_final": len(scene),
        "num_points_exported": exp.n_points_exported,
        "frac_low_confidence": conf_summary.frac_low_confidence,
        "training_backend": train_result.backend,
        "depth_backend": depth_backend,
        "wall_time_seconds": round(time.perf_counter() - t_start, 2),
    }
