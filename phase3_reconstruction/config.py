"""Typed, validated, file-driven configuration.

Every hyperparameter in Phase 3 lives in a YAML file (see ``configs/``).  The CLI
loads one file, optionally applies a handful of ``--set a.b=c`` overrides, then
freezes it into nested dataclasses so the rest of the code never touches raw
dicts or magic numbers.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import Phase3ConfigError


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class DepthConfig:
    backend: str = "auto"               # auto | depth_anything_v2 | mock
    model_size: str = "small"           # small | base | large  (DA-V2 encoder)
    checkpoint: str = ""                # optional explicit weights path / HF id
    input_size: int = 518              # DA-V2 native inference resolution
    device: str = "auto"               # auto | cpu | cuda
    # --- mock / fallback predictor knobs ---
    mock_use_gt_depth: bool = True      # in --mock, read synthetic GT depth as the prior
    mock_noise_std: float = 0.02       # relative noise added to the mock prior
    mock_smooth_px: float = 2.0        # gaussian blur (px) of the mock prior


@dataclass
class FusionConfig:
    # "disparity" aligns s*mono + o to 1/z (MiDaS / Depth-Anything style, robust
    # for affine-invariant *inverse* depth). "depth" aligns s*mono + o to z.
    space: str = "disparity"
    solver: str = "ransac"             # lstsq | huber | ransac
    min_points: int = 20               # per-frame sparse anchors required
    ransac_iters: int = 200
    ransac_thresh_m: float = 0.30      # inlier threshold in metres (depth space)
    huber_delta: float = 1.0
    max_depth_m: float = 500.0         # clamp fused depth
    min_depth_m: float = 0.1
    fallback_to_global: bool = True    # reuse global scale when a frame is short on anchors
    global_from_median: bool = True    # global scale = median of per-frame scales


@dataclass
class MaskConfig:
    enabled: bool = True
    dilate_px: int = 6                 # grow dynamic masks before excluding pixels
    treat_missing_as_static: bool = True


@dataclass
class InitConfig:
    pixel_stride: int = 4             # sample every Nth pixel when back-projecting
    voxel_size_m: float = 0.15        # downsample resolution for the dense init cloud
    max_points: int = 1_500_000
    merge_sparse_cloud: bool = True   # also seed from Phase 2 sparse points
    depth_edge_filter: bool = True    # drop points on strong depth discontinuities
    edge_rel_thresh: float = 0.05
    min_confidence_frame: float = 0.0 # skip frames whose pose confidence < this


@dataclass
class TrainingConfig:
    backend: str = "auto"            # auto | gsplat | fallback
    iterations: int = 7000
    sh_degree: int = 2
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4
    ssim_lambda: float = 0.2
    densify_start_iter: int = 500
    densify_stop_iter: int = 4000
    densify_every: int = 100
    densify_grad_thresh: float = 2e-4
    prune_opacity_thresh: float = 0.05
    reset_opacity_every: int = 3000
    random_bg: bool = False
    train_image_downscale: int = 1   # downscale training images by this factor
    normalize_scene: bool = True     # similarity-normalise world for stability (undone on export)
    normalize_radius: float = 1.0
    eval_every: int = 1000
    # fallback-trainer knobs (numpy, no radiance optimisation):
    fallback_voxel_size_m: float = 0.10
    fallback_outlier_nb: int = 12
    fallback_outlier_std: float = 2.0


@dataclass
class ConfidenceConfig:
    depth_consistency_m: float = 0.5   # a Gaussian is "seen" by a cam if its
                                       # reprojected depth agrees within this
    low_obs_threshold: int = 2         # observation_count <= this  => low-confidence flag
    angle_diversity_deg: float = 5.0   # min spread of viewing rays for "well constrained"
    write_per_point: bool = True


@dataclass
class PreviewConfig:
    enabled: bool = True
    width: int = 960
    height: int = 540
    n_orbit_frames: int = 72
    elevation_deg: float = 28.0
    radius_scale: float = 1.6         # orbit radius = radius_scale * scene extent
    point_radius_px: int = 2
    fps: int = 24
    background: list = field(default_factory=lambda: [8, 10, 14])
    make_video: bool = True
    max_render_points: int = 600_000   # random-subsample the cloud per preview frame


@dataclass
class ExportConfig:
    ply_name: str = "splat_scene.ply"
    native_dir: str = "splat_scene_native"
    write_confidence: bool = True
    drop_low_confidence: bool = False  # NEVER default-on: brief says flag, don't delete
    max_export_points: int = 4_000_000


@dataclass
class Phase3Config:
    depth: DepthConfig = field(default_factory=DepthConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    init: InitConfig = field(default_factory=InitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    seed: int = 1234
    max_frames: int = 0               # 0 = use all frames
    min_blur_score: float = 0.0       # drop frames with manifest blur_score < this (0 = keep all)
    min_pose_confidence: float = 0.0  # drop frames with Phase-2 pose confidence < this

    # -- (de)serialisation ------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase3Config":
        # `from __future__ import annotations` makes Field.type a string, so we
        # keep an explicit section registry instead of introspecting types.
        d = d or {}
        kwargs: Dict[str, Any] = {}
        for name, sub_type in _SECTIONS.items():
            if name not in d:
                continue
            sub = d[name] or {}
            _reject_unknown(sub_type, sub, name)
            kwargs[name] = sub_type(**sub)
        for scalar in ("seed", "max_frames", "min_blur_score", "min_pose_confidence"):
            if scalar in d:
                kwargs[scalar] = d[scalar]
        return cls(**kwargs)


_SECTIONS = {
    "depth": DepthConfig,
    "fusion": FusionConfig,
    "mask": MaskConfig,
    "init": InitConfig,
    "training": TrainingConfig,
    "confidence": ConfidenceConfig,
    "preview": PreviewConfig,
    "export": ExportConfig,
}


def _reject_unknown(dc_type, provided: Dict[str, Any], prefix: str) -> None:
    known = {f.name for f in fields(dc_type)}
    unknown = set(provided) - known
    if unknown:
        raise Phase3ConfigError(
            f"unknown config key(s) under '{prefix}': {sorted(unknown)}. "
            f"valid keys: {sorted(known)}"
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(text: str) -> Any:
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_config(path: str = None, overrides: list = None) -> Phase3Config:
    """Load ``path`` (YAML) over the built-in defaults, then apply CLI
    ``--set section.key=value`` overrides. Returns a validated :class:`Phase3Config`.
    """
    data: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if not p.is_file():
            raise Phase3ConfigError(f"config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise Phase3ConfigError(f"could not parse YAML config {p}: {e}") from e
        if not isinstance(data, dict):
            raise Phase3ConfigError(f"config {p} must be a mapping at top level")

    defaults = Phase3Config().to_dict()
    merged = _deep_merge(defaults, data)

    for item in overrides or []:
        if "=" not in item:
            raise Phase3ConfigError(f"--set expects section.key=value, got '{item}'")
        key, val = item.split("=", 1)
        parts = key.strip().split(".")
        node = merged
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise Phase3ConfigError(f"--set: unknown section '{part}' in '{key}'")
            node = node[part]
        if parts[-1] not in node:
            raise Phase3ConfigError(f"--set: unknown key '{key}'")
        node[parts[-1]] = _coerce(val)

    try:
        cfg = Phase3Config.from_dict(merged)
    except TypeError as e:
        raise Phase3ConfigError(str(e)) from e

    _validate(cfg)
    return cfg


def _validate(cfg: Phase3Config) -> None:
    if cfg.fusion.space not in ("disparity", "depth"):
        raise Phase3ConfigError("fusion.space must be 'disparity' or 'depth'")
    if cfg.fusion.solver not in ("lstsq", "huber", "ransac"):
        raise Phase3ConfigError("fusion.solver must be lstsq | huber | ransac")
    if cfg.depth.backend not in ("auto", "depth_anything_v2", "mock"):
        raise Phase3ConfigError("depth.backend must be auto | depth_anything_v2 | mock")
    if cfg.training.backend not in ("auto", "gsplat", "fallback"):
        raise Phase3ConfigError("training.backend must be auto | gsplat | fallback")
    if cfg.init.pixel_stride < 1:
        raise Phase3ConfigError("init.pixel_stride must be >= 1")
    if cfg.fusion.min_depth_m <= 0 or cfg.fusion.max_depth_m <= cfg.fusion.min_depth_m:
        raise Phase3ConfigError("fusion depth clamp range is invalid")
    if cfg.training.iterations < 0:
        raise Phase3ConfigError("training.iterations must be >= 0")
