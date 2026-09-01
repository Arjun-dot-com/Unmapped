"""Step 1 - data loading layer.

Reads a Phase 1 output folder and a Phase 2 output folder, validates both
against the contract in the brief (Section 3), joins them on ``frame_id`` and
returns a single in-memory :class:`SceneDataset`.

Design rules:
* **Fail loudly.**  A missing file, a bad matrix shape, a frame_id that appears
  in one source but not the other -> a :class:`Phase3InputError` with a message a
  teammate can act on.  Never silently proceed on malformed input.
* **Join on ``frame_id``, never on order.**  Phase 1 and Phase 2 may enumerate
  frames differently.
* Images/masks are loaded lazily (only path validity is checked up front) so the
  loader is cheap even for thousands of frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..exceptions import Phase3InputError
from ..geometry import Camera, Intrinsics
from ..io.ply import read_ply
from ..logging_utils import get_logger

log = get_logger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
@dataclass
class FrameRecord:
    frame_id: str
    image_path: Path
    camera: Camera
    mask_path: Optional[Path] = None
    timestamp: Optional[float] = None
    blur_score: Optional[float] = None
    gps: Optional[dict] = None
    imu: Optional[dict] = None

    # ---- lazy loaders ---------------------------------------------------- #
    def load_image(self, downscale: int = 1) -> np.ndarray:
        """RGB uint8 (H,W,3)."""
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise Phase3InputError(f"frame {self.frame_id}: cannot read image "
                                   f"{self.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if downscale and downscale > 1:
            img = img[::downscale, ::downscale]
        return np.ascontiguousarray(img)

    def load_mask(self, shape_hw: tuple = None, dilate_px: int = 0,
                  downscale: int = 1) -> np.ndarray:
        """Binary dynamic-object mask, bool (H,W). ``True`` = dynamic pixel to
        EXCLUDE.  Returns all-``False`` if the frame has no mask."""
        if self.mask_path is None or not self.mask_path.is_file():
            if shape_hw is None:
                raise Phase3InputError(
                    f"frame {self.frame_id}: no mask and no shape hint given")
            return np.zeros(shape_hw, dtype=bool)
        m = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise Phase3InputError(f"frame {self.frame_id}: cannot read mask "
                                   f"{self.mask_path}")
        mask = m > 127
        if dilate_px and dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * dilate_px + 1, 2 * dilate_px + 1))
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        if downscale and downscale > 1:
            mask = mask[::downscale, ::downscale]
        if shape_hw is not None and mask.shape != tuple(shape_hw):
            mask = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask


@dataclass
class SceneDataset:
    frames: List[FrameRecord]
    intrinsics: Intrinsics
    scale_estimate_m_per_unit: float = 1.0
    accuracy_estimate_cm: float = float("nan")
    sparse_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sparse_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    video_meta: dict = field(default_factory=dict)
    source_paths: dict = field(default_factory=dict)
    drop_stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def cameras(self) -> List[Camera]:
        return [f.camera for f in self.frames]

    def summary(self) -> str:
        cts = np.array([c.center for c in self.cameras])
        span = (cts.max(0) - cts.min(0)) if len(cts) else np.zeros(3)
        return (f"{len(self.frames)} frames | "
                f"intr fx={self.intrinsics.fx:.1f} {self.intrinsics.width}x{self.intrinsics.height} | "
                f"sparse pts={len(self.sparse_xyz)} | "
                f"scale={self.scale_estimate_m_per_unit:g} m/unit | "
                f"cam baseline span=({span[0]:.1f},{span[1]:.1f},{span[2]:.1f}) m")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise Phase3InputError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise Phase3InputError(f"{what} is not valid JSON ({path}): {e}") from e


def _resolve_dir(root: Path, *candidates: str) -> Path:
    """Find the first existing sub-path; tolerate the caller pointing either at
    the ``output/`` dir or its parent."""
    for c in candidates:
        p = (root / c) if c else root
        if p.exists():
            return p
    return root / candidates[0]


def _find_manifest(frames_dir: Path) -> Path:
    for c in ("manifest.json", "output/manifest.json", "../manifest.json"):
        p = (frames_dir / c).resolve()
        if p.is_file():
            return p
    raise Phase3InputError(
        f"Phase 1 manifest.json not found under {frames_dir}. "
        f"Expected {frames_dir / 'manifest.json'}")


def _find_poses(poses_dir: Path) -> Path:
    for c in ("poses.json", "output/poses.json"):
        p = (poses_dir / c).resolve()
        if p.is_file():
            return p
    raise Phase3InputError(
        f"Phase 2 poses.json not found under {poses_dir}. "
        f"Expected {poses_dir / 'poses.json'}")


def _norm_id(fid) -> str:
    """Normalise a frame id to a plain string (accepts ints, '000123', 'frame_000123')."""
    s = str(fid).strip()
    if s.startswith("frame_"):
        s = s[len("frame_"):]
    if s.lower().endswith(_IMG_EXTS):
        s = Path(s).stem
        if s.startswith("frame_"):
            s = s[len("frame_"):]
    return s


def _locate_image(frames_root: Path, frame_id: str, given: Optional[str]) -> Optional[Path]:
    if given:
        p = (frames_root / given) if not Path(given).is_absolute() else Path(given)
        if p.is_file():
            return p
    for stem in (f"frame_{frame_id}", frame_id, f"frame_{frame_id.zfill(6)}"):
        for sub in ("frames", ""):
            for ext in _IMG_EXTS:
                p = frames_root / sub / f"{stem}{ext}"
                if p.is_file():
                    return p
    return None


def _locate_mask(frames_root: Path, frame_id: str, given: Optional[str]) -> Optional[Path]:
    if given:
        p = (frames_root / given) if not Path(given).is_absolute() else Path(given)
        if p.is_file():
            return p
    for stem in (f"frame_{frame_id}_mask", f"frame_{frame_id.zfill(6)}_mask",
                 f"{frame_id}_mask", f"frame_{frame_id}"):
        for sub in ("masks", ""):
            for ext in (".png", ".jpg", ".bmp"):
                p = frames_root / sub / f"{stem}{ext}"
                if p.is_file():
                    return p
    return None


def load_scene(frames_dir: str, poses_dir: str,
               max_frames: int = 0,
               min_blur_score: float = 0.0,
               min_pose_confidence: float = 0.0) -> SceneDataset:
    """Load + validate + join Phase 1 and Phase 2 outputs."""
    frames_dir = Path(frames_dir).expanduser().resolve()
    poses_dir = Path(poses_dir).expanduser().resolve()
    if not frames_dir.exists():
        raise Phase3InputError(f"--frames-dir does not exist: {frames_dir}")
    if not poses_dir.exists():
        raise Phase3InputError(f"--poses-dir does not exist: {poses_dir}")

    manifest_path = _find_manifest(frames_dir)
    poses_path = _find_poses(poses_dir)
    frames_root = manifest_path.parent
    poses_root = poses_path.parent

    manifest = _read_json(manifest_path, "Phase 1 manifest.json")
    poses = _read_json(poses_path, "Phase 2 poses.json")

    # ---- intrinsics ---------------------------------------------------- #
    intr_d = poses.get("intrinsics")
    if not isinstance(intr_d, dict) or not {"fx", "fy", "cx", "cy"} <= set(intr_d):
        raise Phase3InputError(
            "poses.json.intrinsics must contain fx, fy, cx, cy "
            f"(got: {list(intr_d) if isinstance(intr_d, dict) else type(intr_d)})")
    vm = manifest.get("video_meta", {}) or {}
    res = vm.get("resolution")
    width = int(intr_d.get("width", res[0] if res else round(2 * intr_d["cx"])))
    height = int(intr_d.get("height", res[1] if res else round(2 * intr_d["cy"])))
    intrinsics = Intrinsics(fx=float(intr_d["fx"]), fy=float(intr_d["fy"]),
                            cx=float(intr_d["cx"]), cy=float(intr_d["cy"]),
                            width=width, height=height)

    scale_m_per_unit = float(poses.get("scale_estimate_m_per_unit", 1.0))
    accuracy_cm = float(poses.get("accuracy_estimate_cm", float("nan")))

    # ---- index the two sources by normalised frame_id ---------------- #
    man_frames = manifest.get("frames")
    if not isinstance(man_frames, list) or not man_frames:
        raise Phase3InputError("manifest.json.frames must be a non-empty list")
    pose_frames = poses.get("frames")
    if not isinstance(pose_frames, list) or not pose_frames:
        raise Phase3InputError("poses.json.frames must be a non-empty list")

    man_by_id: Dict[str, dict] = {}
    for e in man_frames:
        if "frame_id" not in e:
            raise Phase3InputError(f"manifest frame entry missing 'frame_id': {e}")
        man_by_id[_norm_id(e["frame_id"])] = e
    pose_by_id: Dict[str, dict] = {}
    for e in pose_frames:
        if "frame_id" not in e:
            raise Phase3InputError(f"poses frame entry missing 'frame_id': {e}")
        pose_by_id[_norm_id(e["frame_id"])] = e

    common = [fid for fid in man_by_id if fid in pose_by_id]
    if not common:
        raise Phase3InputError(
            "no overlapping frame_id between manifest.json and poses.json. "
            f"manifest ids sample={list(man_by_id)[:3]}, poses ids sample={list(pose_by_id)[:3]}")
    only_man = sorted(set(man_by_id) - set(pose_by_id))
    only_pose = sorted(set(pose_by_id) - set(man_by_id))
    if only_pose:
        raise Phase3InputError(
            f"{len(only_pose)} frame(s) have a pose but no manifest entry "
            f"(cannot fetch image/mask): {only_pose[:5]}")
    if only_man:
        log.warning("%d manifest frame(s) have no Phase-2 pose; dropping them: %s%s",
                    len(only_man), only_man[:5], " ..." if len(only_man) > 5 else "")

    common.sort(key=lambda s: (len(s), s))

    # ---- build FrameRecords ----------------------------------------- #
    records: List[FrameRecord] = []
    n_blur_drop = n_conf_drop = 0
    for fid in common:
        me = man_by_id[fid]
        pe = pose_by_id[fid]

        R = pe.get("R")
        t = pe.get("t")
        if R is None or t is None:
            raise Phase3InputError(f"pose for frame {fid} missing 'R' or 't'")
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if R.shape != (3, 3):
            raise Phase3InputError(f"pose R for frame {fid} must be 3x3, got {R.shape}")
        if t.shape != (3,):
            raise Phase3InputError(f"pose t for frame {fid} must be length-3, got {t.shape}")
        conf = float(pe.get("confidence", 1.0))

        blur = me.get("blur_score")
        blur = float(blur) if blur is not None else None
        if min_blur_score > 0 and blur is not None and blur < min_blur_score:
            n_blur_drop += 1
            continue
        if min_pose_confidence > 0 and conf < min_pose_confidence:
            n_conf_drop += 1
            continue

        img_path = _locate_image(frames_root, fid, me.get("image_path") or me.get("path"))
        if img_path is None:
            raise Phase3InputError(
                f"frame {fid}: image file not found under {frames_root} "
                f"(looked for frames/frame_{fid}.jpg etc.)")
        mask_path = _locate_mask(frames_root, fid, me.get("mask_path"))

        cam = Camera.from_lists(fid, R, t, intrinsics, confidence=conf)
        records.append(FrameRecord(
            frame_id=fid, image_path=img_path, camera=cam, mask_path=mask_path,
            timestamp=me.get("timestamp"), blur_score=blur,
            gps=me.get("gps"), imu=me.get("imu"),
        ))

    if n_blur_drop:
        log.info("dropped %d frame(s) below blur_score %.1f", n_blur_drop, min_blur_score)
    if n_conf_drop:
        log.info("dropped %d frame(s) below pose confidence %.2f",
                 n_conf_drop, min_pose_confidence)
    if not records:
        raise Phase3InputError("no usable frames left after blur / confidence filtering")

    if max_frames and max_frames > 0 and len(records) > max_frames:
        idx = np.linspace(0, len(records) - 1, max_frames).round().astype(int)
        records = [records[i] for i in sorted(set(idx.tolist()))]
        log.info("subsampled to %d frames (max_frames=%d)", len(records), max_frames)

    # ---- sparse cloud --------------------------------------------- #
    sparse_xyz = np.zeros((0, 3))
    sparse_rgb = np.zeros((0, 3), np.uint8)
    sp_path = None
    for c in ("sparse_cloud.ply", "output/sparse_cloud.ply", "points3D.ply"):
        p = (poses_root / c)
        if p.is_file():
            sp_path = p
            break
    if sp_path is not None:
        try:
            ply = read_ply(str(sp_path))
            sparse_xyz = ply.xyz()
            sparse_rgb = ply.rgb()
        except Exception as e:                       # noqa: BLE001 - want the context
            raise Phase3InputError(f"failed to read sparse cloud {sp_path}: {e}") from e
        if len(sparse_xyz) < 8:
            log.warning("sparse cloud has only %d points - metric fusion will lean "
                        "heavily on the global fallback", len(sparse_xyz))
    else:
        log.warning("no sparse_cloud.ply found under %s - per-frame metric fusion "
                    "will be disabled; falling back to scale_estimate_m_per_unit=%g",
                    poses_root, scale_m_per_unit)

    # Per the Phase 2 contract, poses.json.frames[*].t AND sparse_cloud.ply are
    # already expressed in the SAME (metric-scaled) world frame. We therefore do
    # NOT re-scale either here; `scale_estimate_m_per_unit` is carried as metadata
    # and only used as a last-resort fallback when there is no sparse cloud.
    if abs(scale_m_per_unit - 1.0) > 1e-6:
        log.warning("poses.json scale_estimate_m_per_unit=%g (!=1). Assuming poses "
                    "and sparse cloud are already consistent in that frame; "
                    "fusion anchors to the sparse cloud regardless.", scale_m_per_unit)

    ds = SceneDataset(
        frames=records, intrinsics=intrinsics,
        scale_estimate_m_per_unit=scale_m_per_unit,
        accuracy_estimate_cm=accuracy_cm,
        sparse_xyz=sparse_xyz, sparse_rgb=sparse_rgb,
        video_meta=vm,
        source_paths={"manifest": str(manifest_path), "poses": str(poses_path),
                      "sparse_cloud": str(sp_path) if sp_path else None,
                      "frames_root": str(frames_root)},
        drop_stats={"dropped_blur": n_blur_drop, "dropped_pose_confidence": n_conf_drop,
                    "dropped_no_pose": len(only_man),
                    "manifest_frames": len(man_frames)},
    )
    log.info("loaded scene: %s", ds.summary())
    return ds
