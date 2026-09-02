from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .src.pose_core import (
    _camera_from_imu,
    _compute_confidence,
    _orthonormalize_rotation,
    build_sparse_cloud,
    estimate_intrinsics,
    estimate_scale_from_gps,
    feature_match_pair,
    georeference_pose_records,
    load_manifest,
    load_rgb_image,
    normalize_pose_output,
    reconstruct_with_pycolmap,
    resolve_frame_image,
    resolve_frame_mask,
    write_poses_json,
)

log = logging.getLogger("phase2_pose")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _frame_sort_key(frame: Dict[str, Any]) -> Tuple[float, str]:
    ts = frame.get("timestamp")
    if ts is None:
        return (1e18, str(frame.get("frame_id", "")))
    return (float(ts), str(frame.get("frame_id", "")))


def run_pipeline(frames_dir: str, out_dir: str) -> Dict[str, Any]:
    frames_root = Path(frames_dir).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = frames_root / "manifest.json"
    if not manifest_path.exists() and (frames_root / "output").exists():
        manifest_path = frames_root / "output" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 1 manifest.json not found under {frames_root}")

    manifest = load_manifest(manifest_path)
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(
            "manifest.json must contain a non-empty 'frames' list")
    ordered = sorted(raw_frames, key=_frame_sort_key)

    video_meta = manifest.get("video_meta", {}) or {}
    res = video_meta.get("resolution", [640, 480]) or [640, 480]
    width, height = int(res[0]), int(res[1])
    intrinsics = estimate_intrinsics(width, height)
    K = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                  [0.0, intrinsics["fy"], intrinsics["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    pose_records: List[Dict[str, Any]] = []
    camera_centres: List[np.ndarray] = []
    all_cloud_xyz: List[np.ndarray] = []
    all_cloud_rgb: List[np.ndarray] = []
    prev_R = np.eye(3)
    prev_t = np.zeros(3)
    prev_img = None
    prev_mask = None
    pycolmap_summary = reconstruct_with_pycolmap(
        ordered, output_dir, intrinsics, width, height)
    if pycolmap_summary is not None:
        log.info("Using PyCOLMAP reconstruction path: %s",
                 pycolmap_summary.get("status"))
    else:
        log.info(
            "PyCOLMAP not available or not usable; falling back to ORB+RANSAC geometric reconstruction.")

    for idx, record in enumerate(ordered):
        frame_id = str(record.get("frame_id"))
        try:
            img_path = resolve_frame_image(frames_root, record)
            rgb = load_rgb_image(img_path)
            shape_hw = rgb.shape[:2]
            mask = resolve_frame_mask(frames_root, record, shape_hw=shape_hw)
        except Exception as exc:  # pragma: no cover - error path
            log.warning(
                "Skipping frame %s because it could not be read: %s", frame_id, exc)
            continue

        if idx == 0:
            R = np.eye(3)
            t = np.zeros(3)
            conf = 1.0
        else:
            if prev_img is None or prev_mask is None:
                R = prev_R.copy()
                t = prev_t.copy()
                conf = 0.4
            else:
                rel_R, rel_t, inlier_count, score, pts3d, match_meta = feature_match_pair(
                    prev_img, rgb, prev_mask, mask, K)
                if rel_R is None or rel_t is None or inlier_count < 8:
                    log.warning(
                        "Frame %s had too few inlier matches; using previous pose", frame_id)
                    R = prev_R.copy()
                    t = prev_t.copy()
                    conf = 0.25
                else:
                    imu_R = _camera_from_imu(record.get("imu"))
                    visual_R = rel_R @ prev_R
                    if imu_R is not None:
                        R = _orthonormalize_rotation(imu_R @ visual_R)
                    else:
                        R = visual_R
                    t = rel_R @ prev_t + rel_t
                    gps_consistency = 0.0
                    if isinstance(record.get("gps"), dict):
                        gps_consistency = min(
                            1.0, abs(float(record["gps"].get("alt", 0.0))) / 500.0)
                    reproj_error = 1.0 / max(inlier_count, 1)
                    conf = _compute_confidence(
                        inlier_count, score, gps_consistency, reproj_error)
                    if pts3d.size:
                        pts_world = prev_R.T @ (pts3d.T - prev_t[:, None])
                        all_cloud_xyz.append(pts_world.T)
                        grey = np.full(
                            (pts3d.shape[0], 3), 180, dtype=np.uint8)
                        all_cloud_rgb.append(grey)

        pose_records.append({
            "frame_id": frame_id,
            "R": np.asarray(R, dtype=np.float64).tolist(),
            "t": np.asarray(t, dtype=np.float64).tolist(),
            "confidence": float(np.clip(conf, 0.0, 1.0)),
        })
        camera_centres.append(-R.T @ t)
        prev_R = R.copy()
        prev_t = t.copy()
        prev_img = rgb
        prev_mask = mask

    if not pose_records:
        raise ValueError(
            "No usable frames were processed; check Phase 1 input and masks")

    scale, acc_cm, _ = estimate_scale_from_gps(ordered, camera_centres)
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    acc_cm = float(max(acc_cm, 5.0))

    if any(isinstance(rec.get("gps"), dict) for rec in ordered):
        try:
            scale, acc_cm, _ = georeference_pose_records(
                ordered, pose_records, camera_centres)
        except Exception as exc:  # pragma: no cover
            log.warning(
                "Georeferencing alignment failed; using GPS-only scale fallback: %s", exc)

    normalize_pose_output(pose_records, scale)

    cloud_xyz = np.concatenate(
        all_cloud_xyz, axis=0) if all_cloud_xyz else np.zeros((0, 3), dtype=np.float64)
    cloud_rgb = np.concatenate(
        all_cloud_rgb, axis=0) if all_cloud_rgb else np.zeros((0, 3), dtype=np.uint8)
    if cloud_xyz.size:
        cloud_xyz = cloud_xyz * scale
    output_cloud = output_dir / "sparse_cloud.ply"
    build_sparse_cloud(cloud_xyz, cloud_rgb, out_path=output_cloud)

    poses_path = output_dir / "poses.json"
    write_poses_json(poses_path, pose_records, intrinsics,
                     width, height, scale, acc_cm)

    return {
        "frames": pose_records,
        "intrinsics": intrinsics,
        "scale_estimate_m_per_unit": scale,
        "accuracy_estimate_cm": acc_cm,
        "output_dir": str(output_dir),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate camera poses and a sparse point cloud from Phase 1 output.")
    parser.add_argument("--frames-dir", required=True,
                        help="Directory containing Phase 1 output, e.g. phase1_ingestion/output")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write the Phase 2 outputs")
    args = parser.parse_args()
    result = run_pipeline(args.frames_dir, args.out_dir)
    print(json.dumps({
        "num_frames": len(result["frames"]),
        "scale_estimate_m_per_unit": result["scale_estimate_m_per_unit"],
        "accuracy_estimate_cm": result["accuracy_estimate_cm"],
        "intrinsics": result["intrinsics"],
    }, indent=2))


if __name__ == "__main__":
    main()
