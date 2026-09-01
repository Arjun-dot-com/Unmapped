"""Estimate a metric camera trajectory from Phase 1 telemetry.

When PyCOLMAP is available it can be integrated upstream; this implementation
provides a deterministic GPS/ENU trajectory fallback that satisfies the exact
Phase 2 contract and reports its accuracy limitation explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase3_reconstruction.geometry import look_at
from phase3_reconstruction.io.ply import write_point_cloud


def _enu(gps, origin):
    lat0, lon0, alt0 = origin
    lat, lon, alt = float(gps.get("lat", lat0)), float(gps.get("lon", lon0)), float(gps.get("alt", alt0))
    radius = 6378137.0
    east = np.deg2rad(lon - lon0) * radius * np.cos(np.deg2rad(lat0))
    north = np.deg2rad(lat - lat0) * radius
    return np.array([east, north, alt - alt0], dtype=float)


def estimate(phase1_dir: str | Path, output: str | Path, focal_fraction: float = 0.8) -> dict:
    phase1_dir, output = Path(phase1_dir), Path(output)
    payload = json.loads((phase1_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("Phase 1 manifest has no frames")
    gps_frames = [f for f in frames if isinstance(f.get("gps"), dict) and f["gps"].get("lat") is not None]
    if gps_frames:
        g0 = gps_frames[0]["gps"]
        origin = (float(g0["lat"]), float(g0["lon"]), float(g0.get("alt", 0.0)))
        centers = [_enu(f.get("gps") or g0, origin) for f in frames]
        note = "GPS-derived ENU trajectory; use PyCOLMAP + GPS/IMU fusion for production accuracy"
        accuracy = 999.0
    else:
        origin = (0.0, 0.0, 0.0)
        centers = [np.array([i * 0.5, 0.0, 0.0]) for i in range(len(frames))]
        note = "synthetic fallback trajectory because Phase 1 contains no GPS"
        accuracy = 999.0
    centers_array = np.asarray(centers)
    target = centers_array.mean(axis=0) + np.array([0.0, 0.0, -1.0])
    video_meta = payload.get("video_meta", {})
    width, height = (video_meta.get("resolution") or [640, 384])[:2]
    fx = fy = float(max(width, height) * focal_fraction)
    intrinsics = {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0, "width": int(width), "height": int(height)}
    pose_frames = []
    for frame, center in zip(frames, centers_array):
        rotation = look_at(center, target)
        pose_frames.append({"frame_id": str(frame["frame_id"]), "R": rotation.tolist(), "t": (-rotation @ center).tolist(), "confidence": 0.35 if gps_frames else 0.1})
    output.mkdir(parents=True, exist_ok=True)
    poses = {"frames": pose_frames, "intrinsics": intrinsics, "scale_estimate_m_per_unit": 1.0, "accuracy_estimate_cm": accuracy, "origin": {"lat": origin[0], "lon": origin[1], "alt": origin[2]}, "method": "gps_enu_fallback", "notes": note}
    (output / "poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")
    colors = np.tile(np.array([[80, 160, 220]], dtype=np.uint8), (len(centers_array), 1))
    write_point_cloud(output / "sparse_cloud.ply", centers_array, colors, comments=[note])
    return {"output": str(output), "frames": len(pose_frames), "poses": str(output / "poses.json"), "sparse_cloud": str(output / "sparse_cloud.ply"), "accuracy_estimate_cm": accuracy, "method": "gps_enu_fallback"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 metric camera pose estimation")
    parser.add_argument("--frames-dir", required=True, help="Phase 1 output directory")
    parser.add_argument("--out", default="phase2_pose/output")
    parser.add_argument("--focal-fraction", type=float, default=0.8)
    args = parser.parse_args(argv)
    print(json.dumps(estimate(args.frames_dir, args.out, args.focal_fraction), indent=2))


if __name__ == "__main__":
    main()
