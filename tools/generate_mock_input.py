"""Step 0 - synthetic Phase-1 + Phase-2 output generator.

Builds a tiny, fully-known toy scene (ground + two buildings + a road + trees +
one moving "car") photographed from a **single-pass arc** of cameras with a
limited (~100 deg) viewing angle, then writes it out in the EXACT schema Phase 3
consumes:

    <root>/phase1_ingestion/output/
        frames/frame_000001.jpg ...
        masks/frame_000001_mask.png ...        (255 = dynamic object)
        gt_depth/frame_000001.npy ...          (extra: metric GT depth, used by the
                                                mock monocular-depth stand-in)
        manifest.json                          (GPS w/ noise, IMU, blur_score)
    <root>/phase2_pose/output/
        poses.json                             (R,t, intrinsics, scale estimate)
        sparse_cloud.ply                       (noisy metric SfM points, RGB)

Run directly:
    python tools/generate_mock_input.py --root phase3_reconstruction/mock_data --frames 18
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# allow running as a script from the repo root
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3_reconstruction.geometry import (Camera, Intrinsics, look_at,
                                            rotation_to_euler_zyx)
from phase3_reconstruction.io.ply import write_point_cloud
from phase3_reconstruction.render import rasterize_points

LAT0, LON0, ALT0 = 12.9716, 77.5946, 915.0
_M_PER_DEG_LAT = 111_320.0


# --------------------------------------------------------------------------- #
# Scene geometry -> coloured point cloud
# --------------------------------------------------------------------------- #
def _grid(ax, bx, ay, by, step):
    xs = np.arange(ax, bx + 1e-6, step)
    ys = np.arange(ay, by + 1e-6, step)
    gx, gy = np.meshgrid(xs, ys)
    return gx.reshape(-1), gy.reshape(-1)


def _build_scene(rng) -> "tuple[np.ndarray, np.ndarray]":
    P, C = [], []

    # --- ground plane (40 x 40 m, z=0) --------------------------------- #
    gx, gy = _grid(-20, 20, -20, 20, 0.16)
    gz = np.zeros_like(gx) + rng.normal(0, 0.01, gx.size)
    checker = (((gx // 2).astype(int) + (gy // 2).astype(int)) % 2)
    base = np.where(checker[:, None] == 0,
                    np.array([[90, 120, 70]]), np.array([[110, 140, 85]])).astype(np.float32)
    base += rng.normal(0, 6, base.shape)
    P.append(np.stack([gx, gy, gz], 1)); C.append(base)

    # --- road (asphalt strip along x) -------------------------------- #
    rx, ry = _grid(-20, 20, -1.6, 1.6, 0.13)
    rz = np.zeros_like(rx) + 0.02
    rc = np.tile(np.array([[55, 55, 60]], np.float32), (rx.size, 1))
    rc += rng.normal(0, 4, rc.shape)
    P.append(np.stack([rx, ry, rz], 1)); C.append(rc)

    # --- building A (big block, NORTH of the road) --------------- #
    P2, C2 = _box(-8, 2, 4, 14, 0.0, 9.0, 0.14, (185, 170, 150),
                  windows=True, rng=rng)
    P.append(P2); C.append(C2)

    # --- building B (small red block, SOUTH of the road) -------- #
    P3, C3 = _box(6, 14, -14, -6, 0.0, 4.0, 0.14, (170, 95, 80),
                  windows=False, rng=rng)
    P.append(P3); C.append(C3)

    # --- a few trees (green blobs), clear of the road corridor -- #
    for cx, cy in [(-14, 11), (-4, 15), (16, -11), (-16, -9)]:
        n = 1400
        r = rng.normal(0, 1.2, (n, 3)) * np.array([1, 1, 1.6])
        r[:, 2] = np.abs(r[:, 2]) + 0.5
        pts = r + np.array([cx, cy, 2.2])
        col = np.tile(np.array([[40, 100, 45]], np.float32), (n, 1)) + rng.normal(0, 10, (n, 3))
        P.append(pts); C.append(col)

    xyz = np.concatenate(P, 0).astype(np.float64)
    rgb = np.clip(np.concatenate(C, 0), 0, 255).astype(np.uint8)
    return xyz, rgb


def _box(ax, bx, ay, by, az, bz, step, colour, windows, rng):
    P, C = [], []
    col = np.array(colour, np.float32)

    def add(pts, tint=0.0):
        c = np.tile(col[None], (len(pts), 1)) + rng.normal(0, 7, (len(pts), 3)) + tint
        P.append(pts); C.append(c)

    # 4 walls
    ys, zs = _grid(ay, by, az, bz, step)
    add(np.stack([np.full_like(ys, ax), ys, zs], 1))
    add(np.stack([np.full_like(ys, bx), ys, zs], 1))
    xs, zs = _grid(ax, bx, az, bz, step)
    add(np.stack([xs, np.full_like(xs, ay), zs], 1))
    add(np.stack([xs, np.full_like(xs, by), zs], 1))
    # roof
    xs, ys = _grid(ax, bx, ay, by, step)
    add(np.stack([xs, ys, np.full_like(xs, bz)], 1), tint=-35.0)

    P = np.concatenate(P, 0); C = np.concatenate(C, 0)
    if windows:
        # darken a regular lattice of "windows" on the walls
        onwall = (np.abs(P[:, 2] - np.round(P[:, 2])) < step) & (P[:, 2] > 1.0) & (P[:, 2] < bz - 0.8)
        wx = (np.sin(P[:, 0] * 2.2) > 0.6) & (np.sin(P[:, 1] * 2.2) > 0.6)
        win = onwall & wx
        C[win] = C[win] * 0.35 + np.array([20, 25, 35])
    return P, np.clip(C, 0, 255).astype(np.float32)


def _car_points(center, rng):
    """~4m x 2m x 1.5m bright-blue box on the road."""
    xs, ys = _grid(-2.0, 2.0, -1.0, 1.0, 0.1)
    top = np.stack([xs, ys, np.full_like(xs, 1.5)], 1)
    xs2, zs2 = _grid(-2.0, 2.0, 0.1, 1.5, 0.1)
    s1 = np.stack([xs2, np.full_like(xs2, -1.0), zs2], 1)
    s2 = np.stack([xs2, np.full_like(xs2, 1.0), zs2], 1)
    ys3, zs3 = _grid(-1.0, 1.0, 0.1, 1.5, 0.1)
    s3 = np.stack([np.full_like(ys3, -2.0), ys3, zs3], 1)
    s4 = np.stack([np.full_like(ys3, 2.0), ys3, zs3], 1)
    pts = np.concatenate([top, s1, s2, s3, s4], 0) + np.asarray(center)
    col = np.tile(np.array([[40, 90, 210]], np.float32), (len(pts), 1)) + rng.normal(0, 8, (len(pts), 3))
    return pts.astype(np.float64), np.clip(col, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
def _enu_to_gps(x, y, z, rng, h_sigma=1.8, v_sigma=2.5):
    lat = LAT0 + (y + rng.normal(0, h_sigma)) / _M_PER_DEG_LAT
    lon = LON0 + (x + rng.normal(0, h_sigma)) / (_M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
    alt = ALT0 + z + rng.normal(0, v_sigma)
    return {"lat": round(lat, 8), "lon": round(lon, 8), "alt": round(alt, 3)}


def generate_mock_scene(root: str, n_frames: int = 18, seed: int = 0,
                        img_w: int = 640, img_h: int = 384,
                        pose_noise_deg: float = 0.0,
                        pose_noise_m: float = 0.0) -> dict:
    root = Path(root)
    p1 = root / "phase1_ingestion" / "output"
    p2 = root / "phase2_pose" / "output"
    for d in (p1 / "frames", p1 / "masks", p1 / "gt_depth", p2):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    static_xyz, static_rgb = _build_scene(rng)

    fx = 480.0
    intr = Intrinsics(fx=fx, fy=fx, cx=img_w / 2, cy=img_h / 2, width=img_w, height=img_h)

    look = np.array([0.0, 0.0, 3.0])
    radius, altitude = 38.0, 26.0
    az0, az1 = -50.0, 50.0                      # limited single-pass arc

    manifest_frames, pose_frames = [], []
    fps = 30.0
    t0 = 1_699_999_000.0

    for i in range(n_frames):
        fid = f"{i + 1:06d}"
        az = math.radians(az0 + (az1 - az0) * i / max(n_frames - 1, 1))
        eye = np.array([radius * math.cos(az), radius * math.sin(az), altitude])
        jitter = rng.normal(0, 0.4, 3)
        R = look_at(eye, look + jitter, up=np.array([0.0, 0.0, 1.0]))
        t = -R @ eye
        cam = Camera(frame_id=fid, R=R, t=t, intr=intr)

        # dynamic car position for this frame (drives along the clear road corridor)
        car_c = np.array([-16.0 + 30.0 * i / max(n_frames - 1, 1), 0.0, 0.0])
        car_xyz, car_rgb = _car_points(car_c, rng)

        all_xyz = np.concatenate([static_xyz, car_xyz], 0)
        all_rgb = np.concatenate([static_rgb, car_rgb], 0).astype(np.float32) / 255.0

        rgb, depth, _ = rasterize_points(all_xyz, all_rgb, cam, point_radius_px=2,
                                         background=(155, 175, 200))
        # car-only pass -> dynamic mask
        _, car_depth, _ = rasterize_points(car_xyz, car_rgb.astype(np.float32) / 255.0,
                                           cam, point_radius_px=2, background=(0, 0, 0))
        with np.errstate(invalid="ignore"):
            mask = np.isfinite(car_depth) & (car_depth <= np.nan_to_num(depth, nan=1e9) + 0.3)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

        # --- degrade the RGB a bit (lighting / motion blur / jpeg) ---- #
        gain = float(rng.uniform(0.85, 1.15))
        img = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        heavy_blur = i in (max(0, n_frames // 3), min(n_frames - 1, 2 * n_frames // 3))
        ksz = 7 if heavy_blur else 3
        img = cv2.GaussianBlur(img, (ksz, ksz), 0)
        blur_score = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                                         cv2.CV_64F).var())

        cv2.imwrite(str(p1 / "frames" / f"frame_{fid}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(p1 / "masks" / f"frame_{fid}_mask.png"),
                    (mask.astype(np.uint8) * 255))
        np.save(str(p1 / "gt_depth" / f"frame_{fid}.npy"),
                depth.astype(np.float32))            # NaN where empty

        roll, pitch, yaw = rotation_to_euler_zyx(R)
        imu = None if i == 1 else {
            "roll": round(roll + float(rng.normal(0, 0.01)), 5),
            "pitch": round(pitch + float(rng.normal(0, 0.01)), 5),
            "yaw": round(yaw + float(rng.normal(0, 0.02)), 5),
        }
        manifest_frames.append({
            "frame_id": fid,
            "timestamp": round(t0 + i / fps, 3),
            "gps": _enu_to_gps(eye[0], eye[1], eye[2], rng),
            "imu": imu,
            "blur_score": round(blur_score, 3),
            "mask_path": f"masks/frame_{fid}_mask.png",
        })

        # --- poses.json entry (optionally perturbed) ---------------- #
        Rn, tn = R, t
        if pose_noise_deg > 0 or pose_noise_m > 0:
            ax = rng.normal(0, math.radians(pose_noise_deg), 3)
            th = np.linalg.norm(ax) + 1e-12
            k = ax / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            dR = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
            Rn = dR @ R
            tn = t + rng.normal(0, pose_noise_m, 3)
        pose_frames.append({
            "frame_id": fid,
            "R": [[round(float(v), 9) for v in row] for row in Rn],
            "t": [round(float(v), 6) for v in tn],
            "confidence": round(float(np.clip(rng.normal(0.9, 0.04), 0.6, 0.99)), 3),
        })

    # --- manifest.json ------------------------------------------- #
    (p1 / "manifest.json").write_text(json.dumps({
        "frames": manifest_frames,
        "video_meta": {"fps": int(fps), "resolution": [img_w, img_h], "codec": "h264"},
    }, indent=2), encoding="utf-8")

    # --- poses.json -------------------------------------------- #
    (p2 / "poses.json").write_text(json.dumps({
        "frames": pose_frames,
        "intrinsics": {"fx": fx, "fy": fx, "cx": img_w / 2, "cy": img_h / 2,
                       "width": img_w, "height": img_h},
        "scale_estimate_m_per_unit": 1.0,
        "accuracy_estimate_cm": 8.5,
    }, indent=2), encoding="utf-8")

    # --- sparse_cloud.ply : noisy metric subset of STATIC points --- #
    n_sparse = min(900, len(static_xyz))
    sel = rng.choice(len(static_xyz), n_sparse, replace=False)
    sp_xyz = static_xyz[sel] + rng.normal(0, 0.04, (n_sparse, 3))
    sp_rgb = static_rgb[sel]
    write_point_cloud(str(p2 / "sparse_cloud.ply"), sp_xyz, sp_rgb,
                      comments=["synthetic SfM sparse cloud (mock)",
                                "metric world frame (metres), gaussian noise sigma=0.04 m"])

    info = {
        "root": str(root),
        "phase1_output": str(p1),
        "phase2_output": str(p2),
        "n_frames": n_frames,
        "n_static_points": int(len(static_xyz)),
        "n_sparse_points": int(n_sparse),
        "resolution": [img_w, img_h],
    }
    print(json.dumps(info, indent=2))
    return info


def _cli():
    ap = argparse.ArgumentParser(description="generate a synthetic Phase-1/Phase-2 scene")
    ap.add_argument("--root", default="phase3_reconstruction/mock_data")
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--pose-noise-deg", type=float, default=0.0)
    ap.add_argument("--pose-noise-m", type=float, default=0.0)
    a = ap.parse_args()
    generate_mock_scene(a.root, n_frames=a.frames, seed=a.seed,
                        img_w=a.width, img_h=a.height,
                        pose_noise_deg=a.pose_noise_deg, pose_noise_m=a.pose_noise_m)


if __name__ == "__main__":
    _cli()
