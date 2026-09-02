from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from pyproj import Geod
except Exception:  # pragma: no cover - optional dependency
    Geod = None

try:
    from phase3_reconstruction.io.ply import write_point_cloud
except Exception:  # pragma: no cover - fallback import path
    write_point_cloud = None

try:  # pragma: no cover - optional dependency
    import pycolmap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pycolmap = None

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from lightglue import LightGlue, SuperPoint  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LightGlue = None
    SuperPoint = None


def estimate_intrinsics(width: int, height: int, fallback_scale: float = 1.15) -> Dict[str, float]:
    """Estimate intrinsics from image size when Phase 1 metadata does not provide calibration."""
    width = max(1, int(width))
    height = max(1, int(height))
    fx = fy = max(width, height) * fallback_scale
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
    }


def _normalize_frame_id(frame_id: Any) -> str:
    s = str(frame_id).strip()
    if s.startswith("frame_"):
        s = s[6:]
    if s.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        s = Path(s).stem
    if s.startswith("frame_"):
        s = s[6:]
    return s


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 1 manifest not found: {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc


def resolve_frame_image(frames_root: Path, record: Dict[str, Any]) -> Path:
    given = record.get("image_path") or record.get("path")
    if given:
        p = Path(given)
        if not p.is_absolute():
            p = frames_root / p
        if p.exists():
            return p
    fid = _normalize_frame_id(record.get("frame_id"))
    candidates = [
        frames_root / "frames" / f"frame_{fid}.jpg",
        frames_root / "frames" / f"frame_{fid}.jpeg",
        frames_root / "frames" / f"frame_{fid}.png",
        frames_root / "frames" / f"frame_{fid}.bmp",
        frames_root / f"frame_{fid}.jpg",
        frames_root / f"frame_{fid}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot locate frame image for frame_id={fid!r} under {frames_root}")


def resolve_frame_mask(frames_root: Path, record: Dict[str, Any], shape_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    mask_rel = record.get("mask_path")
    if mask_rel:
        p = Path(mask_rel)
        if not p.is_absolute():
            p = frames_root / p
        if p.exists():
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                arr = (mask > 127).astype(bool)
                if shape_hw is not None and arr.shape[:2] != shape_hw:
                    arr = cv2.resize(arr.astype(
                        np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                return arr
    fid = _normalize_frame_id(record.get("frame_id"))
    candidates = [
        frames_root / "masks" / f"frame_{fid}_mask.png",
        frames_root / "masks" / f"frame_{fid}_mask.jpg",
        frames_root / "masks" / f"frame_{fid}.png",
        frames_root / "masks" / f"frame_{fid}.jpg",
        frames_root / f"frame_{fid}_mask.png",
    ]
    for p in candidates:
        if p.exists():
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                arr = (mask > 127).astype(bool)
                if shape_hw is not None and arr.shape[:2] != shape_hw:
                    arr = cv2.resize(arr.astype(
                        np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                return arr
    if shape_hw is None:
        return np.zeros((0, 0), dtype=bool)
    return np.zeros(shape_hw, dtype=bool)


def load_rgb_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _camera_from_imu(imu: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(imu, dict):
        return None
    yaw = imu.get("yaw")
    pitch = imu.get("pitch")
    roll = imu.get("roll")
    if yaw is None and pitch is None and roll is None:
        return None
    yaw = float(yaw) if yaw is not None else 0.0
    pitch = float(pitch) if pitch is not None else 0.0
    roll = float(roll) if roll is not None else 0.0
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    return R


def _orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def _gps_enu_for_record(record: Dict[str, Any], ref: Dict[str, float]) -> Optional[np.ndarray]:
    gps = record.get("gps")
    if not isinstance(gps, dict):
        return None
    try:
        lat = float(gps["lat"])
        lon = float(gps["lon"])
        alt = float(gps["alt"])
    except (KeyError, TypeError, ValueError):
        return None
    return gps_to_enu(lat, lon, alt, float(ref["lat"]), float(ref["lon"]), float(ref["alt"]))


def gps_to_enu(lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float, ref_alt: float) -> np.ndarray:
    """Convert GPS WGS84 coordinates into a local metric ENU frame rooted at the first valid GPS observation."""
    try:
        if Geod is not None:
            g = Geod(ellps="WGS84")
            x, y, _ = g.inv(ref_lon, ref_lat, lon, lat)
            return np.array([x, y, alt - ref_alt], dtype=np.float64)
    except Exception:  # pragma: no cover - optional fallback
        pass
    lat0 = math.radians(ref_lat)
    x = (lon - ref_lon) * math.cos(lat0) * 111_320.0
    y = (lat - ref_lat) * 111_320.0
    z = alt - ref_alt
    return np.array([x, y, z], dtype=np.float64)


def estimate_scale_from_gps(frame_records: Iterable[Dict[str, Any]], camera_centres: List[np.ndarray]) -> Tuple[float, float, List[float]]:
    entries = list(frame_records)
    if len(entries) < 2 or len(camera_centres) < 2:
        return 1.0, 5000.0, []

    ref = entries[0].get("gps")
    if not ref or not all(k in ref for k in ("lat", "lon", "alt")):
        return 1.0, 5000.0, []
    ref_enu = {"lat": float(ref["lat"]), "lon": float(
        ref["lon"]), "alt": float(ref["alt"])}
    enu_points: List[Optional[np.ndarray]] = []
    for rec in entries:
        enu_points.append(_gps_enu_for_record(rec, ref_enu))

    scale_samples: List[float] = []
    for i in range(1, min(len(entries), len(camera_centres))):
        a = enu_points[i - 1]
        b = enu_points[i]
        if a is None or b is None:
            continue
        gps_dist = float(np.linalg.norm(b - a))
        recon_dist = float(np.linalg.norm(
            camera_centres[i] - camera_centres[i - 1]))
        if gps_dist <= 1e-6 or recon_dist <= 1e-6:
            continue
        scale_samples.append(gps_dist / recon_dist)

    if not scale_samples:
        return 1.0, 5000.0, []
    arr = np.asarray(scale_samples, dtype=np.float64)
    med = float(np.median(arr))
    if not np.isfinite(med) or med <= 1e-9:
        return 1.0, 5000.0, []
    mad = float(np.median(np.abs(arr - med)))
    if mad > 0.0 and np.isfinite(mad):
        keep = np.abs(arr - med) <= 6.0 * mad + 1e-6
        arr = arr[keep]
        if arr.size == 0:
            return 1.0, 5000.0, []
        med = float(np.median(arr))
    acc = float(np.median(np.abs(arr - med)) * 100.0)
    return med, max(acc, 5.0), arr.tolist()


def _lightglue_match_pair(prev_img: np.ndarray, curr_img: np.ndarray, prev_mask: np.ndarray, curr_mask: np.ndarray, K: np.ndarray):
    if LightGlue is None or SuperPoint is None or torch is None:
        return None
    try:
        import lightglue
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        prev_valid = None if prev_mask is None or prev_mask.size == 0 else (
            ~prev_mask).astype(np.uint8) * 255
        curr_valid = None if curr_mask is None or curr_mask.size == 0 else (
            ~curr_mask).astype(np.uint8) * 255
        extractor = SuperPoint(features="superpoint", nms_radius=4)
        feats0 = extractor.extract(
            {"image": prev_gray, "mask": prev_valid} if prev_valid is not None else {"image": prev_gray})
        feats1 = extractor.extract(
            {"image": curr_gray, "mask": curr_valid} if curr_valid is not None else {"image": curr_gray})

        matcher = LightGlue(features="superpoint")
        out = matcher({"image0": feats0, "image1": feats1})
        if not isinstance(out, dict):
            return None
        match_indices = out.get("matches") or out.get(
            "matches0") or out.get("indices")
        if match_indices is None:
            return None
        if hasattr(match_indices, "cpu"):
            match_indices = match_indices.cpu().numpy()
        matches = np.asarray(match_indices, dtype=np.int32)
        if matches.size == 0 or matches.ndim != 2 or matches.shape[1] < 2:
            return None
        kp0 = np.asarray(feats0["keypoints"], dtype=np.float32)
        kp1 = np.asarray(feats1["keypoints"], dtype=np.float32)
        pts1 = kp0[matches[:, 0]]
        pts2 = kp1[matches[:, 1]]
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None:
            return None
        mask = mask.reshape(-1)
        pts1i = pts1[mask > 0]
        pts2i = pts2[mask > 0]
        if len(pts1i) < 8:
            return None
        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1i, pts2i, K)
        if R_rel is None or t_rel is None:
            return None
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R_rel, t_rel))
        pts4d = cv2.triangulatePoints(P1, P2, pts1i.T, pts2i.T)
        pts3d = pts4d[:3] / np.where(np.abs(pts4d[3]) > 1e-8, pts4d[3], 1e-8)
        valid = np.isfinite(pts3d).all(axis=0) & (np.abs(pts3d[2]) > 1e-6)
        pts3d = pts3d[:, valid]
        inlier_count = int(mask.sum())
        score = float(inlier_count / max(len(matches), 1))
        return R_rel, t_rel, inlier_count, score, pts3d.T, {"method": "lightglue"}
    except Exception:
        return None


def feature_match_pair(prev_img: np.ndarray, curr_img: np.ndarray, prev_mask: np.ndarray, curr_mask: np.ndarray, K: np.ndarray):
    if prev_img.size == 0 or curr_img.size == 0:
        return None, None, 0, 0.0, np.empty((0, 3)), {"method": "empty"}

    try:
        light = _lightglue_match_pair(
            prev_img, curr_img, prev_mask, curr_mask, K)
        if light is not None:
            return light
    except Exception:
        pass

    gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
    valid_prev = (~prev_mask).astype(np.uint8) * 255
    valid_curr = (~curr_mask).astype(np.uint8) * 255

    orb = cv2.ORB_create(nfeatures=4000, edgeThreshold=15, patchSize=31)
    kp1, des1 = orb.detectAndCompute(gray_prev, valid_prev)
    kp2, des2 = orb.detectAndCompute(gray_curr, valid_curr)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, 0, 0.0, np.empty((0, 3)), {"method": "orb-fallback"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m is None or n is None:
            continue
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, None, 0, 0.0, np.empty((0, 3)), {"method": "orb-fallback"}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        return None, None, 0, 0.0, np.empty((0, 3)), {"method": "orb-fallback"}
    mask = mask.reshape(-1)
    inlier_count = int(mask.sum())
    if inlier_count < 8:
        return None, None, 0, 0.0, np.empty((0, 3)), {"method": "orb-fallback"}

    pts1i = pts1[mask > 0]
    pts2i = pts2[mask > 0]
    _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1i, pts2i, K)
    if R_rel is None or t_rel is None:
        return None, None, 0, 0.0, np.empty((0, 3)), {"method": "orb-fallback"}

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R_rel, t_rel))
    pts4d = cv2.triangulatePoints(P1, P2, pts1i.T, pts2i.T)
    pts3d = pts4d[:3] / np.where(np.abs(pts4d[3]) > 1e-8, pts4d[3], 1e-8)
    valid = np.isfinite(pts3d).all(axis=0) & (np.abs(pts3d[2]) > 1e-6)
    pts3d = pts3d[:, valid]
    score = float(inlier_count / max(len(good), 1))
    return R_rel, t_rel, inlier_count, score, pts3d.T, {"method": "orb-fallback"}


def reconstruct_with_pycolmap(frame_records: List[Dict[str, Any]], out_dir: Path, intrinsics: Dict[str, float], width: int, height: int) -> Optional[Dict[str, Any]]:
    if pycolmap is None:
        return None
    try:
        tmpdir = out_dir / "pycolmap_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        imgs = tmpdir / "images"
        imgs.mkdir(parents=True, exist_ok=True)
        prepared = 0
        for rec in frame_records:
            src = resolve_frame_image(out_dir.parent, rec)
            if src.exists():
                dst = imgs / f"{str(rec.get('frame_id'))}.jpg"
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
                prepared += 1
        if prepared < 2:
            return None

        for name in ("extract_features", "match_exhaustive", "incremental_mapping", "mapper"):
            fn = getattr(pycolmap, name, None)
            if not callable(fn):
                continue
            try:
                if name == "extract_features":
                    result = fn(database_path=str(
                        tmpdir / "database.db"), image_path=str(imgs))
                elif name == "match_exhaustive":
                    result = fn(database_path=str(tmpdir / "database.db"))
                elif name == "incremental_mapping":
                    result = fn(database_path=str(tmpdir / "database.db"),
                                image_path=str(imgs), output_path=str(tmpdir / "model"))
                else:
                    result = fn(str(tmpdir / "database.db"),
                                str(imgs), str(tmpdir / "model"))
                return {"backend": "pycolmap", "status": f"invoked:{name}", "images": str(imgs), "intrinsics": intrinsics, "width": width, "height": height, "result": result}
            except TypeError:
                try:
                    if name == "extract_features":
                        result = fn(str(tmpdir / "database.db"), str(imgs))
                    elif name == "match_exhaustive":
                        result = fn(str(tmpdir / "database.db"))
                    elif name == "incremental_mapping":
                        result = fn(str(tmpdir / "database.db"),
                                    str(imgs), str(tmpdir / "model"))
                    else:
                        result = fn(str(tmpdir / "database.db"),
                                    str(imgs), str(tmpdir / "model"))
                    return {"backend": "pycolmap", "status": f"invoked:{name}", "images": str(imgs), "intrinsics": intrinsics, "width": width, "height": height, "result": result}
                except Exception:
                    continue
            except Exception:
                continue
        return {"backend": "pycolmap", "status": "pycolmap available but no usable reconstruction call succeeded", "images": str(imgs), "intrinsics": intrinsics, "width": width, "height": height}
    except Exception:
        return None


def align_sfm_to_gps(sfm_centres: np.ndarray, gps_enu: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if sfm_centres.shape[0] != gps_enu.shape[0] or sfm_centres.shape[0] < 2:
        return 1.0, np.eye(3), np.zeros(3)
    A = np.asarray(sfm_centres, dtype=np.float64)
    B = np.asarray(gps_enu, dtype=np.float64)
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A0 = A - a_mean
    B0 = B - b_mean
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = float(np.sum(S) / max(np.sum(A0 * A0), 1e-12))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    t = b_mean - scale * (R @ a_mean)
    return scale, R, t


def georeference_pose_records(frame_records: List[Dict[str, Any]], pose_records: List[Dict[str, Any]], camera_centres: List[np.ndarray]) -> Tuple[float, float, List[float]]:
    ref = None
    valid = []
    for rec in frame_records:
        gps = rec.get("gps")
        if not isinstance(gps, dict):
            continue
        if not all(k in gps for k in ("lat", "lon", "alt")):
            continue
        if ref is None:
            ref = {"lat": float(gps["lat"]), "lon": float(
                gps["lon"]), "alt": float(gps["alt"])}
        p = _gps_enu_for_record(rec, ref)
        if p is not None:
            valid.append((len(valid), rec, p))
    if ref is None or len(valid) < 2:
        return 1.0, 5000.0, []
    sfm_points = np.asarray([camera_centres[idx] for idx, _, _ in valid if idx < len(
        camera_centres)], dtype=np.float64)
    gps_points = np.asarray([p for _, _, p in valid], dtype=np.float64)
    if gps_points.shape[0] < 2 or sfm_points.shape[0] < 2:
        return 1.0, 5000.0, []
    scale, R_align, t_world = align_sfm_to_gps(sfm_points, gps_points)
    residuals = np.linalg.norm(
        gps_points - (scale * (R_align @ sfm_points.T).T + t_world), axis=1)
    accuracy_cm = float(np.median(residuals) * 100.0)
    return scale, max(float(accuracy_cm), 5.0), residuals.tolist()


def build_sparse_cloud(points_xyz: np.ndarray, colors: Optional[np.ndarray] = None, out_path: Optional[Path] = None) -> None:
    points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        colors = np.full((points_xyz.shape[0], 3), 180, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points_xyz.shape[0] != colors.shape[0]:
        colors = np.tile(
            np.array([180, 180, 180], dtype=np.uint8), (points_xyz.shape[0], 1))
    if out_path is not None:
        if write_point_cloud is not None:
            write_point_cloud(str(out_path), points_xyz, colors, comments=[
                              "Phase 2 sparse SfM cloud", "metric world frame (metres)"])
            return
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points_xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(points_xyz.shape[0]):
                x, y, z = points_xyz[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def write_poses_json(out_path: Path, frames: List[Dict[str, Any]], intrinsics: Dict[str, float], width: int, height: int, scale: float, acc_cm: float) -> None:
    payload = {
        "frames": frames,
        "intrinsics": {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "width": int(width),
            "height": int(height),
        },
        "scale_estimate_m_per_unit": float(scale),
        "accuracy_estimate_cm": float(acc_cm),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_pose_output(frames: List[Dict[str, Any]], scale: float) -> None:
    for item in frames:
        item["t"] = [float(v) * float(scale) for v in item["t"]]


def detection_backend_summary() -> Dict[str, Any]:
    return {
        "pycolmap": pycolmap is not None,
        "lightglue": LightGlue is not None and SuperPoint is not None and torch is not None,
        "opencv_orb": True,
        "pyproj": Geod is not None,
    }


def _compute_confidence(inlier_count: int, score: float, gps_consistency: Optional[float] = None, reproj_error: Optional[float] = None) -> float:
    conf = 0.40 * min(1.0, inlier_count / 80.0)
    conf += 0.35 * float(np.clip(score, 0.0, 1.0))
    if gps_consistency is not None:
        conf += 0.15 * float(np.clip(1.0 - gps_consistency, 0.0, 1.0))
    if reproj_error is not None:
        conf += 0.10 * \
            float(np.clip(1.0 / (1.0 + max(reproj_error, 0.0)), 0.0, 1.0))
    return float(np.clip(conf, 0.0, 1.0))
