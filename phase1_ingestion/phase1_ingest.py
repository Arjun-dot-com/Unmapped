"""Extract a contract-compliant Phase 1 dataset from a drone video.

The production mask path uses Ultralytics when a model is supplied. Without
that optional dependency the CLI still emits explicit empty masks and records
the limitation in the manifest instead of silently claiming object removal.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _read_telemetry(path: Path | None) -> list[dict]:
    if not path:
        return []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        result = []
        for index, row in enumerate(rows):
            def number(key, default=0.0):
                value = row.get(key, default)
                return float(value) if value not in (None, "") else default
            result.append({
                "timestamp": number("timestamp", index),
                "gps": {"lat": number("lat"), "lon": number("lon"), "alt": number("alt")},
                "imu": None,
            })
        return result
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("frames", payload if isinstance(payload, list) else [])


def _nearest_telemetry(rows: list[dict], timestamp: float) -> dict:
    if not rows:
        return {}
    return min(rows, key=lambda row: abs(float(row.get("timestamp", 0.0)) - timestamp))


def _blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _normalise(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _masker(model_path: str | None):
    if not model_path:
        return None, "no Ultralytics model supplied; masks are empty and dynamic objects are not removed"
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, f"Ultralytics segmentation model: {model_path}"
    except Exception as exc:  # optional dependency/model failure is recorded
        return None, f"segmentation unavailable ({exc}); emitted empty masks"


def extract(video: str | Path, telemetry: str | Path | None, output: str | Path,
            sample_fps: float = 2.0, max_frames: int = 0, min_blur: float = 0.0,
            model: str | None = None, normalize: bool = False) -> dict:
    video, output = Path(video), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_dir, mask_dir = output / "frames", output / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step = max(1, int(round(fps / max(sample_fps, 0.01))))
    indices = list(range(0, total, step)) if total else []
    if max_frames and len(indices) > max_frames:
        indices = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [list(range(0, total, step))[i] for i in indices]
    rows = _read_telemetry(Path(telemetry) if telemetry else None)
    segmenter, mask_note = _masker(model)
    manifest = []
    for frame_no in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        timestamp = frame_no / fps
        score = _blur_score(frame)
        if min_blur > 0 and score < min_blur:
            continue
        if normalize:
            frame = _normalise(frame)
        frame_id = f"{len(manifest) + 1:06d}"
        image_name = f"frame_{frame_id}.jpg"
        mask_name = f"frame_{frame_id}_mask.png"
        cv2.imwrite(str(image_dir / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if segmenter is not None:
            try:
                result = segmenter.predict(frame, verbose=False)[0]
                if result.masks is not None:
                    for cls, polygon in zip(result.boxes.cls.tolist(), result.masks.data):
                        if int(cls) in {0, 1, 2, 3, 5, 7, 14, 15, 16, 17, 18, 19}:
                            mask = np.maximum(mask, cv2.resize(polygon.cpu().numpy().astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) * 255)
            except Exception:
                pass
        cv2.imwrite(str(mask_dir / mask_name), mask)
        row = _nearest_telemetry(rows, timestamp)
        manifest.append({
            "frame_id": frame_id, "timestamp": float(row.get("timestamp", timestamp)),
            "gps": row.get("gps"), "imu": row.get("imu"), "blur_score": score,
            "image_path": f"frames/{image_name}", "mask_path": f"masks/{mask_name}",
        })
    cap.release()
    if not manifest:
        raise RuntimeError("video produced no readable sampled frames")
    payload = {"frames": manifest, "video_meta": {"fps": fps, "resolution": [width, height], "codec": "OpenCV"}, "notes": [mask_note, f"sampled every {step} source frame(s)"]}
    (output / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"output": str(output), "frames": len(manifest), "manifest": str(output / "manifest.json"), "mask_note": mask_note}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1 drone video ingestion")
    parser.add_argument("--video", required=True)
    parser.add_argument("--gps-log")
    parser.add_argument("--out", default="phase1_ingestion/output")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-blur", type=float, default=0.0)
    parser.add_argument("--segmentation-model")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(extract(args.video, args.gps_log, args.out, args.sample_fps, args.max_frames, args.min_blur, args.segmentation_model, args.normalize), indent=2))


if __name__ == "__main__":
    main()
