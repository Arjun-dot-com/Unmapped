import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from tools.generate_mock_input import generate_mock_scene

from phase2_pose.pipeline import run_pipeline


def _make_mock_scene(tmp_path, n_frames=6, img_w=240, img_h=160):
    root = tmp_path / "mock_scene"
    generate_mock_scene(str(root), n_frames=n_frames, img_w=img_w, img_h=img_h)
    p1 = root / "phase1_ingestion" / "output"
    p2 = root / "phase2_pose" / "output"
    p2.mkdir(parents=True, exist_ok=True)
    return root, p1, p2


def test_phase2_pipeline_emits_poses_and_sparse_cloud(tmp_path):
    _, p1, p2 = _make_mock_scene(tmp_path)

    result = run_pipeline(str(p1), str(p2))

    poses_path = p2 / "poses.json"
    cloud_path = p2 / "sparse_cloud.ply"

    assert poses_path.exists()
    assert cloud_path.exists()
    assert result["frames"]
    assert len(result["frames"]) == 6

    pose0 = result["frames"][0]
    assert set(pose0) >= {"frame_id", "R", "t", "confidence"}
    assert len(pose0["R"]) == 3 and all(len(r) == 3 for r in pose0["R"])
    assert len(pose0["t"]) == 3
    assert 0.0 <= float(pose0["confidence"]) <= 1.0

    payload = json.loads(poses_path.read_text(encoding="utf-8"))
    assert payload["intrinsics"]["fx"] > 0
    assert payload["scale_estimate_m_per_unit"] > 0
    assert payload["accuracy_estimate_cm"] > 0

    cloud = cloud_path.read_bytes()
    assert b"ply" in cloud.lower()
    assert b"element vertex" in cloud.lower()
    assert b"red" in cloud.lower()


def test_phase2_manifest_parsing_and_frame_ordering(tmp_path):
    root, p1, p2 = _make_mock_scene(tmp_path, n_frames=4)
    manifest = json.loads((p1 / "manifest.json").read_text(encoding="utf-8"))

    frame_ids = [frame["frame_id"] for frame in manifest["frames"]]
    assert frame_ids == sorted(frame_ids, key=lambda fid: int(fid))

    result = run_pipeline(str(p1), str(p2))
    output_ids = [frame["frame_id"] for frame in result["frames"]]
    assert output_ids == frame_ids


def test_phase2_missing_telemetry_falls_back_gracefully(tmp_path):
    root = tmp_path / "no_gps_scene"
    root.mkdir(parents=True, exist_ok=True)
    p1 = root / "phase1_ingestion" / "output"
    p2 = root / "phase2_pose" / "output"
    p1.mkdir(parents=True, exist_ok=True)
    p2.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(2):
        frame_id = f"{i + 1:06d}"
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 160, dtype=np.uint8)[
            None, :]  # red channel gradient
        img[:, :, 1] = 80
        img[:, :, 2] = 160
        to = p1 / "frames" / f"frame_{frame_id}.jpg"
        to.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(to), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        mask = p1 / "masks" / f"frame_{frame_id}_mask.png"
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask_img = np.zeros((120, 160), dtype=np.uint8)
        cv2.imwrite(str(mask), mask_img)
        frames.append({
            "frame_id": frame_id,
            "timestamp": float(i),
            "gps": None,
            "imu": None,
            "blur_score": 0.1,
            "mask_path": f"masks/frame_{frame_id}_mask.png",
        })

    manifest = {"frames": frames, "video_meta": {
        "fps": 30, "resolution": [160, 120], "codec": "h264"}}
    (p1 / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = run_pipeline(str(p1), str(p2))
    payload = json.loads((p2 / "poses.json").read_text(encoding="utf-8"))
    assert payload["scale_estimate_m_per_unit"] >= 1.0
    assert payload["accuracy_estimate_cm"] > 0
    assert (p2 / "sparse_cloud.ply").exists()


def test_phase2_cli_generates_outputs(tmp_path):
    _, p1, p2 = _make_mock_scene(tmp_path, n_frames=3)
    cmd = [
        sys.executable,
        "-m",
        "phase2_pose",
        "--frames-dir",
        str(p1),
        "--out-dir",
        str(p2),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(
        Path(__file__).resolve().parents[2]))
    assert result.returncode == 0, result.stderr
    payload = json.loads((p2 / "poses.json").read_text(encoding="utf-8"))
    assert payload["frames"]
    assert (p2 / "sparse_cloud.ply").exists()
