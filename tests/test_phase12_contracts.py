import json

import cv2
import numpy as np

from phase1_ingestion.phase1_ingest import extract
from phase2_pose.pose_estimator import estimate
from phase3_reconstruction.data.dataset import load_scene


def test_phase1_to_phase2_contract_chain(tmp_path):
    video = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 48))
    for index in range(4):
        frame = np.zeros((48, 64, 3), np.uint8)
        cv2.rectangle(frame, (8 + index, 10), (40 + index, 35), (60, 130, 200), -1)
        writer.write(frame)
    writer.release()

    phase1 = tmp_path / "phase1"
    phase2 = tmp_path / "phase2"
    extract(video, None, phase1, sample_fps=2.0)
    result = estimate(phase1, phase2)

    manifest = json.loads((phase1 / "manifest.json").read_text())
    poses = json.loads((phase2 / "poses.json").read_text())
    assert manifest["frames"]
    assert all((phase1 / entry["image_path"]).is_file() for entry in manifest["frames"])
    assert all((phase1 / entry["mask_path"]).is_file() for entry in manifest["frames"])
    assert {f["frame_id"] for f in manifest["frames"]} == {f["frame_id"] for f in poses["frames"]}
    assert result["method"] == "gps_enu_fallback"
    dataset = load_scene(phase1, phase2)
    assert len(dataset) == len(manifest["frames"])
