"""Backend adapter that invokes the canonical Phase 1 and Phase 2 stages."""
from __future__ import annotations

from pathlib import Path


def prepare_phase_inputs(root: Path, max_frames: int = 24):
    from phase1_ingestion.phase1_ingest import extract
    from phase2_pose.pose_estimator import estimate

    video = next((p for p in root.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}), None)
    if video is None:
        raise FileNotFoundError(f"no uploaded video found in {root}")
    telemetry = next((p for p in root.iterdir() if p.suffix.lower() in {".json", ".csv"}), None)
    phase1 = root / "phase1"
    phase2 = root / "phase2"
    extract(video, telemetry, phase1, sample_fps=2.0, max_frames=max_frames)
    estimate(phase1, phase2)
    return phase1, phase2
