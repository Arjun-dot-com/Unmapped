"""Background orchestration for the Phase 1-4 reconstruction pipeline."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..ingestion import prepare_phase_inputs
from .state import DATA, ROOT, tasks


def _log(task: dict, message: str) -> None:
    task["logs"].append(message)


def execute(task_id: str, flight_id: str, request) -> None:
    task = tasks[task_id]
    try:
        task.update(status="running", stage="starting")
        flight_root = DATA / flight_id
        phase3_out = flight_root / "phase3"
        phase4_out = flight_root / "phase4"

        if request.mock:
            _log(task, "mock mode requested; using the checked-in Phase 3 demo output")
            input_ply = ROOT / "phase3_reconstruction" / "output" / "splat_scene.ply"
        elif request.frames_dir and request.poses_dir:
            task["stage"] = "phase3"
            _log(task, "running Phase 3 reconstruction from supplied Phase 1/2 directories")
            input_ply = _run_phase3(request.frames_dir, request.poses_dir, phase3_out)
        elif flight_id != "demo":
            task["stage"] = "ingestion"
            _log(task, "extracting upload frames and creating GPS-derived Phase 1/2 inputs")
            frames, poses = prepare_phase_inputs(flight_root)
            task["stage"] = "phase3"
            input_ply = _run_phase3(
                frames,
                poses,
                phase3_out,
                config=ROOT / "configs" / "default.yaml",
                extra=["--set", "max_frames=24"],
            )
        else:
            _log(task, "no Phase 1/2 directories supplied; using checked-in demo reconstruction")
            input_ply = ROOT / "phase3_reconstruction" / "output" / "splat_scene.ply"

        if not input_ply.is_file():
            raise FileNotFoundError(f"Phase 3 output not found: {input_ply}")

        task["stage"] = "phase4"
        _log(task, "running Phase 4 geospatial conversion")
        from phase4_geospatial.run_phase4 import run_phase4

        result = run_phase4(input_ply, phase4_out)
        task.update(status="complete", stage="done", result=result)
    except Exception as exc:  # background task errors are reported through status
        task.update(status="failed", error=str(exc))


def _run_phase3(frames_dir, poses_dir, output: Path, config: Path | None = None, extra=None) -> Path:
    command = [
        sys.executable,
        "-m",
        "phase3_reconstruction.run",
        "--frames-dir",
        str(frames_dir),
        "--poses-dir",
        str(poses_dir),
        "--out",
        str(output),
    ]
    if config:
        command.extend(["--config", str(config)])
    if extra:
        command.extend(extra)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:] or "Phase 3 failed")
    return output / "splat_scene.ply"


def start(flight_id: str, request) -> dict:
    import threading
    import uuid

    task_id = uuid.uuid4().hex
    tasks[task_id] = {"task_id": task_id, "flight_id": flight_id, "status": "queued", "stage": "queued", "logs": []}
    threading.Thread(target=execute, args=(task_id, flight_id, request), daemon=True).start()
    return tasks[task_id]
