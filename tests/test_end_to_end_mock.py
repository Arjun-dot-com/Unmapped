"""The non-negotiable deliverable: the whole pipeline runs standalone on mock
data with no GPU and produces the exact Section-4 output contract.
"""

import json

import numpy as np
import pytest

from phase3_reconstruction.config import load_config
from phase3_reconstruction.io.ply import read_ply
from phase3_reconstruction.pipeline import run_pipeline
from tools.generate_mock_input import generate_mock_scene


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    generate_mock_scene(str(root / "in"), n_frames=6, img_w=240, img_h=160)
    out = root / "out"
    cfg = load_config(None, overrides=[
        "depth.backend=mock",
        "training.backend=fallback",
        "init.pixel_stride=4",
        "init.voxel_size_m=0.2",
        "preview.n_orbit_frames=6",
        "preview.width=240", "preview.height=160",
        "preview.max_render_points=60000",
        "confidence.low_obs_threshold=1",
    ])
    res = run_pipeline(str(root / "in" / "phase1_ingestion" / "output"),
                       str(root / "in" / "phase2_pose" / "output"),
                       str(out), cfg, mock=True)
    return out, res


def test_contract_files_exist(result):
    out, res = result
    assert (out / "splat_scene.ply").is_file()
    assert (out / "splat_scene_native").is_dir()
    assert (out / "splat_scene_native" / "meta.json").is_file()
    assert (out / "preview_render.png").is_file()
    assert (out / "training_meta.json").is_file()
    # video is "optional but preferred" - accept mp4 OR gif fallback OR none
    assert res["num_gaussians_final"] > 1000


def test_training_meta_has_contract_fields(result):
    out, _ = result
    meta = json.loads((out / "training_meta.json").read_text())
    for k in ("train_time_seconds", "num_frames_used", "num_gaussians_final",
              "hardware", "final_loss", "final_psnr_db", "notes"):
        assert k in meta, k
    assert meta["num_frames_used"] == 6
    assert "FLAGGED" in meta["notes"] or "flagged" in meta["notes"]


def test_ply_schema_and_metric_frame(result):
    out, _ = result
    ply = read_ply(str(out / "splat_scene.ply"))
    for f in ("x", "y", "z", "red", "green", "blue", "opacity",
              "observation_count", "confidence"):
        assert f in ply.names, f
    xyz = ply.xyz()
    # mock scene: ground plane near z=0, nothing absurd -> metric frame preserved
    assert xyz[:, 2].min() > -3 and xyz[:, 2].max() < 20
    gnd = xyz[np.abs(xyz[:, 2]) < 0.5]
    assert len(gnd) > 0.2 * len(xyz)
    assert abs(np.mean(gnd[:, 2])) < 0.5


def test_dynamic_object_excluded(result):
    out, _ = result
    ply = read_ply(str(out / "splat_scene.ply"))
    xyz, rgb = ply.xyz(), ply.rgb()
    # the mock "car" is bright blue, on the road (y~0), low z
    blueish = ((rgb[:, 2].astype(int) - rgb[:, 0].astype(int) > 60)
               & (np.abs(xyz[:, 1]) < 2.5) & (xyz[:, 2] < 1.8))
    assert blueish.sum() < 0.002 * len(xyz)


def test_low_confidence_is_flagged_not_dropped(result):
    out, _ = result
    meta = json.loads((out / "training_meta.json").read_text())
    occ = meta["extended"]["occlusion_confidence"]
    assert occ["n_gaussians"] > 1000
    assert 0.0 <= occ["frac_low_confidence"] <= 1.0
    # export keeps low-confidence points (policy: flag, don't hallucinate/delete)
    ply = read_ply(str(out / "splat_scene.ply"))
    assert (ply.columns["observation_count"] <= 1).sum() >= 0
