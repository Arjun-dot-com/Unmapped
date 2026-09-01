import json
import shutil

import numpy as np
import pytest

from phase3_reconstruction.data import load_scene
from phase3_reconstruction.exceptions import Phase3InputError
from tools.generate_mock_input import generate_mock_scene


@pytest.fixture(scope="module")
def mock_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mock_scene")
    generate_mock_scene(str(root), n_frames=6, img_w=240, img_h=160)
    return root


def _p1p2(root):
    return str(root / "phase1_ingestion" / "output"), str(root / "phase2_pose" / "output")


def test_loads_and_joins(mock_root):
    p1, p2 = _p1p2(mock_root)
    ds = load_scene(p1, p2)
    assert len(ds) == 6
    assert ds.intrinsics.width == 240
    assert len(ds.sparse_xyz) > 100
    fr = ds.frames[0]
    img = fr.load_image()
    assert img.shape[:2] == (160, 240)
    mask = fr.load_mask(shape_hw=(160, 240))
    assert mask.dtype == bool and mask.shape == (160, 240)
    # cameras look roughly at the scene origin
    fwd = fr.camera.forward
    to_origin = -fr.camera.center / np.linalg.norm(fr.camera.center)
    assert float(fwd @ to_origin) > 0.7


def test_max_frames_subsamples(mock_root):
    p1, p2 = _p1p2(mock_root)
    ds = load_scene(p1, p2, max_frames=3)
    assert len(ds) == 3


def test_missing_manifest_raises(tmp_path, mock_root):
    p1, p2 = _p1p2(mock_root)
    broken = tmp_path / "no_manifest"
    (broken).mkdir()
    with pytest.raises(Phase3InputError, match="manifest"):
        load_scene(str(broken), p2)


def test_frame_id_mismatch_raises(tmp_path, mock_root):
    p1, p2 = _p1p2(mock_root)
    # copy phase2, then add a pose whose frame_id has no manifest entry
    p2b = tmp_path / "p2"
    shutil.copytree(p2, p2b)
    poses = json.loads((p2b / "poses.json").read_text())
    poses["frames"].append({"frame_id": "999999", "R": np.eye(3).tolist(),
                            "t": [0, 0, 0], "confidence": 0.9})
    (p2b / "poses.json").write_text(json.dumps(poses))
    with pytest.raises(Phase3InputError, match="no manifest entry"):
        load_scene(p1, str(p2b))


def test_bad_pose_shape_raises(tmp_path, mock_root):
    p1, p2 = _p1p2(mock_root)
    p2b = tmp_path / "p2bad"
    shutil.copytree(p2, p2b)
    poses = json.loads((p2b / "poses.json").read_text())
    poses["frames"][0]["R"] = [[1, 0], [0, 1]]         # not 3x3
    (p2b / "poses.json").write_text(json.dumps(poses))
    with pytest.raises(Phase3InputError, match="3x3"):
        load_scene(p1, str(p2b))
