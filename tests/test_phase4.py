import json, struct
import numpy as np
from phase4_geospatial.georeferencer import enu_to_wgs84
from phase4_geospatial.tileset_converter import write_glb

def test_enu_origin_is_wgs84_origin():
    result = enu_to_wgs84([[0, 0, 0], [1, 0, 2]], (12.0, 77.0, 100.0))
    assert np.allclose(result[0], [77.0, 12.0, 100.0])
    assert result[1, 2] == 102.0

def test_glb_has_valid_header_and_mesh(tmp_path):
    path = tmp_path / "model.glb"
    write_glb(path, np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32), np.array([[0,1,2]], dtype=np.uint32))
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    chunk_len = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + chunk_len])
    assert document["meshes"][0]["primitives"][0]["indices"] == 2
