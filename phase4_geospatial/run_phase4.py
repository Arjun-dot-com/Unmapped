"""CLI: python -m phase4_geospatial.run_phase4 --input ... --out ..."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from phase3_reconstruction.io.ply import read_ply
from .mesh_generator import reconstruct_mesh
from .georeferencer import load_origin, enu_to_wgs84, origin_metadata
from .tileset_converter import write_glb, write_tileset

def run_phase4(input_ply, out_dir, manifest=None, poses=None, method="poisson"):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p=read_ply(str(input_ply)); xyz,rgb=p.xyz(),p.rgb(); v,faces,c=reconstruct_mesh(xyz,rgb,method)
    origin=load_origin(manifest, poses); glb=write_glb(out/"model.glb",v,faces,c); tiles=write_tileset(out/"tileset.json","model.glb",origin)
    # Preserve local vertices for accurate client-side measurements.
    meta={"origin":origin_metadata(origin),"vertex_count":len(v),"triangle_count":len(faces),"source":str(input_ply),"asset":"model.glb","tileset":"tileset.json","coordinate_order":"longitude, latitude, altitude"}
    (out/"geospatial_meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    confidence = p.columns.get("confidence")
    if confidence is not None:
        low = xyz[np.asarray(confidence) < 0.5]
        (out/"low_confidence.json").write_text(json.dumps({"points": low.tolist(), "count": len(low)}), encoding="utf-8")
    return {"glb":glb,"tileset":tiles,"meta":str(out/"geospatial_meta.json"),**meta}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="phase3_reconstruction/output/splat_scene.ply"); ap.add_argument("--out",default="phase4_geospatial/output"); ap.add_argument("--manifest"); ap.add_argument("--poses"); ap.add_argument("--method",choices=["poisson","ball_pivoting"],default="poisson"); a=ap.parse_args(argv)
    print(json.dumps(run_phase4(a.input,a.out,a.manifest,a.poses,a.method),indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
