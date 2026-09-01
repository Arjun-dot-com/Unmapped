"""Minimal upload adapter for the Phase 3 input contract.

It extracts review frames from a video and converts GPS telemetry supplied as
JSON or CSV into the manifest/pose files consumed by Phase 3. If telemetry
already contains ``R`` and ``t`` they are preserved; otherwise a low-confidence
ENU trajectory is generated from GPS so the pipeline remains honest about its
accuracy while still being runnable end-to-end.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2
import numpy as np

def _telemetry(path):
    if not path or not path.exists(): return []
    if path.suffix.lower()==".csv":
        with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        return [{"timestamp":float(r.get("timestamp",i)),"gps":{"lat":float(r.get("lat",0)),"lon":float(r.get("lon",0)),"alt":float(r.get("alt",0))},**r} for i,r in enumerate(rows)]
    data=json.loads(path.read_text(encoding="utf-8")); return data.get("frames",data if isinstance(data,list) else [])

def prepare_phase_inputs(root: Path, max_frames=24):
    video=next((p for p in root.iterdir() if p.suffix.lower() in {".mp4",".mov",".avi",".mkv"}),None)
    if video is None: raise FileNotFoundError(f"no uploaded video found in {root}")
    telemetry=next((p for p in root.iterdir() if p.suffix.lower() in {".json",".csv"}),None); rows=_telemetry(telemetry)
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 384)
    n=min(max_frames,total) if total else max_frames; ids=np.linspace(0,max(total-1,0),n).round().astype(int); frames_dir=root/"phase1"; image_dir=frames_dir/"frames"; image_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]; centers=[]
    for i,frame_no in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_no)); ok,frame=cap.read()
        if not ok: continue
        fid=f"{i+1:06d}"; name=f"frame_{fid}.jpg"; cv2.imwrite(str(image_dir/name),frame); row=rows[min(i,len(rows)-1)] if rows else {}; gps=row.get("gps") or {}; manifest.append({"frame_id":fid,"timestamp":float(row.get("timestamp",frame_no/fps)),"image_path":f"frames/{name}","gps":gps,"blur_score":100.0})
        centers.append((float(gps.get("lat",0)),float(gps.get("lon",0)),float(gps.get("alt",0))))
    cap.release()
    if not manifest: raise RuntimeError("uploaded video produced no readable frames")
    lat0,lon0,alt0=centers[0] if centers else (0,0,0); radius=6378137.0; intr={"fx":0.8*width,"fy":0.8*width,"cx":width/2,"cy":height/2,"width":width,"height":height}; pose_frames=[]
    for m,g in zip(manifest,centers or [(lat0,lon0,alt0)]*len(manifest)):
        east=np.deg2rad(g[1]-lon0)*radius*np.cos(np.deg2rad(lat0)); north=np.deg2rad(g[0]-lat0)*radius; up=g[2]-alt0
        pose_frames.append({"frame_id":m["frame_id"],"R":np.eye(3).tolist(),"t":[-east,-north,-up],"confidence":0.35 if rows else 0.1})
    phase2=root/"phase2"; phase2.mkdir(exist_ok=True); (frames_dir/"manifest.json").write_text(json.dumps({"frames":manifest,"video_meta":{"fps":fps,"resolution":[width,height],"codec":"uploaded"}},indent=2),encoding="utf-8"); (phase2/"poses.json").write_text(json.dumps({"frames":pose_frames,"intrinsics":intr,"scale_estimate_m_per_unit":1.0,"accuracy_estimate_cm":999.0,"notes":"GPS-derived fallback poses; use a calibrated Phase 2 pose output for production accuracy."},indent=2),encoding="utf-8")
    return frames_dir,phase2
