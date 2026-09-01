"""Phase 5 API: upload, run, inspect and serve reconstruction assets."""
from __future__ import annotations
import json, shutil, sqlite3, threading, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parents[3]; DATA=ROOT/"platform"/"backend"/"data"; DATA.mkdir(parents=True,exist_ok=True)
DB=DATA/"flights.sqlite3"; tasks={}
app=FastAPI(title="Unmapped reconstruction platform", version="0.1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
def db():
    c=sqlite3.connect(DB); c.execute("CREATE TABLE IF NOT EXISTS flights(id TEXT PRIMARY KEY, name TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, root TEXT)"); c.commit(); return c
class RunRequest(BaseModel):
    flight_id: str
    phases: list[int]=[1,2,3,4]
@app.get("/api/health")
def health(): return {"ok":True,"service":"unmapped-platform"}
try:
    import multipart  # type: ignore
    HAS_MULTIPART=True
except ImportError:
    HAS_MULTIPART=False

if HAS_MULTIPART:
 @app.post("/api/flights/upload")
 async def upload(video: UploadFile=File(...), telemetry: UploadFile|None=None):
    fid=uuid.uuid4().hex; root=DATA/fid; root.mkdir()
    safe=Path(video.filename or "video.bin").name; vp=root/safe
    with vp.open("wb") as f: shutil.copyfileobj(video.file,f)
    if telemetry:
        with (root/Path(telemetry.filename or "telemetry.json").name).open("wb") as f: shutil.copyfileobj(telemetry.file,f)
    c=db(); c.execute("INSERT INTO flights(id,name,root) VALUES(?,?,?)",(fid,safe,str(root))); c.commit(); c.close()
    return {"flight_id":fid,"status":"uploaded"}
else:
 @app.post("/api/flights/upload")
 async def upload_unavailable(request: Request):
    raise HTTPException(503,"install python-multipart to enable file uploads")
def _run(task_id,fid):
    try:
        tasks[task_id].update(status="running",stage="phase4")
        inp=ROOT/"phase3_reconstruction"/"output"/"splat_scene.ply"; out=DATA/fid/"phase4"
        from phase4_geospatial.run_phase4 import run_phase4
        result=run_phase4(inp,out); tasks[task_id].update(status="complete",stage="done",result=result)
    except Exception as e: tasks[task_id].update(status="failed",error=str(e))
@app.post("/api/pipeline/run")
def run(req:RunRequest):
    tid=uuid.uuid4().hex; tasks[tid]={"task_id":tid,"flight_id":req.flight_id,"status":"queued","stage":"queued","logs":[]}; threading.Thread(target=_run,args=(tid,req.flight_id),daemon=True).start(); return tasks[tid]
@app.get("/api/pipeline/status/{task_id}")
def status(task_id):
    if task_id not in tasks: raise HTTPException(404,"task not found")
    return tasks[task_id]
def asset(flight_id,name):
    p=DATA/flight_id/"phase4"/name
    if flight_id == "demo" and not p.is_file():
        p=ROOT/"phase4_geospatial"/"output"/name
    if not p.is_file(): raise HTTPException(404,"asset not found; run pipeline first")
    return p
@app.get("/api/models/{flight_id}/tileset")
def tileset(flight_id): return FileResponse(asset(flight_id,"tileset.json"),media_type="application/json")
@app.get("/api/models/{flight_id}/meta")
def meta(flight_id):
    p=asset(flight_id,"geospatial_meta.json"); data=json.loads(p.read_text())
    t=ROOT/"phase3_reconstruction"/"output"/"training_meta.json"
    if t.exists(): data["training"]=json.loads(t.read_text())
    return data
@app.get("/api/models/{flight_id}/{asset_name}")
def model_asset(flight_id,asset_name):
    if asset_name not in {"model.glb","tileset.json","geospatial_meta.json"}: raise HTTPException(404,"unsupported asset")
    return FileResponse(asset(flight_id,asset_name))
