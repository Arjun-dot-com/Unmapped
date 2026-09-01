from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from ..services.state import DATA, db

router = APIRouter(prefix="/api/flights", tags=["flights"])


@router.post("/upload")
async def upload(video: UploadFile = File(...), telemetry: UploadFile | None = None):
    flight_id = uuid.uuid4().hex
    root = DATA / flight_id
    root.mkdir(parents=True)
    video_name = Path(video.filename or "video.bin").name
    with (root / video_name).open("wb") as stream:
        shutil.copyfileobj(video.file, stream)
    if telemetry:
        telemetry_name = Path(telemetry.filename or "telemetry.json").name
        with (root / telemetry_name).open("wb") as stream:
            shutil.copyfileobj(telemetry.file, stream)
    connection = db()
    connection.execute("INSERT INTO flights(id,name,root) VALUES(?,?,?)", (flight_id, video_name, str(root)))
    connection.commit()
    connection.close()
    return {"flight_id": flight_id, "status": "uploaded"}
