from fastapi import APIRouter, HTTPException

from ..models.schemas import RunRequest
from ..services.pipeline import start
from ..services.state import DATA, tasks

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run")
def run(request: RunRequest):
    if request.flight_id != "demo" and not (DATA / request.flight_id).is_dir():
        raise HTTPException(404, "flight not found")
    return start(request.flight_id, request)


@router.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "task not found")
    return tasks[task_id]
