from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..services.state import DATA, ROOT

router = APIRouter(prefix="/api/models", tags=["models"])
ALLOWED = {"model.glb", "tileset.json", "geospatial_meta.json", "low_confidence.json"}


def asset(flight_id: str, name: str) -> Path:
    path = DATA / flight_id / "phase4" / name
    if flight_id == "demo" and not path.is_file():
        path = ROOT / "phase4_geospatial" / "output" / name
    if not path.is_file():
        raise HTTPException(404, "asset not found; run pipeline first")
    return path


@router.get("/{flight_id}/tileset")
def tileset(flight_id: str):
    return FileResponse(asset(flight_id, "tileset.json"), media_type="application/json")


@router.get("/{flight_id}/meta")
def meta(flight_id: str):
    metadata = json.loads(asset(flight_id, "geospatial_meta.json").read_text())
    training = DATA / flight_id / "phase3" / "training_meta.json"
    if not training.exists():
        training = ROOT / "phase3_reconstruction" / "output" / "training_meta.json"
    if training.exists():
        metadata["training"] = json.loads(training.read_text())
    return metadata


@router.get("/{flight_id}/{asset_name}")
def model_asset(flight_id: str, asset_name: str):
    if asset_name not in ALLOWED:
        raise HTTPException(404, "unsupported asset")
    return FileResponse(asset(flight_id, asset_name))
