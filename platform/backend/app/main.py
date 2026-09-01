"""Phase 5 API entry point.

The API is deliberately thin: schemas live in ``models``, HTTP handlers in
``routes``, and pipeline orchestration in ``services``.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.flights import router as flights_router
from .routes.models import router as models_router
from .routes.pipeline import router as pipeline_router

app = FastAPI(title="Unmapped reconstruction platform", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(flights_router)
app.include_router(pipeline_router)
app.include_router(models_router)


@app.get("/api/health", tags=["system"])
def health():
    return {"ok": True, "service": "unmapped-platform"}
