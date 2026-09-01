# Phase 5 backend

Run from the repository root:

```bash
python -m uvicorn app.main:app --app-dir platform/backend --reload --port 8000
```

`POST /api/flights/upload` stores a video and optional telemetry under a generated
flight ID. For a real run, call `POST /api/pipeline/run` with that ID plus
`frames_dir` and `poses_dir` pointing to Phase 1 and Phase 2 contract outputs;
the backend executes Phase 3 and then Phase 4 in a background task. Use
`{"flight_id":"demo","mock":true}` to view the checked-in demonstration asset.

`schema.sql` is the production PostgreSQL/PostGIS schema. Local development
uses SQLite automatically so the demo does not require a database server.

From an uploaded flight, the backend invokes the canonical Phase 1 ingestion,
Phase 2 pose estimation, Phase 3 reconstruction, and Phase 4 export stages.
The frontend proxy expects this service at `http://127.0.0.1:8000`.
