# Phase 5 backend

Run from the repository root:

```bash
uvicorn app.main:app --app-dir platform/backend --reload
```

`POST /api/flights/upload` stores a video and optional telemetry under a generated
flight ID. For a real run, call `POST /api/pipeline/run` with that ID plus
`frames_dir` and `poses_dir` pointing to Phase 1 and Phase 2 contract outputs;
the backend executes Phase 3 and then Phase 4 in a background task. Use
`{"flight_id":"demo","mock":true}` to view the checked-in demonstration asset.
