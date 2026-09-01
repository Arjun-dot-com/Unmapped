# Phase 5 backend

Run from the repository root:

```bash
uvicorn app.main:app --app-dir platform/backend --reload
```

The demo pipeline uses the checked-in Phase 3 PLY as its source asset. A production adapter can replace `_run` with the Phase 1→4 job runner without changing the API contract.
