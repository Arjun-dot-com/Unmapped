# Phase 5 viewer

From `platform/frontend`, run `npm install` and then `npm run dev`. Start the
backend separately from the repository root with:

```powershell
python -m uvicorn app.main:app --app-dir platform/backend --reload --port 8000
```

Open <http://localhost:3000>. The viewer loads the georeferenced model, includes
click-to-measure and confidence-layer controls, and uses OpenStreetMap imagery
by default. Copy `.env.example` to `.env.local` and add
`NEXT_PUBLIC_CESIUM_ION_TOKEN` to enable Cesium Ion world terrain.
