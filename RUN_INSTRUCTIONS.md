# Unmapped: Run Instructions

Run these commands from the repository root:

```powershell
cd D:\SIH26\Unmapped
```

## 1. Install Python dependencies

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r platform\backend\requirements.txt
```

## 2. Run the complete mock pipeline

```powershell
python -m phase3_reconstruction.run --mock
python -m phase4_geospatial.run_phase4 `
  --input phase3_reconstruction\output\splat_scene.ply `
  --out phase4_geospatial\output
```

## 3. Run a real flight through Phases 1–4

```powershell
python -m phase1_ingestion.phase1_ingest `
  --video C:\path\to\flight.mp4 `
  --gps-log C:\path\to\flight.csv `
  --out phase1_ingestion\output
python -m phase2_pose.pose_estimator `
  --frames-dir phase1_ingestion\output `
  --out phase2_pose\output
python -m phase3_reconstruction.run `
  --frames-dir phase1_ingestion\output `
  --poses-dir phase2_pose\output `
  --out phase3_reconstruction\output `
  --config configs\default.yaml
python -m phase4_geospatial.run_phase4 `
  --input phase3_reconstruction\output\splat_scene.ply `
  --out phase4_geospatial\output
```

## 4. Generate the Phase 4 GLB and 3D Tiles manually

```powershell
python -m phase4_geospatial.run_phase4 `
  --input phase3_reconstruction\output\splat_scene.ply `
  --out phase4_geospatial\output `
  --manifest phase3_reconstruction\mock_data\phase1_ingestion\output\manifest.json `
  --poses phase3_reconstruction\mock_data\phase2_pose\output\poses.json
```

Generated files:

- `phase4_geospatial/output/model.glb`
- `phase4_geospatial/output/tileset.json`
- `phase4_geospatial/output/geospatial_meta.json`

## 5. Start the backend

Open a second PowerShell terminal:

```powershell
cd D:\SIH26\Unmapped
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir platform/backend --reload --port 8000
```

Backend API documentation: <http://localhost:8000/docs>

## 6. Start the frontend

Open a third PowerShell terminal:

```powershell
cd D:\SIH26\Unmapped\platform\frontend
npm install
npm run dev
```

Open the viewer at <http://localhost:3000>.

The frontend includes Cesium model viewing, click-to-measure, and the low-confidence overlay control. The pre-generated model is available through the `demo` flight ID.

Optional Cesium Ion terrain:

```powershell
Copy-Item .env.example .env.local
# Set NEXT_PUBLIC_CESIUM_ION_TOKEN in platform\frontend\.env.local
```
