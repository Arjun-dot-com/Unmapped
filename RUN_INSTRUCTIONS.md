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

## 2. Run Phase 3 with mock data

```powershell
python -m phase3_reconstruction.run --mock
```

## 3. Generate the Phase 4 GLB and 3D Tiles

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

## 4. Start the backend

Open a second PowerShell terminal:

```powershell
cd D:\SIH26\Unmapped
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir platform/backend --reload --port 8000
```

Backend API documentation: <http://localhost:8000/docs>

## 5. Start the frontend

Open a third PowerShell terminal:

```powershell
cd D:\SIH26\Unmapped\platform\frontend
npm install
npm run dev
```

Open the viewer at <http://localhost:3000>.

The frontend includes Cesium model viewing, click-to-measure, and the low-confidence overlay control. The pre-generated model is available through the `demo` flight ID.
