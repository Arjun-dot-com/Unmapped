# Unmapped architecture

```text
video + GPS/IMU
      │
      ▼
Phase 1: frames + masks + manifest.json
      │
      ▼
Phase 2: poses.json + sparse_cloud.ply
      │
      ▼
Phase 3: metric Gaussian scene + training_meta.json
      │
      ▼
Phase 4: colored mesh.glb + 3D Tiles + geospatial_meta.json
      │
      ▼
Phase 5: FastAPI upload/status/assets ── Next.js + Cesium viewer
```

The JSON schemas in `contracts/` are the stable boundaries. GPU models and
PyCOLMAP are optional adapters; the portable fallbacks preserve the same
outputs and record their limitations in metadata.
