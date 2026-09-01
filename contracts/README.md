# Shared pipeline contracts

These schemas are the stable interfaces between the four pipeline phases. Each
stage may add fields, but must not rename the required fields without updating
the next stage and this directory.

- `phase1_manifest.schema.json`: frames, masks, telemetry, and video metadata.
- `phase2_poses.schema.json`: camera poses, intrinsics, scale, and accuracy.
- `phase3_training_meta.schema.json`: reconstruction/training metrics.
- `phase4_geo_metadata.schema.json`: georeferenced web asset metadata.
