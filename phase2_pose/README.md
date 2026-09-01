# Phase 2 — camera pose and trajectory

```bash
python -m phase2_pose.pose_estimator --frames-dir phase1_ingestion/output --out phase2_pose/output
```

Run from the repository root after Phase 1. The portable GPS/ENU estimator is
included in the core dependencies; optional PyCOLMAP/FilterPy adapters can be
installed with `pip install -r phase2_pose/requirements.txt`.

The stage emits `poses.json` and `sparse_cloud.ply` in the shared Phase 2
contract. The current portable implementation converts GPS to a local ENU
metric frame and records a conservative accuracy estimate. A calibrated
PyCOLMAP/IMU fusion adapter can replace this estimator without changing the
contract.
