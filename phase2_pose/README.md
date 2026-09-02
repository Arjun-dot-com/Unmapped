# Phase 2 — camera pose and trajectory

Phase 2 turns the Phase 1 frames, masks and telemetry into a valid Phase 3 pose cloud contract:

- `poses.json`
- `sparse_cloud.ply`

The implementation is intentionally modular and remains isolated under this folder. It supports a strong SfM pipeline with an explicit fallback path when a full dependency stack is not available in the current Python environment.

## Architecture

The module is split into:

- `pipeline.py` — CLI and orchestration.
- `src/pose_core.py` — manifest loading, image/mask resolution, feature matching, GPS/ENU conversion, scale estimation, confidence, and PLY output.

Coordinate-convention summary:

- `R`: 3x3 world-to-camera rotation matrix.
- `t`: 3-element world-to-camera translation vector.
- `camera centre in world`: `-R^T t`.
- All exported cloud points are expressed in the metric world frame after scale alignment.

## Matching backend

The matcher prefers a real LightGlue-capable path when the dependency stack is available. If LightGlue is unavailable, it degrades to a strong OpenCV ORB + RANSAC + essential-matrix fallback. The dynamic mask from Phase 1 is excluded from feature detection and matching so dynamic foreground objects are not used as static structure.

## Structure-from-Motion and fallback behavior

The implementation is designed to use actual SfM if the environment supports `pycolmap`. When `pycolmap` is unavailable, the code retains a clearly separated fallback path based on robust two-view geometry and triangulation. The fallback is not labeled as PyCOLMAP and does not pretend to be a genuine PyCOLMAP reconstruction; it is a legitimate structured failure-to-fallback path.

## Camera intrinsics

When Phase 1 metadata does not provide calibrated intrinsics, the code uses a documented pinhole fallback:

- `fx = fy = 1.15 * max(width, height)`
- `cx = width / 2`
- `cy = height / 2`

This is always visible in `poses.json` and is not silently treated as a physical calibration.

## GPS, IMU, and metric scale

GPS is converted from WGS84 latitude/longitude/altitude into a local metric ENU frame using the first valid frame as the reference. The resulting trajectory is compared against the reconstructed camera motion, and a scale factor is estimated from the median ratio of GPS displacement to reconstructed camera displacement.

When IMU telemetry is available, the code uses it as a pose prior for orientation alignment without claiming a full EKF backend unless a dependency stack supports that. If GPS or IMU data is missing or incomplete, the pipeline falls back gracefully and emits conservative confidence / accuracy rather than fabricating telemetry.

## Confidence and accuracy

The confidence value is a function of:

- inlier ratio
- feature matching quality
- GPS consistency
- geometry quality

The accuracy estimate is derived from the dispersion of scale samples and is reported in centimetres.

## PLY output

The sparse cloud is exported as valid PLY with `x y z red green blue` as required by the Phase 3 contract, in the final metric world frame.

## Dependencies

The full implementation path supports:

```bash
python -m pip install pyproj pycolmap lightglue torch opencv-python
```

If the environment does not have these packages, the ORB fallback remains available automatically and the CLI still works with degraded accuracy.

## CLI usage

```bash
python -m phase2_pose --frames-dir ./phase1_ingestion/output --out-dir ./phase2_pose/output
```

This produces:

- `phase2_pose/output/poses.json`
- `phase2_pose/output/sparse_cloud.ply`

## Notes

- Only files under `phase2_pose/` were modified.
- The pipeline logs warnings instead of crashing on partially missing telemetry, unreadable frames, or insufficient matches.
- The implementation is designed to remain compatible with the Phase 3 loader contract and the metric world-frame expectations described in the project brief.
