# Phase 1 — ingestion and preprocessing

```bash
python -m phase1_ingestion.phase1_ingest --video flight.mp4 --gps-log flight.csv --out phase1_ingestion/output
```

Run from the repository root after installing `requirements.txt`. For optional
Ultralytics segmentation, install `phase1_ingestion/requirements.txt` and add
`--segmentation-model path/to/model.pt`.

The command extracts uniformly sampled frames, Laplacian blur scores, binary
dynamic-object masks, and `manifest.json`. Pass `--segmentation-model` with an
Ultralytics YOLO segmentation model for real masks; without it, empty masks are
emitted and the limitation is recorded in the manifest.
