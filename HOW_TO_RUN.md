# How to Run "Unmapped" (3D Reconstruction Engine — Phase 3)

This guide provides step-by-step instructions for setting up, running, testing, and troubleshooting the **Unmapped** 3D Terrain Reconstruction Engine (Phase 3 of SIH26158).

---

## 1. Prerequisites & Environment Setup

### 1.1 Python Version
Ensure you have **Python 3.9+** installed (Python 3.10 or 3.11 recommended).

```bash
python --version
```

### 1.2 Create & Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
# From the root repository directory (d:\SIH26\Unmapped)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS (Bash):**
```bash
python -m venv venv
source venv/bin/activate
```

---

## 2. Installing Dependencies

### 2.1 Core Dependencies (CPU-Only / Mock Run)
For running the standalone mock mode and pipeline scaffold (no GPU or model weights needed):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Core packages installed:* `numpy`, `scipy`, `opencv-python`, `Pillow`, `PyYAML`, `tqdm`.

### 2.2 Optional Hardware Accelerations (For Real Data & GPU)
Install these depending on your system capabilities:

```bash
# 1. For real Depth Anything V2 monocular depth inference:
pip install "torch>=2.0" transformers

# 2. For real 3D Gaussian Splatting radiance field training (requires CUDA):
pip install "torch>=2.0" gsplat

# 3. For enhanced point cloud I/O & meshing:
pip install open3d
```

> **Note:** The pipeline degrades gracefully. If `torch`/`transformers` or CUDA/`gsplat` are missing, it automatically falls back to documented NumPy consolidation and stand-in depth models without crashing.

---

## 3. Quickstart: Running in Mock Mode (No GPU Needed, ~1 Min)

You can run the entire pipeline end-to-end on a synthetic test scene without needing real drone footage or heavy model weights:

```bash
python -m phase3_reconstruction.run --mock
```

### What this does:
1. Generates a synthetic mock drone scene (ground + buildings + road + trees + moving dynamic car) with camera poses and sparse points.
2. Applies dynamic object masking (excluding the moving vehicle).
3. Performs robust metric scale-depth fusion.
4. Initializes and consolidates 3D Gaussians.
5. Computes observation counts & confidence flags.
6. Renders visual previews and exports the standard deliverables to `phase3_reconstruction/output/`.

---

## 4. Running on Real Upstream Data

To run on actual video frames from **Phase 1** and camera poses from **Phase 2**:

```bash
python -m phase3_reconstruction.run \
    --frames-dir ./phase1_ingestion/output \
    --poses-dir  ./phase2_pose/output \
    --out        ./phase3_reconstruction/output \
    --config     configs/default.yaml
```

### Input Contract Expected:
- `--frames-dir`: Directory containing `frames/` (e.g. `frame_000001.jpg`), `masks/` (e.g. `frame_000001_mask.png`), and `manifest.json`.
- `--poses-dir`: Directory containing `poses.json` and `sparse_cloud.ply`.

---

## 5. CLI Arguments & Helpful Commands

| Flag | Description | Example Usage |
|---|---|---|
| `--mock` | Run on synthetic toy scene (no external inputs needed) | `python -m phase3_reconstruction.run --mock` |
| `--regen-mock` | Force-regenerate mock dataset even if cached | `python -m phase3_reconstruction.run --mock --regen-mock` |
| `--max-frames <N>` | Subsample and process only the first $N$ frames | `python -m phase3_reconstruction.run --mock --max-frames 6` |
| `--config <path>` | Path to custom YAML configuration file | `python -m phase3_reconstruction.run --config configs/default.yaml` |
| `--set <key=value>`| Override specific configuration parameters directly | `python -m phase3_reconstruction.run --mock --set training.iterations=3000` |
| `--out <dir>` | Specify target output folder | `python -m phase3_reconstruction.run --mock --out ./my_custom_output` |
| `--log-level <LVL>`| Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `python -m phase3_reconstruction.run --mock --log-level DEBUG` |

---

## 6. Output Files & Artifacts

All outputs are saved to the directory specified by `--out` (default: `phase3_reconstruction/output/`):

| File / Folder | Purpose & Description |
|---|---|
| `splat_scene.ply` | Dense 3D point cloud containing `x, y, z`, `r, g, b`, `opacity`, `scale_x/y/z`, `observation_count`, and `confidence` fields in metric frame. |
| `splat_scene_native/` | Raw trained Gaussians (`checkpoint.pt` for gsplat or `gaussians.npz` for NumPy fallback) + `meta.json`. |
| `preview_render.png` | Orbit-view still image rendering of the 3D reconstruction. |
| `preview_orbit.mp4` | Short fly-around orbit animation video (or `.gif` fallback). |
| `preview_confidence.png` | Reconstruction view with low-confidence / occluded points tinted in red. |
| `training_meta.json` | Comprehensive metadata log (training time, Gaussian count, frame count, hardware profile, loss/PSNR, and per-frame fusion metrics). |

---

## 7. Running Unit & Integration Tests

Run the test suite via `pytest`:

```bash
# Install pytest if not already present
pip install pytest

# Run all test suites
python -m pytest -v
```

### Test Coverage includes:
- `test_dataset_loader.py`: Dataset schema validation, matching `frame_id`, handling missing/corrupt inputs.
- `test_depth_scale_fusion.py`: Relative-to-metric scale solver, RANSAC outlier rejection, global scale fallback.
- `test_geometry.py`: SE(3) camera projections, back-projections, similarity transforms.
- `test_ply_io.py`: Dependency-free binary PLY reading/writing.
- `test_end_to_end_mock.py`: Full end-to-end integration test verifying export files and exclusion of moving dynamic objects.

---

## 8. Standalone Mock Data Generator Tool

If you want to manually generate or inspect the mock dataset separately:

```bash
python tools/generate_mock_input.py --out-dir phase3_reconstruction/mock_data
```

---

## 9. Troubleshooting & FAQ

- **ModuleNotFoundError: No module named 'phase3_reconstruction'**
  Ensure you are running commands from the repository root directory (`d:\SIH26\Unmapped`) and using `python -m phase3_reconstruction.run ...`.

- **CUDA/GPU not detected warning:**
  This is expected on CPU-only machines. The system automatically switches to the CPU rasterizer and NumPy consolidation engine without failing.

- **Dynamic objects showing in output:**
  Check `phase1_ingestion/output/masks/` to ensure dynamic pixels are marked as 255. The pipeline dilates and masks them out across fusion, seeding, and confidence estimation.
