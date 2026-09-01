# Phase 3 — 3D Reconstruction Engine

**Project "Unmapped — A 3D Terrain Reconstruction Simulator" · Team Uncharted · SIH26158**
*Single-Pass Drone Video to Accurate 3D Model Generation System*

Phase 3 turns **posed drone-video frames** (from Phase 1 + Phase 2) into a
**dense, metrically-scaled, textured, explorable 3D Gaussian scene**, and hands
Phase 4 a clean point cloud + the raw trained Gaussians.

This is the team's core research bet: **generalise to an unseen scene by fusing a
pretrained monocular-depth prior with SfM-derived metric scale**, rather than
optimising a scene from scratch or needing multi-pass overlap.

---

## TL;DR — run it now (no GPU, no model weights, ~1 minute)

```bash
cd phase3_reconstruction
python -m pip install -r requirements.txt

# generate a synthetic scene AND run the whole pipeline on it:
python -m phase3_reconstruction.run --mock
```

Outputs land in `phase3_reconstruction/output/`:

| file | what |
|---|---|
| `splat_scene.ply` | dense point cloud: `x y z`, `red green blue`, `opacity`, `scale_x/y/z`, **`observation_count`**, **`confidence`** |
| `splat_scene_native/` | raw trained Gaussians (`gaussians.npz` for the NumPy fallback, `checkpoint.pt` for gsplat) + `meta.json` |
| `preview_render.png` | orbit-view still of the reconstruction |
| `preview_orbit.mp4` | short orbiting fly-around (falls back to `.gif` if no codec) |
| `preview_confidence.png` | same view, low-confidence Gaussians tinted red |
| `training_meta.json` | train time, frame count, Gaussian count, hardware, loss/PSNR, caveats |

---

## Installation

**Core (required for the full Phase 1-5 scaffold):**

```bash
pip install -r requirements.txt        # includes pyproj + Open3D for Phase 4 geospatial export
```

**Optional, per hardware:**

```bash
pip install "torch>=2.0" transformers   # real Depth Anything V2 monocular depth
pip install "torch>=2.0" gsplat         # real 3D Gaussian Splatting training (needs CUDA)
# Open3D is already included above; install it separately only if using a minimal environment.
```

The pipeline **degrades gracefully**: if `torch`/`transformers` are missing it
uses a documented depth stand-in; if `gsplat`/CUDA are missing it uses a NumPy
consolidation fallback (see *Trade-offs* below). Every substitution is logged
loudly and recorded in `training_meta.json`.

---

## Running on real upstream data

Same CLI, point it at the real Phase 1 / Phase 2 output folders:

```bash
python -m phase3_reconstruction.run \
    --frames-dir ./phase1_ingestion/output \
    --poses-dir  ./phase2_pose/output \
    --out        ./phase3_reconstruction/output \
    --config     configs/default.yaml
```

Useful flags:

| flag | effect |
|---|---|
| `--mock` | generate + use the synthetic scene; ignore `--frames-dir/--poses-dir` |
| `--regen-mock` | force-rebuild the synthetic scene |
| `--max-frames N` | cap frames (evenly subsampled) — fast iteration |
| `--set a.b=c` | override any config value, repeatable (`--set training.iterations=3000`) |
| `--log-level DEBUG` | full tracebacks + per-stage detail |

Config lives entirely in `configs/*.yaml` (mirrors `phase3_reconstruction/config.py`).
`--mock` defaults to `configs/mock_fast.yaml` (CPU, seconds); real runs default to
`configs/default.yaml`.

---

## Input contract (what Phase 3 consumes)

**Phase 1** (`--frames-dir`): `frames/frame_000001.jpg …`,
`masks/frame_000001_mask.png …` (255 = dynamic object), `manifest.json`
(`frame_id`, `timestamp`, `gps`, `imu|null`, `blur_score`, `mask_path`).

**Phase 2** (`--poses-dir`): `poses.json` (`R` 3×3 world→camera, `t`,
`confidence`, `intrinsics`, `scale_estimate_m_per_unit`, `accuracy_estimate_cm`),
`sparse_cloud.ply` (metric SfM points with RGB).

Frames are joined on **`frame_id`** (string, order-independent). The loader
(`data/dataset.py`) validates both sources and fails with a single actionable
message on any mismatch — it never silently proceeds.

> `tools/generate_mock_input.py` emits **exactly** this schema for a tiny known
> scene (ground + 2 buildings + road + trees + one moving car), photographed from
> a ~100° single-pass camera arc, with GPS noise, motion blur, JPEG artifacts and
> a dynamic object — so Phase 3 can be developed and demoed with zero upstream
> dependencies.

---

## Output contract (what Phase 4 consumes)

Written to `--out` (default `phase3_reconstruction/output/`):

* **`splat_scene.ply`** — `binary_little_endian`. Coordinates stay in the
  **Phase-2 metric world frame** (metres). If the scene was similarity-normalised
  for training stability, the trainer inverts that transform before export and
  the exact transform is written into the PLY header comments + `native/meta.json`.
* **`splat_scene_native/`** — the un-flattened trained Gaussians for maximum
  fidelity (gsplat `checkpoint.pt`, or `gaussians.npz` from the fallback) +
  `meta.json` describing the format.
* **`preview_render.png`**, **`preview_orbit.mp4`**, `preview_confidence.png`.
* **`training_meta.json`** — top-level contract fields
  (`train_time_seconds`, `num_frames_used`, `num_gaussians_final`, `hardware`,
  `final_loss`, `final_psnr_db`, `notes`) **plus** an `extended` block with
  per-frame fusion diagnostics, the occlusion histogram, init breakdown, etc.

---

## Pipeline / module map

| step | module | responsibility |
|---|---|---|
| 0 | `tools/generate_mock_input.py` | synthetic Phase-1/Phase-2 scene in the exact schema |
| 1 | `data/dataset.py` | load + validate + join on `frame_id` → `SceneDataset` |
| 2 | `depth/depth_anything.py` | Depth Anything V2 wrapper (+ documented mock stand-in) |
| 3 | **`fusion/depth_scale_fusion.py`** | **relative depth → metric depth via SfM anchors** |
| 4 | `masking/dynamic_masks.py` | load/dilate Phase-1 masks; exclude dynamic pixels everywhere |
| 5 | `init/gaussian_init.py` | back-project fused metric depth → dense informed Gaussian seed |
| 6 | `training/gsplat_trainer.py` · `training/fallback_trainer.py` | radiance-field optimisation (or NumPy consolidation) |
| 7 | `confidence/observation_count.py` | per-Gaussian observation count + view diversity → confidence flag |
| 8 | `export/exporter.py` · `export/preview.py` | write the output contract + preview render |
| 9 | `metrics/training_meta.py` | performance metrics → console + `training_meta.json` |
| 10 | `run.py` / `pipeline.py` | one CLI that runs 1→9 |

Support: `geometry.py` (camera model, SE(3), similarity transform),
`gaussian_scene.py` (shared Gaussian container), `render.py` (CPU splat
rasteriser for mock + preview), `pointcloud.py` (voxel downsample / outlier
removal), `io/ply.py` (dependency-free PLY I/O), `hardware.py` (quiet GPU probe).

---

## The depth ↔ metric-scale fusion — in plain English (judge Q&A)

**Question a judge will ask:** *"How does this generalise to an unseen scene from
one monocular video with noisy GPS and no ground control points?"*

**Answer — a division of labour:**

1. **Depth Anything V2** was trained on millions of diverse images. Given a
   brand-new frame it predicts a plausible depth map **with zero retraining** —
   that is where *generalisation* comes from. **But** its output is only correct
   *up to an unknown per-image affine transform*: it predicts affine-invariant
   *inverse* depth, roughly `prediction ≈ scale·(1/true_depth) + shift` with
   `scale`, `shift` unknown. It does not know how many **metres** "far" is.

2. **Phase 2 Structure-from-Motion** produces a **metric** sparse point cloud +
   camera poses for *this specific scene*, with scale anchored by the GPS/IMU EKF
   fusion. Metric, but only a few hundred–thousand points.

3. **`depth_scale_fusion.py` is the bridge.** For each frame it:
   * projects the visible SfM points into the image,
   * reads the monocular prediction at those pixels,
   * solves a tiny **robust least-squares** problem for the missing
     `(scale, shift)` — `minimise Σ (scale·mono_i + shift − 1/z_i)²` in
     disparity space (RANSAC / Huber options for pose & triangulation noise),
   * applies that transform to the **dense** map → **dense depth in metres**.

   Frames with too few anchors (narrow FoV, lots of sky, heavy masking) fall back
   to a **global** `(scale, shift)` = the median over the confident frames, so
   every usable frame still gets metres.

So: **pretrained prior → generalisation**, **SfM → metric anchoring**, and the
fusion is a **~2-parameter fit per frame** you can inspect, unit-test and explain
on a slide. It is not a black box and not per-scene training from scratch.
`training_meta.json → extended.fusion.per_frame` reports the recovered scale,
shift, anchor count, inlier ratio and metric RMSE for **every frame**.

Unit tests: `tests/test_depth_scale_fusion.py` recovers a known scale/shift to
<1% from synthetic anchors, stays within 5% with 30% gross-outlier anchors, and
exercises the mask exclusion + global fallback paths.

---

## Occlusion handling policy — **flag, never hallucinate**

The PS calls out "reconstruction of occluded surfaces". For v1 we **do not**
inpaint or invent unseen geometry. Instead `confidence/observation_count.py`
measures, per Gaussian:

* **`observation_count`** — how many cameras actually see it (reprojects in
  frame, in front of the camera, **not** on a dynamic mask, and depth-consistent
  with that frame's fused metric depth).
* **view-angle spread** — a point seen by 5 cameras all from nearly the same
  direction (a single drone pass!) is far less constrained than one seen from a
  wide spread.

These combine into **`confidence ∈ [0,1]`**, written **per point** into
`splat_scene.ply`. Gaussians with `observation_count ≤ confidence.low_obs_threshold`
are tagged low-confidence; `preview_confidence.png` tints them red; the flagged
percentage goes into `training_meta.json → notes`. The export **keeps** these
points (subsampling never drops them) so the team can honestly tell judges
*"these regions are marked low-confidence, not fabricated."*

---

## Dynamic-object exclusion

Phase 1's per-frame masks are loaded once, dilated (`mask.dilate_px`, default
6 px — segmentation edges are loose), and consulted by **every** stage:

* fusion — dynamic pixels excluded from the metric-scale anchor set,
* fused depth — dynamic pixels set to `NaN` before back-projection,
* init — dynamic pixels never seeded as Gaussians,
* training — dynamic pixels excluded from the photometric loss & densification,
* confidence — a camera that only sees a Gaussian through dynamic pixels gives it
  no observation credit.

Verified end-to-end: in the mock scene the moving car (bright blue, on the road)
contributes **0** points to `splat_scene.ply`.

---

## Trade-offs, and what is stubbed / simplified for v1

| area | what we did | why / impact |
|---|---|---|
| **Radiance-field training** | Real `gsplat` trainer is implemented (`training/gsplat_trainer.py`, gsplat ≥ 1.0 API: `ParameterDict` + per-attr Adam + `DefaultStrategy` densification + masked L1/SSIM). **But** this dev machine has no CUDA/gsplat, so the default run uses the **NumPy fallback**: voxel-average the depth-informed cloud, remove statistical outliers, size Gaussians from local spacing, set opacity from local density. **No view-dependent appearance, no photometric optimisation.** Geometry/colour come straight from fused depth + SfM. `radiance_field_optimised: false` is recorded. |
| **`final_psnr_db` in the fallback** | A z-buffer render-vs-image **proxy** (`psnr_is_render_proxy: true`), measured only where the cloud actually splats a point. Honest "does the geometry explain the footage" signal, **not** a real 3DGS PSNR. |
| **Monocular depth in `--mock`** | We don't bundle ~hundreds of MB of DA-V2 weights. In `--mock` the stand-in reads the synthetic scene's ground-truth depth and **degrades it** into a noisy, affine-invariant prior (random gain/bias + noise + blur) so the fusion step has a realistic problem to solve. On **real** data with `torch`+`transformers` installed, the real Depth Anything V2 runs (`depth/depth_anything.py`). Without a model on real data, a crude image-only prior is used and says so loudly. |
| **Similarity normalisation** | Optional; the gsplat trainer normalises the world to a unit-ish box for numerical stability and **inverts it before export** (`geometry.SimilarityTransform`, unit-tested). The NumPy fallback needs no normalisation and reports identity. |
| **Mock poses** | Generated **exact** (no injected SfM pose noise by default; `--pose-noise-deg/-m` knobs exist). Sparse cloud carries σ = 4 cm Gaussian noise, so fusion still solves a non-trivial problem. Real Phase 2 output will be noisier — RANSAC/Huber solvers are already in place for that. |
| **Preview renderer** | CPU point splatter, not a Gaussian rasteriser — discs with a z-buffer, no true alpha. Fine for a proof-of-life orbit; the real look comes from Phase 5's CesiumJS viewer or a gsplat render. |
| **Spherical harmonics** | Only the DC (view-independent) colour term survives into `splat_scene.ply` (that is what Phase 4 meshing wants). Full SH lives in `splat_scene_native/checkpoint.pt` when gsplat runs. |
| **Occluded geometry** | Flagged, not reconstructed (by design — see policy above). |
| **`open3d`** | Avoided in the core path; we ship our own PLY I/O + voxel/outlier ops so the mock path has a tiny dependency footprint. |

### Highest-value next improvement

**Run the real `gsplat` trainer end-to-end on a GPU** against actual Phase 1/2
output. The scaffolding (masked losses, depth-informed init, densification
schedule in `configs/default.yaml`, native checkpoint export, PSNR tracking) is
all wired — it needs a CUDA box and a tuning pass on the densification
hyper-parameters for single-pass (limited-baseline) captures. Second: replace the
`--mock` GT-depth stand-in with a small real DA-V2 checkpoint in CI so the
fusion numbers in tests reflect a genuine monocular prior.

---

## Tests

```bash
pip install pytest
python -m pytest -q          # 25 tests, ~10 s, no GPU
```

Covers: PLY round-trip, camera projection/back-projection + similarity-transform
invariants, the fusion module (scale recovery, RANSAC robustness, mask
exclusion, global fallback), the dataset loader (happy path + 3 failure modes),
and a full **end-to-end mock run** asserting the exact output contract, the
metric frame is preserved, and the dynamic object is excluded.

---

## Non-negotiable deliverables — checklist

- [x] **Runs standalone end-to-end on mock data with one command** (`--mock`), **no GPU required** — verified on CPU-only Windows, ~50 s for 12 frames.
- [x] **Runs end-to-end on real Phase 1 + Phase 2 output**, same CLI (`--frames-dir/--poses-dir`). Loader validates the exact Section-3 schema.
- [x] **`depth_scale_fusion.py` is a standalone, documented, independently testable module** — no dependency on the rest of Phase 3 beyond `geometry`/`config`; dedicated unit-test file; the "how does this generalise" answer lives here in prose + code.
- [x] **Occlusion / low-confidence regions are explicitly flagged** (`observation_count` + `confidence` per point in `splat_scene.ply`, `preview_confidence.png`, `%` in `training_meta.json`) and **never silently filled in** — export keeps them.
- [x] **Dynamic-object pixels excluded** using Phase-1 masks, at fusion + init + training + confidence. Verified: 0 car points in the mock export.
- [x] **Output exactly matches the Section-4 contract** — file names, formats, and all seven `training_meta.json` top-level fields.
- [x] **`training_meta.json` + console report** train time, hardware, frame count, Gaussian count (+ loss/PSNR, fusion diagnostics, occlusion stats).
- [x] **This README** — how to run (mock + real), fusion approach in plain English, occlusion policy, trade-offs & failure modes.
- [x] **Modular** — one module per step, not a mega-script.
- [x] **Configuration-driven** — all hyperparameters in `configs/*.yaml` via typed `Phase3Config`; no hardcoded magic numbers in the stages.
- [x] **Actionable error handling** — typed `Phase3Error` hierarchy; the CLI prints one clear line (not a raw traceback) on missing/malformed upstream input.
- [~] **Real `gsplat` radiance-field optimisation** — *implemented but not run here* (no CUDA on the dev box); default run uses the NumPy consolidation fallback, clearly labelled everywhere. This is the one item gated on hardware.

---

## Known limitations & failure modes

* **No GPU here → no true radiance field.** The fallback output is a clean,
  metric, coloured point cloud with confidence flags — good enough to unblock
  Phase 4/5 and to demo, but not photorealistic and with no view-dependent
  shading. `radiance_field_optimised: false` in the metadata.
* **Fusion needs sparse points in view.** If Phase 2's sparse cloud is very thin
  or a frame is mostly sky, that frame relies on the global scale/shift; if
  *every* frame is anchor-starved and no global can be formed, those frames are
  dropped (logged). Mitigation: `fusion.min_points`, `fusion.fallback_to_global`.
* **Single-pass = permanent occlusion.** Back faces of buildings, undersides,
  anything the one flight never saw, simply won't exist — and will be sparse /
  low-confidence at the edges of what *was* seen. This is inherent to the
  problem; we flag it rather than hide it.
* **Depth discontinuities → flying pixels.** Back-projection at silhouettes
  creates stragglers; `init.depth_edge_filter` + statistical outlier removal cut
  most, not all.
* **Mock depth ≠ real depth.** `--mock` fusion accuracy reflects a *degraded GT*
  prior, not Depth Anything V2. Treat mock PSNR/RMSE as pipeline-plumbing checks,
  not quality numbers.
* **Preview is a point splat**, not a Gaussian render — expect visible disc
  aliasing and gaps; it is a proof-of-life, not the final visual.
* **Pose noise sensitivity.** Large Phase-2 pose error shifts every back-projected
  point; RANSAC fusion tolerates anchor outliers but not a globally wrong pose.
  EKF/GPS quality upstream matters.
