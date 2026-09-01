# SIH26158 — "Unmapped: A 3D Terrain Reconstruction Simulator" (Team Uncharted)
### Judge Review, Mentor Notes & 4-Member Work Breakdown

---

## PART 1 — JUDGE'S EVALUATION

**Verdict:** Strong, well-differentiated pitch. This is a "top quartile" PPT for this PS — it has a real technical stack (not buzzword soup), it correctly identifies why existing tools (Pix4D, DroneDeploy, SkyeBrowse, SLAM) don't solve the *single-pass* constraint, and it commits to a specific, defensible USP (feed-forward reconstruction + click-to-measure). The weaknesses are typical of every SIH deck at this stage: unproven claims, an underspecified accuracy story, and a scope that is larger than what 4 people can demo convincingly in a hackathon window.

### Strengths
- **Correct problem framing.** They didn't just restate the PS — they explicitly contrast against Pix4D/DroneDeploy/SkyeBrowse/SLAM, which shows domain research, not just a Google search of "3D reconstruction tools."
- **A real, coherent pipeline.** Slide 3's 5-phase architecture (Ingestion → Pose → Reconstruction → Meshing/GIS → Platform) maps cleanly onto known SOTA components (COLMAP, Depth Anything V2, gsplat/3D Gaussian Splatting, Open3D, CesiumJS). This is not vapourware — every block names a real library.
- **A genuinely hard, genuinely current USP.** "Generalizable, feed-forward reconstruction with no per-scene retraining" is the single biggest research bet in this deck. If they pull it off even partially, it's the differentiator vs. classic SfM/MVS pipelines (Pix4D etc.) which *are* per-scene optimization, not learned generalization.
- **Object-level interactivity (click-to-measure distance)** is a smart, demo-friendly feature — judges remember things they can click on stage.
- **Feasibility slide is honest** — it lists real risks (motion blur, pose error, compute cost) instead of pretending the problem is easy.

### Weaknesses / Gaps a judge will flag
1. **"Feed-forward, no retraining" is the hardest possible claim in this space and is currently a research-frontier problem** (this is what things like DUSt3R/MASt3R/pixelSplat/Flash3D are trying to solve in academia). Pitching this as a solved capability for a hackathon build is risky — if a judge with CV background probes it, "how does your model generalize with *zero* retraining across unseen scenes with a single monocular video and noisy GPS?" is a question they must have a real answer to, or it reads as marketing.
2. **No accuracy/metrics story.** The PS explicitly says "metrically accurate" and lists GCP-free accuracy as a key challenge — but the deck never states a target error tolerance (e.g., "±X cm using GPS/IMU fusion, validated against Y ground-truth"). Judges will ask "accurate compared to what, measured how?"
3. **Gaussian Splatting ≠ a metric mesh out of the box.** gsplat produces a great *photorealistic* radiance field, but converting splats into a metrically usable, georeferenced polygonal mesh (Phase 4) is non-trivial and glossed over as one box in the diagram. This is probably the single biggest technical risk in the whole pipeline.
4. **Dynamic objects, occlusion, and real-time processing** (explicitly called out as PS "Key Challenges") appear once (YOLOv10-Seg masking) but there's no story for occluded-surface completion or for what "near real-time" actually means in seconds/minutes for their pipeline.
5. **Scope is large for team size.** 5 pipeline phases + DB + backend + a CesiumJS 3D web frontend is a lot for 4 people to build *and* integrate *and* demo reliably. This is the #1 execution risk, not a slide risk.
6. **The narrative shifted mid-deck.** Slides 1–4 talk drones/terrain/infrastructure (matching the PS). Slide 5 ("Impact and Benefits") suddenly pivots to caves, speleologists, and underground heritage sites — that's a different application (looks like content reused from a cave-mapping project). This is the kind of inconsistency a judge *will* catch and use to question whether the team actually understands their own PS scope.
7. **No baseline/benchmark comparison numbers.** Claims of being better than Pix4D/SkyeBrowse/SLAM are qualitative only — no table of "processing time," "# of passes needed," "point density," etc.

### Questions to be ready for on stage
- "Walk me through what happens when your model sees a scene type it's never seen before — show us, don't tell us."
- "What's your accuracy without GCPs — how do you validate metric scale from a single monocular pass?"
- "How do you turn Gaussian splats into a usable, measurable mesh? What library/algorithm, and what's the failure mode?"
- "What happens to a person, car, or occluded rooftop section your single pass never saw?"
- "What's your actual processing time from video-in to model-out, on what hardware?"
- "Is this cave slide a mistake, or do you support underground/GPS-denied environments too?"

---

## PART 2 — MENTOR'S RECOMMENDATIONS

**Before the next round, fix (in priority order):**
1. **Kill the cave/speleology content on Slide 5** or explicitly reframe it as *one example vertical* clearly labelled under the drone/infrastructure umbrella (e.g., "GPS-denied subterranean/void mapping as a stretch use-case"), otherwise it looks like a copy-paste error.
2. **Add the "Desired Output" and "Evaluation Criteria" table** the PS explicitly asks for (this is literally called out in the PS text you were given — it's an easy, visible point to lose for missing it).
3. **Add one slide/box with a concrete accuracy target and validation method** — e.g., "Target: <2% relative scale error using GPS+IMU-fused SfM scale recovery, validated against N surveyed control points on test flights." Even an estimated/planned number is better than none.
4. **De-risk the "no retraining" claim** — reframe as "generalizes across scenes using pretrained monocular depth + SfM priors, with lightweight per-scene refinement (not full retraining)" if that's closer to what you'll actually build. Judges respect a well-scoped, honest claim far more than an overreached one that collapses under one question.
5. **Scope the MVP explicitly.** For the actual build, pick ONE hero demo scene (e.g., a small building + surrounding terrain from one clean flight) and get that pipeline bulletproof end-to-end before generalizing. A judge would much rather see one flawless reconstruction live than a shaky "works on everything" claim.

**What to demo vs. fake for the hackathon build:**
- **Must be real and live:** single-pass video → point cloud/mesh → CesiumJS viewer with click-to-see-distance. This is your core USP; it must work on stage.
- **Can be partially precomputed:** Gaussian Splatting training (it's slow) — precompute for the demo scene, but show the pipeline/logs to prove it's real, and have one small live run on a short clip if GPU time allows.
- **Can be simplified for v1:** full occlusion completion, dynamic object *removal* (masking + flagging is enough; don't promise full inpainting), true real-time (near-real-time / a few minutes is fine if you're upfront about it).

---

## PART 3 — TEAM WORK BREAKDOWN (4 MEMBERS)

### 3.1 How the pipeline maps to people

Your Slide 3 diagram already gives you 5 phases. With 4 people, the cleanest split — the one that keeps each person's git folder self-contained and matches the diagram's own boundaries — is:

| Member | Owns | Phases | One-line mission |
|---|---|---|---|
| **M1 — Vision & Preprocessing Engineer** | Ingestion & Preprocessing | Phase 1 | Turn raw drone video into a clean, indexed, masked frame set with synced telemetry |
| **M2 — Geometry & Pose Engineer** | Camera Pose & Trajectory | Phase 2 | Recover accurate camera poses/trajectory and metric scale for every frame |
| **M3 — 3D Reconstruction / AI Engineer** | 3D Reconstruction Engine | Phase 3 | Turn posed frames into a dense, textured 3D scene representation (depth + Gaussian splats) |
| **M4 — Geospatial & Platform Engineer** | Meshing/GIS + Backend + Frontend | Phase 4 + Phase 5 | Turn the 3D scene into a georeferenced mesh and serve it in an interactive web viewer |

**Why M4 gets two phases instead of one:** Phase 3 (AI reconstruction) is the single hardest, most research-heavy phase in the whole system — it deserves one person's full, undivided attention. Phases 4 and 5, by contrast, are integration/engineering work (meshing library calls, a REST API, a web viewer) that don't block on each other sequentially the way 1→2→3 does — M4 can build the FastAPI backend, DB schema, and the CesiumJS frontend shell **in week 1, in parallel, using mock/dummy data**, before Phase 4's real mesh output ever exists. This is the standard way to balance "5 phases, 4 people" without anyone being idle. See the sprint plan (3.5) for how M1 also folds back in to help M4/M3 once Phase 1 is done early.

---

### 3.2 Detailed Role Cards

Each card below is written so a member can work almost entirely inside their own repo folder, as long as they respect the **exact input/output contract** in section 3.3. Treat the contract as an API — if your output format changes, you must tell the next person before you push.

---

#### 🟦 M1 — Vision & Preprocessing Engineer (Phase 1: Ingestion & Preprocessing)

**Mission:** Be the front door of the pipeline. Everything downstream depends on clean, well-labeled frames.

**Responsibilities:**
- Ingest raw drone video (1080p/4K) + GPS/flight-metadata log; parse and time-sync them.
- Frame extraction using FFmpeg/PyAV — hardware-accelerated decode, uniform frame sampling (define FPS-to-frame-interval logic, e.g. sample every Nth frame based on flight speed/overlap needed).
- Deblurring/quality filtering — Laplacian-variance blur detection to drop or flag unusable frames; basic exposure/illumination normalization.
- Dynamic object masking — run YOLOv10-Seg to produce a binary mask per frame flagging vehicles/humans/animals for exclusion in Phase 2/3.
- Emit a per-frame metadata manifest joining: frame ID, timestamp, GPS lat/lon/alt, (IMU if available), blur score, mask path.
- Write a small CLI (`python phase1_ingest.py --video ... --gps-log ... --out ./output/phase1/`) so M2 can run your stage standalone.
- Own the sample test video/dataset the whole team develops against (pick and publish it on Day 1).

**Tech stack:** Python, FFmpeg/PyAV, OpenCV, YOLOv10-Seg (Ultralytics), NumPy/Pandas for metadata.

**Repo folder:** `/phase1_ingestion/`

**Definition of done:** Given a raw `.mp4` + GPS log, your script deterministically outputs a frame folder + mask folder + `manifest.json` matching the schema in 3.3, runnable by anyone on the team with one command.

**Key risk:** Frame sampling rate is the #1 variable that affects every downstream stage's speed and quality — agree on it with M2 before locking it in, don't decide alone.

---

#### 🟩 M2 — Geometry & Pose Engineer (Phase 2: Camera Pose & Trajectory Estimation)

**Mission:** Answer "where was the camera, and how is it moving?" for every usable frame — this is what gives the final model *metric* scale and georeferencing, which is explicitly called out in the PS as a hard requirement.

**Responsibilities:**
- Feature matching across consecutive/overlapping frames (LightGlue or equivalent).
- Structure-from-Motion using PyColmap to get relative camera poses + a sparse point cloud.
- Fuse SfM's relative-scale poses with absolute GPS/IMU telemetry (e.g., via an Extended Kalman Filter) to recover **metric scale and geo-registration** without relying on Ground Control Points — this is directly the PS's hardest listed challenge ("maintaining metric accuracy without extensive GCPs"), so this stage is your team's answer to that specific judge question.
- Handle GPS inaccuracy / sensor noise (explicitly listed PS challenge) — implement outlier rejection/smoothing on the GPS track before fusion.
- Exclude/down-weight masked (dynamic-object) regions from feature matching using M1's masks.
- Emit per-frame camera pose (rotation + translation), camera intrinsics, and the sparse/scaled point cloud.
- Write a standalone CLI reading M1's manifest and producing `poses.json` + `sparse_cloud.ply`.

**Tech stack:** Python, PyColmap, LightGlue/OpenCV feature matching, SciPy/FilterPy (EKF), PyProj/GDAL for coordinate transforms.

**Repo folder:** `/phase2_pose/`

**Definition of done:** Given M1's frame+mask+manifest output, your script outputs a pose file + sparse point cloud with an estimated scale/accuracy metric printed to console (this number is your answer to the judge's accuracy question).

**Key risk:** This stage is the accuracy bottleneck for the entire system — bad poses here silently corrupt everything M3 and M4 build on top. Build a quick visual sanity-check (plot the camera trajectory over the GPS track) early, don't wait until integration to notice divergence.

---

#### 🟨 M3 — 3D Reconstruction / AI Engineer (Phase 3: 3D Reconstruction Engine)

**Mission:** This is the team's core research bet — turn posed frames into a dense, textured, explorable 3D scene. This is what a judge means when they ask "show me the AI."

**Responsibilities:**
- Monocular depth estimation per frame using Depth Anything V2, calibrated/scaled against M2's sparse point cloud (this is how you reconcile "generalizable monocular depth" with "metric scale from SfM" — be ready to explain this fusion explicitly, it's your answer to the "how does this generalize" judge question).
- Initialize 3D Gaussian Splatting from the fused depth + poses (gsplat), rather than random/SfM-point-only init, for faster convergence on a single pass.
- Radiance field optimization/training (gsplat) to produce the final textured 3D Gaussian scene representation.
- Explicitly handle the PS's "reconstruction of occluded surfaces" challenge — document your fallback (e.g., leave gaps flagged/marked rather than hallucinating geometry, at least for v1) so the team can state this honestly to judges.
- Export the trained splat scene in a form M4 can consume for meshing (e.g., `.ply` point cloud with color/opacity, or gsplat's native export).
- Log training time/hardware used — you need this number for the "how fast is your pipeline" question.

**Tech stack:** Python, PyTorch, Depth Anything V2, gsplat, CUDA-capable GPU (this member should get priority GPU access/time in the team).

**Repo folder:** `/phase3_reconstruction/`

**Definition of done:** Given M2's poses + sparse cloud + M1's frames, your pipeline outputs a trained Gaussian splat scene file + a renderable preview (a still image or short orbit render is enough to prove it works, even before M4's viewer exists).

**Key risk:** This is the slowest, most GPU-bound, and most likely-to-slip stage. Get a minimal end-to-end version running on Day 1–2 on a tiny/toy scene (even 10 frames) so integration isn't blocked — optimize quality later.

---

#### 🟥 M4 — Geospatial & Platform Engineer (Phase 4: Surface Generation & GIS + Phase 5: Platform & Web UI)

**Mission:** Turn the AI output into something a judge can click, rotate, and measure in a browser — this is what makes the demo land.

**Responsibilities — Phase 4 (Geospatial):**
- Convert M3's dense Gaussian/point-cloud scene into a polygonal mesh (Open3D — Poisson or ball-pivoting surface reconstruction).
- Georeference the mesh using GDAL/PyProj, tying it back to the GPS coordinate frame established in Phase 2, so the model has real-world coordinates, not just relative ones.
- Package the geo-registered mesh into 3D Tiles (Cesium's tiling format) for efficient web streaming.

**Responsibilities — Phase 5 (Platform & Web UI):**
- Design and build the FastAPI backend: endpoints to upload a flight, trigger/track pipeline runs, and serve model + metadata.
- Design the PostgreSQL + PostGIS schema (flights, frames, poses, model assets, object-click metadata) — **this schema should be drafted and shared with the whole team by Day 1**, since M1–M3's manifest formats should align with what the DB will eventually store.
- Build the Next.js + CesiumJS frontend: load the 3D Tiles model, implement the "click an object → show distance from camera" interaction (your team's headline USP from Slide 2) and basic measurement/visualization tools.
- Own the demo build/deployment (what actually runs on stage).

**Tech stack:** Open3D, GDAL/PyProj, FastAPI, PostgreSQL/PostGIS, Next.js, TypeScript, CesiumJS.

**Repo folder:** `/phase4_geospatial/` + `/platform/backend/` + `/platform/frontend/`

**Definition of done (staged):** (a) Backend + DB + frontend shell working end-to-end against **mock/dummy 3D Tiles data by end of week 1**, independent of the AI pipeline being finished; (b) real mesh ingestion from Phase 4 wired in once M3 has a working export; (c) click-to-measure feature working against real georeferenced coordinates.

**Key risk:** You are the integration point for everyone else's output — you cannot start the *real* pipeline wiring until M2/M3 have working exports, so front-load your mock-data build so you're never blocked, and communicate schema needs to the rest of the team early rather than at the end.

---

### 3.3 Interface Contracts (what gets pushed to the shared repo, exactly)

Define these as literal files in a shared `/contracts/` folder on Day 1 (e.g., JSON Schema or Pydantic models) — agree on them as a team *before* anyone starts coding, since this is what makes 4 independently-built folders merge into one working system.

**M1 → M2** (`/phase1_ingestion/output/`)
```
frames/frame_000001.jpg, frame_000002.jpg, ...
masks/frame_000001_mask.png, ...
manifest.json:
{
  "frames": [
    {"frame_id": "000001", "timestamp": 1699999999.0,
     "gps": {"lat": .., "lon": .., "alt": ..},
     "imu": {...} | null,
     "blur_score": 12.4, "mask_path": "masks/frame_000001_mask.png"}
  ],
  "video_meta": {"fps": .., "resolution": [w,h], "codec": ".."}
}
```

**M2 → M3** (`/phase2_pose/output/`)
```
poses.json:
{
  "frames": [
    {"frame_id": "000001", "R": [3x3], "t": [3], "confidence": 0.9}
  ],
  "intrinsics": {"fx":.., "fy":.., "cx":.., "cy":..},
  "scale_estimate_m_per_unit": ..,
  "accuracy_estimate_cm": ..
}
sparse_cloud.ply
```

**M3 → M4** (`/phase3_reconstruction/output/`)
```
splat_scene.ply   (or gsplat native export)
preview_render.png (or short .mp4 orbit)
training_meta.json: {"train_time_s": .., "num_frames": .., "hardware": ".."}
```

**M4 internal (Phase 4 → Phase 5)**
```
mesh.glb (or 3D Tiles bundle: tileset.json + .b3dm/.glb tiles)
geo_metadata.json: {"crs": "EPSG:4326", "origin": {...}, "bbox": {...}}
```

Everyone should also push a **1-page `README.md` in their own folder** stating: how to run their stage standalone, what input it expects, what output it produces, and known limitations — this is what lets teammates (and later, judges reading your repo) understand the system without asking each other.

---

### 3.4 Suggested repo layout

```
/contracts/            <- shared schemas (edit only with team agreement)
/phase1_ingestion/      (M1)
/phase2_pose/           (M2)
/phase3_reconstruction/ (M3)
/phase4_geospatial/     (M4)
/platform/
  /backend/             (M4)
  /frontend/            (M4)
/data/                  <- sample flight(s) everyone tests against (git-lfs or external link, not raw video in git)
/docs/                  <- architecture diagram, this doc, demo script
README.md               <- pipeline overview + how to run end-to-end
```

### 3.5 Suggested sprint plan (keeps everyone unblocked)

- **Week 1:** Everyone builds their stage against *mock* inputs/outputs matching the contracts (M1 has real video, so starts for real; M2–M4 use dummy/synthetic data matching the schema). M4 gets backend+DB+frontend shell running against mock 3D Tiles. Team locks the sample test flight.
- **Week 2:** Chain M1→M2 for real; M3 starts on toy scenes; M4 continues platform build.
- **Week 3:** Chain M2→M3→M4 end-to-end on the one hero demo scene; fix integration breaks. M1 (usually finishes earliest) shifts to helping M3 with data/eval or helping M4 with georeferencing QA.
- **Week 4 (buffer):** Polish the demo, precompute the heavy Gaussian-splat training for the hero scene, rehearse the pitch, prepare answers to the judge questions in Part 1, fix the Slide 5 cave-content inconsistency, add the missing "Desired Output/Evaluation Criteria" table.

