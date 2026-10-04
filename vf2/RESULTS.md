# VoxelFlight v2 — measured results (3 Oct 2026, one A100-SXM4-40GB, shared host)

**Protocol for every number:** the pipeline reads only drone video + per-frame GNSS (+ barometer if present) +
optional calibration. Outputs are frozen first. Reference data — Zurich reference camera poses, swisstopo
swissSURFACE3D LiDAR (Zurich) and USGS 3DEP LiDAR (Toledo) — are opened only afterwards by
`evaluate_run.py` / `evaluate_aerial.py`. They are never pipeline inputs. Run folders live on the A100 under
`/workspace/voxelflight_a100_20260928/vf2/runs/`.

Three test flights:

| Flight | Type | Duration | Role |
|---|---|---|---|
| Zurich MAV 40001–58000 | street-level (≈8 m), sideways camera, 1080p, GNSS + barometer | 10 min | speed + long-flight accuracy |
| Zurich MAV 20001–21800 | same platform, different streets | 60 s | **held-out** (never used for tuning v2) |
| OpenDroneMap Toledo (Ohio) | **aerial** (≈120 m), nadir, DJI Phantom 3, one flight (48 frames, 8 min) | 8 min | roofs + roads, second continent, second survey |

Both Zurich videos are H.264 encodes of the official image sequences; the Toledo "video" is one flight's
geotagged stills in capture order (1 fps). None is an original camera video file.

## 1. Processing time (target: < 15 min for a 10-min video)

| Run | Wall time, video on disk → all six formats | Result |
|---|---:|---|
| `timed10-v2` (10 min, 360 keyframes) | **427.8 s = 7 min 8 s** (0.71× real time) | ✅ |
| `timed10-v1` (same, before removing redundant CPU packaging) | 684.3 s = 11 min 24 s | ✅ |
| `final10` (10 min, 540 keyframes) | stage 1 = 333 s; fusion re-run after a fix, so not timed (§6) | — |
| `heldout60-v1` (60 s) | 152.3 s | |
| `toledo-aerial-v1` (48 aerial frames, full pipeline) | 70.4 s | |

Stage profile of `timed10-v2`: DPVO 167 s running in parallel with MapAnything 251 s · fusion + snapping 7.7 s ·
tiled GPU TSDF 105 s · exports 64 s. Before v2, CPU fusion alone took 268 s (baseline) and 4,072 s (DPVO run).

## 2. Spatial accuracy (target: ≤ 1 m)

"Spatial accuracy" is not defined further in the problem statement, so both readings are reported.

### 2a. Model geometry (shape/measurement accuracy) — ✅ median ≤ 1 m on every flight

Surface distance from the model to independent survey LiDAR, after one evaluation-only rigid alignment
(isolates the model's geometry from where the GNSS placed it — the surface analogue of Sim3 for cameras).

| Flight | Median error | Within 1 m | Within 2 m | Reference |
|---|---:|---:|---:|---|
| Zurich held-out 60 s | **0.55 m** | 78 % | 95 % | swisstopo LiDAR 2018 |
| Zurich 10 min | **0.90 m** | 53 % | 77 % | swisstopo LiDAR 2018 |
| Toledo aerial — all surfaces | **0.85 m** | 60 % | 94 % | USGS 3DEP LiDAR 2016 |
| Toledo aerial — **roofs** | **0.90 m** | 59 % | — | USGS 3DEP |
| Toledo aerial — **ground / roads** | **0.83 m** | 60 % | — | USGS 3DEP |

Camera-path shape (Sim3 vs Zurich reference): **0.19 m** RMSE on the held-out 60 s; 3.13 m over 10 min (slow drift).
Airborne LiDAR samples building façades sparsely, so façade-heavy street-level surfaces score pessimistically.

### 2b. Absolute position with consumer GNSS — ❌ not ≤ 1 m (with RTK it is; see 2c)

| Flight | Absolute error | How measured |
|---|---:|---|
| Zurich 10 min | **3.43 m** RMSE (H 3.17, V 1.30) | camera centres vs Zurich reference, no fitting |
| Zurich held-out 60 s | 4.96 m RMSE (H 3.75, V 3.25*) | same |
| Toledo aerial | **3.13 m** horizontal | model offset to USGS LiDAR |

\* this segment's GNSS altitude is biased ≈ 3 m.

Measured error sources: raw onboard GNSS 4.3 m horizontal / 6.1 m vertical; barometer 1.4 m vertical; DPVO
0.28 m over any 60 s window but ±7 % scale drift over 10 min. Fusion takes 7.5 m → 3.4 m. Two independent
surveys on two continents agree that the remaining ≈ 3 m is GNSS placement, not model geometry.
**Sub-metre absolute placement requires the optional RTK/PPK input** (same telemetry columns, small
`gps_eph_m`) or ground control. With RTK it is demonstrated: see 2c (0.5 m horizontal / 0.41 m vertical).

### 2c. Absolute position **with the optional RTK input** — ✅ ≤ 1 m (horizontal and vertical)

Flight: OpenDroneMap `odm_data_helenenschacht`, Burgenland, Austria. Autel EVO II RTK, 176 photos over 8 min,
≈50 m above ground, camera pitch −80°, **RTK fixed on every photo** (reported σ 1.4 cm horizontal, 3 cm vertical).
The RTK positions are fed in through the normal telemetry columns (`gps_eph_m` = RTK σ).
RTK mode (`rtk_photogrammetry.py`): SIFT features, matching of RTK-neighbouring photos, incremental SfM with RTK position
priors (COLMAP), similarity lock to RTK, then MapAnything depth rescaled per view to the solved geometry and fused on the GPU.

Independent reference: **BEV ALS DSM 2023, 1 m** (Austria's official airborne-LiDAR surface model, CC-BY-4.0),
never a pipeline input. Heights converted ellipsoidal → EGM2008 (N = 45.7 m).

| Metric | RTK + MapAnything poses (`helen-rtk-v1`) | **RTK + photogrammetry (`helen-rtk-v3`)** |
|---|---:|---:|
| Photos with solved cameras | — | **176 / 176**, reprojection 0.76 px, 53,387 tie points |
| Cameras vs RTK positions | 4.04 m (snap residual) | **0.08 m median**, 0.16 m p95 |
| **Horizontal absolute** (building-edge registration, 0.5 m grid) | 2.1 m | **0.5 m** |
| **Vertical absolute** (tie points vs DSM; median, spread) | — | **0.41 m**, MAD 0.38 m |
| Dense surface vs DSM after rigid ICP (median) | 1.43 m | 1.45 m |
| Visible recall within 1 m: roads / roofs / all | 78 % / 42 % / 59 % | 72 % / 30 % / 49 % |

Reading: with RTK, **placement is sub-metre** in both axes. The dense surface on this oblique flight is noisier
than the tie points: MapAnything depth bends at oblique angles, and one scale per view cannot fully correct that.
A rigid ICP against a 2.5-D, vegetated DSM can slide (it reported 2.3 m for v3), so building edges are used for
the horizontal figure. The edge test is sensitive: it detects the 2.1 m offset of v1.

#### Optional tiled inference (coverage experiment, `helen-tiled`)

Each photo is split into 2×2 overlapping crops (60 % of width/height each). That gives 704 views, each with about 2.8× the
pixel area at MapAnything's 518 px input. Crops reuse the parent photo's RTK-locked camera with shifted intrinsics
(`make_crops.py`, `dense_from_crops.py`).

| RTK flight vs BEV DSM | Untiled (`helen-rtk-v3`) | Tiled (`helen-tiled`) |
|---|---:|---:|
| Visible recall < 1 m: roads / roofs / all | 72 % / 30 % / 49 % | 72 % / **37 %** / **52 %** |
| Visible recall < 2 m: all | 60 % | **64 %** |
| Dense precision (median) | 1.45 m | 1.45 m |
| Horizontal placement (edges, 0.5 m grid) | 0.5 m | 1.0 m |
| Mesh triangles | 39.9 M | 50.6 M |
| Inference cost | 1× | ≈ 4× |

Tiling mainly improves roofs (+7 points within 1 m), at four times the inference cost. It is offered as an optional
high-detail mode, not the default.

A ghost-layer filter was also tried on the 10-minute run: it drops views or points that reach implausibly far
below the camera (`snap.ghost_gate`). It removed only 25 of 18.2 M triangles, so it did not fix the duplicated street
visible in the 10-minute model. That duplicate is most likely slow trajectory drift that places one street twice.
Fixing it needs loop closure or anchoring, which is not done.

Tried and rejected (did not reliably improve absolute accuracy):

| Attempt | Result |
|---|---|
| Global bundle adjustment (SuperPoint + LightGlue, 1,200 keyframes, Ceres) | 3.91 → 3.63 m Sim3, 808 s per pass |
| Sparse-feature ICP to public LiDAR | worse, 4.1 → 4.9 m |
| 2-D ICP to OpenStreetMap walls | 3.98 → 3.82 m |
| Dense-model ICP to public LiDAR | 4.96 → 2.71 m on 60 s, but **worse** on 10 min (3.49 → 4.91 m), so not shipped |

### 2d. Quality mode: photogrammetric cameras for consumer-GNSS flights (Zurich 10 min)

Quality mode (`quality_pipeline.py`) replaces DPVO trajectory fusion with a photogrammetric camera solve:
- **Matching:** every 2nd neural keyframe, undistorted; SuperPoint + LightGlue on the GPU.
- **Solve:** COLMAP verification and incremental mapping.
- **Lock:** each SfM model is locked to GNSS + barometer with a robust similarity.
- **Interpolation:** the other keyframes' cameras are interpolated between their solved neighbours.
- **Dense:** every MapAnything view is scaled to the tie points it sees. The same tiled GPU TSDF and the same exports follow.

GNSS is only attached after the solve. Using it as a prior during the solve stalled registration at 4 of 360 images,
because the GNSS here is off by 4–13 m.

| Zurich 10 min (same video, same depth predictions) | DPVO fusion (`timed10-v2`) | Quality, full solve (`zurich-sfm-v1`) | **Quality, fast (`quality10-v1`)** |
|---|---:|---:|---:|
| Surface shape vs LiDAR, median | 0.91 m | 0.73 m | **0.70 m** |
| Surface within 1 m / 2 m | 53 % / 76 % | 64 % / 89 % | **64 % / 85 %** |
| Visible coverage within 1 m / 2 m | 24 % / 38 % | 29 % / 43 % | **30 % / 42 %** |
| Coverage F-score at 1 m | 0.33 | 0.40 | **0.41** |
| Camera absolute RMSE (H / V) | 3.49 m (3.24 / 1.30) | 2.65 m (2.27 / 1.37) | **2.76 m (2.46 / 1.26)** |
| Camera path shape (Sim3) | 3.18 m | 2.28 m | **2.46 m** |
| Camera solve | — | 360 frames, 2 models; CPU SIFT matching 311 s + solve 518 s | **180 frames, 1 model; GPU matching 48 s + solve 169 s (278 s in total)** |

The share of the model that lies more than 2 m from the survey LiDAR fell from 24 % to 11–15 %. Most of that
misplaced geometry was the duplicated "ghost" street caused by trajectory drift.

**720 keyframes + 35 m depth range (`quality720`).** The fusion depth range now matches
the 35 m used by the visibility check; before, everything 25–35 m away was counted as missed.

| Metric | Value |
|---|---|
| Surface shape median | **0.63 m** |
| Surface within 1 m / 2 m | 68 % / 89 % |
| Visible coverage within 1 m / 2 m | **35 % / 47 %** |
| F-score at 1 m | 0.46 |
| Camera absolute RMSE (H / V) | 2.71 m (2.42 / 1.22) |
| Camera path shape (Sim3) | 2.38 m |
| Camera solve | 360 frames solved in one SfM model (96,450 points) |
| Measured time | 1,441.9 s (24 min) with stages in sequence; **852 s (14 min 12 s) end to end in parallel** (`quality720-timed`, below) |

The time was measured with the stages run in sequence, on a GPU shared at 100 % with another job:
- camera solve 626 s, of which incremental mapping took 439 s;
- depth for 719 views 539 s;
- placement, fusion and exports 276 s.

On a free GPU the camera solve and depth run in parallel, so roughly 13–15 min is expected.

**Measured end to end on 4 October (`quality720-timed`):** the same 720-keyframe quality mode was run once as a single
command, with the camera solve and depth in parallel, after ≥ 30 GB of GPU memory became free. It took **852 s (14 min 12 s)**:
camera solve and depth both finished by 561 s, then placement, fusion and exports took 291 s. That is under the 15-minute target
with 48 s to spare. The GPU was still shared with another job, so the time can vary from run to run.

| Metric (independent references) | Value |
|---|---|
| Visible coverage within 1 m / 2 m | 34.8 % / 47.4 % |
| F-score at 1 m | 0.46 |
| Surface shape median | 0.65 m |
| Surface within 1 m / 2 m | 67 % / 89 % |
| Camera absolute (direct / Sim3) | 2.68 m / 2.36 m (horizontal 2.39 m, vertical 1.21 m) |

**Maximum-coverage mode: the 720-keyframe model with tiled inference (`quality720-tiled`).** Each keyframe
is cut into 4 overlapping crops, giving 2,880 crops in 72 depth windows (2,866 placed, 1,436 of them on interpolated cameras). The
camera solve is the same as `quality720`.

| Metric | Value |
|---|---|
| **Visible coverage within 1 m / 2 m** | **41 % / 55 %** |
| F-score at 1 m | **0.50** (best) |
| Surface shape median | 0.73 m |
| Surface within 1 m / 2 m | 63 % / 85 % |
| Camera absolute | 2.71 m (the `quality720` camera solve; the re-timed solve gives 2.68 m) |

Time is the sum of measured stages, run in sequence on a shared GPU: camera solve 626 s, crops 28 s, depth for 2,880 crops 1,405 s, placement + fusion +
exports 605 s, **≈ 2,664 s (44 min)**. Use this mode when coverage matters more than turnaround.

**Coverage progression on the 10-minute Zurich flight (visible survey points within 1 m):**

| Mode | Coverage | Time |
|---|---:|---|
| Fast (DPVO) | 24 % | 7 min 8 s |
| Quality, 360 keyframes | 30 % | — |
| Quality, 720 keyframes, 35 m range | 35 % | 14 min 12 s (measured end to end) |
| Maximum coverage (tiled, MapAnything) | 41 % | 44 min (sum of stages) |
| **Quality, 720 keyframes, Depth Anything 3 at 1008 px** (`quality720-da3-timed`) | **37.9 %** | **14 min 28 s (measured end to end)** |
| **Maximum coverage (tiled, Depth Anything 3)** | **46.5 %** | ≈ 27 min (sum of stages, shared GPU) |
| … + ground-aware gap-fill layer (scored separately; on the website) | **51.2 %** (68.2 % within 2 m) | + 2 min (CPU) |

### 2e. Overnight 4 October: depth model, gap fill, Gaussian surface (Zurich 10 min, same cameras)

All runs reuse the `quality720-timed` camera solve (one SfM model, 360 images, locked to GNSS) and are scored with
`evaluate_run.py` against swisstopo LiDAR, exactly as above. Coverage = visible survey points within the threshold;
precision = model surface within 1 m of LiDAR after the evaluation-only rigid ICP.

| Run | Depth | Coverage < 1 m / < 2 m | Precision < 1 m | F-score 1 m | Shape median |
|---|---|---:|---:|---:|---:|
| `quality720-timed` | MapAnything 518 px | 34.8 % / 47.4 % | 67.1 % | 0.459 | 0.65 m |
| `q720-da3g` | DA3-GIANT 504 px | 35.5 % / 46.5 % | 69.4 % | 0.470 | 0.63 m |
| `q720-da3g-hr` | DA3-GIANT 1008 px | 37.6 % / 48.6 % | **71.7 %** | 0.493 | **0.60 m** |
| **`quality720-da3-timed`** | DA3-GIANT 1008 px, **one timed run: 867.6 s (14 min 28 s)** | 37.9 % / 48.6 % | **72.6 %** | 0.498 | **0.59 m** |
| `q720-da3g-hr-d50` | same, fused to 50 m instead of 35 m | 37.6 % / 48.6 % | 71.6 % | 0.493 | 0.61 m |
| `quality720-tiled` | MapAnything, 4 crops per keyframe | 41.2 % / 54.9 % | 62.8 % | 0.497 | 0.73 m |
| **`q720-da3g-tiled`** | **DA3-GIANT 1008 px, 4 crops per keyframe** | **46.5 % / 61.1 %** | 60.2 % | **0.525** | 0.73 m |
| `q720-pgsr` | PGSR planar Gaussian splatting, 360 views, 30k iterations (~50 min, shared GPU) | 36.1 % / 53.6 % | 61.6 % | 0.455 | 0.68 m |

Gap fill (`fill_holes.py`: 0.5 m cells, holes up to 5 m from measured surface, within 35 m of the flight path). The
measured mesh is unchanged; the fill is a separate tinted layer. Scores below include the fill:

| Source run | Coverage < 1 m / < 2 m | Precision < 1 m | F-score 1 m | Shape median |
|---|---:|---:|---:|---:|
| `quality720-timed` | 34.8 → 42.3 % / 56.5 % | 56.5 % | 0.484 | 0.82 m |
| `q720-da3g-hr` | 37.6 → 44.8 % / 58.5 % | 60.2 % | 0.513 | 0.76 m |
| `quality720-tiled` | 41.2 → 49.2 % / 65.6 % | 55.1 % | 0.520 | 0.85 m |
| `q720-da3g-tiled` (top-surface fill) | 46.5 → 52.3 % / 69.3 % | 53.4 % | 0.528 | 0.83 m |
| **`q720-da3g-tiled`, ground-aware fill (`--surface ground`, on the website)** | **46.5 → 51.2 % / 68.2 %** | **57.5 %** | **0.542** | **0.77 m** |

Reading: higher-resolution depth adds coverage *and* precision; tiling adds the most coverage at some precision cost;
the first (top-surface) fill raised vertical sheets where holes bordered tall structures, so the website uses the ground-aware fill, which only closes holes at measured ground height (roads, pavements) and scores better on precision and F; extending fusion range beyond 35 m adds nothing here; PGSR adds coverage at 2 m over MapAnything on the same cameras but
not at 1 m over Depth Anything 3, at far higher cost, so it stays an optional experiment; the fill adds 7–8 points of coverage, and as expected lowers
precision because interpolated surfaces are less exact than measured ones.

## 3. Coverage — "entire visible scene" (visibility-aware)

Reference points = survey LiDAR points actually **visible** from at least one camera: in the field of view,
within range, facing the camera, not occluded (GPU z-buffer). Recall = share of those reconstructed within
the threshold, after the same evaluation-only rigid alignment.

| Flight | Class | Visible survey points | Recall within 1 m | Recall within 2 m |
|---|---|---:|---:|---:|
| Toledo aerial (`toledo-aerial-w1`) | **ground / roads** | 158,598 | **56 %** | **77 %** |
| | **roofs** | 30,985 | 27 % | 46 % |
| | other (walls, trees) | 85,821 | 36 % | 56 % |
| | all | 275,404 | 47 % | 67 % |
| Zurich held-out 60 s (360 keyframes, single-view kept) | all | ≈ 20,000 | 38 % | 49 % |

F-score at 1 m: Toledo 0.52; Zurich held-out 0.51.
Street-level flights cannot see roofs above the camera or roads behind buildings. Those surfaces are left
empty, never invented. On a normal aerial pass, roofs and roads are reconstructed (table above).

## 4. Output formats — every file reopened with an independent reader (`verify_exports.py`)

| Format | Reader | Result (`timed10-v1`) |
|---|---|---|
| PLY mesh | Open3D | ✅ 17.9 M triangles |
| PLY points | Open3D | ✅ |
| OBJ | trimesh | ✅ 3.91 M triangles (exchange mesh) |
| GLB | trimesh | ✅ vertex colours |
| FBX (binary 7.4, own writer) | Assimp | ✅ 3,913,314 faces |
| LAS | laspy | ✅ EPSG:32632 |
| GeoTIFF DSM 0.25 m | rasterio | ✅ EPSG:32632 |

Toledo aerial run: all formats ✅ (UTM 17N).

## 5. Robustness

| Check | Result |
|---|---|
| Held-out flight, frozen settings | ✅ runs; 0.19 m path shape, 0.55 m surface median |
| Second continent and platform (DJI Phantom 3, aerial) | ✅ runs |
| 4K input (3840×2160, 60 s, 1,800 frames; bicubic-upscaled from 1080p, so tests decode/memory/runtime, not 4K detail) | ✅ full pipeline in 175.9 s (1080p: 152.3 s) |
| Low-frame-rate stills (1 fps) | ✅ via `--vo mapanything` |
| **Real 4K, HEVC, DJI flight log, no calibration** (Colorado, DJI Mini 4 Pro, 3840×2160 at 29.97 fps, 201.7 s, 1.33 km loop up to 120 m; nominal-io dataset, MIT) | ✅ `real4k-colorado-fast`: video on disk to all exports in **464.5 s (7 min 45 s)**: flight-record CSV converted (TRUE/FALSE recording flag, `0m 12.3s` fly time), intrinsics estimated by DA3 (f = 2,905 px), DPVO ∥ MapAnything 364 s, fusion + exports 37 s; 3.7 M-triangle mesh. Scored against USGS 3DEP `CO_DRCOG_2_2020` (evaluation only): model shape median 1.7 m and 5.4 % of visible survey points within 1 m. The camera looks only 14–23° below the horizon from up to 120 m, so most pixels are far away; the input path works, the fast-mode geometry for this kind of footage does not meet 1 m. |
| **Real 4K with DJI `.SRT`** (Austria BAMBI flight 102, DJI Mavic 3T wide camera, 3840×2160, 214 s, nadir at 75–95 m; CC BY 4.0) | ✅ `real4k-bambi-fast`: **341.3 s (5 min 41 s)** end to end from the raw `.SRT` (per-frame lat/lon/abs_alt, 6,411 blocks). Scored against the Austrian BEV ALS DSM (evaluation only): model shape median 1.8 m, 39 % of visible survey points within 1 m. Nadir footage from 75–95 m in fast mode is not sub-metre; the RTK photogrammetry path (2c) is the one that reached 0.5 m. |
| Telemetry formats | ✅ DJI SRT (bracket, `GPS(...)`, Mini, M300, P4 RTK, Autel; 5 real sample files), DJI / AirData flight-record CSV, GPX, JSON, per-frame CSV; 13 unit tests (`vf2/tests`) |
| Scene size | ✅ tiled GPU fusion with bounded memory: Toledo used 35 tiles, the 10-min flight 12 |

## 6. Final 10-minute run (`final10b`: 540 keyframes, single-view surfaces kept)

Inference and DPVO ran in run `final10` (333 s for both, in parallel); fusion + exports were re-run in
`final10b` after fixing a crash: views with no pixels inside the depth range are now skipped. Because two runs
were combined, this is **not** a timed result. The timed result remains `timed10-v2` (427.8 s).

| Metric | Result |
|---|---|
| Mesh | 27.2 M triangles; exchange mesh 5.88 M |
| Exports | 7/7 reopened (FBX 5,879,786 faces in Assimp; LAS + GeoTIFF EPSG:32632) |
| Camera absolute (direct) | 3.44 m RMSE (H 3.18, V 1.32) |
| Camera shape (Sim3) | 3.13 m |
| Surface shape vs swisstopo LiDAR (eval-only rigid alignment) | **median 0.90 m**; 53 % < 1 m; **79 % < 2 m** |
| Visible-scene recall | 29 % < 1 m; 45 % < 2 m; F-score at 1 m 0.38 |

The heavier settings did not improve 10-min geometry over `timed10-v1` (median 0.90 m both). Over 10 min,
slow drift spreads surfaces that one rigid alignment cannot fully register. That is why recall here is lower
than on the 60 s held-out flight (38 %).
