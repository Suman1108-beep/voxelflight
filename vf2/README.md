# VoxelFlight v2 — single-pass drone video → georeferenced 3-D model

SIH26158 (NTRO, Drone/Robotics). This folder is the A100 pipeline built on 3 Oct 2026. It replaces the
CPU fusion path and adds trajectory fusion, metric snapping, tiled GPU fusion and the full export set.
Every number in this file names the run it came from; see `RESULTS.md` for the evidence ledger.

## One command

```bash
python vf2/run_pipeline.py --video flight.mp4 --telemetry flight.SRT --output runs/my-flight
```

`--telemetry` accepts a DJI `.SRT` sidecar, a DJI / AirData flight-record CSV, GPX, a JSON list or the per-frame CSV below.
`--calibration camera.json` is optional: without it the camera intrinsics are estimated from the video with Depth Anything 3.
Video that OpenCV cannot decode (for example DJI HEVC) is converted to H.264 at full resolution. All three steps run
inside the timed pipeline.

Modes (same inputs; see `RESULTS.md` for the numbers):

| Mode | Command | Use |
|---|---|---|
| Fast | `run_pipeline.py` | DPVO ∥ MapAnything, GNSS fusion, GPU TSDF; the speed-compliant default |
| High quality | `quality_pipeline.py --keyframes 720 --depth-max 35` | SuperPoint + LightGlue + COLMAP camera solve ∥ depth, cameras locked to GNSS |
| Maximum coverage | quality cameras, then `make_video_crops.py` + `da3_infer.py --images` + `dense_multi_crops.py` | 4 overlapping crops per keyframe for higher-resolution depth |
| Gap-free surface | `fill_holes.py --run <run> --out <run>-fill` | adds a tinted, interpolated top surface over holes; reported separately |
| Photo texture | `texture_atlas.py --run <run> ...` | 400k-triangle mesh with an 8K atlas baked from the source frames |
| Gaussian surface (optional) | `prep_gs.py`, PGSR `train.py`, `gs_fuse.py` | planar Gaussian splatting refinement |

Profiles used for the measured runs:

| Flight type | Extra flags |
|---|---|
| Street-level / low altitude video (Zurich) | defaults (`--voxel 0.05 --depth-max 25`); `VF_WEIGHT=1` keeps single-view surfaces (more coverage) |
| Aerial pass ~100 m (Toledo, BAMBI) | `--voxel 0.15 --depth-max 150..300 --exchange-voxel 0.3`, `VF_WEIGHT=1` |
| Low frame rate / geotagged stills | `--vo mapanything` (neural poses replace DPVO), stills → video with `prepare_stills_video.py` |
| 4K video | no change (inference downsizes internally; tested on real DJI 4K H.264 and HEVC) |
| **RTK/PPK positions** (`gps_eph_m` ≈ cm) | run `run_pipeline.py --vo mapanything` once for the depth predictions, then `rtk_photogrammetry.py --predictions <run>/inference`. This is photogrammetric SfM with RTK priors; measured **0.5 m horizontal / 0.41 m vertical absolute** vs the Austrian national DSM |

## Setup

Package versions are in `requirements-a100.txt`. Paths default to `ROOT=/workspace/voxelflight_a100_20260928` in
`common.py`; `cloud/root_env.patch` makes this an environment variable. Weights download from Hugging Face on first use
(`facebook/map-anything-apache`, `depth-anything/DA3-GIANT-1.1`); DPVO needs its own environment and `dpvo.pth`
(see the DPVO README). `cloud/modal_app.py` builds the same environment as a container image.

Evaluation (never part of the pipeline): `evaluate_run.py` (Zurich reference poses + swisstopo LiDAR,
visibility-aware coverage) and `evaluate_aerial.py` (USGS 3DEP LiDAR, per-class roofs / roads).

### Input contract (matches the problem statement)

| Input | Status in SIH | Format |
|---|---|---|
| Drone video 1080p/4K | mandatory | MP4/MOV, H.264 or HEVC |
| GPS coordinates | mandatory | DJI `.SRT`, flight-record CSV, GPX, JSON, or `latitude, longitude, altitude_m` per video frame (`source_frame, timestamp_s`) |
| Flight metadata | mandatory | `gps_eph_m` (horizontal accuracy) if available, else 5 m assumed |
| Barometric altitude | optional | `baro_alt_m` column — used for height when present |
| Camera intrinsics | optional | `camera.json`: `intrinsic_matrix`, `distortion_coefficients`, `width`, `height`; estimated when absent |
| IMU, RTK/PPK | optional | RTK/PPK positions go in the same GPS columns with a small `gps_eph_m`; the fusion weights them accordingly |

No survey, LiDAR or reference-pose data are ever read by the pipeline.

## What happens inside

```
MP4 ──┬─► DPVO visual odometry (every 5th frame, GPU)           ┐ run concurrently
      └─► MapAnything metric depth + poses (360 keyframes, GPU)  ┘ on one A100
                │
GPS + baro ─────┴─► trajectory fusion: robust Sim(3) VO→GNSS, then sparse least squares
                    (GNSS corrects slow visual drift; barometer fixes height)
                │
                ├─► snap every neural view onto the fused metric trajectory
                │   (local rotation average + scale/translation from ±20 neighbours)
                │
                ├─► tiled GPU TSDF fusion (60 m tiles, Open3D CUDA voxel blocks, 5 cm voxels)
                │   each tile cropped exactly + fragment-cleaned in parallel threads
                │
                └─► exports: PLY mesh + points, LAS (UTM), GeoTIFF DSM (UTM),
                    OBJ / GLB / FBX (decimated exchange mesh), metadata.json with hashes
```

Key design decisions, each backed by a measurement:

* **Why fuse GNSS + barometer + visual odometry.** DPVO is accurate locally (0.28 m over any 60 s window on
  the 10-min Zurich flight) but its scale drifts ±7 % over 10 min. Raw GNSS is 4.3 m horizontal / 6.1 m
  vertical. The barometer is 1.4 m vertical. Fusion: 7.5 m → 3.3 m absolute camera error.
* **Why snap MapAnything views to the fused trajectory.** MapAnything's native metric scale was 22 % short on this
  flight (snap scale median 1.29); windows stitched on their own drift (6.5 m Sim3 error in the old baseline).
* **Why GPU tiled fusion.** CPU TSDF took 268 s (and 4,072 s in the DPVO run). GPU TSDF integrates and extracts 360
  views in 14.5 s; tiling keeps memory bounded so the same code scales to any area and shares a GPU.
* **Why our own FBX writer.** No FBX SDK on the server; `fbx.py` writes binary FBX 7.4, verified by importing in
  Assimp (triangle counts match the PLY exactly).

## Output files (`<output>/model/`)

| File | CRS / frame | Purpose |
|---|---|---|
| `model_mesh.ply` | local metres (UTM − origin) | full-resolution coloured mesh |
| `model_points.ply` | local metres | fused point cloud |
| `model_points_utm.las` | EPSG:326xx (UTM), heights = GNSS MSL | GIS / survey software |
| `dsm_utm.tif` | EPSG:326xx, 0.25 m | digital surface model (max observed height; unobserved = nodata) |
| `model_mesh.obj/.glb/.fbx` | local metres (GLB is Y-up) | exchange/viewer mesh (10 cm vertex clustering) |
| `metadata.json` | — | UTM origin, EPSG, SHA-256 of every artifact |

`verify_exports.py <dir>` reopens every file with an independent reader (Open3D, trimesh, laspy, rasterio, assimp).

## Honest limits (do not remove)

* Absolute accuracy is bounded by the GNSS: ≈ 2.7–3.5 m with the consumer GPS in the Zurich logs, depending on mode. Sub-metre absolute
  positioning needs RTK/PPK (an optional SIH input) or ground control. Relative/local geometry is far better.
* A single pass cannot see backsides or roofs above the camera; these are left empty, never hallucinated. The optional
  gap-free layer (`fill_holes.py`) is tinted and reported separately from measured geometry.
* The Zurich video is a derived H.264 encode of the official image sequence, not an original camera file.
