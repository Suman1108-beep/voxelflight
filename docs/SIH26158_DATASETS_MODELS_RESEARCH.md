# SIH26158 — datasets, model inventory, and research directions

**Audit date:** 3 October 2026. Read the [project handoff](SIH26158_READ_THIS_FIRST.md) first. “Present” means observed in this audit; “historical” means referenced by old scripts/docs but current storage was not verified. No official organizer evaluation dataset has been verified locally or on the A100.

## Input and label discipline

The deployed inference contract should take a video plus timestamped GPS/flight metadata, and optionally intrinsics, distortion, IMU, barometer, RTK/PPK. It must normalize timestamps, coordinate reference system, axis/altitude conventions, units, camera-to-GNSS lever arm, and missing intervals. Training/evaluation references such as Zurich `GroundTruth*.csv` and ETH3D scans are **labels for evaluation**, not allowed inputs for the unseen-video pipeline. A derived MP4 from official image sequences is useful for engineering the video decoder, but it is not proof of robustness to arbitrary camera-produced MOV/MP4 files.

## Dataset inventory and intended role

| Dataset / subset | Current evidence | Inputs and labels | Correct use | Limitation / next step |
|---|---|---|---|---|
| **Zurich Urban MAV** frames `61201–63000` | **Local and A100 present.** Local folder about 750.9 MiB extracted; 1,800 1920×1080 JPEGs + 10 metadata files, archive CRC and SHA-256 checked in `datasets/zurich-61201-63000/download_manifest.json`. | Real UAV imagery, GPS/telemetry/calibration; released camera reference for post-hoc evaluation. | Saved public demo; 90/180-view 60-second experiments. | No surveyed 3D surface reference. Segment has been used repeatedly for tuning, so not a clean final holdout. |
| **Zurich Urban MAV** frames `20001–21800` | **A100 present** under `/workspace/voxelflight_a100_20260928/datasets/`; local A100 ledger/report present. | 1,800 sequential 1080p frames, onboard GPS and calibration; reference camera positions. Derived constant-frame-rate 60-second H.264 input. | Disjoint short-video/GPS and speed test. | Also used for parameter selection; no independent surface GT. |
| **Zurich Urban MAV** frames `40001–58000` | **A100 present**; 18,000 sequential frames; derived 600-second H.264, exact reports saved locally. | Video, onboard GPS/calibration, released reference camera positions. | Ten-minute A100 runtime and long-trajectory gate. | Not an untouched test anymore; output surface cannot be scored from camera positions alone. |
| **ETH3D courtyard** | **A100 present**; 38-image test and local JSON evaluation reports. | DSLR imagery, camera reference and laser evaluation scans. | Independent *diagnostic* of surface distance/completeness after outputs frozen. | Not a single UAV flight, not GPS input, sampled proxy is not official free-space-aware ETH3D score; possible overlap with pretrained corpus not excluded. |
| **Organizer SIH evaluation data** | **Not present/verified.** Supplied PDF says data will be provided in real time. | Unknown video, GPS, metadata, reference and evaluator. | Final blind acceptance only. | Obtain schema/evaluator immediately when released; keep labels segregated. |
| **Tanks and Temples / Barn** | **Historical mention only** in `datasets/README.md`; not found in the audited local/A100 dataset inventory. | Public imagery and laser reference. | Possible geometry diagnostic. | Do not claim downloaded/current. Confirm license, access, and domain relevance before use. |
| **TartanAir V2 / ArchVizTinyHouseDay** | **Historical mention only**; not found in audited current storage. | Synthetic RGB/depth/pose/semantics/IMU. | Regression/sensor unit tests, not substitute for real UAV. | Do not claim current training/evaluation from it. |
| **UseGeo** | **Proposed, not downloaded.** [Project](https://usegeo.fbk.eu/). | UAV imagery with geographic/survey context. | More relevant independent UAV surface/geo validation, subject to exact access/license. | Obtain a bounded subset and establish a blind split only if time permits. |
| **UAVFF3D** | **Proposed, not downloaded.** [Paper](https://arxiv.org/abs/2605.17942), [official repository](https://github.com/yanxian-ll/UAVFF3D). | UAV-oriented reconstruction benchmark with real and synthetic sequences. | Domain-specific stress/evaluation and possible fine-tuning research. | Download/weight/license friction; not a prerequisite for the final blind test. |

Official Zurich source: [Zurich Urban MAV Dataset](https://rpg.ifi.uzh.ch/zurichmavdataset.html). ETH3D: [ETH3D benchmark](https://www.eth3d.net/). The local `datasets/README.md` has an older statement about a full 28 GB archive on a different remote instance; **current reachability of that old archive is unverified**. The present A100 inventory lists only the three Zurich subsets, ETH3D courtyard, and intermediate caches—not Tanks and Temples, TartanAir, UseGeo, or UAVFF3D.

## What model is actually running?

| Component | Status | Purpose | Training / caveat |
|---|---|---|---|
| `facebook/map-anything-apache` | **Pinned/pretrained checkpoint used in A100 experiments.** [Official repository](https://github.com/facebookresearch/map-anything). | Feed-forward camera/depth/3D prediction in overlapping windows. | It is not a new model trained from scratch by the team. The Apache-licensed variant is intentionally distinct from other checkpoint licenses. |
| **PyCOLMAP/COLMAP** | Used in saved SfM-assisted demo / research runs. [Documentation](https://colmap.github.io/). | Feature matching, camera registration/bundle adjustment, calibration anchors. | 180-view SfM alone once took 1,198.6 s; not the desired 10-minute fast path. |
| **DPVO** | Official pretrained checkpoint present on A100 in `thirdparty/DPVO/dpvo.pth`; 10-minute pose runs recorded. [Paper/repository](https://github.com/princeton-vl/DPVO). | Long-sequence visual odometry/camera shape. | Arbitrary visual scale, drift remains; 3.907 m post-hoc Sim(3) on the 10-minute segment. Runtime must be counted when integrated. |
| **TSDF + multiview support + Open3D** | Used in local/A100 fusion. | Combine observed depths into real mesh/point cloud and suppress outliers. | CPU-heavy; filtering can erase valid surfaces. Huge meshes/components do not imply quality. |
| **GPS/UTM export** | Tested on selected A100 results. | Convert local model into a geospatial reference; LAS/GeoTIFF/GLB outputs. | Ordinary GNSS bias is not corrected by assigning an EPSG code. GeoTIFF currently represents occupancy/derived raster, not a surveyed DEM. |
| **SegFormer / MASt3R-SLAM / older stack** | Present in historical L40S research path, not the current A100 baseline or public new-upload default. | Semantic masking / visual anchors. | Do not put them in the current run's tech stack unless the exact code path and checkpoint are used. |
| **2D Gaussian Splatting / 3DGS** | Optional appearance research. 7k-step 2DGS result exists. | Photoreal novel-view rendering. | Scene-specific optimization; cannot replace independent surface accuracy or a fast unseen-input core. |
| **Mac Apple-MPS preview** | Public upload service code's current default. | Bounded user-upload inference when a Mac worker/ingress is online. | Separate from A100 research quality; current configured temporary hostname did not resolve in this audit. |

The pipeline is not accurately described as “a 3D CNN.” Its neural geometry component is a modern pretrained transformer-style architecture; camera tracking and image encoders may include convolutional operations, followed by classical optimization, multiview geometry, TSDF fusion, and geospatial conversion. The team's contribution is chiefly **system integration, alignment, measurement, and an inspectable product workflow**, not training a proprietary foundation model.

## Research that may actually move the SIH score

Rank by chance of improving **accuracy + completeness (50%)** before the deadline, not novelty for its own sake.

1. **Long-sequence pose stability and anchored coordinates.** [GeoFF3D](https://arxiv.org/abs/2608.28288) and its [official repository](https://github.com/yanxian-ll/GeoFF3D) study coordinate-anchored UAV reconstruction and spatial chunking. The lesson is to prevent per-window drift from accumulating, not to paste their published timing into our measured pipeline. Their checkpoint/access requirements and our reproducibility are not verified. Their code notes that sequential footprint estimation avoids a GT-depth dependency.
2. **UAV-domain validation before fine-tuning.** [UAVFF3D](https://arxiv.org/abs/2605.17942) is relevant to a real UAV/metric benchmark. First establish baseline score and legal data access; fine-tune only if the gain is measurable on a distinct holdout. A100 compute alone cannot create ground truth.
3. **Long-sequence transformer/memory alternative.** [VGGT-Long repository](https://github.com/DengKaiCQ/VGGT-Long) may offer a coherent visual global context. Treat as a controlled ablation: same held-out flight, same camera/surface metrics and stopwatch. Not “SOTA” merely because the paper is recent.
4. **Sensor fusion where available.** [MASt3R-Fusion](https://github.com/GREAT-WHU/MASt3R-Fusion) uses visual/inertial coupling, but SIH makes IMU optional. The default must still work with mandatory video+GPS+metadata alone; use IMU/RTK as an improvement branch, not a hidden requirement.
5. **Surface-aware appearance.** [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) can improve render/surface regularity in some settings. It should run after robust camera/geometry inference and must be included in the <15-minute budget if submitted as core.

Do **not** spend the remaining time training a large model from scratch, adding a GAN to hallucinate occluded buildings, or optimizing splats until a render looks perfect. Those directions risk creating convincing but unmeasured geometry. The decisive model experiment is one that improves held-out **absolute/local camera alignment, independent 3D surface error, visible-scene recall, and runtime** together.

## Data governance and reproducibility checklist

- Record source URL, license, acquisition date, original checksum, extracted file count, frame interval, resolution, timestamps, camera calibration, coordinate datum and whether video is original or derived.
- Keep dataset subsets, pretrained weights, generated caches and private uploads outside the public Git repository. Public code may provide download/verification scripts and manifests without bundling large or restricted assets.
- Pin checkpoint revision and Python/package versions; the Mac DINOv2 architecture dependency is currently noted as `main` in the older setup doc and should be pinned for a reproducible release.
- Record train/validation/test provenance. Any Zurich segment used for hyperparameter choice is no longer a clean final holdout. ETH3D pretraining overlap is unverified.
- Freeze a predicted artifact before reading a reference camera trajectory or LiDAR scan. Save a hash and evaluation command.
- Measure the same output under **local shape**, **absolute georeference**, **surface precision**, **surface completeness**, and **appearance** protocols; never combine them into an invented “SIH score.”

See the [run-by-run evidence](SIH26158_RESULTS_EVIDENCE_LEDGER.md) for measured numbers and the [execution plan](SIH26158_EXECUTION_PLAN_TO_OCT5.md) for tests to run next.
