# VoxelFlight: 48-hour A100 reconstruction sprint

Date: 28 September 2026. This is an execution proposal backed by a host audit and a source-code review, not a claim that the experiments below have already run.

## Decision

Prioritize a better measured surface and one repeatable video-to-model pipeline. Keep a fast, pretrained reconstruction path. Test geometry-aware Gaussian optimization as an optional quality path. Do not spend the two-day window training a foundation model from scratch or editing the website.

A single A100 can support serious inference, scene optimization and bounded fine-tuning experiments. It does not imply unlimited model size, training data or attainable accuracy.

## Verified starting point

- Bridge authentication succeeded at the user-provided LAN Jupyter endpoint. Credentials were used locally and not printed or uploaded.
- GPU: NVIDIA A100-SXM4-40GB, idle at audit; CUDA matrix-multiplication smoke test passed.
- Default interpreter: Python 3.12.3, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121.
- Host exposes 256 logical CPUs and approximately 1 TiB RAM; these are shared-host observations, not exclusive allocations. Start with bounded CPU workers and memory use.
- Open3D, PyCOLMAP, MapAnything and gsplat are absent from the audited default interpreter; ffmpeg and COLMAP executables were not on PATH. Other environments were not exhaustively searched.
- Isolated audit directory created: /workspace/voxelflight_a100_20260928. No model installation, weight download, training or full reconstruction has started.
- Local source dataset is present: 1,800 original 1920x1080 Zurich frames plus real sensor logs and calibration; approximately 0.75 GiB. Original data are images, not an original camera MP4.
- Saved baseline: 90 views; 420-pixel inference; 24-view windows with only two shared frames; 0.08 m TSDF voxel setting.
- Saved output: 646,019 mesh triangles, 119 connected components; largest component contains about 35.5% of triangles. Separate buildings can legitimately form separate components, so this is a diagnostic, not a quality score.
- Recorded trajectory: 0.397 m Sim(3)-aligned, 0.451 m SE(3)-aligned, 8.649 m absolute. Surface error is unmeasured. Those trajectory values are not surface-accuracy evidence.
- Current runner is hard-coded to MPS, at most 96 selected frames and at most 420-pixel inference. Merely launching it on Linux will not use the A100.

## Priority 1 — stronger, coherent geometry

1. Port the existing inference runner to an explicit CUDA/MPS/CPU device abstraction. Preserve checkpoint revisions, input hashes, camera conventions and the unchanged baseline.
2. Replace hard-coded limits with tested experiment settings. Try 90, 180 and 360 well-chosen frames from the same input interval; reject excessive blur and redundant views while retaining useful parallax.
3. Keep actual intrinsics/distortion consistent through every crop, undistortion and resize. Do not reuse Zurich calibration for arbitrary uploaded videos.
4. Re-estimate one globally consistent camera trajectory using COLMAP and bundle adjustment. Use learned matching only if feature matching/registration diagnostics justify the extra dependency.
5. Test supported 518-pixel model inference first. Sweep 24/48/72-view windows with 25–50% overlap only after a VRAM pilot; fit the actual measured 40 GB budget. Larger windows are an experiment, not a promised improvement.
6. Maintain one scene frame through all windows. Fuse overlapping predictions with consistency/confidence weights; check both camera and surface agreement before fusion.
7. Compare the pretrained neural route against CUDA COLMAP PatchMatch stereo plus stereo fusion. Use the same camera/input protocol, valid image masks and evaluation crop. Keep whichever produces better measurable geometry, not whichever has the newest name.
8. Use occlusion-aware checks; exclude sky and dynamic-object evidence where justified. Avoid smoothing away thin structures or falsely treating all vegetation as reliable planar surfaces.

Expected benefit: fewer inconsistent patches and better supported visible surfaces. Additional sampling cannot recover a backside never observed by the flight.

## Priority 2 — one bounded model comparison

Compare the pinned MapAnything baseline with Pi3X on identical selected frames and estimated camera priors. Pi3X supports pose/intrinsic/depth conditioning and a convolutional output head intended to reduce grid artifacts. Its metric scale remains approximate.

Select by camera registration, view consistency, surface accuracy/completeness and runtime. Do not run a large catalogue of models or merge their outputs without geometric agreement.

Pi3/Pi3X weights are non-commercial; keep license/provenance records and do not silently replace a commercially licensed deployment checkpoint.

## Priority 3 — actual surface-focused Gaussian optimization

Start with 2D Gaussian Splatting (2DGS); consider PGSR only if it is practical to install and the first baseline exposes a reason to switch.

- 2DGS represents a 3D scene using oriented surface disks. “2D” does not mean a flat output image.
- Use estimated cameras and observed imagery, not reference/test poses.
- Freeze a held-out set of image views before optimizing the scene.
- Run a short pilot, then a bounded longer run only if rendered held-out views and extracted geometry improve.
- Test depth/normal consistency and confidence-weighted geometric supervision as controlled additions; call them experiments, not validated novelty.
- Export a real mesh and evaluate it. A beautiful Gaussian render alone is insufficient.
- Report scene optimization time separately. This route must remain optional if it cannot meet the mandatory processing-time requirement.

This is training/optimization of a scene representation, not learning a universal video-to-3D model from scratch. It must be repeated for each new scene unless a separate feed-forward model replaces it.

## Priority 4 — metric evaluation and sensor handling

- Use Zurich for the real aerial input, synchronization and trajectory audit.
- Use one or two ETH3D outdoor scenes with scan ground truth for surface diagnostics, or Tanks and Temples Barn with its laser reference. These are geometry benchmarks, not exact SIH drone/GPS-equivalent tests.
- Keep reference meshes and reference camera poses out of reconstruction. Give the evaluator access only after outputs are frozen.
- Report any evaluation-only rigid/similarity alignment explicitly. No per-window reference fitting or alignment presented as operational georeferencing.
- Measure surface accuracy AND completeness at declared thresholds, report F-scores and outlier statistics, and preserve the evaluation region. Do not improve a score merely by deleting difficult surfaces.
- Report local surface geometry, absolute geolocation, and camera trajectory as separate results.
- Audit timestamp offsets, GNSS outliers, world/camera axes, altitude datum and any camera-to-GNSS lever arm. Add IMU constraints only if calibration and timing are trustworthy.
- Noisy GPS can limit absolute accuracy independently of visual quality. GPU power alone does not guarantee sub-metre positioning.
- Reserve a different contiguous flight interval for a ten-minute end-to-end test. Document whether the selected pretrained checkpoint may have seen benchmark scenes; a local holdout is not proof of unseen pretraining data.

## Would we fine-tune?

Only after the reconstruction baseline works and if a specific generalization defect can be measured.

Optional, time-boxed experiment: freeze most of the image encoder and train a small decoder/adapter or confidence correction on scene-separated depth/pose-labelled data. Validate on different scenes; keep the original model as control. Intrinsics/poses/depth must transform correctly under augmentations.

Plausible augmentations: exposure/colour changes, compression and moderate motion blur, consistent resizing/cropping, frame dropout and simulated sensor noise. Do not apply flips/crops independently of camera geometry or use the held-out flight as training data.

Budget no more than six hours for this optional branch, including validation. Abandon it if data preparation, memory, stability or held-out results are poor. The recommended core training experiment is geometry-aware scene optimization, not full foundation-model training.

## Relative 48-hour schedule

| Window | Main task | Gate / deliverable |
| --- | --- | --- |
| 0–4 h | Isolated environment, data staging, CUDA port, baseline replay | Same input and provenance; valid poses, depths and exports; CUDA memory pilot |
| 4–12 h | Frame/overlap improvements, global cameras, MapAnything/Pi3X and MVS pilots | Select a geometry route on frozen diagnostics, not appearance alone |
| 12–24 h | Surface-aware Gaussian pilot and mesh extraction | Compare extracted surfaces and held-out rendering against the unchanged baseline |
| 24–34 h | Strongest route refinement; optional <=6 h adaptation only if justified | Accuracy/coverage/runtime ablations; keep unsuccessful runs documented |
| 34–42 h | Independent surface evaluation and different-flight/ten-minute test | Honest surface metrics, absolute positioning and complete processing time |
| 42–48 h | Freeze dependencies/checkpoints; connect the chosen processing route to the job runner | Repeatable unseen-input command; artifacts, reports and recovery instructions |

These are work budgets, not guaranteed experiment durations. Cut optional methods if environment setup or data transfer consumes the margin.

## End-state acceptance

1. One command processes a new video plus supported metadata into a point cloud and inspectable textured surface.
2. Geometry is visibly and quantitatively compared with the existing saved result.
3. No ground-truth pose/mesh leakage into reconstruction, training holdouts or georeferencing.
4. Model code/weight revisions, runtime, peak VRAM, input selection and coordinate reference are logged.
5. Fast and optional refined modes are distinguished; all optimization time is counted.
6. Spatial-accuracy or SIH compliance is claimed only after the corresponding independent measurement. No fabricated roofs, hidden backsides or GAN-filled geometry labelled as observed.

## Reproducible primary sources

- [MapAnything code, model interfaces and inference](https://github.com/facebookresearch/map-anything)
- [MapAnything training: multi-node reference recipe and memory controls](https://github.com/facebookresearch/map-anything/blob/main/train.md)
- [Pi3/Pi3X implementation and license](https://github.com/yyfz/Pi3)
- [COLMAP dense reconstruction](https://colmap.github.io/)
- [2DGS paper/project](https://surfsplatting.github.io/) and [implementation](https://github.com/hbb1/2d-gaussian-splatting)
- [PGSR implementation](https://github.com/zju3dv/PGSR)
- [ETH3D inputs and scan references](https://www.eth3d.net/datasets)
- [Tanks and Temples videos and geometry](https://www.tanksandtemples.org/download/)

