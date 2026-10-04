# SIH3D — real Apple-GPU reconstruction

The Mac now runs pretrained MapAnything inference locally. This is a **separate experimental pipeline**, not the earlier MASt3R-SLAM / MapAnything / SegFormer / Gaussian-splat NVIDIA pipeline.

## Run it

Double-click **Start Mac Engine.command** in this directory. Keep the terminal open and the Mac awake, then open:

http://127.0.0.1:8133/engine.html

Choose an MP4/MOV video, optional GPS CSV/JSON/SRT, and optional pinhole-camera calibration JSON. Start reconstruction. Progress, cancellation, persistent local job history, actual input thumbnails, interactive textured mesh, points, measurements and downloads are available. Only one GPU job runs at a time. Press Control-C in the terminal to stop the engine.

No new training is necessary. The first setup downloads approximately 4.9 GB of pretrained weights; they are already cached on this Mac. New inference uses those weights, not answers for a known scene.

The hosted website includes a real Mac-generated sample and a link to the local engine. **The hosted page does not run GPU inference or remotely access this Mac.** Someone else's localhost is their own computer, not yours. No public tunnel, cloud GPU or auto-start daemon has been installed. The hosted site's existing private audience was preserved.

## What was actually tested

- Machine: MacBook Air M4, 16 GB unified memory; PyTorch 2.14.0, MPS available.
- Four actual saved Zurich frames, 350 px: 186,197 reconstructed points, 98,145 surface triangles.
- The textured four-view test completed in approximately 26 seconds with Python TCP connections explicitly blocked. The GLB embeds an atlas of actual source pixels, not generated artwork.
- Twelve saved frames / five overlapping windows, 350 px: 603,609 points, 280,061 triangles, approximately 36 seconds including input and model loading. Approximately 5.1 GiB GPU allocation was observed at the end; this is not a measured peak.
- Real API upload → new GPU inference → status → downloads succeeded. The test MP4 was assembled from 19 existing Zurich frames; it was **not the original flight recording or an unseen-flight benchmark**. That 12-keyframe run took approximately 33 seconds.
- Security, cancellation, geometry operations and GIS file metadata have automated tests. Browser click/visual QA has not been performed in this turn.

Do not extrapolate these small tests to the SIH 10-minute-video runtime target.

## Architecture

1. Decode and sample frames across the full clip, with a configurable cap (12–96 in the UI). Optionally undistort using supplied calibration. Small inference resolution keeps memory bounded.
2. Infer depths, camera intrinsics and camera-to-world poses using pretrained MapAnything on Apple MPS. Four-view windows share two frames. Unsupported operations may use PyTorch's CPU fallback; the model runs on MPS.
3. Register windows using robust 3D correspondences at identical pixels of the overlapping views. No ground-truth trajectory is supplied. This is not global bundle adjustment, and drift/seams can remain.
4. Filter ambiguous depth and depth edges. Triangulate supported image neighborhoods, embed source-image texture pixels, and average cloud observations in occupied voxels. No GAN, hole hallucination or unseen-backside completion.
5. Optionally align the estimated camera path to timestamped GPS. Export a provisional UTM LAS cloud and observed-cell GeoTIFF DSM only when the path constrains a usable 3D alignment.

## Honest limits

The Mac mode has **no independently measured trajectory or surface accuracy**. It does not inherit the old 0.810 m aligned trajectory score. It is not certified to meet SIH's ≤1 m spatial requirement, complete-visible-scene coverage or <15 minutes per 10-minute video target. More points or realistic textures do not prove geometry accuracy.

Without GPS: coordinates have learned metric scale, local Z-up and unknown north. Measurements are estimates. With GPS: the alignment residual is a fit to the same GPS used for alignment, **not an independent accuracy score**. Unknown time offset, GPS noise and altitude datum can still make absolute positions wrong. Nearly straight/stationary GPS paths, insufficient timestamps or missing altitude cause georeferencing to be withheld.

This mode does not currently perform IMU/RTK factor-graph fusion, semantic dynamic-object removal, calibrated uncertainty estimation, global bundle adjustment, occluded-surface inference or Gaussian-splat refinement. FBX is not exported. GLB is textured; OBJ is a self-contained colored geometry export. LAS and GeoTIFF are conditional on usable telemetry.

Use steady, overlapping footage and a short clip first. Input compatibility is not universal: codecs must be supported by OpenCV. GPS input must have recognizable latitude/longitude fields, numeric synchronized timestamps and altitude for 3D alignment. Calibration JSON needs matching width/height and either a 3×3 `intrinsic_matrix` or `fx`, `fy`, `cx`, `cy`; optional OpenCV `distortion_coefficients` are supported. Fisheye calibration is rejected explicitly.

## Files and durability

- `.venv-mac/`: isolated Python runtime; the original environment was not changed.
- `cache-mac/huggingface/`: pinned full model checkpoint.
- `cache-mac/torch/hub/facebookresearch_dinov2_main/`: cached encoder source; reconstruction forces local loading.
- `vendor/map-anything/`: official source pinned at `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
- `runs-mac/jobs/<id>/`: uploaded inputs, worker log, progress, output and provenance. These persist across server/Mac restarts. A reboot interrupts an active job; restart processing with a new job afterward.
- `runs-mac/offline-textured-proof/`: actual four-view offline proof.
- `viewer/assets/mac-proof/`: a publication-safe copy of that actual sample, not private user uploads.

Back up this directory, especially `cache-mac` and the model source, before replacing or wiping the Mac. A server restart does not delete these local files. The website does not serve model weights, upload inputs, Python source or logs.

## Reproduce setup / tests

Use an ARM64 Python 3.12 environment, install `requirements-mac.lock.txt`, then run `setup_mac.py` once with internet. The current `.venv-mac` is already ready. The virtual environment is not portable to a different operating system.

```sh
.venv-mac/bin/python setup_mac.py
.venv-mac/bin/python -m unittest discover -s tests -p 'test_mac_engine.py' -v
.venv-mac/bin/python mac_reconstruct.py --video /path/to/flight.mp4 --output /path/to/new-empty-result --max-frames 24 --size 350
```

`scripts/smoke_mac_api.py` exercises the running real local API. `scripts/verify_mac_offline.py` runs actual GPU inference with outbound TCP blocked; its output path must be absent/empty because existing results are never overwritten.

References: [MapAnything official source](https://github.com/facebookresearch/map-anything), [Apache model weights](https://huggingface.co/facebook/map-anything-apache), [DINOv2](https://github.com/facebookresearch/dinov2). Model checkpoint revision: `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`.
