# SIH26158 requirements and measured evidence

Source: the problem-statement document supplied to the team (printed pages 37–39), **Single-Pass Drone Video to Accurate
3D Model Generation System**, National Technical Research Organisation, category Software, theme Drone/Robotics.
Dataset availability is stated as real time.

Every number below was measured by the VoxelFlight v2 pipeline (`vf2/`) on one NVIDIA A100-40GB and scored against
**independent references the pipeline never reads** (national airborne LiDAR and the Zurich reference poses). Run names
and full tables are in [`vf2/RESULTS.md`](../vf2/RESULTS.md). This is an evidence checklist, not an official score.

| Requirement | Target | Measured evidence | Status |
|---|---|---|---|
| Input | 1080p/4K video, GPS and flight metadata | 10-min 1080p Zurich flight; real DJI 4K: H.264 with `.SRT` (Austria, BAMBI) and HEVC with a DJI flight-record CSV (Colorado) in 5 min 41 s and 7 min 45 s end to end (fast-mode geometry on these high-altitude clips is ≈ 1.7–1.8 m shape, so not sub-metre). Telemetry: DJI SRT (all common firmware styles), DJI/AirData flight records, GPX, JSON, per-frame CSV; HEVC is transcoded; missing intrinsics are estimated | Met |
| Output | Textured mesh or point cloud | Coloured mesh (up to 52 M triangles), photo-textured mesh (atlas baked from the source frames; 6K on the website, 8K in the run folder), coloured point cloud | Met |
| Speed | < 15 min for a 10-min video | Fast mode **7 min 8 s**; high-quality mode **14 min 28 s** with Depth Anything 3 (14 min 12 s with MapAnything), each one continuous run from video on disk to all exports. The maximum-coverage model is slower (≈ 37 min, sum of stages) | Met |
| Spatial accuracy | ≤ 1 m | Model shape 0.55–0.91 m (0.59 m for the 15-minute high-quality model, 0.73 m for the maximum-coverage model) median vs survey LiDAR; **absolute 0.5 m horizontal / 0.41 m vertical with RTK** (Austria); ≈ 2.7 m with consumer GNSS | Met with RTK; shape met |
| Coverage | Entire visible scene | Visible survey points within 1 m: fast 24 %, high quality 37.9 % (DA3), maximum coverage **46.5 %**; with the tinted ground-aware gap-fill layer **51.2 %** (68.2 % within 2 m). Surfaces never seen by the camera are left empty or marked as interpolated | Partial (46.5 % measured, 51.2 % with fill) |
| Formats | OBJ, PLY, LAS, GeoTIFF, GLB/glTF, FBX | All six from every run (FBX by our own binary 7.4 writer), each reopened by an independent reader (`vf2/verify_exports.py`) | Met |
| Visualisation | Web or desktop | https://voxelflight-3d.web.app: flight video, 3D orbit, layers (incl. the gap-fill layer), measurement, downloads | Met |
| New input | Single moving-drone pass | Zurich (street level), Toledo (aerial stills), Austria helenenschacht (RTK), BAMBI and Colorado (real 4K) | Met on 5 flights |
| Optional sensors | IMU, barometer, intrinsics, RTK/PPK | Barometer fused for height; intrinsics used when given, estimated otherwise; RTK/PPK via position accuracy weights. IMU not yet fused | Partial |

## Evaluation weights in the problem statement

Reconstruction accuracy 30 %; model completeness 20 %; processing speed 20 %; innovation 15 %; scalability 10 %; user
interface 5 %.

## What is still open

- **Coverage of the whole visible scene.** No single-pass method measures every visible surface; grazing facades, ground
  under canopy and long-range surfaces stay thin. The gap-fill layer closes holes visually but is interpolation, and is
  scored separately.
- **Absolute ≤ 1 m without RTK.** Consumer GNSS limits placement to ≈ 2.7 m; the model *shape* is sub-metre either way.
- **Live uploads.** Offline during judging; a GPU backend for the same pipeline is prepared in [`vf2/cloud`](../vf2/cloud/README.md).
- **IMU fusion** is not implemented.

The September prototype's checklist is preserved in [`docs/results.md`](results.md) for provenance.
