# SIH26158 requirements and current status

Source: the problem-statement document supplied to the team, printed pages 37-39, titled **Single-Pass Drone Video to Accurate 3D Model Generation System**. The supplied extract identifies the organisation as National Technical Research Organisation, category Software, theme Drone/Robotics. Dataset availability is stated as real time.

This is an evidence checklist, not an official score or certificate.

| Requirement | Requested target | Current evidence |
|---|---|---|
| Input | 1080p/4K video, GPS and flight metadata | Video/image-sequence processing and supported telemetry/calibration parsers exist. Arbitrary formats and 4K robustness are not exhaustively tested. |
| Output | 3D mesh / point cloud | Actual saved meshes and point clouds are available. |
| Speed | **Less than 15 minutes for a 10-minute video** | Not benchmarked on a ten-minute input. Short-sequence timings are not proof. |
| Spatial accuracy | **≤1 m** | Not verified against independent surface ground truth. Aligned trajectory RMSE is a different metric. |
| Coverage | Entire visible scene | Partial coverage with holes and artifacts. |
| Formats | OBJ, PLY, LAS, GeoTIFF, GLB/glTF, FBX | OBJ, PLY, LAS, GeoTIFF and GLB are available. FBX is not demonstrated. |
| Visualization | Web or desktop | Public browser viewer with video, orbit controls, layers, measurement and downloads. |
| New/unseen input | Single moving-drone pass | Preview inference code exists; broad unseen-flight robustness is not established. |
| Optional sensors | IMU, barometer, camera intrinsics, RTK/PPK | Calibration and selected telemetry support exist. Full inertial integration and all optional sensor formats are not established. |

The model targets observed façades, structures, road/terrain context and vegetation. It cannot truthfully present wholly occluded roofs or backsides as measured surfaces. Additional validation is required for moving objects, blur, shadows, compression, GPS noise and weak viewing geometry.

## Supplied evaluation weights

Reconstruction accuracy 30%; model completeness 20%; processing speed 20%; innovation 15%; scalability 10%; user interface 5%. These weights describe the supplied rubric, not scores awarded to VoxelFlight.

## Next acceptance tests

1. Freeze the model/checkpoints and run an unseen ten-minute video on specified hardware.
2. Measure surface error against an independently surveyed reference, using the organizer's alignment and coverage protocol.
3. Measure completeness over the entire visible scene and report failure cases.
4. Validate each required export in an independent viewer, including an FBX implementation.
5. Test reconstruction across supported upload formats and sensor conditions.
