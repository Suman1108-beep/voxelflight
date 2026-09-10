# System architecture

VoxelFlight has a public inspection website and a separate reconstruction service. Firebase serves the website and authenticates users. It does not execute the reconstruction model.

## Data flow

```text
Reviewer or operator
  |
  +-- Public saved demo -- Firebase Hosting -- Video + Three.js viewer
  |                                           |
  |                                           +-- Mesh, telemetry, exports, evidence
  |
  +-- New reconstruction -- Firebase sign-in -- Bearer-authenticated FastAPI API
                                               |
                                       Validate inputs and ownership
                                               |
                                       Bounded queue / one worker
                                               |
                                   Decode video and parse calibration/GPS
                                               |
                                     Pretrained MapAnything inference
                                               |
                                   Window alignment and observed geometry
                                               |
                                    Owner-restricted files and job status
                                               |
                                       Interactive result viewer
```

The default upload worker is `mac_reconstruct.py`, using Apple MPS and cached checkpoints. It is a preview pipeline. The higher-quality saved reconstruction follows the separate experiment below; uploading a video does not automatically run that experiment.

## Latest saved geometry experiment

1. Select 90 calibrated images from a continuous 60-second Zurich Urban MAV sequence.
2. Estimate visual camera anchors with calibrated COLMAP SfM (`scripts/run_video_sfm.py`).
3. Predict dense geometry using overlapping MapAnything windows and visual pose priors.
4. Apply multiview support checks and TSDF integration (`mac_refuse.py`).
5. Texture the supported mesh with actual source camera images (`scripts/texture_fused_mesh.py`).
6. Produce separate local-visual and provisional GPS/UTM-referenced deliverables.
7. Evaluate frozen predicted camera positions against the reference. Never use reference poses in reconstruction.

The earlier GPU experiment uses MASt3R-SLAM, MapAnything, SegFormer and a position-only telemetry fusion graph. Optional Gaussian splatting fits appearance per scene. That older experiment is retained separately in [results.md](results.md). The final six-slide presentation describes the latest saved reconstruction. Its geometry, timings and image metrics must not be combined with the earlier experiment.

## Components

| Component | Code | Responsibility |
|---|---|---|
| Input contracts | `src/sih3d/` | Canonical video, telemetry and calibration metadata |
| Public API | `public_server.py`, `public_security.py`, `mac_server.py` | Token verification, ownership, upload bounds, queue and artifacts |
| Preview reconstruction | `mac_reconstruct.py`, `mac_geometry.py` | Cached model inference, alignment and geometry exports |
| Surface refinement | `mac_refuse.py`, `scripts/texture_fused_mesh.py` | Supported TSDF surface and source-image texture |
| Website | `viewer/` | Login, original video, 3D viewer, accuracy and downloads |
| Authentication | `viewer/firebase/` | Google sign-in and current visitor profile |
| Recorded evidence | `docs/results.md`, `docs/evaluation.json` | Metrics and their evaluation scope |

## State, latency and failure handling

Jobs and outputs are files in an operator-managed directory. There is no Firestore or SQL database in the current implementation. The API scopes jobs by verified Firebase user ID. It limits uploads and admits a bounded number of jobs to a single inference worker. On restart it marks unfinished runs interrupted rather than claiming completion.

The viewer initially loads the video interface and requests 3D modules and mesh data only when needed. Firebase caches static scene assets. Checkpoints are prepared once, keyframe budgets bound inference, and optional appearance optimization stays outside the default upload path. These are implemented design choices, not a measured production latency or availability guarantee.

## Limits

- A temporary HTTPS tunnel needs a running processing machine. It is not durable cloud GPU hosting.
- The viewer's grid and metric labels do not certify spatial accuracy.
- Single-pass occlusions, low texture and motion can leave holes and inaccurate surfaces.
- The original-video preview is downscaled for playback. It is not evidence of a tested 4K, ten-minute reconstruction.
- See [requirements and verification](requirements.md) before making SIH compliance claims.
