# VoxelFlight

Single-pass drone video reconstruction with an inspectable 3D workspace.

**[Live website](https://voxelflight-3d.web.app/)** · **[Presentation](submission/PRESENTATION.md)** · **[Demo](submission/DEMO.md)** · **[Setup](docs/setup.md)** · **[Requirements checklist](docs/requirements.md)**

![Actual reconstructed surface in the VoxelFlight viewer](assets/screenshots/03-reconstructed-surface.png)

*Photo-textured maximum-coverage model of the 10-minute Zurich flight in the live viewer.*

## Results at a glance (4 October 2026, VoxelFlight v2 on one A100)

| SIH26158 target | Measured result (independent references, never pipeline inputs) |
|---|---|
| < 15 min for a 10-min video | **7 min 8 s** fast mode; **14 min 28 s** high-quality mode (Depth Anything 3), each one continuous run to all six formats |
| ≤ 1 m spatial accuracy | Model shape **0.59 m** median (15-min high-quality model) vs swisstopo LiDAR; **0.5 m horizontal / 0.41 m vertical absolute with RTK**; ≈ 2.7 m with consumer GNSS |
| Entire visible scene | **46.5 %** of visible survey points within 1 m on the maximum-coverage model, **51.2 %** with the interpolated ground-level gap-fill layer (scored separately) |
| OBJ, PLY, LAS, GeoTIFF, GLB, FBX | All six from every run, reopened by independent readers; plus a photo-textured mesh |
| Input | 1080p and real DJI 4K (H.264 + `.SRT`, HEVC + flight log), estimated intrinsics when no calibration is given |

Full tables: [`docs/requirements.md`](docs/requirements.md) and [`vf2/RESULTS.md`](vf2/RESULTS.md). Pipeline: [`vf2/README.md`](vf2/README.md).

## 1. Project information

| Field | Project |
|---|---|
| Product | VoxelFlight |
| PS ID | SIH26158 |
| PS title | Single-Pass Drone Video to Accurate 3D Model Generation System |
| Organisation | National Technical Research Organisation |
| Category | Software |
| Theme | Drone / Robotics |
| Registered team | Awaiting confirmation; see [team details](submission/TEAM.md) |

## 2. Problem statement

Conventional 3D mapping often needs multiple drone passes and extensive image overlap. Disaster assessment, infrastructure inspection and rapid mapping may offer only one pass. The challenge asks for georeferenced, metrically accurate terrain and structures from a moving UAV's video, GPS coordinates and flight metadata, with usable meshes or point clouds.

The requested spatial accuracy is **≤1 m**, the speed target is **less than 15 minutes for a 10-minute video**, and coverage should include the entire visible scene. The [requirement-by-requirement audit](docs/requirements.md) states what the current prototype does and does not demonstrate.

## 3. Proposed solution

VoxelFlight combines pretrained multiview geometry, visual camera estimation and available flight telemetry. It exports observed surfaces and keeps the original imagery, 3D result and accuracy evidence accessible in one browser workspace.

The public website opens on a login page. Google sign-in supports account-specific reconstruction jobs; **Explore as a guest** opens the saved demo without an account. New processing uses a separate operator-managed inference worker, not Firebase's hosting servers.

## 4. Key features

- The real 10-minute flight video (4× time-lapse) synchronized with 719 reconstructed views.
- Three.js viewer with camera presets, layers (measured surface, optional interpolated gap fill, flight path) and measurement.
- Photo-textured mesh baked from the source frames; full-resolution coloured mesh and point cloud for download.
- Model/evidence downloads with local and provisional UTM coordinate descriptions.
- Video ingestion (H.264/HEVC, 1080p/4K) and DJI SRT, flight-record, GPX and CSV telemetry.
- Firebase Google sign-in and current-user profile display.
- Authenticated FastAPI jobs, owner-restricted artifacts, bounded queue and cancellation.
- Explicit separation of input imagery, reconstructed geometry and optional appearance renders.

A ruler uses the model's estimated scale; it does not certify measured dimensions. Missing roofs and hidden backsides are not filled in and presented as observations.

## 5. Technology stack

| Layer | Implementation |
|---|---|
| Frontend | JavaScript, HTML/CSS, Three.js, Vite |
| Hosting and identity | Firebase Hosting and Firebase Authentication |
| Backend | Python, FastAPI, token verification, filesystem job records |
| v2 reconstruction (A100) | DPVO, SuperPoint + LightGlue + COLMAP, Depth Anything 3 / MapAnything, Open3D CUDA TSDF, xatlas texture; optional PGSR |
| Default preview model (September) | Pretrained MapAnything on Apple MPS |
| September saved experiment | Calibrated COLMAP anchors, MapAnything, multiview-supported TSDF, source-image texture |
| Earlier GPU experiment | MASt3R-SLAM, MapAnything, SegFormer, telemetry fusion, optional gsplat |
| Geometry / GIS | Open3D, trimesh, PyCOLMAP, pyproj, laspy, rasterio |

## 6. Architecture

[System architecture and data flow](docs/architecture.md) distinguishes the default upload preview from the improved saved experiment. [Security and deployment boundaries](docs/security.md) explains account isolation and processing availability.

## 7. Repository structure

```text
voxelflight/
├── README.md
├── SUBMISSION_GUIDE.md
├── LICENSE
├── requirements.txt
├── submission/
│   ├── PRESENTATION.md
│   ├── DEMO.md
│   ├── TEAM.md
│   ├── VoxelFlight_SIH2026_Presentation.pptx
│   └── VoxelFlight_SIH2026_Presentation.pdf
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   ├── results.md
│   ├── evaluation.json
│   ├── requirements.md
│   ├── security.md
│   └── references.md
├── assets/screenshots/
├── src/                       # Shared ingestion and geometry contracts
├── viewer/                    # Frontend and Firebase integration
├── scripts/                   # Reconstruction and evaluation experiments
├── config/
├── tests/
├── mac_reconstruct.py         # Default preview inference
├── mac_refuse.py              # Supported-surface refusion
├── mac_server.py
├── public_server.py           # Authenticated API entry point
└── public_security.py
```

This preserves the application's normal code folders, as permitted by the [NSUT reference repository](https://github.com/NSUT-SIH-26/NSUT-SIH-DEMO), and uses its required documentation and submission locations.

## 8. Final presentation

[Six-slide PowerPoint and six-page PDF](submission/PRESENTATION.md) are committed in `submission/`. The cover still needs the registered team name and ID. The final deck describes the latest 90-view, source-textured result and the actual model/web stacks, with editable architecture and detailed speaker notes. [Results.md](docs/results.md) keeps the earlier GPU experiment separate.

## 9. Demo

[Demo instructions and video links](submission/DEMO.md). A guest can inspect the saved result without a Google or GitHub account. The real source-camera video is labelled as input, not a reconstruction demo.

## 10. Screenshots

| Login-first entrance | Source-video workspace |
|---|---|
| ![VoxelFlight login](assets/screenshots/01-login.png) | ![Original flight video](assets/screenshots/02-flight-workspace.png) |

See [all screenshots](assets/screenshots/README.md) for the actual mesh and accuracy page. Login artwork is decorative, not scientific evidence.

## 11. Installation

Node.js 22.15+ and pnpm 10:

```sh
git clone https://github.com/Suman1108-beep/voxelflight.git
cd voxelflight/viewer
pnpm install --frozen-lockfile
node scripts/fetch-demo-assets.mjs
pnpm test
pnpm run build:firebase
```

The asset restore downloads approximately 122 MB of public demo files with SHA-256 verification. Model weights, original dataset archives and private uploads are excluded. See [setup.md](docs/setup.md) for Python/model dependencies and platform requirements.

## 12. Run

```sh
# From viewer/
pnpm dlx firebase-tools emulators:start --only hosting --project voxelflight-3d
```

Open `http://127.0.0.1:8140/` for the guest demo. Production sign-in needs a configured Firebase project. See [setup.md](docs/setup.md) to run the separate reconstruction API, use your own project or deploy the frontend.

### Recorded evidence (September prototype, archived; current results are at the top)

| September saved experiment | Result |
|---|---:|
| Selected views | 90 |
| Mesh triangles | 646,019 |
| Absolute trajectory RMSE | 8.649 m |
| Sim(3)-aligned trajectory RMSE | 0.397 m |
| Surface accuracy | Not measured |

Reference poses enter evaluation only. The aligned camera-path score removes global alignment errors and **does not establish ≤1 m surface accuracy**. The [full evaluation protocol](docs/results.md) includes interpolation details and the earlier-run comparison.

## 13. Future scope

- Sub-metre absolute placement without RTK (ground control or map-based registration).
- Coverage beyond the measured 46.5 %: more viewpoints per pass, longer-range depth, better facade recovery.
- Sub-metre geometry for high-altitude 4K footage, which fast mode does not yet reach (≈ 1.7–1.8 m).
- Live uploads on the prepared GPU backend (`vf2/cloud`), and IMU fusion.

## Team and submission

[Team members and roles](submission/TEAM.md) still need confirmation. [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) lists the final checks. Repository publication is separate from submitting the college form.

## Research and licensing

[References and attribution](docs/references.md) credit the models, libraries and Zurich Urban MAV data. Third-party licenses remain in force. No blanket relicensing is implied. [LICENSE](LICENSE) records the current project licensing status.

No credentials, private user jobs, model caches or private Git history are included.

## VoxelFlight v2 (3–4 October 2026): A100 pipeline and measured results

The `vf2/` folder holds the A100 pipeline built after the September submission (`vf2/README.md` explains how to run it).
Every number below is measured against reference data the pipeline never reads (`vf2/RESULTS.md` has the full protocol):

| Target | Result |
|---|---|
| < 15 min for a 10-min video | **7 min 8 s** fast, **14 min 28 s** high quality, end to end on one A100 (video → OBJ, PLY, LAS, GeoTIFF DSM, GLB, FBX) |
| Spatial accuracy, model geometry | 0.55 m (held-out), 0.85 m (aerial), 0.59–0.91 m (10 min, depending on mode) median vs national airborne LiDAR |
| Spatial accuracy, absolute | 0.5 m horizontal / 0.41 m vertical with RTK input; ≈ 2.7–3.5 m with consumer GNSS |
| Visible-scene coverage | 46.5 % of visible survey points within 1 m in maximum-coverage mode, 51.2 % with the gap-fill layer (24 % fast, 37.9 % high quality); 59 % on the Toledo aerial survey |

The public demo at <https://voxelflight-3d.web.app/workspace> shows the 10-minute run in maximum-coverage mode. Live uploads are offline during
judging; a GPU upload backend for the same pipeline is prepared in [`vf2/cloud`](vf2/cloud/README.md).
