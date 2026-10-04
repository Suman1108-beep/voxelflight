# SIH26158 / VoxelFlight — project handoff

**Status date:** 3 October 2026 (Asia/Kolkata)  
**Deadline used for planning:** 5 October 2026, as reported by the team; confirm the organizer's exact submission time and time zone.  
**Project:** Single-Pass Drone Video to Accurate 3D Model Generation System  
**Problem owner / category:** National Technical Research Organisation (NTRO) / Software / Drone & Robotics  
**Product name:** VoxelFlight  
**Evidence standard:** A result is called *verified* here only for the specific input, code, hardware, and metric actually measured. This is not an organizer score or a claim that all SIH requirements are met.

## Read these four files in order

1. **This file:** exact problem, system, requirements-to-evidence matrix, and the honest current verdict.
2. [Datasets, models, and research](SIH26158_DATASETS_MODELS_RESEARCH.md): what data/checkpoints are present versus merely proposed; provenance and research directions.
3. [Results and evidence ledger](SIH26158_RESULTS_EVIDENCE_LEDGER.md): every important run with its own metric protocol and artifact paths.
4. [Execution and acceptance plan](SIH26158_EXECUTION_PLAN_TO_OCT5.md): prioritized work, pass/fail gates, release plan, and definition of done.
5. [Website code and loading flow](WEBSITE_CODE_AND_LOADING_2026-10-03.md): public GitHub versus current local source, route-by-route files, lazy 3D loading, Firebase sign-in, and the separate upload API.

Existing source documents are [requirements.md](requirements.md), [architecture.md](architecture.md), [results.md](results.md), the [A100 experiment ledger](../analysis/a100_20260928/EXPERIMENT_RESULTS.md), and the [previous sprint plan](../analysis/a100_20260928/OCT5_EXECUTION_PLAN.md). The older documents are historical and some of their “not yet benchmarked” statements predate the 10-minute A100 run. The supplied problem-statement extract is the local PDF at `/Users/booshi1234/Documents/Codex/2026-08-27/rea/tmp/pdfs/SIH26158.pdf`, printed pages 37–39, “Problem Statement - 17.” The first line of the PDF is not a project ID; **SIH26158** is the ID recorded in the team repository/presentation, not independently reconfirmed against a current organizer portal in this audit.

## One-paragraph truth

VoxelFlight already has a public saved demonstration, real source-frame imagery, a browser 3D viewer, pretrained-model inference code, actual meshes/point clouds, a GPS-referenced export path, and A100 experiments. A **derived** 10-minute Zurich flight was processed offline in **666.7 seconds** on one A100 run, beating the supplied <15-minute speed target *for that run*. However, its camera trajectory was **6.469 m after evaluation-only Sim(3) alignment and 7.925 m in direct absolute coordinates**; its mesh is fragmented and has **no independent surveyed surface score**. A later DPVO-conditioned run improved trajectory shape to **3.9405 m Sim(3)** but its completed fusion alone took **4072.4 seconds** and was not georeferenced. The public upload path is a different Mac preview worker, not this A100 pipeline. The official unseen dataset has not arrived. Therefore the project is a meaningful prototype, **not a verified ≤1 m, complete, production-ready solution**.

## What the organizer actually asks for

From the supplied statement, the system receives **one moving UAV pass**, not an arbitrary single still image. Mandatory inputs are a **1080p/4K drone video**, **GPS coordinates**, and **flight metadata**. IMU, barometric altitude, camera intrinsics, and RTK/PPK corrections are optional. It should reconstruct visible terrain/structures, building façades and roofs, roads/infrastructure, vegetation/obstacles, and output a textured 3D mesh or point cloud usable for visualization, measurement, and analysis. Key difficulties are limited views, blur/compression, varying light, moving objects, GPS/sensor error, near-real-time operation, occlusion, and metric scale without extensive ground-control points.

The annex's desired-output table adds the following concrete acceptance targets. The wording “spatial accuracy” is **not accompanied by a precise alignment/evaluator definition**; ask whether it means local shape, absolute geolocation, surface-to-survey distance, or all three. Do not substitute any one for another.

| Desired output | Supplied target | Interpretation for our acceptance testing |
|---|---|---|
| Geometry | 3D mesh or point cloud | Export an actual inspectable model, not a generated image or only camera poses. |
| Speed | **<15 min** for a **10-minute** video | Time the full video-to-final-export workflow; record hardware and preprocessing boundary. |
| Spatial accuracy | **≤1 m** | Needs independent metric reference and the organizer's alignment rules. A camera-path RMSE is not mesh accuracy. |
| Coverage | Entire visible scene | Measure observed-region coverage; do not fabricate occluded backsides. |
| Interoperability | OBJ, PLY, LAS, GeoTIFF, GLB/glTF, FBX | Verify each requested format independently, including CRS/units where applicable. |
| Inspection | Web or desktop viewer | Allow real orbit/inspection, scale-aware measurement, and provenance. |

The supplied scoring rubric is **accuracy 30%, completeness 20%, processing speed 20%, innovation 15%, scalability 10%, UI 5%**. These are category *weights*, not a VoxelFlight score. Accuracy plus completeness account for **50%**, so visual polish cannot compensate for missing geometry validation. The source says an evaluation dataset will be provided “real time”; no verified organizer dataset or scoring script is present in this workspace as of this audit.

## The actual system, with three different paths

```text
                         PUBLIC PRODUCT (September release)
User -> Firebase-hosted site -> guest saved demo / Google sign-in
                                   |                   |
                                   |                   +-> authenticated upload API
                                   |                        -> bounded Mac MPS preview worker
                                   |                        -> temporary HTTPS processing address
                                   +-> source video, timeline, Three.js model,
                                        layers, measurements, saved exports

                       RESEARCH / HIGH-QUALITY PATH (A100)
Prepared MP4 + GPS/metadata + optional intrinsics
    -> decode and choose keyframes
    -> optional DPVO visual odometry (pretrained)
    -> overlapping MapAnything inference (pretrained transformer-based geometry)
    -> window alignment, scale/GPS transform, consistency filtering
    -> TSDF fusion -> local mesh/point cloud -> optional UTM/LAS/GeoTIFF exports
    -> separate, post-hoc camera or surface evaluation

                       EARLIER RESEARCH PATH (archival)
MASt3R-SLAM / MapAnything / segmentation / telemetry fusion
    -> separate geometry and optional 2D/3D Gaussian appearance refinement
```

**Important separation:** The public saved demo, the Mac upload preview, and the newer A100 experiments are not the same run. A reviewer uploading a new video today should **not** expect the A100 result. Firebase hosts/authenticates the interface; it does **not** run the 3D model itself. Pretrained weights are used for per-scene inference; a scene-specific Gaussian appearance optimization is optional and is not a prerequisite for the core pipeline. Calling the entire system “a 3D CNN trained from scratch” would be inaccurate. The main learned geometry is transformer-based; convolutional components may exist inside backbones/embeddings, but the system is a hybrid inference + geometry + optimization pipeline.

## Requirement-by-requirement audit

Legend: **Demonstrated** = an artifact or measured run exists; **Partial** = some code/output exists but exact acceptance is not proved; **Open** = missing or unverified. The table is not a percentage score.

| Requirement | Status on 3 Oct | Evidence and limitation | Next acceptance proof |
|---|---|---|---|
| Single-pass moving-UAV input | **Partial** | Real consecutive Zurich MAV frames; 60 s and 600 s H.264 MP4s were derived from official sequential JPEGs, not original camera MP4s. | Unseen organizer/original MP4/MOV, no reference poses at inference. |
| Video + GPS + flight metadata | **Partial** | Zurich onboard GPS and calibration/telemetry accepted in tested paths. Arbitrary vendor formats not exhaustively handled. | Input-contract tests covering timestamps, coordinate systems, malformed files. |
| 1080p | **Demonstrated on derived input** | 1920×1080 Zurich source used in A100 benchmark. Neural keyframes were resized for inference. | Record original size and internal resize explicitly. |
| 4K | **Open** | No verified end-to-end 4K input test. | Bounded 4K smoke and memory/runtime report. |
| Terrain and roads | **Partial** | Some observed road/ground context in Zurich meshes; no class-specific score. | Annotated visual/surface coverage on independent flight. |
| Façades and roofs | **Partial** | Visible façades appear; roofs and backsides may be absent due to the single path. | Separate visible-facade/roof coverage, no hallucinated “measured” occlusions. |
| Vegetation and obstacles | **Partial** | Visible foliage/obstacles are represented inconsistently; no quantified semantic quality. | Object-class inspection and dynamic-object exclusion test. |
| Textured mesh or point cloud | **Demonstrated, run-dependent** | September saved 646,019-triangle mesh has a real source-image texture atlas; A100 TSDF outputs are vertex-coloured/geometry outputs, not necessarily UV-textured. | Identify which output is submitted and verify material/texture in another viewer. |
| Geographic reference / metric scale | **Partial** | UTM EPSG:32632 LAS/GeoTIFF exports reopen; GPS-relative transform transfer consistency measured. Ordinary GPS bias leaves multi-metre direct error. | Independent surveyed anchor/RTK if absolute ≤1 m is required. |
| ≤1 m spatial accuracy | **Open** | No independent single-pass UAV mesh surface RMSE ≤1 m. Post-hoc camera trajectory scores are different. | Organizer or held-out UAV LiDAR/survey surface test, exact protocol. |
| Entire visible scene | **Open** | A100 10-minute baseline mesh has 3,065 components; largest has 52.2% of triangles. DPVO refusion still 734 components. | Coverage mask + surface recall, connectedness/failure examples. |
| <15 min for 10-minute video | **Partial** | One baseline A100 derived flight: 666.7 s end-to-end by its recorded boundary; newer DPVO refusion: 4072.4 s fusion alone. | Repeat on frozen improved pipeline and unseen input, include all stages. |
| OBJ / PLY / LAS / GeoTIFF / GLB | **Partial** | Export paths and selected independent reopen checks exist. Formats vary by run; GeoTIFF is an occupied-cell raster, not a surveyed terrain DEM. | Final output matrix and automated reopen test for every claimed format. |
| FBX | **Open** | No demonstrated FBX export. | Generate and independently import with units/material audit. |
| Web/desktop viewer | **Demonstrated for saved demo** | Firebase Hosting responds with HTTP 200; guest source/video/mesh workspace exists in code and prior release. Current browser flow not fully re-audited in this turn. | Manual desktop/mobile end-to-end check, export and measurement tests. |
| New upload -> model -> result | **Open for A100 quality** | Mac MPS preview service exists, but its configured temporary `trycloudflare.com` processing hostname did not resolve from this Mac on 3 Oct. A100 pipeline not deployed behind the site. | Authenticated fresh upload, durable worker, produced artifacts, owner-only retrieval. |
| Offline reconstruction | **Partial** | Prepared A100 checkpoint runs offline; online Firebase auth/upload is separate. | Fresh environment/cached-weights run with network disconnected. |
| Optional sensors | **Partial** | Some calibration/GPS handling; optional IMU/barometer/RTK inputs are not all fused/tested. | Unit and integration tests per claimed modality. |

## Current artifacts and ownership

| Artifact | Location | State |
|---|---|
| Working project | Repository workspace root `work/sih3d-gpu/` | Local research/source directory; not itself one Git repository. |
| Public submission checkout | `deliverables/github-submission-20260910/` | Separate clean Git checkout, `origin` = [Suman1108-beep/voxelflight](https://github.com/Suman1108-beep/voxelflight), last observed commit `ea7c74e` dated 10 Sep. It does **not** automatically contain later A100 experiments. |
| Website | [voxelflight-3d.web.app](https://voxelflight-3d.web.app/) | Root returned HTTP 200 on 3 Oct. Functional auth/job test still needed. |
| Viewer source | `viewer/` | Separate local checkout with working-tree changes; preserve and review before any release. |
| Local Zurich data | `datasets/zurich-61201-63000/` | 1,800 original images plus metadata, verified manifest. |
| A100 workspace | `/workspace/voxelflight_a100_20260928` on the connected A100 | Three Zurich segments, ETH3D courtyard, checkpoints, run outputs observed read-only on 3 Oct. This is not permanent cloud hosting. |
| New wrapper | `scripts/run_a100_pipeline.py` | Local offline orchestrator code; not verified end-to-end on remote and not the public upload worker. |
| Presentation | September submission checkout `deliverables/` | Historical six-slide deck/README; metric claims must be updated from this ledger before reuse. |

## What “done” means for this project

There is no defensible single “percent complete” because **the hardest criteria are unverified**. The website and exports are much further along than surface accuracy, coverage, and unseen-video reliability. A defensible final claim is: “We built a real single-pass video-to-3D prototype and an inspectable public demo; we measured a sub-15-minute A100 run on one 10-minute derived flight, and we transparently report the remaining geometry and georeference limitations.” Do **not** claim “≤1 m SIH achieved,” “all visible geometry complete,” “fully cloud-hosted model,” or “SOTA” without the corresponding independent test.

## Source and evidence rules for the team

- **Never feed Zurich GroundTruthAGL camera poses, ETH3D laser scans, or organizer reference surfaces into reconstruction.** Open them only after freezing predictions for evaluation.
- **Never interchange** camera-path RMSE, direct absolute geolocation RMSE, surface-distance metrics, completeness/recall, and PSNR/SSIM/LPIPS. They answer different questions.
- **Always pair a number with run ID, dataset segment, input duration, hardware, alignment, and whether it is a held-out or parameter-tuned scene.** The 60-second `0.304 m` score is not a 10-minute or surface score.
- A beautiful Gaussian render proves appearance from that camera, not a complete, metric mesh. A high triangle count proves file size, not accuracy.
- Do not publish private bridge credentials, API tokens, user uploads, or unlicensed model/data assets in GitHub. The existing temporary ingress is not durable service infrastructure.
- Cite the supplied PDF separately from research papers; organizer requirements and our implementation results are different sources of truth.
