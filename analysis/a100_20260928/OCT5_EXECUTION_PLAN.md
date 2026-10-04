# VoxelFlight: 5 October acceptance gates

This is an engineering checklist, not a promise that the SIH numerical targets
have already been met. The official unseen input and exact scoring protocol may
only become available at evaluation. Reference camera poses and laser scans
must never enter reconstruction.

## Where we stand on 30 September

| Gate | Evidence today | Verdict |
|---|---|---|
| A100 availability | A100-SXM4 40 GB reachable; checkpoint restored under the persistent project directory | Ready for experiments, not a durable public worker |
| New-scene inference | Image-only ETH3D courtyard reconstructed in 73.1 s; a separate derived 60 s drone MP4 plus onboard GPS produced GLB/OBJ/PLY/LAS/GeoTIFF in 120.3 s at 90 views | Short video path works; unknown-file robustness and 10-minute run unproven |
| Camera trajectory | Short video: 0.304 m **post-hoc Sim(3)-aligned**; 10-minute video: 6.469 m aligned, 7.925 m direct absolute | Long-flight drift fails; neither number proves surface accuracy |
| Surface accuracy | ETH3D sampled laser-scan proxy: 0.414 m median raw cloud, 0.390 m median fused cloud after **evaluation-only** camera alignment | First independent geometry diagnostic; not official ETH3D/SIH score |
| Surface coverage | At 0.5 m, raw ETH3D cloud 90.5% sampled scan coverage; strict fused cloud 59.9%; relaxed/no-filter fusion still under raw F1 | Quality/coverage trade-off unresolved |
| Speed | Clean derived 10-minute 1080p flight, 360 selected views: inference 386 s + fusion 268 s + GIS 12 s = **666.7 s** | Under-15-minute processing demonstrated on this one flight; not generalized to all unseen videos |
| Public product | Firebase-hosted saved viewer works; arbitrary new-video GPU processing is not durably connected | Required workflow gap |
| Formats | Ten-minute fused output independently reopened as GLB, LAS and GeoTIFF with EPSG:32632; base output has OBJ/PLY | FBX remains an export gap |

## Non-negotiable tests by the deadline

1. **Unseen video → actual 3D, one command.** Accept MP4/MOV (1080p and a
   bounded 4K case), GPS, timestamped flight metadata, optional calibration.
   The output must include poses, a textured observed-surface mesh or colored
   cloud, provenance, a clear coordinate system, and explicit missing regions.
   Run fully offline once checkpoints are prepared. A single image is not a
   valid substitute for the specified moving-video input.
2. **Independent geometry score.** Freeze outputs before accessing a surveyed
   scan. Report precision/accuracy, completeness/recall, F1 at declared metric
   thresholds and distance quantiles. Use the official evaluator where feasible;
   keep our sampled nearest-neighbour proxy separately labelled. Never replace
   a surface score with trajectory RMSE, GPS fit, or PSNR.
3. **Metric and geographic audit.** Validate camera/GNSS timestamp offset,
   intrinsics, lever arm, altitude datum, UTM zone, and axis conventions.
   Report local metric scale and absolute geolocation separately. Ordinary GPS
   cannot be assumed to certify ≤1 m absolute positioning; request RTK/PPK or
   surveyed anchors if the organizer's metric requires it.
4. **Real 10-minute stopwatch.** On a continuous held-out flight, include video
   decoding, frame selection, poses, neural inference, fusion, texturing and
   export in one wall-clock time. The supplied target is under 15 minutes.
   Optimize the pose stage first; the 180-frame SfM currently exceeds that
   budget alone. Record A100, CPU, VRAM and frame budget.
5. **Website trigger, not a static demo.** An authenticated user uploads a new
   video and metadata, starts a bounded A100 job, sees real queue/progress/errors,
   and opens/downloads only their own finished outputs. Durable job state must
   survive browser refresh and worker restart. The public saved demo stays as
   a clearly labelled example. No temporary tunnel is a submission backend.
6. **Deliverable verification.** Reopen exported GLB/OBJ/PLY/LAS/GeoTIFF in an
   independent viewer or GIS tool; add and validate FBX if the supplied annex
   requires it. Verify road/terrain, visible façades, vegetation, and obstacles
   on the same output, while marking unobserved backsides/rooftops honestly.

## Execution order

| Date | Main work | Stop/go evidence |
|---|---|---|
| 30 Sep | Completed: persistent A100 environment, independent ETH3D surface proxy, short MP4/GPS test, and a 10-minute processing benchmark | Speed passes at 666.7 s; long trajectory drift is 6.469 m aligned / 7.925 m absolute |
| 1 Oct | Test a separately licensed/pretrained fast visual SLAM pose prior, stabilize long-flight windows, retest camera path without reference fitting | Improve held-out long-flight trajectory, not just visual cleanliness |
| 2 Oct | Wire the A100-quality *pretrained inference* path to one new-video job; test unknown video/telemetry parsing; audit GPS/RTK transform | Complete private job from upload to downloadable geometry |
| 3 Oct | Repeat 10-minute processing and camera/surface protocol after pose changes; 4K smoke; blur/shadow/dynamic-object cases | Timestamped runtime, independent geometry report and no hidden scene-specific training |
| 4 Oct | Validate formats, ownership/security, restart recovery, public user flow; freeze weights and code | Clean install/restart works; no unverified metric claims in UI or slides |
| 5 Oct | Independent dry run, capture honest demo evidence, final public-link/access check | Submit only measured results and accessible artifacts |

The priority is geometry evidence and reliable on-demand inference. More
Gaussian-splat iterations, GANs or training a foundation model from scratch are
not substitutes for the missing acceptance tests. Scene-specific appearance
refinement remains optional and must count toward total processing time.
