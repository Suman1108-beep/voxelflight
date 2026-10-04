# SIH26158 — measured results and evidence ledger

**Audited:** 3 October 2026. This ledger distinguishes six different quantities: (1) camera-path shape after **post-hoc Sim(3)** alignment to released reference positions; (2) camera-path **direct absolute** error without reference-based alignment; (3) independent **surface** distance; (4) **coverage/completeness**; (5) **appearance** metrics; and (6) wall-clock **runtime**. Only the first two are available for the Zurich flight. None of the camera-path numbers is the SIH ≤1 m surface score.

## Metric definitions and leakage boundary

- **Sim(3)-aligned camera RMSE:** one similarity transform (scale, rotation, translation) chosen with released reference camera positions *after predictions were frozen*. Good for trajectory shape, unusable as a deployed absolute georeference. A 0.3–0.4 m short-flight result says nothing by itself about the mesh surface.
- **SE(3)-aligned/GPS-scale camera RMSE:** rigid rotation/translation alignment without scale correction, still post-hoc if reference-fit. Direct absolute error is different.
- **Direct absolute camera RMSE:** compare GPS/georeferenced predicted cameras to reference positions without reference-based fitting. This reflects onboard GPS bias plus drift and is often several metres here.
- **Surface precision/accuracy:** predicted surface-to-survey distance. **Surface completeness/recall:** fraction of independently surveyed *visible* surface reconstructed within a declared threshold. ETH3D below is only a deterministic nearest-neighbour proxy, not its official evaluator and not SIH UAV certification.
- **Appearance:** PSNR/SSIM/LPIPS compare rendered held-out images. They do not prove metrically accurate or complete 3D.
- **Speed:** report the exact start/stop boundary, hardware, input duration, pre-encoding, number of keyframes, and whether fusion/export are included. A short-video extrapolation is invalid.

## A. Public September saved demo — separate from A100 runs

| Field | Recorded result |
|---|---|
| Source | Zurich Urban MAV continuous frames **61201–63000**, 60 s, 90 selected views. |
| Reconstruction | **4,208,069** predicted points; **646,019-triangle** mesh textured from actual source camera images. |
| Camera path | **0.397 m** post-hoc Sim(3); **0.451 m** post-hoc SE(3); **8.649 m** direct absolute. |
| Surface score / coverage | Independent surface RMSE **not measured**; visible geometry partial, with holes/artifacts. |
| Time | Approximately 6m10s sum of separately timed stages on this **one-minute** sequence; not a 10-minute full stopwatch. |
| Product | Saved result displayed in public Three.js viewer. New uploads do not automatically reproduce this run. |

Evidence: [existing results doc](results.md), [machine-readable evaluation](evaluation.json), [public repository](https://github.com/Suman1108-beep/voxelflight). Note that public GitHub was last observed at 10 September commit `ea7c74e`; later A100 experiments are not automatically included.

The *earlier* L40S 180-view experiment must stay distinct: about **450,000 mesh triangles**, **0.810 m** post-hoc Sim(3), **8.797 m** absolute camera error, optional Gaussian appearance W000 **20.41 dB PSNR / 0.631 SSIM / 0.226 LPIPS**. Do not attribute those image scores or run times to the newer 646k-triangle source-textured mesh.

## B. A100 short-flight development runs

All A100 results below used an A100-SXM4 **40 GB** with pretrained `facebook/map-anything-apache`, unless a row states otherwise. Ground-truth camera references were for post-hoc evaluation, not construction. See [original A100 ledger](../analysis/a100_20260928/EXPERIMENT_RESULTS.md) for detailed ablations.

| Zurich source / run | Views and processing | Post-hoc Sim(3) camera RMSE | Other result | Honest interpretation |
|---|---|---:|---|---|
| `61201–63000`, `90-518-w48-o8` | 90 views, 518 px model side, 48-frame windows / 8 overlap | **0.38792 m** | 0.44406 m GPS-scale rigid-aligned; 8.651 m direct absolute | Best tuned **short-flight camera shape**, not surface ≤1 m. |
| Same subset, 0.04 m supported TSDF | 90 views | Same frozen cameras | 1.686 M triangles; 98.65% internal supported pixels | Internal consistency is not independent accuracy/coverage. |
| `20001–21800`, 180-view SfM-heavy | 180 views | See reports | 1,198.6 s visual SfM alone, 190.1 s inference, 563.4 s fusion; 18.27 M triangles, 10,635 components, largest 32.4% | Fails speed; many triangles do not make a coherent model. |
| `20001–21800`, derived 60 s MP4 v2 | 90 views, 48-frame windows; onboard GPS | **0.304 m** | 120.3 s inference/run, 83.1 s subsequent TSDF, **6.408 m absolute**, 1,564 fused components, largest 60.9% | Good short input proof and export test, not a blind 10-minute or surface pass. |

The 60-second MP4 was **encoded from 1,800 original contiguous 1080p Zurich JPEGs**; it is not an original drone-camera MP4. The v2 fused output received a separate audited onboard-GPS transform and exports GLB/PLY/LAS/GeoTIFF in **EPSG:32632**. A near-zero camera-transform transfer residual checks consistency of two code paths; it does *not* make the model geographically accurate to one metre. Source evidence: [disjoint video evaluation](../analysis/a100_20260928/zurich-disjoint-video60-90w48-evaluation.json) and [georeference report](../analysis/a100_20260928/zurich-disjoint-video60-90w48-fused-georef-report.json).

## C. Decisive 10-minute A100 speed/quality run

**Input:** Zurich frames **40001–58000**, 18,000 original 1920×1080 consecutive JPEGs encoded *before timing* as a **derived 600-second H.264 MP4**; original onboard GPS and calibration; 360 selected frames; MapAnything 48-frame windows/8 overlap; 0.05 m TSDF. This is a real sequential UAV source but not a blind untouched organizer input. No ground-truth references, scene-specific training, or Gaussian appearance fitting were used during processing.

| Timed stage | Wall time |
|---|---:|
| Decode, select, MapAnything inference, original geometry/cache | **386.3 s** |
| 360-view TSDF fusion | **268.0 s** |
| Apply frozen onboard-GPS transform and GIS export | **12.3 s** |
| **Total** | **666.7 s = 11m07s** |

This **passes the supplied <15-minute processing target for this one derived 10-minute flight on this hardware and this baseline**. Peak allocated GPU memory was **9.90 GiB**. The exported GLB reopened; LAS contained 7.54 M points; LAS and GeoTIFF reported EPSG:32632. The mesh had **14.06 M triangles and 3,065 connected components**; the largest held **52.2%** of triangles. That fragmentation is a serious completeness/usability warning.

The **same frozen run** evaluated after the fact at **6.469 m Sim(3)-aligned camera RMSE** and **7.925 m direct absolute camera RMSE**. Neither is the surface score, and neither is sub-metre. There is no independent surveyed Zurich surface reference in the downloaded segment. The important combined conclusion is “**speed pass on one test, geometry/geolocation not verified to required accuracy**,” not “SIH passed.” Evidence: [benchmark report](../analysis/a100_20260928/zurich-10min-benchmark-report.json), [trajectory evaluation](../analysis/a100_20260928/zurich-10min-trajectory-evaluation.json), [fusion report](../analysis/a100_20260928/zurich-10min-fusion-report.json), [GIS report](../analysis/a100_20260928/zurich-10min-georeferenced-report.json).

## D. DPVO long-trajectory attempt and newly completed fusion

The official pretrained DPVO model was run on the *same* derived 10-minute video, using released calibration but **not released reference poses**. Post-hoc shape evaluation:

| Pose source | Frames processed | Sim(3)-aligned camera RMSE |
|---|---:|---:|
| Original 360-keyframe MapAnything windows | 360 selected | **6.469 m** |
| DPVO, every 10th video frame | 1,800 | **4.099 m** |
| DPVO, every 5th video frame | 3,600 | **3.907 m** |
| DPVO5 priors + MapAnything, 360 keyframes | 360 neural views (DPVO processed 3,600) | **3.940510 m**; **4.445232 m direct absolute** |

DPVO meaningfully reduced long-path drift but is still not ≤1 m and by itself has arbitrary visual scale. Conditioning MapAnything on DPVO priors did not beat the DPVO-only camera shape. The conditioned neural inference took **584.4 s**, already consuming most of the 900-second budget. Evidence: [DPVO stride-5 evaluation](../analysis/a100_20260928/dpvo-zurich10m-stride5-evaluation.json) and [conditioned run evaluation](../analysis/a100_20260928/zurich-10min-dpvo5-trajectory-evaluation.json).

**3 October remote audit, new result not in the older local experiment ledger:** A100 file `/workspace/voxelflight_a100_20260928/runs/zurich-10min-dpvo5-mapanything360/fusion-v2/report.json` shows a completed refusion of those 360 cached views at 0.05 m voxels. It took **4,072.394 s (67m52s) for fusion alone**, produced **6,131,006 triangles**, **3,274,291 vertices**, **734 connected components**, and largest-component triangle fraction **60.9%**. It reports `ground_truth_used: false`, `georeferenced: false`, `surface_rmse_m: null`, coordinate system **prediction-local Z-up with north unknown**. Support validation was skipped (`min_other_view_support: 0`), so no cross-view-supported fraction should be claimed. This is a partially less fragmented local surface, **not a valid geographic final export, surface score, or <15-minute pipeline**. The reason for the unusually long fusion time has not yet been isolated; profile before claiming algorithmic speed or regression. The newly written local wrapper is not a measured substitute for this run.

One controlled older baseline shortcut—skipping a support-count calculation when requested minimum support is zero—reduced cached **baseline** 360-view fusion from **265.7 s to 189.6 s** while preserving identical output bytes. That is a **fusion-stage-only** ablation, not the DPVO refusion result and not a new end-to-end time.

## E. Independent surface diagnostic: ETH3D courtyard

An image-only **38-view DSLR scene**, not a drone/GPS flight, was reconstructed and then compared post-hoc against ETH3D laser evaluation scans. Evaluation applied **one global reference-based Sim(3)** to place the result for a deterministic 200k-point nearest-neighbour proxy. Camera shape was **0.629 m RMSE** after that alignment. Do not call this an absolute georeference or official ETH3D benchmark score.

| Geometry output | Predicted-to-scan median | Sampled surface F1 at 0.5 m | Other |
|---|---:|---:|---|
| Raw predicted cloud | **0.414 m** | **0.726** | 0.5 m scan completeness about **90.5%**. |
| Strict 0.05 m TSDF filter | **0.390 m** | **0.625** | Cleaner dominant component, but scan completeness dropped to **59.9%**. |
| Relaxed TSDF support threshold | See JSON | **0.646** | Recover some coverage. |
| No-filter TSDF | **0.407 m** | **0.709** | At 1 m F1 about 0.899 vs raw 0.894. |

At a strict 0.1 m threshold, only **6.6%** of sampled raw-cloud predicted points and **16.7%** of scan points had a counterpart. The scene may overlap the foundation model's unknown pretraining distribution. The key lesson is that a cleaner-looking mesh can lose correct surface: optimize precision **and** completeness under one fixed evaluator. Evidence: [ETH3D raw proxy](../analysis/a100_20260928/eth3d-courtyard38-surface-proxy.json), [strict TSDF proxy](../analysis/a100_20260928/eth3d-courtyard38-tsdf005-surface-proxy.json), [no-filter proxy](../analysis/a100_20260928/eth3d-courtyard38-tsdf005-no-filter-surface-proxy.json), and [evaluator source](../analysis/a100_20260928/evaluate_eth3d_courtyard.py).

## F. Appearance-only experiment

Optional 2D Gaussian Splatting on the first Zurich segment, with every eighth view withheld, reached **24.23 dB PSNR / 0.810 SSIM / 0.190 LPIPS** at 7,000 optimization steps on 12 held-out views. Extending to 15,000 steps reduced PSNR to **23.50 dB**. This branch needs optimization for every new scene; it is useful for presentation and novel-view rendering but does not satisfy metric surface accuracy, scene completeness, or immediate unseen-input processing by itself. Never mix it with a TSDF mesh count or claim that the geometry “looks like” the splat render.

## Current public/operational status (checked 3 October)

- `https://voxelflight-3d.web.app/` returned **HTTP 200** over HTTPS; its `last-modified` header reflected a 10 September release. That is an availability check for the hosted frontend, **not** a successful authenticated job test.
- `viewer/firebase/settings.json` points `apiOrigin` to a temporary `trycloudflare.com` hostname. A DNS lookup/request for that hostname from this Mac failed (`Could not resolve host`). The A100 research path was not observed as the public API worker. Hence new-user upload → A100 result is **unverified/unavailable from this test point**.
- Public-mode upload limit is currently **60 MiB** in [setup.md](setup.md); a 10-minute/4K input may exceed that. There is no durable production queue/worker proven here.
- The public GitHub checkout is clean, but later local A100 changes are not automatically pushed. Preserve current local modifications before release work.

## Never put these claims on a slide without a qualifier

| Tempting claim | Accurate wording |
|---|---|
| “We achieved <1 m accuracy” | “We measured 0.304–0.397 m **post-hoc camera-path RMSE on tuned one-minute segments**; independent single-pass UAV surface ≤1 m remains unverified.” |
| “We passed the SIH speed requirement” | “A baseline A100 processed **one derived ten-minute flight in 11m07s**; the newer DPVO-conditioned fusion failed that speed target. Generalization is untested.” |
| “The website runs the A100 model” | “The site hosts a saved demo; current new-upload worker is a separate Mac preview path and the configured processing ingress failed DNS in this audit.” |
| “We have complete 3D reconstruction” | “We reconstruct observed surfaces; roofs/backsides and parts of visible scene remain missing. Baseline mesh fragmentation is documented.” |
| “A 24 dB render means metric 3D” | “24.23 dB is a held-out **image appearance** measurement for optional scene-specific splatting.” |

Next: [prioritized execution plan](SIH26158_EXECUTION_PLAN_TO_OCT5.md).
