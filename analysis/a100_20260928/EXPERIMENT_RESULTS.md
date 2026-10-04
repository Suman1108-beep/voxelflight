# A100 reconstruction experiments — 28 September 2026

This is a reproducible experiment ledger, not a claim of SIH surface-accuracy
certification. The official Zurich Urban MAV camera-position reference is used
only for post-hoc evaluation. It is not a LiDAR or 3D surface reference.

## Data and machine

- A100-SXM4 40 GB. CPU has limited concurrency, so SfM is often the bottleneck.
- Official Zurich Urban MAV frames 61201–63000 and a disjoint segment
  20001–21800 (1,800 full-resolution frames each, 1920 × 1080).
- 90- and 180-view uniformly sampled experiments from the first segment.
- Camera calibration/poses: PyCOLMAP; learned geometry: pinned MapAnything
  `facebook/map-anything-apache` checkpoint; fusion: observed-view-supported
  TSDF. Optional scene-specific 2D Gaussian Splatting is being evaluated.

## First-segment results

| Run | Views | Input side | Window / overlap | Sim(3)-aligned camera-path RMSE | GPS-scale, rotation/translation-aligned camera-path RMSE | Direct absolute georeference RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline90-420-w24-o2 | 90 | 420 | 24 / 2 | 0.39742 m | 2.5586 m | — |
| 90-420-w24-o6 | 90 | 420 | 24 / 6 | 0.39719 m | 0.45112 m | 8.65 m |
| 90-518-w24-o6 | 90 | 518 | 24 / 6 | 0.39391 m | 0.44827 m | 8.65 m |
| 90-518-w48-o8 | 90 | 518 | 48 / 8 | **0.38792 m** | **0.44406 m** | 8.651 m |
| calibrated-sfm180-v2 | 180 | original | global BA | 0.39518 m | 0.46278 m | 8.665 m |
| 180-518-w24-o6 | 180 | 518 | 24 / 6 | 0.39332 m | 0.46124 m | 8.665 m |

The 90-view, 48-frame-window setting currently has the best camera-path
shape. The absolute score remains poor because the source GPS is biased;
Sim(3) alignment is an evaluation operation and is not available at deployment.
No independent surface GT has been used, so none of these camera metrics proves
sub-metre 3D surface accuracy.

## Observed-surface fusion

| Run | Voxel | Valid depth pixels | Internal multiview-supported fraction | Mesh triangles |
|---|---:|---:|---:|---:|
| baseline90-refuse-006 | 0.06 m | 5.71 M | 96.0% | 1.217 M |
| 90-518-w48-refuse-004 | 0.04 m | 10.26 M | 98.65% | 1.686 M |
| 180-518-refuse-004 | 0.04 m | 20.58 M | 99.45% | 2.942 M |

The supported fraction is internal multiview consistency, not reconstruction
accuracy. It can increase simply by adding duplicate/nearby views. All meshes
are partial, observed surfaces. Unseen backsides and rooftops are not inferred
as if measured.

## Disjoint Zurich segment and speed experiment (30 September)

The second Zurich segment (source images 20001–21800) was processed separately
with 180 selected views. Its visual SfM registered all 180 images, but took
**1198.6 s**; model inference took **190.1 s** and full 0.04 m TSDF refusion
took **563.4 s**. Those stages alone exceed the supplied 15-minute target. The
180-view mesh has 18.27 M triangles and 10,635 connected components; its largest
component is only 32.4% of triangles. High internal support (98.4%) did not
make it visually coherent or certify surface accuracy.

A controlled fast-fusion variant reused the **same frozen predictions** and
selected every second view, with an 8-selected-view temporal support radius.
Fusion decreased from 563.4 to **208.6 s** (2.70× faster). It still has 15.28 M
triangles and 8,809 components, so it is a speed improvement, not a winning
geometry result. The full reference stays intact. See
`unseen20001-90-stride2-neighbor8-refusion-report.json`.

## Scene-specific appearance experiment

2D Gaussian Splatting was optimized on the first segment with every eighth
view withheld. The 7,000-step checkpoint reached **24.23 dB PSNR, 0.810 SSIM,
0.190 LPIPS** on 12 held-out views. This measures image appearance, not mesh
accuracy. At 15,000 steps held-out PSNR degraded to 23.50 dB, so longer
optimization was not automatically better. Every unseen scene would require its
own optimization; the splat branch is not the default immediate-upload path.

## Actual video + onboard-GPS path (30 September)

The second Zurich segment was separately encoded from **1,800 contiguous
official drone JPEGs into a 60-second, 1080p H.264 MP4**. This is a derived
constant-frame-rate video, not an original camera MP4. Its original onboard GPS
and released intrinsics were passed to the same frozen MapAnything inference
path; the GroundTruthAGL camera reference was opened only after each output was
complete. The output includes GLB/OBJ/PLY, UTM LAS and a GeoTIFF occupied-cell
surface. No new model training or scene-specific 3DGS was used.

| Video run | Frames / window | GPU-run wall time | Eval-only Sim(3) camera RMSE | Direct absolute camera RMSE | 0.05 m no-filter TSDF time | Largest fused component |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 180 / 24 | 221.9 s | 0.348 m | 6.331 m | 89.1 s after inference | 36.1% |
| v2 | 90 / 48 | **120.3 s** | **0.304 m** | 6.408 m | 83.1 s after inference | **60.9%** |

V2 is the better *speed and camera-shape* candidate and has a more coherent
dominant fused surface on this segment. It does not solve absolute position or
surface accuracy, and its fused mesh still has 1,564 connected components.
The fused geometry initially remained in prediction-local coordinates; a
separate audited export now transfers the frozen onboard-GPS transform to the
V2 fused mesh (maximum pre/post camera-pose residual **0.000003 m**) and writes
GLB, PLY, UTM LAS and GeoTIFF. The GLB reopened, and the LAS/GeoTIFF both report
EPSG:32632. This fixes a deliverable-coordinate gap, **not** the 6.408 m
absolute position error. The 0.304 m result is
post-hoc trajectory alignment, **not** the SIH ≤1 m surface test. This segment
has already been used for parameter choice, so the pending third segment must
serve as the next clean speed/quality check.

A frame-for-frame decoder benchmark on the same derived 60-second H.264 input
identified a major avoidable cost: 180 repeated OpenCV random seeks took
**51.20 s**, while one-pass `grab`/selected-frame `retrieve` took **3.33 s**;
all **180 decoded images matched byte-for-byte**. The one-pass path is now in
`mac_reconstruct.py` with selection tests. The inference wall times above were
measured **before** this change, so do not subtract 48 seconds and present it
as an end-to-end measurement; the clean ten-minute run will measure that.

## Ten-minute continuous-video gate (30 September)

An additional Zurich segment, frames **40001–58000**, was downloaded from the
official archive with CRC and SHA-256 verification. Its 18,000 contiguous
1080p drone frames and onboard GPS were packaged as a **derived 600-second
H.264 MP4**. Preparation/encoding from released JPEGs happened before the
stopwatch, as an actual submitted input would already be MP4/MOV. The full
video-to-fused-GIS pipeline then ran continuously and offline on the A100 with
360 selected frames, 48-frame inference windows, 8-frame overlaps, and a 5 cm
TSDF voxel. No GT, scene-specific training or 3DGS was used in processing.

| Stage | Measured wall time |
|---|---:|
| Decode + MapAnything + original geometry/cache/export | 386.3 s |
| 360-view TSDF fusion | 268.0 s |
| Transfer frozen onboard-GPS transform + GLB/PLY/LAS/GeoTIFF export | 12.3 s |
| **Total** | **666.7 s = 11 min 7 s** |

This passes the supplied **under-15-minute processing target for this derived
10-minute 1080p flight**. Peak allocated GPU memory was 9.90 GiB. The fused
mesh has 14.06 M triangles and 3,065 components; the largest contains 52.2%
of triangles. The exported GLB reopened, LAS has 7.54 M points, and LAS and
GeoTIFF both report EPSG:32632. The camera/GPS transform transfer residual
was under 0.00005 m, but this merely verifies coordinate consistency.

The **frozen-output, post-hoc** camera evaluation is the decisive warning:
6.469 m RMSE after reference-based Sim(3) alignment and **7.925 m direct
absolute** RMSE. Neither is a surface score, and neither passes a sub-metre
trajectory criterion. No surveyed surface exists in this Zurich release, so
sub-metre mesh accuracy is still unverified. Long-range camera drift is now
the principal modeling problem; adding mesh triangles or a prettier renderer
would not solve it. The speed pass and the accuracy failure must be reported
together.

## Visual-odometry experiment (2 October)

The official pretrained DPVO checkpoint was installed in a separate A100
environment and run on the **same derived 10-minute video**, using released
camera calibration but no reference poses. Its output was frozen before the
released camera-position reference was loaded for post-hoc evaluation.

| Pose source | Processed video frames | Post-hoc Sim(3) camera RMSE |
|---|---:|---:|
| MapAnything windows, prior baseline | 360 selected | 6.469 m |
| DPVO, every 10th frame | 1,800 | 4.099 m |
| DPVO, every 5th frame | 3,600 | **3.907 m** |

DPVO improves the *trajectory shape* but does not pass a sub-metre criterion.
It has arbitrary visual scale and no independent absolute georeference. A new
MapAnything run conditioned on the frozen every-5th-frame DPVO poses is being
measured separately; its score must not be inferred from the DPVO-only result.
The visual pose files explicitly declare no ground truth and no metric scale.

The 360-view 5 cm TSDF pass was also repeated on cached baseline predictions
with an implementation shortcut: when the requested minimum cross-view support
is zero, skip the unused support-count calculation. The output still has
14,061,861 triangles and 3,065 connected components; fusion time decreased
from **265.7 s to 189.6 s**. This is a controlled fusion-stage result, not a
new end-to-end stopwatch or an accuracy improvement. The 195 MB PLY cloud and
276 MB GLB mesh are byte-for-byte SHA-256 identical to the original outputs.

## Independent surface diagnostic: ETH3D courtyard

An image-only, 38-view MapAnything run on ETH3D courtyard took **73.1 s** and
created 2.54 M triangles. Camera and laser-scan references were withheld until
after inference. A single global **evaluation-only Sim(3) camera alignment** gave
0.629 m camera RMSE. In a deterministic 200k-point nearest-neighbour proxy
against the official laser evaluation scans, median reconstructed-to-scan
distance was **0.414 m**. At 1 m, 83.4% of sampled reconstructed points and
96.3% of sampled scan points had a neighbour within the threshold; at 0.1 m,
the fractions were only 6.6% and 16.7%. This is **not the official ETH3D
free-space-aware score**, not an absolute georeference and not SIH drone/GPS
compliance. The DSLR scene is not a single-pass flight. The run/evaluation
reports and evaluator script are saved beside this ledger. Potential overlap
with the pretrained model's original training corpus has not been ruled out,
so this is not a strict held-out foundation-model benchmark.

Re-fusing those same predictions at 0.05 m took another **31.8 s** and yielded
711,523 triangles, 56 connected components, and 96.3% of triangles in the
largest component. Median reconstructed-to-scan distance improved slightly
from 0.414 m to **0.390 m**, but the stricter multiview filter removed valid
surface: scan completeness at 0.5 m fell from **90.5% to 59.9%**, and the 0.5 m
F1 proxy fell from 0.726 to **0.625**. A cleaner-looking mesh is therefore not
automatically a better reconstruction. The next geometry experiment should
improve coverage without reintroducing unsupported floating fragments.

The two follow-up ablations reused exactly the same predictions, camera poses,
scan and evaluation alignment. Relaxing depth agreement to 3% / 0.15 m raised
the 0.5 m F1 proxy to **0.646**. Disabling multiview filtering entirely raised
it to **0.709** (median 0.407 m), still below the original unfiltered point
cloud's **0.726**. At the looser 1 m threshold, the no-filter TSDF narrowly
exceeded the raw cloud (0.899 versus 0.894 F1). The correct default is therefore
not established by visual cleanliness alone. The four score reports are saved
as `eth3d-courtyard38*-surface-proxy.json`.

## Still unverified

- Independent surface accuracy on a real single-pass drone flight with a
  surveyed reference and the organizer's exact evaluation protocol.
- Absolute georeferencing to ≤1 m; Zurich onboard GPS has metre-scale bias.
- Entire-visible-scene completeness. Occluded surfaces remain unobserved.
- Generalizing the measured under-15-minute run to an unseen original MP4/MOV,
  especially with an added visual-odometry stage. The fast refusion alone does
  not establish the combined runtime.
- Unseen-upload execution from the public website using the same A100-quality
  pipeline, with durable jobs and restart-safe checkpoints.
