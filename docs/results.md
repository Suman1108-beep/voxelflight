# Recorded results and evidence: September prototype (archived)

> **Archived.** This page records the 10 September Mac/MPS prototype and is kept unchanged for provenance. The current
> pipeline, its measured results (10-minute video in 7 min 8 s / 14 min 12 s, sub-metre model shape against survey
> LiDAR, 0.5 m absolute with RTK, all six export formats) and the model on the website are in
> [`vf2/RESULTS.md`](../vf2/RESULTS.md) and [`docs/requirements.md`](requirements.md).

## September public saved reconstruction

Source: Zurich Urban MAV images 61201-63000, a continuous 60-second camera sequence. The latest experiment selects 90 views, predicts 4,208,069 points and exports a 646,019-triangle source-textured mesh. Construction does not use ground-truth camera poses.

| Metric | Result | Meaning |
|---|---:|---|
| Absolute trajectory RMSE | 8.649 m | Camera-position error in the GPS-referenced frame |
| SE(3)-aligned trajectory RMSE | 0.451 m | Error after rigid reference alignment |
| Sim(3)-aligned trajectory RMSE | 0.397 m | Error after reference alignment of rotation, translation and scale |
| Surface RMSE | Not measured | Requires an independent surface reference |
| Coverage | Partial | Holes, missing surfaces and fusion artifacts remain |

The [machine-readable evaluation](evaluation.json) retains the full precision and protocol. There are 3 exact and 87 interpolated reference matches, with source-frame gaps no greater than 30. An aligned camera-path score cannot demonstrate absolute map accuracy or surface accuracy. None of these results establishes SIH compliance or a state-of-the-art benchmark result.

The ~6 minute 10 second sum of separately timed stages covers a one-minute sequence. It is not one measured end-to-end run including upload/queue time, and it does not satisfy the requested ten-minute-video speed benchmark by extrapolation.

## Earlier GPU experiment, retained separately

The **earlier 180-keyframe NVIDIA L40S experiment** remains here for provenance. The final six-slide presentation uses the latest 90-view saved experiment. The two runs have different outputs and processing paths:

| Property | Earlier GPU experiment | Current saved experiment |
|---|---:|---:|
| Selected views | 180 | 90 |
| Mesh triangles | 450,000 | 646,019 |
| Sim(3)-aligned trajectory RMSE | 0.810 m | 0.397 m |
| Absolute trajectory RMSE | 8.797 m | 8.649 m |
| Texture/appearance | Colored geometry plus optional 3DGS | Actual camera-image texture atlas |

Earlier timings are 3 min 29 s for core reconstruction and 5 min 34 s including optional Gaussian appearance fitting. They are not timings for the newer TSDF/textured model. Earlier W000 appearance scores are PSNR 20.41 dB, SSIM 0.631 and LPIPS 0.226. These belong only to the archived Gaussian-render comparison.

## Input, rendering and geometry

- `02-flight-workspace.png` shows a source-camera video, not a generated output.
- `03-reconstructed-surface.png` shows the actual exported, source-textured model.
- The login page artwork is decorative pixel art, not reconstruction evidence.
- An optional appearance render may look more complete than the supported mesh. It does not certify a measured surface.

## Reproduction boundary

The repository ships inference code and evaluation scripts, not the original full dataset or pretrained weights. Model preparation and dataset acquisition require network access. Cached core inference can run offline. The Firebase login/public API requires internet access and is not part of an offline evaluation claim.

The higher-quality saved SfM-assisted experiment is separate from the default public upload preview. A reviewer should not expect a new upload to reproduce its exact quality automatically.
