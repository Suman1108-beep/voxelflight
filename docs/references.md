# Research, assets and attribution

The system integrates existing research. The team does not claim invention of these models or proof of superiority over them.

## September prototype (Mac / MPS; archived)

- [MapAnything](https://github.com/facebookresearch/map-anything): pretrained multiview geometry. See `requirements-mac.lock.txt` and `setup_mac.py` for code/checkpoint revisions.
- [COLMAP](https://github.com/colmap/colmap): calibrated structure from motion for the September camera anchors.
- [Open3D](https://www.open3d.org/): geometric integration and mesh processing.
- [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM): visual tracking in the earlier GPU experiment.
- [SegFormer](https://arxiv.org/abs/2105.15203): semantic masks in the earlier pipeline, not complete motion estimation.
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and [gsplat](https://github.com/nerfstudio-project/gsplat): the earlier optional appearance stage.
- [Zurich Urban MAV](https://rpg.ifi.uzh.ch/zurichmavdataset.html): original camera sequence, flight information and reference poses. Reference trajectory is not independent surveyed surface ground truth.
- [Three.js](https://github.com/mrdoob/three.js): browser visualization. Upstream licensing applies to the restored browser modules.
- [Firebase](https://firebase.google.com/docs): hosting and authentication.
- [NSUT SIH repository reference](https://github.com/NSUT-SIH-26/NSUT-SIH-DEMO): required submission organization.

## VoxelFlight v2 (A100 pipeline in `vf2/`)

| Component | Role in v2 | Licence of what we use |
|---|---|---|
| [DPVO](https://github.com/princeton-vl/DPVO) | Visual odometry in fast mode | MIT |
| [MapAnything](https://github.com/facebookresearch/map-anything), checkpoint `facebook/map-anything-apache` | Multi-view metric depth (fast and quality modes) | Apache-2.0 code and weights |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), checkpoint `DA3-GIANT-1.1` | Higher-resolution multi-view depth and intrinsics estimation | Apache-2.0 code; **CC BY-NC 4.0 weights** (non-commercial); `DA3-BASE` is Apache-2.0 |
| [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) + [LightGlue](https://github.com/cvg/LightGlue) | Feature matching for the quality-mode camera solve | LightGlue Apache-2.0; **SuperPoint weights non-commercial** (Magic Leap) |
| [COLMAP / pycolmap](https://github.com/colmap/colmap) | Incremental SfM, undistortion | BSD-3-Clause |
| [Open3D](https://www.open3d.org/) | GPU TSDF fusion, meshing, ray casting | MIT |
| [PGSR](https://github.com/zju3dv/PGSR) | Optional planar Gaussian-splatting surface | **Non-commercial research licence** (Inria Gaussian-splatting terms) |
| [xatlas](https://github.com/jpcy/xatlas), [fast-simplification](https://github.com/pyvista/fast-simplification) | UV atlas and decimation for the photo-textured mesh | MIT |

Independent references used only for evaluation, never as pipeline inputs:

| Reference | Used for | Terms |
|---|---|---|
| [swisstopo swissSURFACE3D](https://www.swisstopo.admin.ch/en/height-model-swisssurface3d) | Zurich surface accuracy and coverage | Swiss open government data, attribution required |
| [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) (EPT) | Toledo aerial run | US public domain |
| [BEV ALS DSM](https://data.bev.gv.at) | Austrian runs (helenenschacht RTK, BAMBI 4K) | CC BY 4.0 |

Test flights: [Zurich Urban MAV](https://rpg.ifi.uzh.ch/zurichmavdataset.html) (10-minute run, reference poses); OpenDroneMap sample datasets (Toledo aerial, helenenschacht RTK); [BAMBI](https://zenodo.org/records/18885436) flight 102 (real 4K + DJI SRT, CC BY 4.0); [nominal-io/dji-drone-telemetry](https://huggingface.co/datasets/nominal-io/dji-drone-telemetry) Colorado clip (real 4K HEVC + DJI flight record, MIT). Each keeps its own terms.

The non-commercial weights (DA3-GIANT, SuperPoint, PGSR) suit this hackathon prototype. A deployed or commercial system would swap them for the Apache-licensed alternatives listed above, or license them.

Downloaded models, datasets, fonts and libraries retain their own licenses. Do not treat this repository's public visibility as permission to redistribute or relicense those upstream assets. The original dataset and model weights are not committed. The public source-image previews and screenshots are credited to the Zurich dataset. Pixel-art login decoration is separate from scientific evidence.
