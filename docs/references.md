# Research, assets and attribution

The system integrates existing research. The team does not claim invention of these models or proof of superiority over them.

- [MapAnything](https://github.com/facebookresearch/map-anything): pretrained multiview geometry. See `requirements-mac.lock.txt` and `setup_mac.py` for code/checkpoint revisions.
- [COLMAP](https://github.com/colmap/colmap): calibrated structure from motion for the latest saved camera anchors.
- [Open3D](https://www.open3d.org/): geometric integration and mesh processing.
- [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM): visual tracking in the earlier GPU experiment.
- [SegFormer](https://arxiv.org/abs/2105.15203): semantic masks in the earlier pipeline, not complete motion estimation.
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and [gsplat](https://github.com/nerfstudio-project/gsplat): the earlier optional appearance stage.
- [Zurich Urban MAV](https://rpg.ifi.uzh.ch/zurichmavdataset.html): original camera sequence, flight information and reference poses. Reference trajectory is not independent surveyed surface ground truth.
- [Three.js](https://github.com/mrdoob/three.js): browser visualization. Upstream licensing applies to the restored browser modules.
- [Firebase](https://firebase.google.com/docs): hosting and authentication.
- [NSUT SIH repository reference](https://github.com/NSUT-SIH-26/NSUT-SIH-DEMO): required submission organization.

Downloaded models, datasets, fonts and libraries retain their own licenses. Do not treat this repository's public visibility as permission to redistribute or relicense those upstream assets. The original dataset and model weights are not committed. The public source-image previews and screenshots are credited to the Zurich dataset. Pixel-art login decoration is separate from scientific evidence.
