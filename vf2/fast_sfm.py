"""Fast photogrammetric camera solve for quality mode: every k-th neural keyframe, undistorted; SuperPoint + LightGlue
matching on the GPU; COLMAP geometric verification + incremental mapping on the CPU. Writes <out>/sparse/<model>/."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--keyframes-from", required=True, help="prediction report.json or a JSON list of source frames")
ap.add_argument("--calibration"); ap.add_argument("--output", required=True); ap.add_argument("--every", type=int, default=2)
ap.add_argument("--offsets", default="1,2,3,4,5,6,8,10,12,15")
a = ap.parse_args(); os.makedirs(a.output, exist_ok=True); T = {}
t0 = time.time()
src = json.load(open(a.keyframes_from)); kf = [f["source_frame"] for f in src["frames"]] if isinstance(src, dict) else src
kf = kf[::a.every]
from features import decode_keyframes, extract_superpoint, match_pairs_batched
t = time.time(); imgs, (W, H) = decode_keyframes(a.video, np.array(kf), workers=16); T["decode"] = time.time() - t
if a.calibration:
    cal = json.load(open(a.calibration)); K0, D0 = np.array(cal["intrinsic_matrix"]), np.array(cal.get("distortion_coefficients", []))
    if D0.size:
        imgs = np.stack([cv2.undistort(im, K0, D0) for im in imgs])
fdir = os.path.join(a.output, "frames"); os.makedirs(fdir, exist_ok=True)
names = [f"f_{k:05d}.jpg" for k in kf]
for n, im in zip(names, imgs):
    cv2.imwrite(os.path.join(fdir, n), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
T["frames"] = time.time() - t0
# database with image/camera records (tiny SIFT pass only to create them; keypoints are replaced below)
db_path = os.path.join(a.output, "colmap.db")
if os.path.exists(db_path): os.remove(db_path)
t = time.time()
ro = pycolmap.ImageReaderOptions(); ro.camera_model = "SIMPLE_RADIAL"
eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = 32
try: eo.sift.max_num_features = 64
except Exception: pass
pycolmap.extract_features(db_path, fdir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
T["db_records"] = time.time() - t
t = time.time()
gray = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs])
feats, size = extract_superpoint(gray, max_kp=2048, long_side=1280)
offs = [int(x) for x in a.offsets.split(",")]
pairs = [(i, i + o) for i in range(len(names)) for o in offs if i + o < len(names)]
matches = match_pairs_batched(feats, pairs, size)
T["superpoint_lightglue"] = time.time() - t
t = time.time()
db = pycolmap.Database.open(db_path)
ids = {im.name: im.image_id for im in db.read_all_images()}
db.clear_keypoints(); db.clear_descriptors(); db.clear_matches(); db.clear_two_view_geometries()
for n, f in zip(names, feats):
    db.write_keypoints(ids[n], f["keypoints"].float().cpu().numpy().astype(np.float32))
with open(os.path.join(a.output, "pairs.txt"), "w") as fp:
    for (i, j), mm in matches.items():
        db.write_matches(ids[names[i]], ids[names[j]], mm.astype(np.uint32)); fp.write(f"{names[i]} {names[j]}\n")
db.close()
pycolmap.verify_matches(db_path, os.path.join(a.output, "pairs.txt"))
T["write_verify"] = time.time() - t
t = time.time()
o = pycolmap.IncrementalPipelineOptions(); o.num_threads = 96; o.multiple_models = True
recs = pycolmap.incremental_mapping(db_path, fdir, os.path.join(a.output, "sparse"), o)
T["incremental_mapping"] = time.time() - t
T["total"] = time.time() - t0
summary = {"keyframes_solved_from": len(names), "pairs": len(pairs), "matched_pairs": len(matches),
           "models": {str(k): [r.num_reg_images(), r.num_points3D()] for k, r in recs.items()}, "seconds": {k: round(v, 1) for k, v in T.items()}}
json.dump(summary, open(os.path.join(a.output, "fast_sfm.json"), "w"), indent=1)
print(json.dumps(summary))
