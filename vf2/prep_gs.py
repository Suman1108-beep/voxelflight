"""Gaussian-splatting surface input (PGSR / 3DGS COLMAP layout) from a quality-mode run: the solved SfM model is locked to
GNSS/barometer exactly as in dense_multi_sfm.py, so the trained surface lands in the same local UTM frame as the pipeline."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pycolmap
from common import robust_sim3
from run_pipeline import load_telemetry
from telemetry import GeoFrame

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--telemetry", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
tel = load_telemetry(a.telemetry); geo = GeoFrame(tel["lat"], tel["lon"], tel["alt"]); gl = geo.to_local(tel["lat"], tel["lon"], tel["alt"])
if tel.get("baro") is not None and np.isfinite(tel["baro"]).all():
    gl[:, 2] = tel["baro"] - np.median(tel["baro"] - tel["alt"]) - geo.origin[2]
fidx = {int(f): i for i, f in enumerate(tel["frame"])}; sig = np.maximum(tel["eph"], 0.5) / 3.0
models = sorted(glob.glob(a.run + "/sfm/sparse/*"), key=lambda d: -pycolmap.Reconstruction(d).num_reg_images())
rec = pycolmap.Reconstruction(models[0])
C, G, S = [], [], []
for im in rec.images.values():
    if im.has_pose:
        i = fidx[int(im.name.split("_")[1].split(".")[0])]; C.append(im.projection_center()); G.append(gl[i]); S.append(sig[i])
s, R, t = robust_sim3(np.array(C), np.array(G), np.array(S))
rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
os.makedirs(a.out + "/locked", exist_ok=True); rec.write(a.out + "/locked")
pycolmap.undistort_images(a.out + "/undist", a.out + "/locked", a.run + "/sfm/frames")   # PINHOLE cameras for splatting
os.makedirs(a.out + "/sparse", exist_ok=True)
if not os.path.exists(a.out + "/sparse/0"): os.rename(a.out + "/undist/sparse", a.out + "/sparse/0")
if not os.path.exists(a.out + "/images"): os.rename(a.out + "/undist/images", a.out + "/images")
cams = {c.model.name for c in pycolmap.Reconstruction(a.out + "/sparse/0").cameras.values()}
json.dump({"source_run": a.run, "model": os.path.basename(models[0]), "images": rec.num_reg_images(), "points": rec.num_points3D(),
           "camera_models": sorted(cams), "sim3_scale": s, "utm_origin": geo.origin.tolist(), "epsg": geo.epsg},
          open(a.out + "/prep.json", "w"), indent=1)
print(open(a.out + "/prep.json").read())
