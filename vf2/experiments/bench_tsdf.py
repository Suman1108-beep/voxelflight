"""Benchmark GPU TSDF against the cached 360-view baseline predictions (CPU took 268 s)."""
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import open3d as o3d
print("open3d", o3d.__version__, "cuda", o3d.core.cuda.is_available())
from fusion import gpu_tsdf, clean_mesh, mesh_stats
from common import dump
src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/voxelflight_a100_20260928/runs/zurich-10min-360w48o8-v1/inference/predictions"
out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/voxelflight_a100_20260928/vf2/runs/tsdf-bench"
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(src + "/frame_*.npz"))
tic = time.time()
def views():
    for f in files:
        z = np.load(f)
        d = np.where(z["mask"], z["depth"], 0).astype(np.float32)
        yield {"depth": d, "color": z["color"], "K": z["intrinsics"], "c2w": z["pose"]}
mesh, pcd = gpu_tsdf(views(), voxel=0.05)
t_int = time.time() - tic
raw = mesh_stats(mesh)
mesh = clean_mesh(mesh, 500)
t_all = time.time() - tic
st = mesh_stats(mesh)
o3d.io.write_triangle_mesh(out + "/mesh.ply", mesh)
res = {"views": len(files), "integrate_extract_s": t_int, "total_s": t_all, "raw": raw, "clean": st, "points": len(pcd.points)}
dump(out + "/report.json", res); print(res)
