import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, open3d as o3d
from export import export_all
print(subprocess.run("which assimp || (apt-get install -y -q assimp-utils >/dev/null 2>&1 || (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q assimp-utils >/dev/null 2>&1)); which assimp", shell=True, capture_output=True, text=True).stdout)
src = "/workspace/voxelflight_a100_20260928/vf2/runs/tsdf-bench/mesh.ply"
mesh = o3d.io.read_triangle_mesh(src)
mesh = mesh.simplify_vertex_clustering(0.15)  # small test mesh
P = np.asarray(mesh.vertices); C = np.asarray(mesh.vertex_colors)
t = time.time()
meta = export_all("/workspace/voxelflight_a100_20260928/vf2/runs/export-test", mesh, P, C, np.array([465600.0, 5247900.0, 470.0]), 32632)
print("export s", time.time() - t, {k: v["bytes"] for k, v in meta["artifacts"].items()})
print(subprocess.run([sys.executable, os.path.dirname(os.path.abspath(__file__)) + "/verify_exports.py", "/workspace/voxelflight_a100_20260928/vf2/runs/export-test"], capture_output=True, text=True).stdout)
