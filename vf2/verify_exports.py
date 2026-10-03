"""Independently reopen every exported format and report counts/CRS. Usage: verify_exports.py <dir>"""
import json, os, subprocess, sys
import numpy as np
d = sys.argv[1]
res = {}
def ok(name, **kw): res[name] = {"ok": True, **kw}
def bad(name, e): res[name] = {"ok": False, "error": str(e)[:300]}
try:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(f"{d}/model_mesh.ply"); ok("ply_mesh", triangles=len(m.triangles))
    p = o3d.io.read_point_cloud(f"{d}/model_points.ply"); ok("ply_points", points=len(p.points))
except Exception as e: bad("ply", e)
try:
    import trimesh
    g = trimesh.load(f"{d}/model_mesh.glb", force="mesh"); ok("glb", triangles=len(g.faces), vertex_colors=bool(g.visual.kind == "vertex"))
    o = trimesh.load(f"{d}/model_mesh.obj", force="mesh", process=False); ok("obj", triangles=len(o.faces))
except Exception as e: bad("glb/obj", e)
try:
    import laspy
    l = laspy.read(f"{d}/model_points_utm.las"); crs = l.header.parse_crs()
    ok("las", points=int(l.header.point_count), epsg=crs.to_epsg() if crs else None, x_range=[float(l.x.min()), float(l.x.max())])
except Exception as e: bad("las", e)
try:
    import rasterio
    with rasterio.open(f"{d}/dsm_utm.tif") as r:
        a = r.read(1); ok("geotiff", epsg=r.crs.to_epsg(), shape=list(a.shape), res=list(r.res), valid=float(np.isfinite(a).mean()))
except Exception as e: bad("geotiff", e)
r = subprocess.run(["assimp", "info", f"{d}/model_mesh.fbx"], capture_output=True, text=True)
if r.returncode == 0:
    lines = [x for x in r.stdout.splitlines() if any(k in x for k in ("Meshes", "Faces", "Vertices", "Primitive"))]
    ok("fbx_assimp", info=lines)
else:
    bad("fbx_assimp", (r.stdout + r.stderr)[-400:])
print(json.dumps(res, indent=2))
json.dump(res, open(f"{d}/export_verification.json", "w"), indent=2)
