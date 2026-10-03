"""Write OBJ / PLY / GLB / FBX (local metric frame) and LAS / GeoTIFF DSM (UTM) from one fused model."""
import hashlib
import json
import os

import numpy as np


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_obj(path, V, F, C):
    c = (C if C.max() <= 1.0 else C / 255.0)
    with open(path, "w") as f:
        f.write("# VoxelFlight mesh: local metric frame (x=east, y=north, z=up), see metadata.json for UTM origin\n")
        np.savetxt(f, np.column_stack([V, c]), fmt="v %.4f %.4f %.4f %.4f %.4f %.4f")
        np.savetxt(f, F + 1, fmt="f %d %d %d")


def write_glb(path, V, F, C):
    import trimesh
    # glTF is Y-up: (x, y, z)_ENU -> (x, z, -y)
    Vg = np.column_stack([V[:, 0], V[:, 2], -V[:, 1]])
    c8 = (C * 255 if C.max() <= 1.0 else C).astype(np.uint8)
    m = trimesh.Trimesh(Vg, F, vertex_colors=np.column_stack([c8, np.full(len(c8), 255, np.uint8)]), process=False)
    m.export(path)


def write_las(path, P, C, origin, epsg):
    import laspy
    from pyproj import CRS
    hdr = laspy.LasHeader(point_format=2, version="1.4")
    hdr.offsets = origin
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.add_crs(CRS.from_epsg(epsg))
    las = laspy.LasData(hdr)
    W = P + origin
    las.x, las.y, las.z = W[:, 0], W[:, 1], W[:, 2]
    c16 = ((C if C.max() <= 1.0 else C / 255.0) * 65535).astype(np.uint16)
    las.red, las.green, las.blue = c16[:, 0], c16[:, 1], c16[:, 2]
    las.write(path)


def write_dsm(path, P, origin, epsg, res=0.25, fill_iters=3):
    """Digital surface model: highest observed surface per cell (UTM, metres). NaN where unobserved."""
    import rasterio
    from rasterio.transform import from_origin
    from scipy import ndimage
    W = P + origin
    x0, y1 = np.floor(W[:, 0].min()), np.ceil(W[:, 1].max())
    nx = int(np.ceil((W[:, 0].max() - x0) / res)) + 1
    ny = int(np.ceil((y1 - W[:, 1].min()) / res)) + 1
    ix = ((W[:, 0] - x0) / res).astype(int); iy = ((y1 - W[:, 1]) / res).astype(int)
    dsm = np.full(ny * nx, -np.inf, np.float32)
    np.maximum.at(dsm, iy * nx + ix, W[:, 2].astype(np.float32))
    dsm = dsm.reshape(ny, nx)
    valid = np.isfinite(dsm)
    observed = valid.copy()
    # fill only 1-3 cell pinholes (never large unobserved areas)
    for _ in range(fill_iters):
        filled = ndimage.grey_dilation(np.where(valid, dsm, -np.inf), size=3)
        hole = ~valid & np.isfinite(filled) & (ndimage.uniform_filter(valid.astype(float), 3) >= 0.5)
        dsm[hole] = filled[hole]; valid |= hole
    dsm[~valid] = np.nan
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
                       crs=f"EPSG:{epsg}", transform=from_origin(x0, y1, res, res), nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        dst.write(dsm, 1)
        dst.update_tags(DESCRIPTION="VoxelFlight DSM: max observed surface height per cell; unobserved = nodata",
                        VERTICAL="barometric/GPS-referenced ellipsoidal-ish altitude (m)")
    return {"cells": int(nx * ny), "observed_fraction": float(observed.mean()), "valid_fraction": float(valid.mean()), "resolution_m": res}


def export_all(out_dir, mesh, pcd_points, pcd_colors, origin, epsg, extra_meta=None, formats=("ply", "obj", "glb", "fbx", "las", "tif")):
    import open3d as o3d
    from fbx import write_fbx
    os.makedirs(out_dir, exist_ok=True)
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles); C = np.asarray(mesh.vertex_colors)
    if len(C) != len(V):
        C = np.full((len(V), 3), 0.7)
    C = np.clip(C, 0, 1)
    arts = {}
    if "ply" in formats:
        p = os.path.join(out_dir, "model_mesh.ply"); o3d.io.write_triangle_mesh(p, mesh); arts["model_mesh.ply"] = p
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points)); pc.colors = o3d.utility.Vector3dVector(np.clip(pcd_colors, 0, 1))
        p = os.path.join(out_dir, "model_points.ply"); o3d.io.write_point_cloud(p, pc); arts["model_points.ply"] = p
    if "obj" in formats:
        p = os.path.join(out_dir, "model_mesh.obj"); write_obj(p, V, F, C); arts["model_mesh.obj"] = p
    if "glb" in formats:
        p = os.path.join(out_dir, "model_mesh.glb"); write_glb(p, V, F, C); arts["model_mesh.glb"] = p
    if "fbx" in formats:
        p = os.path.join(out_dir, "model_mesh.fbx"); write_fbx(p, V, F, C); arts["model_mesh.fbx"] = p
    meta = {"coordinate_system_local": "x=UTM easting - origin, y=UTM northing - origin, z=altitude - origin (metres)",
            "utm_epsg": int(epsg), "utm_origin": [float(o) for o in origin]}
    if "las" in formats:
        p = os.path.join(out_dir, "model_points_utm.las"); write_las(p, np.asarray(pcd_points), np.asarray(pcd_colors), origin, epsg); arts["model_points_utm.las"] = p
    if "tif" in formats:
        p = os.path.join(out_dir, "dsm_utm.tif"); meta["dsm"] = write_dsm(p, np.asarray(pcd_points), origin, epsg); arts["dsm_utm.tif"] = p
    meta.update(extra_meta or {})
    mpath = os.path.join(out_dir, "metadata.json")
    prev = json.load(open(mpath)) if os.path.exists(mpath) else {}
    meta = {**prev, **meta}
    meta["artifacts"] = {**prev.get("artifacts", {}), **{k: {"bytes": os.path.getsize(v), "sha256": _sha(v)} for k, v in arts.items()}}
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
