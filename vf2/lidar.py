"""Public reference surface (swissSURFACE3D LiDAR) -> local UTM/ellipsoidal frame, with normals."""
import glob
import os

import numpy as np


def lv95_to_utm_ellipsoidal(E, N, H, epsg_utm, vertical="egm96"):
    """LV95 (EPSG:2056) + LHN95 heights (EPSG:5728) -> UTM easting/northing + height.
    vertical='egm96' matches GNSS receiver MSL altitude (what drone logs report); 'ellipsoidal' gives WGS84 h."""
    import pyproj
    pyproj.network.set_network_enabled(True)
    t1 = pyproj.Transformer.from_crs("EPSG:2056+5728", "EPSG:4979", always_xy=True)
    lon, lat, h = t1.transform(E, N, H)
    if vertical == "egm96":
        t3 = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True)
        _, _, h = t3.transform(lon, lat, h)
    t2 = pyproj.Transformer.from_crs(4326, epsg_utm, always_xy=True)
    x, y = t2.transform(lon, lat)
    return np.column_stack([x, y, h])


def load_reference_cloud(las_dir, geo, bbox_local, voxel=0.15, cache=None, classes=None):
    """Crop LiDAR to bbox_local=(xmin,ymin,xmax,ymax) in the local frame, voxel-downsample, estimate normals."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["points"], z["normals"]
    import laspy
    import open3d as o3d
    from pyproj import Transformer
    # bbox corners in LV95 for cropping before the (slow) datum transform
    inv = Transformer.from_crs(geo.epsg, 2056, always_xy=True)
    xs = np.array([bbox_local[0], bbox_local[2]]) + geo.origin[0]
    ys = np.array([bbox_local[1], bbox_local[3]]) + geo.origin[1]
    ce, cn = inv.transform(np.repeat(xs, 2), np.tile(ys, 2))
    pts = []
    for f in sorted(glob.glob(os.path.join(las_dir, "*.las"))):
        las = laspy.read(f)
        E, N, H = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        keep = (E >= ce.min() - 5) & (E <= ce.max() + 5) & (N >= cn.min() - 5) & (N <= cn.max() + 5)
        if classes is not None:
            keep &= np.isin(np.asarray(las.classification), classes)
        if keep.any():
            pts.append(np.column_stack([E[keep], N[keep], H[keep]]))
    P = np.concatenate(pts)
    U = lv95_to_utm_ellipsoidal(P[:, 0], P[:, 1], P[:, 2], geo.epsg) - geo.origin
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(U))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30))
    Pts, Nrm = np.asarray(pc.points), np.asarray(pc.normals)
    if cache:
        np.savez(cache, points=Pts, normals=Nrm)
    return Pts, Nrm
