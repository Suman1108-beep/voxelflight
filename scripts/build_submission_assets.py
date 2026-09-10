#!/usr/bin/env python3
"""Clean fused geometry and export SIH submission-ready 3D/GIS assets."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
import rasterio
import trimesh
from rasterio.transform import from_origin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("relative_point_cloud", type=Path)
    parser.add_argument("utm_point_cloud", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--epsg", type=int, default=32632)
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--normal-radius", type=float, default=0.75)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--density-quantile", type=float, default=0.025)
    parser.add_argument("--minimum-triangles", type=int, default=250)
    parser.add_argument("--target-triangles", type=int, default=350_000)
    parser.add_argument("--dsm-resolution", type=float, default=0.25)
    return parser.parse_args()


def clean_cloud(
    path: Path, voxel_size: float, normal_radius: float
) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=True)
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.25)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=48
        )
    )
    cloud.normalize_normals()
    return cloud


def reconstruct_mesh(
    cloud: o3d.geometry.PointCloud,
    *,
    depth: int,
    density_quantile: float,
    minimum_triangles: int,
    target_triangles: int,
) -> o3d.geometry.TriangleMesh:
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, scale=1.08, linear_fit=True, n_threads=2
    )
    density = np.asarray(density)
    if len(density):
        threshold = float(np.quantile(density, density_quantile))
        mesh.remove_vertices_by_mask(density < threshold)
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    if len(counts):
        mesh.remove_triangles_by_mask(counts[clusters] < minimum_triangles)
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()
    return mesh


def open3d_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    colors = np.asarray(mesh.vertex_colors)
    vertex_colors = None
    if len(colors) == len(mesh.vertices):
        vertex_colors = np.column_stack(
            (np.clip(colors * 255.0, 0, 255).astype(np.uint8),
             np.full(len(colors), 255, dtype=np.uint8))
        )
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
        vertex_colors=vertex_colors,
        process=False,
    )


def export_las(cloud: o3d.geometry.PointCloud, path: Path) -> None:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.001, 0.001, 0.001])
    header.offsets = np.floor(points.min(axis=0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = points.T
    if len(colors) == len(points):
        rgb = np.clip(colors * 65535.0, 0, 65535).astype(np.uint16)
        las.red, las.green, las.blue = rgb.T
    las.write(path)


def export_dsm(
    cloud: o3d.geometry.PointCloud, path: Path, resolution: float, epsg: int
) -> dict[str, float | int]:
    points = np.asarray(cloud.points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    width = max(1, int(np.ceil((maximum[0] - minimum[0]) / resolution)) + 1)
    height = max(1, int(np.ceil((maximum[1] - minimum[1]) / resolution)) + 1)
    columns = np.clip(
        ((points[:, 0] - minimum[0]) / resolution).astype(int), 0, width - 1
    )
    rows = np.clip(
        ((maximum[1] - points[:, 1]) / resolution).astype(int), 0, height - 1
    )
    nodata = np.float32(-9999.0)
    raster = np.full((height, width), nodata, dtype=np.float32)
    flat = rows * width + columns
    order = np.argsort(flat)
    sorted_flat = flat[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_flat)) + 1]
    cells = sorted_flat[starts]
    raster.reshape(-1)[cells] = np.maximum.reduceat(points[order, 2], starts)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=from_origin(minimum[0], maximum[1], resolution, resolution),
        nodata=float(nodata),
        compress="deflate",
        predictor=3,
    ) as dataset:
        dataset.write(raster, 1)
        dataset.set_band_description(1, "Digital surface elevation (m)")
    return {
        "resolution_m": resolution,
        "width": width,
        "height": height,
        "valid_cells": int(np.count_nonzero(raster != nodata)),
    }


def main() -> int:
    args = parse_args()
    if args.voxel_size <= 0 or args.normal_radius <= 0:
        raise ValueError("Voxel size and normal radius must be positive")
    if not 0 <= args.density_quantile < 1:
        raise ValueError("Density quantile must be in [0, 1)")
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relative = clean_cloud(
        args.relative_point_cloud, args.voxel_size, args.normal_radius
    )
    cleaned_ply = args.output_dir / "reconstruction_clean.ply"
    o3d.io.write_point_cloud(str(cleaned_ply), relative, compressed=True)

    mesh = reconstruct_mesh(
        relative,
        depth=args.poisson_depth,
        density_quantile=args.density_quantile,
        minimum_triangles=args.minimum_triangles,
        target_triangles=args.target_triangles,
    )
    mesh_ply = args.output_dir / "reconstruction_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh, write_vertex_colors=True)
    tri_mesh = open3d_to_trimesh(mesh)
    tri_mesh.export(args.output_dir / "reconstruction_mesh.obj")
    tri_mesh.export(args.output_dir / "reconstruction_mesh.glb")

    utm = clean_cloud(args.utm_point_cloud, args.voxel_size, args.normal_radius)
    export_las(utm, args.output_dir / "reconstruction_utm.las")
    dsm_report = export_dsm(
        utm,
        args.output_dir / "surface_model.tif",
        args.dsm_resolution,
        args.epsg,
    )

    if args.trajectory:
        shutil.copy2(args.trajectory, args.output_dir / "trajectory.csv")
    if args.evaluation:
        shutil.copy2(args.evaluation, args.output_dir / "evaluation.json")

    bounds = np.asarray(relative.get_axis_aligned_bounding_box().get_extent())
    report = {
        "construction_uses_ground_truth": False,
        "source": str(args.relative_point_cloud),
        "coordinate_reference_system": f"EPSG:{args.epsg}",
        "clean_point_count": len(relative.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "local_extent_m": bounds.tolist(),
        "dsm": dsm_report,
        "exports": {
            "point_cloud": "reconstruction_clean.ply",
            "las": "reconstruction_utm.las",
            "mesh_ply": "reconstruction_mesh.ply",
            "mesh_obj": "reconstruction_mesh.obj",
            "mesh_glb": "reconstruction_mesh.glb",
            "geotiff": "surface_model.tif",
        },
        "runtime_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "asset_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
