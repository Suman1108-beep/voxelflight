"""Re-fuse cached predictions into one observed-surface mesh, without training.

Uses independently predicted views to reject inconsistent depth, then Open3D
TSDF integration. This is an opt-in geometry experiment, not an accuracy claim.
Point maps are rasterized into the predicted pinhole camera: the model's learned
rays need not exactly match its approximate intrinsics. We never silently replace
those point maps with unprojected depth on a different set of camera rays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def rasterize_prediction(prediction, max_depth=80.0):
    """Z-buffer actual predicted points; missing pixels remain missing."""
    pose = prediction['pose'].astype(np.float64)
    k = prediction['intrinsics'].astype(np.float64)
    h, w = prediction['depth'].shape
    world = prediction['points'].reshape(-1, 3)
    camera = (world-pose[:3, 3]) @ pose[:3, :3]
    valid = (prediction['mask'].reshape(-1).astype(bool)
             & np.isfinite(camera).all(1) & (camera[:, 2] > .05)
             & (camera[:, 2] < max_depth))
    indices = np.flatnonzero(valid)
    projected = camera[indices] @ k.T
    uv = np.rint(projected[:, :2]/projected[:, 2:]).astype(np.int64)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices, uv = indices[inside], uv[inside]
    # Nearest surface wins where learned rays project to the same pinhole pixel.
    order = np.argsort(camera[indices, 2], kind='stable')
    indices, uv = indices[order], uv[order]
    pixels = uv[:, 1]*w+uv[:, 0]
    _, first = np.unique(pixels, return_index=True)
    indices, pixels = indices[first], pixels[first]
    depth = np.zeros(h*w, np.float32)
    color = np.zeros((h*w, 3), np.uint8)
    depth[pixels] = camera[indices, 2]
    color[pixels] = prediction['color'].reshape(-1, 3)[indices]
    return dict(depth=depth.reshape(h, w), color=color.reshape(h, w, 3),
                pose=pose, intrinsics=k)


def unproject(view):
    d, k, pose = view['depth'], view['intrinsics'], view['pose']
    v, u = np.indices(d.shape)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
    return (rays*d[..., None]) @ pose[:3, :3].T+pose[:3, 3]


def multiview_support(views, relative_tolerance=.015, absolute_tolerance=.08,
                      neighbor_radius=None):
    """Count other views agreeing in depth; report internal consistency only.

    Out-of-frustum and occluded observations do not count as agreements. The
    source view never votes for itself. All candidate maps are frozen throughout.
    """
    if neighbor_radius is not None and neighbor_radius < 1:
        raise ValueError('Neighbor radius must be positive or omitted.')
    counts, observations, errors = [], [], []
    for i, source in enumerate(views):
        points = unproject(source).reshape(-1, 3)
        valid_source = source['depth'].reshape(-1) > 0
        support = np.zeros(len(points), np.uint16)
        observed = np.zeros(len(points), np.uint16)
        residuals = []
        first = max(0, i-neighbor_radius) if neighbor_radius is not None else 0
        last = min(len(views), i+neighbor_radius+1) if neighbor_radius is not None else len(views)
        for j in range(first, last):
            if i == j:
                continue
            target = views[j]
            pose, k = target['pose'], target['intrinsics']
            camera = (points-pose[:3, 3]) @ pose[:3, :3]
            positive = valid_source & (camera[:, 2] > .05)
            indices = np.flatnonzero(positive)
            q = camera[indices] @ k.T
            uv = np.rint(q[:, :2]/q[:, 2:]).astype(np.int64)
            h, w = target['depth'].shape
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            indices, uv = indices[inside], uv[inside]
            target_depth = target['depth'][uv[:, 1], uv[:, 0]]
            valid = target_depth > 0
            indices, target_depth = indices[valid], target_depth[valid]
            delta = np.abs(camera[indices, 2]-target_depth)
            tolerance = absolute_tolerance+relative_tolerance*target_depth
            agrees = delta <= tolerance
            support[indices[agrees]] += 1
            # Points behind a nearer surface are occluded, not free-space errors.
            visible = camera[indices, 2] <= target_depth+tolerance
            observed[indices[visible]] += 1
            residuals.extend((delta[visible]/np.maximum(target_depth[visible], .05))[::32])
        counts.append(support.reshape(source['depth'].shape))
        observations.append(observed.reshape(source['depth'].shape))
        errors.append(float(np.median(residuals)) if len(residuals) else None)
    return counts, observations, errors


def run(args):
    started = time.monotonic()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite existing geometry.')
    paths = sorted((args.input/'predictions').glob('frame_*.npz'))
    if args.view_stride < 1:
        raise ValueError('View stride must be positive.')
    paths = paths[::args.view_stride]
    if len(paths) < 2:
        raise ValueError('Need at least two cached predicted views (--save-predictions).')
    if args.voxel <= 0 or args.min_support < 0:
        raise ValueError('Voxel size must be positive; support must be nonnegative.')
    source_report = json.loads((args.input/'report.json').read_text())
    if source_report.get('ground_truth_used') is not False:
        raise ValueError('Prediction provenance must explicitly exclude ground truth.')
    args.output.mkdir(parents=True, exist_ok=True)
    views = []
    for path in paths:
        with np.load(path) as pred:
            views.append(rasterize_prediction(pred, args.max_depth))
    before = sum(int(np.count_nonzero(v['depth'])) for v in views)
    if args.min_support == 0:
        # No pixel can fail a zero-support threshold. Avoid an otherwise costly
        # O(views × neighbours × pixels) pass that cannot change the geometry.
        print('No multiview support filter requested; integrating all valid depths', flush=True)
        after = before
        errors = [None] * len(views)
    else:
        print(f'Checking cross-view depth agreement across {len(views)} actual predictions', flush=True)
        counts, observations, errors = multiview_support(
            views, args.relative_tolerance, args.absolute_tolerance, args.neighbor_radius)
        after = 0
        for i, (view, support) in enumerate(zip(views, counts)):
            view['depth'][support < args.min_support] = 0
            after += int(np.count_nonzero(view['depth']))
            np.savez_compressed(args.output/f'support_{i:05d}.npz', support=support, visible=observations[i])
    print(f'Retained {after:,}/{before:,} observed depth pixels', flush=True)
    if after < 100:
        raise ValueError('Insufficient cross-view agreement; no credible fused mesh.')
    import open3d as o3d
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=4*args.voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=2)
    for i, view in enumerate(views):
        h, w = view['depth'].shape
        k = view['intrinsics']
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(view['color'])),
            o3d.geometry.Image(np.ascontiguousarray(view['depth'])),
            depth_scale=1., depth_trunc=args.max_depth, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, k[0,0], k[1,1], k[0,2], k[1,2])
        volume.integrate(rgbd, intrinsic, np.linalg.inv(view['pose']))
        print(f'Integrating supported surfaces {i+1}/{len(views)}', flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_triangles().remove_degenerate_triangles().remove_unreferenced_vertices()
    # Remove only small disconnected fragments. Do not keep only the largest
    # building or close open boundaries; preserve legitimate separate structures.
    labels, sizes, areas = mesh.cluster_connected_triangles()
    labels, sizes = np.asarray(labels), np.asarray(sizes)
    faces_before_cleanup = len(mesh.triangles)
    if len(sizes):
        mesh.remove_triangles_by_mask(sizes[labels] < args.min_component_triangles)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise ValueError('No supported connected surfaces survived fusion.')
    o3d.io.write_triangle_mesh(str(args.output/'reconstruction_mesh.ply'), mesh)
    import trimesh
    visual = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                            vertex_colors=np.clip(np.asarray(mesh.vertex_colors)*255, 0, 255).astype(np.uint8), process=False)
    visual.export(args.output/'reconstruction_mesh.glb')
    visual.export(args.output/'reconstruction_mesh.obj')
    cloud = o3d.geometry.PointCloud(mesh.vertices)
    cloud.colors = mesh.vertex_colors
    o3d.io.write_point_cloud(str(args.output/'pointcloud.ply'), cloud)
    final_labels, final_sizes, _ = mesh.cluster_connected_triangles()
    final_sizes = np.asarray(final_sizes)
    report = dict(schema='voxelflight.refusion.v1', input=str(args.input.resolve()),
        method=('Cached predictions + TSDF surface integration' if args.min_support == 0
                else 'Cached multiview predictions + depth support + TSDF surface integration'),
        source_input=source_report.get('input'),source_window_size=source_report.get('window_size'),
        source_model=source_report['model'], source_inference_device=source_report['device'],
        inference_runtime_s=source_report['runtime_s'], refusion_runtime_s=time.monotonic()-started,
        source_views=len(views), selected_view_stride=args.view_stride,
        support_neighbor_radius=args.neighbor_radius,
        support_validation_skipped=args.min_support == 0,
        voxel_length_m=args.voxel, min_other_view_support=args.min_support,
        relative_tolerance=args.relative_tolerance, absolute_tolerance_m=args.absolute_tolerance,
        max_depth_m=args.max_depth, input_depth_pixels=before,
        supported_depth_pixels=after if args.min_support else None,
        supported_fraction=after/before if args.min_support else None,
        retained_depth_pixels=after, retained_fraction=after/before,
        median_relative_crossview_residual_per_view=errors,
        raw_triangles=faces_before_cleanup, triangles=len(mesh.triangles), vertices=len(mesh.vertices),
        connected_components=len(final_sizes),
        largest_component_triangle_fraction=float(final_sizes.max()/len(mesh.triangles)),
        ground_truth_used=False, training_performed=False, georeferenced=False,
        coordinate_system='Prediction-local Z-up; learned metric scale, north unknown',
        surface_rmse_m=None, trajectory_rmse_m=None,
        limitations=['Internal agreement is not independent spatial accuracy.',
            'Filtering can remove genuine surfaces with insufficient views.',
            'TSDF averages local evidence; it does not reconstruct unseen backsides.',
            'Cached local visual poses only; this experiment does not apply GPS.',
            'Input sampling and inference resolution are recorded in the source run; this is not an unseen-flight validation.'],
        artifact_sha256={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                         for name in ['reconstruction_mesh.glb','reconstruction_mesh.ply','pointcloud.ply']})
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--voxel', type=float, default=.06)
    p.add_argument('--min-support', type=int, default=1)
    p.add_argument('--relative-tolerance', type=float, default=.015)
    p.add_argument('--absolute-tolerance', type=float, default=.08)
    p.add_argument('--max-depth', type=float, default=80.)
    p.add_argument('--min-component-triangles', type=int, default=100)
    p.add_argument('--view-stride', type=int, default=1,
                   help='Use every Nth cached prediction, preserving time order.')
    p.add_argument('--neighbor-radius', type=int, default=None,
                   help='Check depth agreement only against this many adjacent selected views on each side; default checks all views.')
    run(p.parse_args())
