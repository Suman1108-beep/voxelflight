"""Package verified real surfaces, frozen GPS transform and evaluation together."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import trimesh


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a):
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Output must be new or empty.')
    source=json.loads((a.georeferenced/'report.json').read_text())
    frozen=json.loads((a.georeferenced/'gps_transform.json').read_text())
    evaluation=json.loads((a.georeferenced/'evaluation.json').read_text())
    fusion=json.loads((a.fusion/'report.json').read_text());texture=json.loads((a.textured/'report.json').read_text())
    if any(r.get('ground_truth_used') is not False for r in [source,frozen,fusion,texture]):raise ValueError('Ground-truth construction forbidden.')
    if Path(fusion['input']).resolve()!=Path(source['source_run']).resolve() or Path(texture['input']).resolve()!=a.fusion.resolve():
        raise ValueError('Geometry and camera lineage mismatch.')
    if evaluation['frozen_report_sha256']!=sha(a.georeferenced/'report.json'):raise ValueError('Evaluation does not match frozen run.')
    for name,digest in source['checksums'].items():
        if sha(a.georeferenced/name)!=digest:raise ValueError(f'Frozen artifact changed: {name}')
    if sha(a.textured/'reconstruction_mesh.glb')!=texture['sha256']:raise ValueError('Textured mesh changed.')
    if sha(a.fusion/'reconstruction_mesh.ply')!=texture['source_mesh_sha256']:raise ValueError('Fused mesh changed.')
    a.output.mkdir(parents=True,exist_ok=True)
    matrix=np.eye(4);matrix[:3,:3]=frozen['scale']*np.asarray(frozen['rotation']);matrix[:3,3]=frozen['translation']
    scene=trimesh.load(a.textured/'reconstruction_mesh.glb',force='scene',process=False)
    scene.apply_transform(matrix);scene.export(a.output/'VoxelFlight_Georeferenced_Textured.glb')
    mesh=trimesh.load(a.fusion/'reconstruction_mesh.ply',process=False);mesh.apply_transform(matrix)
    mesh.export(a.output/'VoxelFlight_Surface.ply');mesh.export(a.output/'VoxelFlight_Surface.obj')
    shutil.copy2(a.textured/'reconstruction_mesh.glb',a.output/'VoxelFlight_Local_Textured.glb')
    if (a.textured/'overview.png').exists():shutil.copy2(a.textured/'overview.png',a.output/'Actual_Reconstruction.png')
    copies={'pointcloud.ply':'Observed_Point_Cloud.ply','reconstruction_utm.las':'VoxelFlight_UTM32N.las',
        'surface_model.tif':'Observed_Surface_Model.tif','camera_poses.npz':'Camera_Poses_LocalUTM.npz',
        'trajectory.csv':'Trajectory_UTM.csv','gps_transform.json':'GPS_Transform.json',
        'evaluation.json':'Trajectory_Evaluation.json','report.json':'Source_GPS_Run.json'}
    for original,name in copies.items():shutil.copy2(a.georeferenced/original,a.output/name)
    shutil.copy2(a.fusion/'report.json',a.output/'Surface_Fusion_Report.json')
    shutil.copy2(a.textured/'report.json',a.output/'Source_Texturing_Report.json')
    model=trimesh.load(a.output/'VoxelFlight_Georeferenced_Textured.glb',force='scene',process=False)
    actual_faces=sum(len(g.faces) for g in model.geometry.values())
    if actual_faces!=fusion['triangles']:raise ValueError('Final GLB lost surface triangles.')
    areas=0.
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node]
        geometry=scene.geometry[name].copy();geometry.apply_transform(transform)
        areas+=geometry.area
    if not np.isclose(areas,mesh.area,rtol=1e-5):raise ValueError('Textured/untextured surface areas differ.')
    crs=dict(horizontal_crs=frozen['horizontal_crs'],local_origin=frozen['utm_origin'],
        local_models='GLB, surface PLY/OBJ, observed PLY and camera NPZ use projected metres relative to local_origin.',
        absolute_models='LAS, GeoTIFF and trajectory CSV contain absolute projected coordinates.',
        local_appearance_model='VoxelFlight_Local_Textured.glb is in the pre-GPS visual frame; use its source run for camera poses.',
        vertical_datum=frozen['vertical_datum'],provisional=True)
    (a.output/'Coordinate_Reference.json').write_text(json.dumps(crs,indent=2)+'\n')
    readme=f'''# VoxelFlight — actual reconstructed result

This package contains real geometry reconstructed from the recovered Zurich
flight segment (original frames 61201–63000, approximately 60 seconds).
No generated buildings, hidden-surface completion or reference-pose construction.

## Open the result

- `VoxelFlight_Georeferenced_Textured.glb`: fused surface with original-image textures, provisional GPS coordinates relative to a local UTM origin.
- `VoxelFlight_Local_Textured.glb`: identical surface in the visual reconstruction frame, before noisy GPS alignment.
- `Actual_Reconstruction.png`: real full-bounds mesh render, not a mockup.
- Surface PLY/OBJ: same fused geometry with vertex colors; OBJ color support varies by viewer.
- LAS, GeoTIFF and trajectory CSV: projected GIS outputs. GeoTIFF is an observed-surface raster, **not a bare-earth terrain model**.
- `Coordinate_Reference.json`: required origin and vertical-datum information.

## Measured evidence

- 90 source cameras; 4 joint Apple-GPU inference windows; {source['points']:,} observed points.
- {fusion['triangles']:,} fused triangles; {texture['source_textured_triangles']:,} use selected original-image textures.
- Camera refinement: 240.57 s; dense inference/export: {source['runtime_s']:.2f} s; fusion: {fusion['refusion_runtime_s']:.2f} s; texturing: {texture['runtime_s']:.2f} s.
- These stage times exclude original-data download, preview rendering and final packaging. They are not a measured 10-minute-video end-to-end benchmark.
- Absolute GPS-referenced camera RMSE: {evaluation['absolute_trajectory_error']['rmse_m']:.3f} m.
- Rigid-aligned camera RMSE: {evaluation['se3_aligned_trajectory_error']['rmse_m']:.3f} m.
- Similarity-aligned camera RMSE: {evaluation['sim3_aligned_trajectory_error']['rmse_m']:.3f} m.
- Aligned scores use one global reference alignment for evaluation only, over 90 sampled cameras. They are not full 1,800-frame scores or surface-accuracy scores.

## Still missing / unverified

1. Independently verified <=1 m surface accuracy and sub-metre absolute positioning.
2. Complete roofs, back sides, occluded areas and scene coverage; gaps remain explicit.
3. An unseen-flight, 10-minute-video runtime/completeness benchmark.
4. FBX export; this package provides GLB, PLY, OBJ, LAS and GeoTIFF instead.
5. General calibrated-SfM refinement integrated into the public upload worker; the current high-quality path uses separately refined cameras.
6. Inertial fusion, reliable dynamic-object removal and measured rolling-shutter correction in this Mac path.

The 90-view high-quality run consumes calibrated images extracted from the
original sequence. A separate MP4 smoke test checks video/GPS ingestion; its
coarser output is **not** substituted for this model. No website deployment or
default saved result was changed by this package.

Do not describe the aligned 0.397 m camera score as a <=1 m mapping certification.
All source reports, evaluation details and file hashes are included.
'''
    (a.output/'README.md').write_text(readme)
    manifest={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}
    (a.output/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    if archive.exists():raise FileExistsError('Archive already exists; it will not be overwritten.')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(a.output.iterdir()):
            if path.is_file():z.write(path,arcname=a.output.name+'/'+path.name)
    with zipfile.ZipFile(archive) as z:
        failure=z.testzip()
        if failure:raise IOError(f'Archive verification failed: {failure}')
    print(json.dumps(dict(output=str(a.output.resolve()),archive=str(archive.resolve()),
        archive_bytes=archive.stat().st_size,verified_files=len(manifest),triangles=actual_faces),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['georeferenced','fusion','textured','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
