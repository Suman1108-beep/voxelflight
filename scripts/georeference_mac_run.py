"""Freeze a GPS-only transform and export an existing real Mac reconstruction.

Consumes only prediction artifacts and image-synchronised onboard telemetry.
Ground-truth files are not accepted. Accuracy is measured by a separate command.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mac_geometry import transform_points
from mac_reconstruct import export_gis


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_similarity(source,target,sigma):
    """Positive-scale Umeyama with bounded Huber residual weights, no GT."""
    x,y=np.asarray(source,float),np.asarray(target,float)
    sigma=np.asarray(sigma,float)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<6 or sigma.shape!=(len(x),):
        raise ValueError('Need six or more matched camera/GPS positions and uncertainties.')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('Non-finite coordinates or invalid uncertainty.')
    base=1/np.maximum(sigma,1.)**2;weights=base.copy()
    for _ in range(8):
        w=weights/weights.sum();xm=w@x;ym=w@y;xc=x-xm;yc=y-ym
        variance=np.sum(w*np.sum(xc*xc,axis=1))
        if variance<1e-9:raise ValueError('Stationary cameras cannot constrain a GPS transform.')
        u,d,vt=np.linalg.svd((yc*w[:,None]).T@xc)
        sign=np.ones(3);sign[-1]=np.linalg.det(u@vt);rotation=(u*sign)@vt
        scale=float(d@sign/variance);translation=ym-scale*rotation@xm
        if scale<=0 or not np.isfinite(scale):raise ValueError('Invalid GPS scale.')
        residual=np.linalg.norm(transform_points(x,scale,rotation,translation)-y,axis=1)
        weights=base*np.minimum(1.,1.5*sigma/np.maximum(residual,1e-9))
    singular=np.linalg.svd(y-y.mean(0),compute_uv=False)
    return scale,rotation,translation,dict(fit_rmse_m=float(np.sqrt(np.mean(residual**2))),
        max_residual_m=float(residual.max()),median_eph_m=float(np.median(sigma)),
        lateral_spread_ratio=float(singular[1]/max(singular[0],1e-9)),
        correspondence_count=len(x),independent_accuracy=False)


def load_gps(path,frames):
    if 'groundtruth' in path.name.lower():raise ValueError('Use onboard telemetry, never a ground-truth table.')
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    required={'image','timestamp_us','latitude','longitude','altitude_m','gps_fix_type','gps_eph_m'}
    if not rows or not required.issubset(rows[0]):raise ValueError('Expected canonical image-synchronised onboard telemetry.')
    lookup={}
    for row in rows:
        if row['image'] in lookup:raise ValueError('Duplicate image telemetry.')
        lookup[row['image']]=row
    selected=[]
    for frame in frames:
        name=frame.get('source_name')
        if name not in lookup:raise ValueError(f'Missing timestamped GPS for {name}')
        row=lookup[name]
        if int(row['gps_fix_type'])<3:raise ValueError(f'No 3D GPS fix for {name}')
        selected.append(row)
    lat=np.array([float(r['latitude']) for r in selected]);lon=np.array([float(r['longitude']) for r in selected])
    alt=np.array([float(r['altitude_m']) for r in selected]);sigma=np.array([float(r['gps_eph_m']) for r in selected])
    times=np.array([int(r['timestamp_us']) for r in selected],np.int64)
    if not np.isfinite(np.c_[lat,lon,alt,sigma]).all() or np.any(np.abs(lat)>90) or np.any(np.abs(lon)>180):
        raise ValueError('Invalid geographic coordinates.')
    if np.any(np.diff(times)<=0):raise ValueError('Camera timestamps must strictly increase.')
    zones=np.clip(((lon+180)//6+1).astype(int),1,60)
    if len(set(zones))!=1 or np.any((lat>=0)!=(lat[0]>=0)):raise ValueError('This export requires a single UTM zone/hemisphere.')
    epsg=int((32600 if lat[0]>=0 else 32700)+zones[0])
    project=Transformer.from_crs(4326,epsg,always_xy=True)
    e,n=project.transform(lon,lat)
    return np.c_[e,n,alt],sigma,times,epsg


def run(a):
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Output must be new or empty.')
    report=json.loads((a.input/'report.json').read_text())
    if report.get('ground_truth_used') is not False or report.get('georeferenced'):
        raise ValueError('Expected a non-GT local prediction; cannot georeference twice.')
    with np.load(a.input/'camera_poses.npz') as data:
        poses=data['camera_to_world'].astype(float);intrinsics=data['intrinsics']
    frames=report['frames']
    if len(poses)!=len(frames):raise ValueError('Camera/frame count mismatch.')
    gps,sigma,times,epsg=load_gps(a.telemetry,frames)
    origin=np.median(gps,axis=0)
    s,r,t,quality=weighted_similarity(poses[:,:3,3],gps-origin,sigma)
    if not .02<s<50:raise ValueError('Implausible GPS/model scale disagreement.')
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    transform=dict(schema='voxelflight.gps-transform.v1',scale=s,rotation=r.tolist(),translation=t.tolist(),
        utm_origin=origin.tolist(),horizontal_crs=f'EPSG:{epsg}',vertical_datum='Onboard GPS altitude; dataset documents MSL; not independently surveyed',
        ground_truth_used=False,construction_inputs={str(a.input/'report.json'):sha(a.input/'report.json'),
            str(a.input/'camera_poses.npz'):sha(a.input/'camera_poses.npz'),str(a.telemetry):sha(a.telemetry)},gps_fit=quality)
    (a.output/'gps_transform.json').write_text(json.dumps(transform,indent=2)+'\n')
    matrix=np.eye(4);matrix[:3,:3]=s*r;matrix[:3,3]=t
    import trimesh
    scene=trimesh.load(a.input/'reconstruction_mesh.glb',force='scene',process=False)
    scene.apply_transform(matrix);scene.export(a.output/'reconstruction_mesh.glb')
    cloud=trimesh.load(a.input/'pointcloud.ply',process=False)
    points=transform_points(np.asarray(cloud.vertices),s,r,t);colors=np.asarray(cloud.colors)[:,:3]
    trimesh.PointCloud(points,colors=colors).export(a.output/'pointcloud.ply')
    geo=dict(origin=origin,epsg=epsg)
    gis=export_gis(a.output,points,colors,geo)
    poses[:,:3,3]=transform_points(poses[:,:3,3],s,r,t);poses[:,:3,:3]=r@poses[:,:3,:3]
    np.savez_compressed(a.output/'camera_poses.npz',camera_to_world=poses,intrinsics=intrinsics)
    with (a.output/'trajectory.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['image','timestamp_us','easting','northing','altitude','gps_easting','gps_northing','gps_altitude'])
        for frame,stamp,pose,g in zip(frames,times,poses,gps):writer.writerow([frame['source_name'],int(stamp),*(pose[:3,3]+origin),*g])
    report.update(schema='voxelflight.gps-run.v1',source_run=str(a.input.resolve()),
        georeferenced=True,georeference_status='Provisional onboard GPS alignment, not surveyed accuracy',
        coordinate_system=f'EPSG:{epsg} projected metres plus local origin; Z is onboard altitude',
        utm_origin=origin.tolist(),gps_fit_residual=quality,trajectory_rmse_m=None,surface_rmse_m=None,
        sih_requirements_verified=False,georeference_runtime_s=time.monotonic()-started,
        source_trajectory=report.get('trajectory'),trajectory=[dict(frame=i,position=p[:3,3].tolist()) for i,p in enumerate(poses)])
    report['warnings']=[*report.get('warnings',[]),'GPS was joined by exact source image ID/timestamp, never by reference trajectory.',
        'GPS uncertainty and altitude datum limit absolute accuracy. GPS-fit residual is not ground-truth RMSE.',
        'No 3D surface reference is available for this run; model holes are not filled.']
    if quality['lateral_spread_ratio']<.05:report['warnings'].append('Near-straight GPS trajectory weakly constrains rotation around the travel direction.')
    report['artifacts']=['reconstruction_mesh.glb','pointcloud.ply','camera_poses.npz','trajectory.csv','gps_transform.json',*gis,'report.json']
    report['checksums']={name:sha(a.output/name) for name in report['artifacts'] if name!='report.json'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),scale=s,gps_fit=quality,exports=report['artifacts']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--telemetry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
