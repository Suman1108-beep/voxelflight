"""Calibrated feature-based camera refinement from original Zurich flight imagery.

Original intrinsics/distortion are supplied; no ground-truth poses, street-view
images, or surveyed scene geometry are used. SfM has arbitrary scale until aligned
to independently supplied GPS or a model scale estimate in a separate stage.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import pycolmap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=90);p.add_argument('--width',type=int,default=1280)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--max-runtime',type=int,default=600)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Output must be new or empty.')
    source=sorted((a.dataset/'MAV Images').glob('*.jpg'))
    if len(source)<a.frames:raise ValueError('Wait for the requested source data before running refinement.')
    ids=np.unique(np.linspace(0,len(source)-1,a.frames).round().astype(int))
    selected=[source[i] for i in ids]
    with np.load(a.dataset/'calibration_data.npz') as calib:
        k=calib['intrinsic_matrix'];d=calib['distCoeff'].reshape(-1)
    first=cv2.imread(str(selected[0]));h,w=first.shape[:2]
    if (w,h)!=(1920,1080):raise ValueError('Expected original 1920×1080 flight images.')
    new_k,_=cv2.getOptimalNewCameraMatrix(k,d,(w,h),0,(w,h))
    mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,new_k,(w,h),cv2.CV_32FC1)
    output_size=(a.width,round(h*a.width/w));scaled_k=new_k.copy()
    scaled_k[0]*=output_size[0]/w;scaled_k[1]*=output_size[1]/h
    a.output.mkdir(parents=True,exist_ok=True);images=a.output/'images';images.mkdir()
    print('Preparing calibrated full-resolution source frames',flush=True)
    started=time.monotonic()
    for source in selected:
        original=cv2.imread(str(source))
        corrected=cv2.remap(original,mapx,mapy,cv2.INTER_LINEAR)
        corrected=cv2.resize(corrected,output_size,interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(images/source.name),corrected,[cv2.IMWRITE_JPEG_QUALITY,96]):raise IOError('Image encoding failed')
    config=dict(model='pinhole',width=output_size[0],height=output_size[1],intrinsic_matrix=scaled_k.tolist(),
        distortion_coefficients=[],source='Original calibration_data.npz; undistorted with alpha=0',source_images=[s.name for s in selected])
    (a.output/'camera.json').write_text(json.dumps(config,indent=2)+'\n')
    reader=pycolmap.ImageReaderOptions();reader.camera_model='PINHOLE'
    reader.camera_params=','.join(str(x) for x in [scaled_k[0,0],scaled_k[1,1],scaled_k[0,2],scaled_k[1,2]])
    extraction=pycolmap.FeatureExtractionOptions();extraction.num_threads=a.threads;extraction.max_image_size=a.width
    extraction.sift.max_num_features=6000
    database=a.output/'database.db'
    pycolmap.extract_features(database,images,camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,extraction_options=extraction,device=pycolmap.Device.cpu)
    matching=pycolmap.FeatureMatchingOptions();matching.num_threads=a.threads;matching.guided_matching=True
    pairing=pycolmap.SequentialPairingOptions();pairing.overlap=10;pairing.quadratic_overlap=True
    pycolmap.match_sequential(database,matching_options=matching,pairing_options=pairing,device=pycolmap.Device.cpu)
    options=pycolmap.IncrementalPipelineOptions();options.num_threads=a.threads;options.random_seed=17
    options.ba_refine_focal_length=False;options.ba_refine_principal_point=False;options.ba_refine_extra_params=False
    options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
    options.max_runtime_seconds=a.max_runtime;options.min_model_size=8
    options.mapper.init_min_tri_angle=8.
    sparse=a.output/'sparse';sparse.mkdir()
    reconstructions=pycolmap.incremental_mapping(database,images,sparse,options=options)
    if not reconstructions:raise RuntimeError('No camera reconstruction registered; input matching failed.')
    best_key=max(reconstructions,key=lambda key:reconstructions[key].num_reg_images())
    reconstruction=reconstructions[best_key];best=a.output/'best';best.mkdir();reconstruction.write(best)
    reconstruction.export_PLY(a.output/'sparse_points.ply')
    registered=[]
    for image_id in reconstruction.reg_image_ids():
        image=reconstruction.images[image_id]
        matrix=np.eye(4);matrix[:3,:]=image.cam_from_world().inverse().matrix()
        registered.append(dict(image=image.name,camera_to_world=matrix.tolist(),camera_id=image.camera_id))
    registered.sort(key=lambda value:value['image'])
    report=dict(method='Calibrated SIFT matching + incremental SfM + bundle adjustment',
        source_dataset=str(a.dataset.resolve()),input_images=len(selected),registered_images=len(registered),
        sparse_points=reconstruction.num_points3D(),mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
        mean_track_length=reconstruction.compute_mean_track_length(),runtime_s=time.monotonic()-started,
        ground_truth_used=False,metric_scale_established=False,poses=registered,
        warning='Image reprojection error is not trajectory or surface error in metres.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='poses'},indent=2),flush=True)


if __name__=='__main__':main()
