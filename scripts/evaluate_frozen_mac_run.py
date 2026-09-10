"""Score a frozen GPS-georeferenced trajectory, without changing its geometry.

Ground truth is read ONLY here, after all construction artifacts are frozen.
Absolute, rigid-aligned and similarity-aligned camera errors are separate.
None measures point-cloud or mesh surface accuracy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from align_metric import load_reference,umeyama,apply_similarity,error_summary


def run(a):
    if a.output.exists():raise ValueError('Refusing to overwrite an existing evaluation.')
    report=json.loads((a.input/'report.json').read_text())
    if report.get('ground_truth_used') is not False or report.get('georeferenced') is not True:
        raise ValueError('Expected a frozen non-GT georeferenced run.')
    for name,digest in report['checksums'].items():
        if hashlib.sha256((a.input/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Artifact changed: {name}')
    with (a.input/'trajectory.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    ids=np.asarray([int(Path(row['image']).stem) for row in rows])
    pred=np.asarray([[float(row[k]) for k in ['easting','northing','altitude']] for row in rows])
    gps=np.asarray([[float(row[k]) for k in ['gps_easting','gps_northing','gps_altitude']] for row in rows])
    reference_ids,reference,_=load_reference(a.reference)
    order=np.argsort(reference_ids);reference_ids=reference_ids[order];reference=reference[order]
    if len(np.unique(reference_ids))!=len(reference_ids) or ids.min()<reference_ids.min() or ids.max()>reference_ids.max():
        raise ValueError('Reference must uniquely cover all evaluated cameras; no extrapolation allowed.')
    upper=np.searchsorted(reference_ids,ids,side='left');lower=np.maximum(upper-1,0)
    gaps=np.where(reference_ids[upper]==ids,0,reference_ids[upper]-reference_ids[lower])
    if np.any(gaps>30):raise ValueError('Reference interpolation gap exceeds 30 source frames.')
    gt=np.column_stack([np.interp(ids,reference_ids,reference[:,axis]) for axis in range(3)])
    s,r,t=umeyama(pred,gt);shape=apply_similarity(pred,s,r,t)
    pc=pred-pred.mean(0);gc=gt-gt.mean(0);u,_,vt=np.linalg.svd(gc.T@pc)
    sign=np.ones(3);sign[-1]=np.linalg.det(u@vt);rigid_r=(u*sign)@vt
    rigid=pc@rigid_r.T+gt.mean(0)
    report_out=dict(schema='voxelflight.frozen-trajectory-evaluation.v1',frames=len(ids),
        image_id_range=[int(ids.min()),int(ids.max())],
        protocol=dict(ground_truth_used_in_construction=False,reference_use='Scoring only; geometry remains frozen',
            evaluated_quantity='Camera centres, NOT reconstructed surface points',reference_interpolation='Linear by source image ID; no extrapolation; gaps <=30 frames',
            interpolated_queries=int(np.sum(gaps>0)),exact_reference_queries=int(np.sum(gaps==0))),
        absolute_trajectory_error=error_summary(pred,gt),raw_gps_absolute_error=error_summary(gps,gt),
        se3_aligned_trajectory_error=error_summary(rigid,gt),
        sim3_aligned_trajectory_error=error_summary(shape,gt),evaluation_only_sim3_scale=s,
        surface_rmse_m=None,sih_spatial_accuracy_verified=False,
        frozen_run=str(a.input.resolve()),frozen_report_sha256=hashlib.sha256((a.input/'report.json').read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        warning='Aligned scores remove global errors using reference data at evaluation time. They are not absolute positioning or surface-accuracy scores.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report_out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report_out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
