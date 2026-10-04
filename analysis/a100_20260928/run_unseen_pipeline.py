"""Evaluate the frozen method on a disjoint Zurich flight segment.

Ground truth is passed only to scoring commands after each reconstruction is
complete. All outputs live in the isolated experiment workspace.
"""

import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/workspace/voxelflight_a100_20260928')
dataset = root / 'datasets/zurich-20001-21800'
sfm = root / 'runs/unseen20001-sfm180'
reconstruction = root / 'runs/unseen20001-180-518-w24-o6'
fusion = root / 'runs/unseen20001-180-518-tsdf004'
reference = dataset / 'Log Files/GroundTruthAGL.csv'
python = str(root / '.venv/bin/python')


def run(command, env=None, timeout=3600):
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, env=env, check=True, timeout=timeout)


deadline = time.monotonic() + 1200
while not (dataset / 'download_manifest.json').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Official unseen-segment download did not complete.')
    time.sleep(5)
manifest = json.loads((dataset / 'download_manifest.json').read_text())
if manifest.get('frames') != 1800 or manifest.get('first_image_id') != 20001:
    raise ValueError('Unexpected official dataset manifest.')
print('Verified official unseen segment:', dataset, flush=True)

if not (sfm / 'report.json').exists():
    run([python, str(root / 'repo/scripts/refine_zurich_cameras.py'),
         '--dataset', str(dataset), '--output', str(sfm), '--frames', '180',
         '--width', '1280', '--threads', '8', '--max-runtime', '1200'], timeout=2400)
if not (sfm / 'evaluation.json').exists():
    run([python, str(root / 'tools/score_visual_poses.py'),
         '--report', str(sfm / 'report.json'), '--reference', str(reference),
         '--output', str(sfm / 'evaluation.json')], timeout=120)

env = os.environ.copy()
env.update({'HF_HOME': str(root / 'cache/huggingface'),
            'TORCH_HOME': str(root / 'cache/torch'),
            'DINO_SOURCE': str(root / 'dinov2'),
            'OMP_NUM_THREADS': '4',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
if not (reconstruction / 'report.json').exists():
    run([python, str(root / 'repo/mac_reconstruct.py'),
         '--images', str(sfm / 'images'),
         '--calibration', str(sfm / 'camera.json'),
         '--visual-pose-priors', str(sfm / 'report.json'),
         '--max-frames', '180', '--size', '518', '--window', '24',
         '--overlap', '6', '--save-predictions', '--output', str(reconstruction)],
        env=env, timeout=1500)
if not (reconstruction / 'evaluation.json').exists():
    run([python, str(root / 'tools/evaluate_cuda_run.py'),
         '--run', str(reconstruction), '--reference', str(reference),
         '--output', str(reconstruction / 'evaluation.json')], timeout=120)

env['LD_LIBRARY_PATH'] = str(root / 'runtime-packages/root/usr/lib/x86_64-linux-gnu')
if not (fusion / 'report.json').exists():
    run([python, str(root / 'repo/mac_refuse.py'),
         '--input', str(reconstruction), '--output', str(fusion),
         '--voxel', '0.04'], env=env, timeout=1500)
print('Unseen-segment reconstruction, independent pose score, and TSDF complete.', flush=True)
