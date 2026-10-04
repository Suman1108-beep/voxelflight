"""Launch the locked 90-view CUDA baseline as a recoverable background job."""

import json
import os
from pathlib import Path
import subprocess

root = Path('/workspace/voxelflight_a100_20260928')
run = root / 'runs/baseline90-420-w24-o2'
if run.exists():
    raise SystemExit(f'Existing output is preserved: {run}')
data = root / 'datasets/zurich90_calibrated'
args = [
    str(root / '.venv/bin/python'), str(root / 'repo/mac_reconstruct.py'),
    '--images', str(data / 'input90'),
    '--calibration', str(data / 'camera.json'),
    '--visual-pose-priors', str(data / 'report.json'),
    '--max-frames', '90', '--size', '420', '--window', '24', '--overlap', '2',
    '--save-predictions', '--output', str(run),
]
env = os.environ.copy()
env.update({
    'HF_HOME': str(root / 'cache/huggingface'),
    'TORCH_HOME': str(root / 'cache/torch'),
    'DINO_SOURCE': str(root / 'dinov2'),
    'OMP_NUM_THREADS': '4',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})
log = root / 'logs/baseline90-420-w24-o2.log'
with log.open('ab') as stream:
    job = subprocess.Popen(args, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, start_new_session=True)
(root / 'logs/baseline90-420-w24-o2.pid.json').write_text(
    json.dumps({'pid': job.pid, 'command': args, 'log': str(log)}, indent=2))
print(json.dumps({'pid': job.pid, 'output': str(run), 'log': str(log)}))
