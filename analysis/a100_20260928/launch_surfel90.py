"""Train optional 2D Gaussian surface appearance with a fixed held-out split."""

import json
import os
from pathlib import Path
import subprocess

root = Path('/workspace/voxelflight_a100_20260928')
source = root / '2d-gaussian-splatting'
dataset = root / 'datasets/zurich90_splat'
output = root / 'runs/surfel90-30k-v2'
if output.exists():
    raise SystemExit(f'Existing training run is preserved: {output}')
command = [
    str(root / '.venv/bin/python'), str(source / 'train.py'),
    '-s', str(dataset), '-m', str(output), '-r', '2', '--eval',
    '--iterations', '30000', '--lambda_dist', '100',
    '--lambda_normal', '0.05', '--depth_ratio', '0',
    '--test_iterations', '3000', '7000', '15000', '30000',
    '--save_iterations', '7000', '15000', '30000',
    '--checkpoint_iterations', '7000', '15000',
    '--quiet',
]
environment = os.environ.copy()
environment['LD_LIBRARY_PATH'] = str(root / 'runtime-packages/root/usr/lib/x86_64-linux-gnu')
environment['OMP_NUM_THREADS'] = '8'
environment['TORCH_CUDA_ARCH_LIST'] = '8.0'
log = root / 'logs/surfel90-30k-v2.log'
with log.open('ab') as stream:
    process = subprocess.Popen(command, cwd=source, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
(root / 'logs/surfel90-30k-v2.pid.json').write_text(
    json.dumps({'pid': process.pid, 'command': command, 'log': str(log)}, indent=2))
print(json.dumps({'pid': process.pid, 'output': str(output), 'log': str(log)}))
