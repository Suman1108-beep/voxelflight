"""Render the best saved 2DGS checkpoint on genuinely held-out frames."""

import json
import os
from pathlib import Path
import subprocess


root = Path('/workspace/voxelflight_a100_20260928')
source = root / '2d-gaussian-splatting'
model = root / 'runs/surfel90-30k-v2'
dataset = root / 'datasets/zurich90_splat'
python = str(root / '.venv/bin/python')
environment = os.environ.copy()
environment['LD_LIBRARY_PATH'] = str(root / 'runtime-packages/root/usr/lib/x86_64-linux-gnu')
environment['OMP_NUM_THREADS'] = '4'
environment['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
log = root / 'logs/surfel7k-finalize.log'

commands = [
    [python, str(source / 'render.py'), '-s', str(dataset), '-m', str(model),
     '-r', '2', '--eval', '--iteration', '7000', '--skip_train', '--skip_mesh'],
    [python, str(source / 'metrics.py'), '-m', str(model)],
]
with log.open('w') as stream:
    for command in commands:
        print('+', ' '.join(command), file=stream, flush=True)
        subprocess.run(command, cwd=source, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=1800)
print(json.dumps({'result': str(model / 'results.json'), 'log': str(log)}))
