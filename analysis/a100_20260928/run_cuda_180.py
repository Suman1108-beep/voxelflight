"""Higher-density 180-view A100 reconstruction with independent pose scoring."""

import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/workspace/voxelflight_a100_20260928')
source = root / 'runs/calibrated-sfm180-v2'
output = root / 'runs/180-518-w24-o6'
if output.exists():
    raise SystemExit(f'Existing output preserved: {output}')
if len(list((source / 'images').glob('*.jpg'))) != 180:
    raise SystemExit('Expected 180 calibrated source frames.')
env = os.environ.copy()
env.update({'HF_HOME': str(root / 'cache/huggingface'),
            'TORCH_HOME': str(root / 'cache/torch'),
            'DINO_SOURCE': str(root / 'dinov2'),
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'OMP_NUM_THREADS': '4'})
command = [str(root / '.venv/bin/python'), str(root / 'repo/mac_reconstruct.py'),
           '--images', str(source / 'images'), '--calibration', str(source / 'camera.json'),
           '--visual-pose-priors', str(source / 'report.json'),
           '--max-frames', '180', '--size', '518', '--window', '24', '--overlap', '6',
           '--save-predictions', '--output', str(output)]
log = root / 'logs/180-518-w24-o6.log'
started = time.time()
with log.open('ab') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
record = {'inference_returncode': result.returncode,
          'inference_wall_s': time.time() - started, 'command': command}
if result.returncode == 0:
    evaluation = [str(root / '.venv/bin/python'), str(root / 'tools/evaluate_cuda_run.py'),
                  '--run', str(output), '--reference',
                  str(root / 'datasets/zurich-61201-63000/Log Files/GroundTruthAGL.csv'),
                  '--output', str(output / 'evaluation.json')]
    with log.open('ab') as stream:
        scored = subprocess.run(evaluation, stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    record['evaluation_returncode'] = scored.returncode
(root / 'runs/180_cuda_job.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
