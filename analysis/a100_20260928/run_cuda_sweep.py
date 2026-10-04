"""Run controlled A100 inference ablations and score only after outputs freeze."""

import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/workspace/voxelflight_a100_20260928')
data = root / 'datasets/zurich90_calibrated'
reference = root / 'datasets/zurich-61201-63000/Log Files/GroundTruthAGL.csv'
experiments = [
    ('90-420-w24-o6', 420, 24, 6),
    ('90-518-w24-o6', 518, 24, 6),
    ('90-518-w48-o8', 518, 48, 8),
]
env = os.environ.copy()
env.update({
    'HF_HOME': str(root / 'cache/huggingface'),
    'TORCH_HOME': str(root / 'cache/torch'),
    'DINO_SOURCE': str(root / 'dinov2'),
    'OMP_NUM_THREADS': '4',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})
outcomes = []
for tag, size, window, overlap in experiments:
    output = root / 'runs' / tag
    if output.exists():
        outcomes.append({'tag': tag, 'status': 'existing_output_preserved'})
        continue
    command = [
        str(root / '.venv/bin/python'), str(root / 'repo/mac_reconstruct.py'),
        '--images', str(data / 'input90'),
        '--calibration', str(data / 'camera.json'),
        '--visual-pose-priors', str(data / 'report.json'),
        '--max-frames', '90', '--size', str(size),
        '--window', str(window), '--overlap', str(overlap),
        '--save-predictions', '--output', str(output),
    ]
    log = root / 'logs' / f'{tag}.log'
    started = time.time()
    with log.open('ab') as stream:
        result = subprocess.run(command, stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    item = {'tag': tag, 'inference_returncode': result.returncode,
            'runtime_wall_s': time.time() - started}
    if result.returncode == 0:
        eval_command = [str(root / '.venv/bin/python'),
                        str(root / 'tools/evaluate_cuda_run.py'),
                        '--run', str(output), '--reference', str(reference),
                        '--output', str(output / 'evaluation.json')]
        with log.open('ab') as stream:
            evaluation = subprocess.run(eval_command, stdout=stream,
                                        stderr=subprocess.STDOUT, env=env)
        item['evaluation_returncode'] = evaluation.returncode
        if evaluation.returncode == 0:
            score = json.loads((output / 'evaluation.json').read_text())
            item['camera_sim3_rmse_m'] = score['sim3_aligned_camera_rmse_m']['rmse_m']
            item['camera_se3_rmse_m'] = score['se3_aligned_camera_rmse_m']['rmse_m']
    outcomes.append(item)
    (root / 'runs/a100_90view_sweep.json').write_text(json.dumps(outcomes, indent=2))
    print(json.dumps(item), flush=True)
