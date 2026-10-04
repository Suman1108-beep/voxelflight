"""Attempt the official 2DGS CUDA build in the isolated reconstruction venv."""

import os
from pathlib import Path
import subprocess

root = Path('/workspace/voxelflight_a100_20260928')
source = root / '2d-gaussian-splatting'
python = str(root / '.venv/bin/python')
env = os.environ.copy()
env.update({'CUDA_HOME': '/usr/local/cuda',
            'TORCH_CUDA_ARCH_LIST': '8.0', 'MAX_JOBS': '6'})


def install(*args):
    command = [python, '-m', 'pip', 'install', '--disable-pip-version-check', *args]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=source, env=env, check=True, timeout=1200)


install('mediapy==1.1.2', 'lpips==0.1.4', 'scikit-image>=0.21')
install('--no-build-isolation', '--no-deps', str(source / 'submodules/simple-knn'))
install('--no-build-isolation', '--no-deps',
        str(source / 'submodules/diff-surfel-rasterization'))
subprocess.run([python, '-c',
                'import simple_knn,diff_surfel_rasterization;print("2DGS CUDA imports OK")'],
               env=env, check=True, timeout=60)
