"""Publish only actual Mac test outputs, never model weights or private inputs."""
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
source=root/'runs-mac/offline-textured-proof'
target=root/'viewer/assets/mac-proof'
report=json.loads((source/'report.json').read_text())
assert report['device']=='Apple MPS' and not report['ground_truth_used']
assert report['surface_rmse_m'] is None
target.mkdir(parents=True,exist_ok=True)
for name in report['artifacts']:
    assert Path(name).name==name
    shutil.copy2(source/name,target/name)
shutil.copytree(source/'frames',target/'frames',dirs_exist_ok=True)
print(f"Packaged actual {report['keyframes']}-frame Mac reconstruction. No accuracy claim.")
