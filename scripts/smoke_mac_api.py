"""Exercise the live local API using a video assembled from actual saved source frames.

This is an API/inference smoke test, NOT an original flight video or accuracy benchmark.
"""
import json
from pathlib import Path
import sys
import time
import cv2
import httpx

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'mac-smoke-input';folder.mkdir(exist_ok=True)
path=folder/'zurich-source-frames-API-smoke-not-original-flight.mp4'
frames=sorted((ROOT/'viewer/assets/frames').glob('*.png'))
first=cv2.imread(str(frames[0]));h,w=first.shape[:2]
writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),1,(w,h))
if not writer.isOpened():raise RuntimeError('MP4 test encoder unavailable')
for frame in frames:
    image=cv2.imread(str(frame));writer.write(cv2.resize(image,(w,h)))
writer.release()
with httpx.Client(base_url='http://127.0.0.1:8133',timeout=120) as client:
    health=client.get('/api/health').json()
    with path.open('rb') as video:
        response=client.post('/api/jobs',headers={'X-SIH3D-Session':health['session']},files={'video':(path.name,video,'video/mp4')},data={'max_frames':12,'size':350})
    response.raise_for_status();job=response.json();print('Created real inference job',job['id'],flush=True)
    for _ in range(240):
        value=client.get(f"/api/jobs/{job['id']}").json()
        if value['state'] in {'complete','failed','cancelled','interrupted'}:break
        time.sleep(1)
    print(json.dumps(value,indent=2),flush=True)
    if value['state']!='complete':sys.exit(1)
    report=client.get(f"/api/jobs/{job['id']}/assets/report.json").json()
    for name in report['artifacts']:
        artifact=client.get(f"/api/jobs/{job['id']}/assets/{name}");artifact.raise_for_status()
        if len(artifact.content)<10:raise RuntimeError(f'Empty artifact: {name}')
    print(json.dumps({k:report[k] for k in ['points','triangles','runtime_s','ground_truth_used','trajectory_rmse_m','surface_rmse_m']},indent=2))
