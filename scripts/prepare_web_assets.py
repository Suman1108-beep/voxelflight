"""Create lightweight copies of saved outputs for the portable viewer.

Coordinates and colors are preserved; only point coordinates change from float64
to float32. Original reconstruction assets are never overwritten.
"""
from pathlib import Path
import json
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'viewer' / 'assets'

def main():
    source = ASSETS / 'reconstruction_clean.ply'
    with source.open('rb') as stream:
        header = []
        while True:
            line = stream.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        count = int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
        types = {'double': '<f8', 'float': '<f4', 'uchar': 'u1', 'uint8': 'u1'}
        fields = [(s.split()[2], types[s.split()[1]]) for s in header if s.startswith('property ')]
        points = np.fromfile(stream, dtype=np.dtype(fields), count=count)
    xyz = np.column_stack([points[a] for a in ('x', 'y', 'z')])
    compact = np.empty(count, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for name in compact.dtype.names:
        compact[name] = points[name]
    out_header = f'ply\nformat binary_little_endian 1.0\ncomment Same saved points, float32 coordinates\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with (ASSETS / 'pointcloud.ply').open('wb') as stream:
        stream.write(out_header.encode('ascii'))
        compact.tofile(stream)
    # A fixed-resolution occupancy map of actual points, not a confidence map.
    resolution = 0.5
    minimum, maximum = xyz.min(0), xyz.max(0)
    cells = np.floor((xyz[:,:2] - minimum[:2]) / resolution).astype(int)
    width, height = cells.max(0) + 1
    occupancy = np.zeros((height, width), dtype=np.int32)
    np.add.at(occupancy, (cells[:,1], cells[:,0]), 1)
    filled = np.argwhere(occupancy > 0)
    coverage = {'resolution_m': resolution, 'bounds': [minimum.tolist(), maximum.tolist()], 'width': int(width), 'height': int(height), 'cells': [[int(x), int(y), int(occupancy[y,x])] for y,x in filled]}
    (ASSETS / 'coverage.json').write_text(json.dumps(coverage, separators=(',',':')))
    protocol = json.loads((ROOT / 'results/e2e_delivery_v3/construction_protocol.json').read_text())
    manifest = {'id': 'zurich-urban-01', 'name': 'Zurich urban corridor', 'dataset': 'Zurich Urban MAV', 'recorded_at': '2026-09-01', 'input_duration_s': 60, 'keyframes': 180, 'thumbnail_frames': list(range(0,180,10))+[179], 'units': 'metres', 'coordinates': 'Local ENU', 'crs': 'EPSG:32632', 'utm_origin': protocol['utm_translation'], 'inference_available': False, 'geometry_status': 'Partial reconstruction', 'source_url': 'https://rpg.ifi.uzh.ch/zurichmavdataset.html', 'windows': [{'id': 'w000','name':'West facade','range':[0,71]}, {'id':'w054','name':'Central facade','range':[54,125]}, {'id':'w108','name':'East facade','range':[108,179]}]}
    (ASSETS / 'scene.json').write_text(json.dumps(manifest, indent=2)+'\n')
    difference = float(np.max(np.abs(xyz - np.column_stack([compact[a] for a in ('x','y','z')]))))
    print(json.dumps({'points': count, 'source_bytes': source.stat().st_size, 'web_bytes': (ASSETS/'pointcloud.ply').stat().st_size, 'max_coordinate_rounding_m':difference, 'coverage_cells':len(filled)}))

if __name__ == '__main__':
    main()
