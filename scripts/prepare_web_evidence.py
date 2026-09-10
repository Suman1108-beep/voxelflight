"""Package real reconstruction evidence for the viewer; never synthesize geometry."""
from pathlib import Path
import csv, hashlib, io, json, shutil, struct, subprocess
import numpy as np
from PIL import Image
import imageio_ffmpeg
import open3d as o3d
import laspy

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'deliverables/VoxelFlight_Reconstruction_Verified_20260910'
RUN = ROOT / 'runs-mac/sfm-anchored90-420-20260910'
OUT = ROOT / 'viewer/assets/latest'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'frames').mkdir(exist_ok=True)

def compact_glb(source, destination):
    data = source.read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20+size])
    start = 20 + size + 8
    binary = data[start:]
    image_views = {v['bufferView']: v for v in document.get('images', [])}
    output = bytearray()
    geometry_hash = hashlib.sha256()
    for i, view in enumerate(document['bufferViews']):
        chunk = binary[view.get('byteOffset', 0):view.get('byteOffset', 0)+view['byteLength']]
        if i in image_views:
            texture = Image.open(io.BytesIO(chunk)).convert('RGB')
            texture.thumbnail((4096,4096))
            encoded = io.BytesIO()
            texture.save(encoded, format='JPEG', quality=82, optimize=True)
            chunk = encoded.getvalue()
            image_views[i]['mimeType'] = 'image/jpeg'
        else:
            geometry_hash.update(chunk)
        output.extend(b'\0' * ((-len(output)) % 4))
        view['byteOffset'], view['byteLength'] = len(output), len(chunk)
        output.extend(chunk)
    output.extend(b'\0' * ((-len(output)) % 4))
    document['buffers'][0]['byteLength'] = len(output)
    header = json.dumps(document, separators=(',', ':')).encode()
    header += b' ' * ((-len(header)) % 4)
    result = struct.pack('<III', 0x46546c67, 2, 28+len(header)+len(output))
    result += struct.pack('<II', len(header), 0x4e4f534a) + header
    result += struct.pack('<II', len(output), 0x004e4942) + output
    assert len(result) < 25*1024**2, f'{destination}: {len(result)} bytes'
    # All non-image buffer views are byte-identical, including indices and positions.
    check = hashlib.sha256()
    for i, view in enumerate(document['bufferViews']):
        if i not in image_views:
            check.update(output[view['byteOffset']:view['byteOffset']+view['byteLength']])
    assert check.digest() == geometry_hash.digest()
    destination.write_bytes(result)
    return {'bytes': len(result), 'geometry_buffer_sha256': check.hexdigest(), 'geometry_changed': False, 'texture': 'Original camera atlas, web-resized to 4096 px maximum; JPEG quality 82'}

report = json.loads((RUN / 'report.json').read_text())
web = {'local': compact_glb(PACKAGE/'VoxelFlight_Local_Textured.glb', OUT/'reconstruction_mesh.glb'),
       'georeferenced': compact_glb(PACKAGE/'VoxelFlight_Georeferenced_Textured.glb', OUT/'georeferenced_mesh.glb')}
for i, frame in enumerate(sorted((RUN/'frames').glob('*.jpg'))):
    with Image.open(frame) as im:
        if i == 0:
            im.save(OUT/'poster.jpg', quality=88)
        im.thumbnail((480, 270))
        im.save(OUT/'frames'/f'frame_{i:05d}.jpg', quality=84)

rows = list(csv.DictReader((RUN/'trajectory.csv').open()))
with (OUT/'trajectory.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['frame_id','camera_x','camera_y','camera_z'])
    writer.writerows([i, r['x'], r['y'], r['z']] for i, r in enumerate(rows))

cloud = o3d.io.read_point_cloud(str(RUN/'pointcloud.ply'))
original_count = len(cloud.points)
ids = np.linspace(0, original_count-1, min(400000, original_count), dtype=np.int64)
sample = cloud.select_by_index(ids.tolist())
o3d.io.write_point_cloud(str(OUT/'pointcloud.ply'), sample, write_ascii=False)
xy = np.asarray(sample.points)[:, :2]
lower, upper = xy.min(axis=0), xy.max(axis=0)
cells, counts = np.unique(np.floor((xy-lower)/.5).astype(int), axis=0, return_counts=True)
coverage = {'bounds':[lower.tolist(),upper.tolist()], 'resolution_m':.5,
            'cells':np.column_stack([cells,counts]).tolist(), 'frame':'Local visual frame, north unknown', 'scene_completeness':None}
(OUT/'coverage.json').write_text(json.dumps(coverage))

# GIS sample preserves the original LAS coordinate scale, offset and CRS.
las = laspy.read(PACKAGE/'VoxelFlight_UTM32N.las')
las.points = las.points[np.linspace(0, len(las.points)-1, min(400000,len(las.points)), dtype=np.int64)]
las.write(OUT/'reconstruction_utm.las')
shutil.copy2(PACKAGE/'Observed_Surface_Model.tif', OUT/'surface_model.tif')
shutil.copy2(PACKAGE/'Trajectory_UTM.csv', OUT/'trajectory_utm.csv')
for name, target in [('Trajectory_Evaluation.json','evaluation.json'),('Coordinate_Reference.json','coordinates.json'),('Surface_Fusion_Report.json','fusion.json'),('Source_Texturing_Report.json','texturing.json'),('GPS_Transform.json','gps_transform.json')]:
    value=json.loads((PACKAGE/name).read_text())
    # Remove local filesystem details from public reports, retaining protocol and hashes.
    for key in ['frozen_run','input','prediction_source']:
        if isinstance(value.get(key),str) and value[key].startswith('/'):
            value[key]=Path(value[key]).name
    (OUT/target).write_text(json.dumps(value,indent=2))

scene = {'keyframes':90,'duration_s':60,'source_images':1800,'run_date':'2026-09-10',
         'thumbnail_frames':list(range(0,90,9))+[89],
         'frame_times':[(int(f['source_name'].split('.')[0])-61201)/30 for f in report['frames']],
         'point_count':original_count,'web_point_count':len(sample.points),'mesh_triangles':646019,
         'texture_source':'Observed camera images only',
         'timings':{'visual_bundle_adjustment':240.5745,'depth_and_export':98.282699709,'surface_fusion':25.884232375,'source_texturing':5.132419458},
         'runtime_scope':'Sum of separately measured stages; excludes data download, browser packaging and GPS export. Not a 10-minute-video benchmark.',
         'web_assets':web,'geometry_coordinate_frame':'Pre-GPS visual frame; learned metric scale, north unknown',
         'provenance':'Zurich Urban MAV images 61201–63000; original 1080p frames. Web video is a 720p encoding, not an original camera MP4.'}
(OUT/'scene.json').write_text(json.dumps(scene,indent=2))
video=OUT/'flight-preview.mp4'
if not video.exists():
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','warning','-i',str(ROOT/'data/zurich-recovered-61201-63000/flight.mp4'),'-vf','scale=1280:720','-c:v','libx264','-preset','fast','-b:v','2100k','-maxrate','2400k','-bufsize','4200k','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(video)],check=True)
assert video.stat().st_size < 25*1024**2
manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'files':len(manifest),'mesh':web,'video_bytes':video.stat().st_size,'web_points':len(sample.points)},indent=2))
