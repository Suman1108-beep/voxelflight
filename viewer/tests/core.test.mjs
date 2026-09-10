import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseCSV,parseTrajectory,validateTelemetry,escapeHTML,formatTime} from '../core.js';
const asset=name=>readFile(new URL('../assets/'+name,import.meta.url));
test('CSV supports BOM, CRLF, quotes, commas, escaped quotes and multiline cells',()=>{
  assert.deepEqual(parseCSV('\uFEFFname,value\r\n"a, b","line1\nline2"\r\n"say ""hi""",2'),[{name:'a, b',value:'line1\nline2'},{name:'say "hi"',value:'2'}]);
});
test('malformed CSV is rejected',()=>{for(const text of ['x,x\n1,2','x,y\n1','x,y\n"1,2',',y\n1,2'])assert.throws(()=>parseCSV(text));});
test('GPS validates aliases and numeric bounds, and requires time',()=>{
  assert.equal(validateTelemetry('timestamp,GPS_latitude,longitude\n0,47.3,8.5','csv').records,1);
  for(const text of ['time,lat,lon\n0,91,8','time,lat,lon\n0,,8','time,lat,lon\n,47,8','lat,lon\n47,8'])assert.throws(()=>validateTelemetry(text,'csv'));
});
test('JSON accepts supported record arrays and rejects invalid coordinates',()=>{
  assert.equal(validateTelemetry(JSON.stringify({records:[{frame:0,lat:47,lon:8}]}),'json').records,1);
  assert.throws(()=>validateTelemetry('[{"frame":0,"lat":null,"lon":8}]','json'));
});
test('SRT is explicitly a field check, not complete synchronization',()=>{
  const result=validateTelemetry('1\n00:00:00,000 --> 00:00:00,033\n[latitude: 47.1] [longitude: 8.2]','srt');
  assert.equal(result.status,'header-check');assert.equal(result.records,1);assert.throws(()=>validateTelemetry('1\nno GPS','srt'));
});
test('trajectory reads 180 saved frames with finite coordinates',async()=>{
  const rows=parseTrajectory((await asset('trajectory.csv')).toString());assert.equal(rows.length,180);assert.equal(rows.at(-1).frame,179);
  assert.throws(()=>parseTrajectory('frame_id,camera_x,camera_y,camera_z\n0,,1,2'));
});
test('reports distinguish aligned shape, absolute error and no-GT construction',async()=>{
  const value=JSON.parse(await asset('evaluation.json'));assert.equal(value.construction_uses_ground_truth,false);
  assert.equal(value.trajectory_shape_error_after_global_sim3.rmse_m.toFixed(3),'0.810');assert.ok(value.absolute_georeferencing_error.rmse_m>8);
});
test('compact point cloud retains all 518786 points and has a complete binary payload',async()=>{
  const buf=await asset('pointcloud.ply');const end=buf.indexOf('end_header\n')+11;
  const header=buf.subarray(0,end).toString();assert.match(header,/element vertex 518786/);assert.match(header,/property float x/);assert.equal(buf.length-end,518786*15);
  for(let i=0;i<518786;i+=101)for(let axis=0;axis<3;axis++)assert.ok(Number.isFinite(buf.readFloatLE(end+i*15+axis*4)));
});
test('GLB has complete geometry matching the asset report',async()=>{
  const b=await asset('reconstruction_mesh.glb');assert.equal(b.readUInt32LE(0),0x46546c67);assert.equal(b.readUInt32LE(8),b.length);
  const doc=JSON.parse(b.subarray(20,20+b.readUInt32LE(12)).toString());let vertices=0,triangles=0;
  for(const mesh of doc.meshes)for(const p of mesh.primitives){vertices+=doc.accessors[p.attributes.POSITION].count;triangles+=(p.indices===undefined?doc.accessors[p.attributes.POSITION].count:doc.accessors[p.indices].count)/3;}
  assert.equal(vertices,224316);assert.equal(triangles,450000);
});
test('each saved thumbnail and comparison is included',async()=>{
  const scene=JSON.parse(await asset('scene.json'));assert.equal(scene.thumbnail_frames.length,19);assert.equal(scene.inference_available,false);
  for(const n of scene.thumbnail_frames)assert.ok((await asset(`frames/frame_${String(n).padStart(5,'0')}.png`)).length>100);
  for(const w of scene.windows)assert.ok((await asset(`photoreal_preview_${w.id}.png`)).length>100);
});
test('labels and timing format are escaped and stable',()=>{assert.equal(escapeHTML('<script>"&'),'&lt;script&gt;&quot;&amp;');assert.equal(formatTime(333.935),'5m 34s');});
