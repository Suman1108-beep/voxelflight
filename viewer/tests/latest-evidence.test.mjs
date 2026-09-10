import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,stat} from 'node:fs/promises';
import {parseTrajectory} from '../core.js';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
test('latest metrics separate reference alignment from absolute and surface accuracy',async()=>{
 const e=JSON.parse(await read('assets/latest/evaluation.json'));
 assert.equal(e.frames,90);assert.equal(e.protocol.ground_truth_used_in_construction,false);
 assert.ok(e.sim3_aligned_trajectory_error.rmse_m<.4);assert.ok(e.absolute_trajectory_error.rmse_m>8);
 assert.equal(e.surface_rmse_m,null);assert.equal(e.sih_spatial_accuracy_verified,false);
 const script=await read('latest-run.js');assert.match(script,/Surface accuracy','Not measured/);
});
test('latest flight video, trajectory and selected frames share one real sequence',async()=>{
 const s=JSON.parse(await read('assets/latest/scene.json'));
 const trajectory=parseTrajectory(await read('assets/latest/trajectory.csv'));
 assert.equal(trajectory.length,s.keyframes);assert.equal(s.keyframes,90);
 assert.equal(s.frame_times.length,90);assert.equal(s.frame_times[0],0);assert.ok(s.frame_times.at(-1)<60);
 for(const f of s.thumbnail_frames)await stat(new URL(`../assets/latest/frames/frame_${String(f).padStart(5,'0')}.jpg`,import.meta.url));
 assert.match(await read('index.html'),/<video id="flight-video" controls playsinline preload="metadata"/);
 assert.match(await read('app.js'),/mode='video'/);
});
test('web mesh preserves source geometry and all new assets fit the hosting limit',async()=>{
 const s=JSON.parse(await read('assets/latest/scene.json')),m=JSON.parse(await read('assets/latest/manifest.json'));
 assert.equal(s.web_assets.local.geometry_changed,false);assert.equal(s.web_assets.georeferenced.geometry_changed,false);
 assert.equal(s.web_assets.local.geometry_buffer_sha256,s.web_assets.georeferenced.geometry_buffer_sha256);
 assert.equal(s.mesh_triangles,646019);
 for(const [name,file] of Object.entries(m)){const data=await readFile(new URL('../assets/latest/'+name,import.meta.url));assert.equal(data.length,file.bytes);assert.ok(data.length<25*1024*1024);}
});
test('public root is the login page, with the real workspace preserved after sign-in',async()=>{
 assert.match(await read('firebase/build.mjs'),/\['index.html','login.html'\]/);
 assert.match(await read('firebase/build.mjs'),/\['workspace.html','index.html'\]/);
 assert.match(await read('firebase/auth.js'),/get\('return_to'\)\|\|'\/workspace'/);
 assert.match(await read('index.html'),/Layers, camera &amp; measurements|Layers, camera & measurements/);
 assert.match(await read('index.html'),/Earlier appearance experiment/);
 assert.doesNotMatch(await read('index.html'),/0\.810 m|5m34 recorded|Recorded run: 180/);
});
