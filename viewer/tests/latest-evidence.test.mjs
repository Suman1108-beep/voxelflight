import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,stat} from 'node:fs/promises';
import {parseTrajectory} from '../core.js';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
test('latest metrics come from independent references and keep absolute and shape accuracy separate',async()=>{
 const e=JSON.parse(await read('assets/latest/evaluation.json'));
 assert.equal(e.protocol.ground_truth_used_in_construction,false);assert.equal(e.protocol.survey_or_lidar_used_in_construction,false);
 assert.ok(e.surface_shape_vs_lidar.median_m<1,'model shape is sub-metre against survey LiDAR');
 assert.ok(e.absolute_camera_error.rmse_m>1,'consumer-GNSS absolute error is reported, not hidden');
 assert.ok(e.visible_completeness.recall_1m>0&&e.visible_completeness.recall_1m<1);
 const script=await read('latest-run.js');
 assert.match(script,/Consumer GNSS/);assert.doesNotMatch(script,/'Not measured'/);
});
test('latest flight video, trajectory and selected frames share one real 10-minute sequence',async()=>{
 const s=JSON.parse(await read('assets/latest/scene.json'));
 const trajectory=parseTrajectory(await read('assets/latest/trajectory.csv'));
 assert.equal(trajectory.length,s.keyframes);assert.ok(s.keyframes>=360);assert.ok(s.duration_s>=595);
 assert.equal(s.frame_times.length,s.keyframes);assert.equal(s.frame_times[0],0);
 assert.ok(s.frame_times.at(-1)<=s.duration_s/s.preview_speed+1,'frame times address the time-lapse preview');
 for(const f of s.thumbnail_frames)await stat(new URL(`../assets/latest/frames/frame_${String(f).padStart(5,'0')}.jpg`,import.meta.url));
 assert.match(await read('index.html'),/<video id="flight-video" controls playsinline preload="metadata"/);
 assert.match(await read('app.js'),/mode='video'/);
});
test('the saved run is a continuous sub-15-minute benchmark and every asset fits the hosting limit',async()=>{
 const s=JSON.parse(await read('assets/latest/scene.json')),m=JSON.parse(await read('assets/latest/manifest.json'));
 assert.ok((s.mode==='quality'?s.fast_mode_seconds:s.processing_wall_seconds)<900,'a measured end-to-end run meets the 15-minute target');assert.ok(s.mesh_triangles>s.web_mesh_triangles);
 for(const name of ['reconstruction_mesh.glb','reconstruction_mesh.fbx','pointcloud.ply','reconstruction_utm.las','surface_model.tif'])assert.ok(m[name],'manifest lists '+name);
 for(const [name,file] of Object.entries(m)){const data=await readFile(new URL('../assets/latest/'+name,import.meta.url));assert.equal(data.length,file.bytes);assert.ok(data.length<25*1024*1024);}
});
test('public root is the 3D explorer; sign-in and the full workspace stay available',async()=>{
 assert.match(await read('firebase/build.mjs'),/\['index.html','explore.html'\]/);
 assert.match(await read('firebase/build.mjs'),/\['login.html','login.html'\]/);
 assert.match(await read('firebase/build.mjs'),/\['workspace.html','index.html'\]/);
 assert.match(await read('firebase/auth.js'),/get\('return_to'\)\|\|'\/workspace'/);
 assert.match(await read('index.html'),/Layers, camera &amp; measurements|Layers, camera & measurements/);
 assert.match(await read('index.html'),/Earlier appearance experiment/);
 assert.doesNotMatch(await read('index.html'),/0\.810 m|5m34 recorded|Recorded run: 180|FBX is not available/);
});
test('the explorer syncs exact camera poses with the video and colours the model by survey error',async()=>{
 const cams=JSON.parse(await read('assets/latest/cameras.json')),s=JSON.parse(await read('assets/latest/scene.json'));
 assert.equal(cams.c2w.length,s.keyframes);assert.equal(cams.c2w[0].length,12);assert.equal(s.frame_times.length,s.keyframes);
 const errors=await readFile(new URL('../assets/latest/errors.bin',import.meta.url));assert.ok(errors.length>1000);
 const js=await read('explore.js');assert.match(js,/slerp/);assert.match(js,/errors\.bin/);assert.match(js,/evaluation-only alignment/);
 assert.match(await read('explore.html'),/Compare with real frame/);
});
test('the studio is the signed-in home: checks inputs locally, submits to the job API, opens local models',async()=>{
 const html=await read('studio.html'),js=await read('studio.js');
 assert.match(html,/data-page="studio"/);assert.match(html,/id="video"[^>]*type="file"/);assert.match(html,/id="telemetry"[^>]*\.srt/);
 assert.match(js,/api\('jobs',\{method:'POST'/);assert.match(js,/sessionHeaders/);assert.match(js,/importFile/);
 assert.match(await read('firebase/build.mjs'),/\['studio.html','studio.html'\]/);
 assert.match(await read('firebase.json'),/"source": "\/workspace", "destination": "\/studio"/);
});
test('operators find the same navigation and a step-by-step guide on every working page',async()=>{
 const explore=await read('explore.html'),studio=await read('studio.html'),js=await read('explore.js');
 for(const html of [explore,studio])for(const label of ['Studio','Guide','Evidence','How it works'])assert.match(html,new RegExp(`>${label}<`));
 assert.match(explore,/id="tab-guide"/);assert.match(explore,/Keep the GPS log/);assert.match(explore,/GeoTIFF surface model/);
 assert.match(js,/TABS=\['guide','evidence','how'\]/);assert.match(studio,/Before you upload/);assert.match(explore,/class="signin" data-account-link/);
});
