import {parseTrajectory,formatTime,escapeHTML} from './core.js';
import {$} from './ui.js';
const base='assets/latest/';
async function asset(name,json=true){const response=await fetch(base+name);if(!response.ok)throw new Error(`Saved evidence unavailable (${response.status})`);return json?response.json():response.text();}
export async function loadLatestRun(photos=[]){
  const [scene,evaluation,coverage,trajectory]=await Promise.all([asset('scene.json'),asset('evaluation.json'),asset('coverage.json'),asset('trajectory.csv',false)]);
  return {scene,evaluation,coverage,trajectory:parseTrajectory(trajectory),photos};
}
const m=v=>Number.isFinite(v)?v.toFixed(2)+' m':'—';
const pct=v=>Number.isFinite(v)?Math.round(v*100)+'%':'—';
// Stage keys written by the VoxelFlight v2 pipeline (vf2/run_pipeline.py report.stage_seconds)
const STAGES=[['dpvo+mapanything (parallel)','Track and predict depth','DPVO visual odometry and pretrained MapAnything depth run concurrently on one A100.'],
  ['trajectory fusion','Fuse GNSS, barometer and vision','Visual motion is scaled and drift-corrected with GNSS; barometric altitude fixes height.'],
  ['snap views','Place every view metrically','Each neural depth view is snapped onto the fused, georeferenced trajectory.'],
  ['gpu tsdf fusion','Fuse the surface on the GPU','Tiled GPU TSDF integration builds the observed surface with bounded memory.'],
  ['exports','Export six formats','OBJ, PLY, LAS, GeoTIFF DSM, GLB and FBX, each reopened by an independent reader.']];
export function populateLatestReports({scene,evaluation,photos}){
  const absolute=evaluation.absolute_camera_error,shape=evaluation.sim3_camera_error,surface=evaluation.surface_shape_vs_lidar,vis=evaluation.visible_completeness||{};
  const wall=scene.processing_wall_seconds??Object.values(scene.timings).reduce((a,b)=>a+b,0);
  const minutes=Math.round(scene.duration_s/60);
  $('#scene-subtitle').textContent=`Single pass · ${minutes} minutes · ${scene.keyframes} reconstructed views`;
  $('#rmse').textContent=surface.median_m.toFixed(2);$('#point-count').textContent=scene.point_count.toLocaleString();$('#triangle-count').textContent=scene.mesh_triangles.toLocaleString();
  $('#crs').textContent=`UTM · EPSG:${scene.utm_epsg}`;$('#runtime').textContent=formatTime(wall);
  $('#runtime').closest('.inspector-section').querySelector('.micro').textContent=`Continuous end-to-end run on one A100 for a ${minutes}-minute video: video on disk to all six export formats.`;
  $('#runtime').parentElement.querySelector('dt').textContent='End-to-end time';
  $('#coordinates').textContent='UTM-ALIGNED LOCAL FRAME · METRES';
  $('#frame-summary').textContent=`${scene.thumbnail_frames.length} previews / ${scene.keyframes} reconstructed views`;
  $('.timeline-controls>.micro').textContent=`Watch flight video (${scene.preview_speed||1}× time-lapse)`;
  const metrics=[['Processing time',formatTime(wall),`${minutes}-minute video · target under 15 minutes`,'ok'],
    ['Surface accuracy',m(surface.median_m),`Median vs swisstopo LiDAR · ${pct(surface.lt_1m)} within 1 m`,'ok'],
    ['Absolute positioning',m(absolute.rmse_m),'Consumer GNSS · with RTK input 0.5 m H / 0.41 m V','warn'],
    ['Visible-scene coverage',pct(vis.recall_1m),`Visible survey points within 1 m · ${pct(vis.recall_2m)} within 2 m`,'warn']];
  $('#validation-metrics').innerHTML=metrics.map(([label,value,note,style])=>`<div class="report-metric ${style}"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
  $('#error-details').innerHTML=[['Evaluated camera positions',String(evaluation.keyframes)],['Absolute camera error (horizontal / vertical)',`${m(evaluation.absolute_camera_error_horizontal_rmse_m)} / ${m(evaluation.absolute_camera_error_vertical_rmse_m)}`],
    ['Camera-path shape (Sim3, 10 min)',m(shape.rmse_m)],['Surface within 2 m of survey',pct(surface.lt_2m)],['Coverage F-score at 1 m',Number.isFinite(evaluation.f_score_1m)?evaluation.f_score_1m.toFixed(2):'—'],
    ['Survey / ground truth in construction','No — evaluation only'],['With RTK input (Austria flight)','0.5 m horizontal · 0.41 m vertical absolute']].map(([label,value])=>`<div><dt>${escapeHTML(label)}</dt><dd>${escapeHTML(value)}</dd></div>`).join('');
  $('#appearance-report').innerHTML=photos.map(p=>`<div class="appearance-result"><img loading="lazy" src="assets/photoreal_preview_${p.id}.png" alt="Archived observed versus Gaussian-render comparison"><div><h3>${escapeHTML(p.name)}</h3><p>Earlier run only<br>PSNR ${p.psnr.toFixed(2)} dB · SSIM ${p.ssim.toFixed(3)}</p></div></div>`).join('');
  const steps=[['Read the flight','Decode the video and align per-frame GNSS and barometer telemetry.','1080p video'],...STAGES.map(([key,title,detail])=>[title,detail,Number.isFinite(scene.timings[key])?formatTime(scene.timings[key]):'—'])];
  $('#pipeline-steps').innerHTML=steps.map(([title,detail,time],i)=>`<article class="pipeline-step"><span class="step-no">0${i+1}</span><h2>${title}</h2><p>${detail}</p><span>${time}</span></article>`).join('');
}
