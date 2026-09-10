import {parseTrajectory,formatTime,escapeHTML} from './core.js';
import {$} from './ui.js';
const base='assets/latest/';
async function asset(name,json=true){const response=await fetch(base+name);if(!response.ok)throw new Error(`Saved evidence unavailable (${response.status})`);return json?response.json():response.text();}
export async function loadLatestRun(photos=[]){
  const [scene,evaluation,coverage,trajectory]=await Promise.all([asset('scene.json'),asset('evaluation.json'),asset('coverage.json'),asset('trajectory.csv',false)]);
  return {scene,evaluation,coverage,trajectory:parseTrajectory(trajectory),photos};
}
export function populateLatestReports({scene,evaluation,photos}){
  const aligned=evaluation.sim3_aligned_trajectory_error,absolute=evaluation.absolute_trajectory_error;
  const stageTime=Object.values(scene.timings).reduce((a,b)=>a+b,0);
  $('#scene-subtitle').textContent='Single pass · 60 seconds · 90 selected views';
  $('#rmse').textContent=aligned.rmse_m.toFixed(3);$('#point-count').textContent=scene.point_count.toLocaleString();$('#triangle-count').textContent=scene.mesh_triangles.toLocaleString();
  $('#crs').textContent='Visual frame';$('#runtime').textContent=formatTime(stageTime);
  $('#runtime').closest('.inspector-section').querySelector('.micro').textContent='Latest 90-view experiment. Sum of measured SfM, inference, fusion and texturing stages; not end-to-end latency.';
  $('#runtime').parentElement.querySelector('dt').textContent='Measured stages';
  $('#coordinates').textContent='LOCAL VISUAL FRAME · ESTIMATED METRES';
  $('#frame-summary').textContent=`${scene.thumbnail_frames.length} previews / ${scene.keyframes} reconstructed views`;
  $('.timeline-controls>.micro').textContent='Watch flight video';
  const metrics=[['Absolute positioning',absolute.rmse_m.toFixed(3)+' m','GPS-georeferenced camera error · lower is better','warn'],['Camera-path shape',aligned.rmse_m.toFixed(3)+' m','After reference-based Sim(3) alignment',''],['Surface accuracy','Not measured','No independent surveyed surface reference','warn'],['Reconstructed surface',scene.mesh_triangles.toLocaleString(),'Triangles · coverage remains partial','']];
  $('#validation-metrics').innerHTML=metrics.map(([label,value,note,style])=>`<div class="report-metric ${style}"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
  $('#error-details').innerHTML=[['Evaluated camera positions','90'],['Exact reference matches','3'],['Interpolated reference matches','87 · gaps ≤30 source frames'],['Rigid-aligned trajectory RMSE',evaluation.se3_aligned_trajectory_error.rmse_m.toFixed(3)+' m'],['Raw onboard GPS RMSE',evaluation.raw_gps_absolute_error.rmse_m.toFixed(3)+' m'],['Ground truth in construction','No — evaluation only'],['Independent surface accuracy','Not measured']].map(([label,value])=>`<div><dt>${label}</dt><dd>${value}</dd></div>`).join('');
  $('#appearance-report').innerHTML=photos.map(p=>`<div class="appearance-result"><img loading="lazy" src="assets/photoreal_preview_${p.id}.png" alt="Archived observed versus Gaussian-render comparison"><div><h3>${escapeHTML(p.name)}</h3><p>Earlier run only<br>PSNR ${p.psnr.toFixed(2)} dB · SSIM ${p.ssim.toFixed(3)}</p></div></div>`).join('');
  const steps=[['Read the flight','Decode video, align GPS timestamps and use camera calibration.','1080p source images'],['Solve camera motion','Calibrated feature matching and bundle adjustment connect the 90 views.',formatTime(scene.timings.visual_bundle_adjustment)],['Predict depth','Pretrained MapAnything reconstructs overlapping windows using visual camera anchors.',formatTime(scene.timings.depth_and_export)],['Fuse the surface','Cross-view depth checks and TSDF integration form the observed surface.',formatTime(scene.timings.surface_fusion)],['Project real textures','Choose depth-tested source cameras; retain vertex colors where no source projection is valid.',formatTime(scene.timings.source_texturing)],['Locate and inspect','GPS-only alignment produces provisional UTM exports. Reference poses are used only afterwards for evaluation.','Separate evaluation']];
  $('#pipeline-steps').innerHTML=steps.map(([title,detail,time],i)=>`<article class="pipeline-step"><span class="step-no">0${i+1}</span><h2>${title}</h2><p>${detail}</p><span>${time}</span></article>`).join('');
}
