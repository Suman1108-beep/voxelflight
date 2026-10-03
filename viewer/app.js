import {showTransition,hideTransition,motionDisabled} from './pixel.js';
import {createViewNavigation,normalizeView} from './navigation.js';
import {parseTrajectory,saveFile,escapeHTML} from './core.js';
import {$,icon,hydrateIcons,toast,status,openDialog,setupDialogs,drawCoverage,populateReports} from './ui.js';
import {setupInputs} from './inputs.js';
import {loadLatestRun,populateLatestReports} from './latest-run.js';

let viewer,viewerPromise,run,frameIndex=0,mode='video',playTimer,ready=false,imported=false,windowIndex=0;
const all=s=>[...document.querySelectorAll(s)];
const imagePath=frame=>`assets/latest/frames/frame_${String(frame).padStart(5,'0')}.jpg`;
const inspectorDialog=document.createElement('dialog');
inspectorDialog.id='inspector-dialog';inspectorDialog.className='inspector-drawer';inspectorDialog.setAttribute('aria-label','Saved reconstruction details');
inspectorDialog.append($('.inspector'));document.body.append(inspectorDialog);
$('.inspector-heading').innerHTML='<h2>RUN DETAILS</h2><button class="icon-button close-dialog" aria-label="Close run details"><span data-icon="close"></span></button>';
hydrateIcons();setupDialogs();setupInputs();
$('#open-details').addEventListener('click',()=>openDialog('#inspector-dialog'));
function stop(){clearInterval(playTimer);playTimer=null;$('#play-button').innerHTML=icon('play');$('#play-button').setAttribute('aria-label','Play sampled frames');}
function renderPage(name){
  if(inspectorDialog.open)inspectorDialog.close();
  document.body.dataset.view=name;
  all('.page').forEach(el=>el.hidden=el.id!==name);
  all('[data-page]').forEach(b=>{b.classList.toggle('active',b.dataset.page===name);if(b.dataset.page===name)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  if(name!=='workspace'){stop();$('#flight-video').pause();}else viewer?.resize();
}
const navigation=createViewNavigation({render:renderPage,motionDisabled,
  begin:view=>showTransition(view.title,view.detail,{destination:view.id,inApp:true}),
  finish:name=>{hideTransition();const heading=$(`#${name} h1`);if(heading&&!document.hidden){heading.tabIndex=-1;heading.focus({preventScroll:true});}}
});
function page(name,{push=false,animate=true}={}){name=normalizeView(name);if(push&&location.hash!==`#${name}`)history.pushState(null,'',`#${name}`);navigation.go(name,{animate});}
all('[data-page]').forEach(b=>b.addEventListener('click',()=>page(b.dataset.page,{push:true})));
addEventListener('hashchange',()=>page(location.hash.slice(1)));
document.addEventListener('sih3d:transition-dismissed',()=>navigation.settle());
document.addEventListener('sih3d:motionchange',event=>{if(event.detail.paused)navigation.settle();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)navigation.settle();});
page(location.hash.slice(1),{animate:false});
for(const id of ['#explain-metric','#view-validation'])$(id).addEventListener('click',()=>page('validation',{push:true}));
for(const [button,dialog] of [['#open-import','#import-dialog'],['#open-input','#input-dialog'],['#pipeline-input','#input-dialog'],['#open-export','#export-dialog']])$(button).addEventListener('click',()=>openDialog(dialog));

function setMode(value){
  if(value==='video'&&imported)return toast('Restore the saved flight to watch its video.');
  if(value==='appearance'&&(!run||imported))return toast('Restore the saved run to view its recorded renders.');
  mode=value;all('[data-mode]').forEach(b=>{const active=b.dataset.mode===mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  $('#viewport').hidden=mode!=='geometry';$('#appearance-view').hidden=mode!=='appearance';
  $('#video-view').hidden=mode!=='video';
  if(mode!=='video')$('#flight-video').pause();
  $('#geometry-settings').hidden=mode!=='geometry';$('#appearance-settings').hidden=mode!=='appearance';
  document.body.dataset.mode=mode;
  $('#stage-mode').textContent=mode==='video'?'Watch the flight':mode==='geometry'?'Explore the reconstruction':'Earlier appearance experiment';
  $('#result-explanation').textContent=imported?'Your local model. Scale, coverage and accuracy have not been verified.':mode==='video'?'The real 10-minute flight (4× time-lapse preview), encoded from the official camera images. Next, explore the 3D surface reconstructed from this sequence.':mode==='geometry'?'Maximum-coverage surface from 720 frames (4 crops each) of a 10-minute flight. Unseen regions remain open; shape is about 0.7 m median against survey LiDAR.':'Archived Gaussian-render comparison from the earlier 180-view run. These image scores do not describe the latest 3D mesh.';
  $('#scout-hint').innerHTML=mode==='video'?'Play the real flight.<br>Then explore in 3D.':mode==='geometry'?'Drag to orbit.<br>Scroll to explore.':'Slide to compare.<br>Earlier experiment.';
  $('#reset-view').disabled=mode!=='geometry'||!ready;
  if(mode==='geometry'){viewer?.resize();if(run&&!ready)ensureViewer();}else if(mode==='appearance')selectWindow(windowIndex);
}
all('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
function selectWindow(index){
  if(!run)return;windowIndex=index;const w=run.photos[index];
  $('#appearance-window-label').textContent=`${w.name} / ${w.id.toUpperCase()}`;
  for(const id of ['#render-image','#source-image'])$(id).style.backgroundImage=`url("assets/photoreal_preview_${w.id}.png")`;
  $('#appearance-stats').textContent=`PSNR ${w.psnr.toFixed(2)} dB · SSIM ${w.ssim.toFixed(3)} · LPIPS ${w.lpips.toFixed(3)}`;
  all('[data-window]').forEach(b=>{const active=Number(b.dataset.window)===index;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
}
$('#compare-slider').addEventListener('input',e=>{$('#source-image').style.clipPath=`inset(0 ${100-Number(e.target.value)}% 0 0)`;$('#comparison-line').style.left=e.target.value+'%';});
function selectFrame(index){
  if(!run||imported)return;frameIndex=(index+run.scene.thumbnail_frames.length)%run.scene.thumbnail_frames.length;
  const frame=run.scene.thumbnail_frames[frameIndex];
  all('.frame-thumb').forEach(b=>{const active=Number(b.dataset.frame)===frame;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  $('#source-preview').src=imagePath(frame);$('#source-preview').alt=`Observed drone frame ${frame}`;
  $('#source-index').textContent=String(frame).padStart(3,'0');$('#frame-counter').innerHTML=`${String(frame+1).padStart(3,'0')} <em>/ ${run.scene.keyframes}</em>`;
  viewer?.setFrame(frame);drawCoverage(run.coverage,run.trajectory,frame);
}
$('#previous-button').addEventListener('click',()=>{stop();selectFrame(frameIndex-1);});
$('#next-button').addEventListener('click',()=>{stop();selectFrame(frameIndex+1);});
$('#play-button').addEventListener('click',()=>{if(!run||imported)return;setMode('video');const video=$('#flight-video');if(video.paused)video.play().catch(()=>toast('Use the video play control to start playback.'));else video.pause();});
$('#flight-video').addEventListener('timeupdate',()=>{if(!run)return;const t=$('#flight-video').currentTime;let index=0;for(let i=0;i<run.scene.thumbnail_frames.length;i++)if(run.scene.frame_times[run.scene.thumbnail_frames[i]]<=t)index=i;if(index!==frameIndex)selectFrame(index);});
$('#flight-video').addEventListener('play',()=>{$('#play-button').innerHTML=icon('pause');$('#play-button').setAttribute('aria-label','Pause flight video');});
$('#flight-video').addEventListener('pause',stop);
$('#flight-video').addEventListener('error',()=>{$('#video-feedback').textContent='Video could not load. Reload the page, or continue to the 3D result.';});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
function openFrame(){if(!run||imported)return;const frame=run.scene.thumbnail_frames[frameIndex];$('#frame-dialog-title').textContent=`Observed source · Frame ${frame}`;$('#frame-dialog-image').src=imagePath(frame);openDialog('#frame-dialog');}
$('#open-frame').addEventListener('click',openFrame);$('#source-preview-button').addEventListener('click',openFrame);
all('button[data-view]').forEach(b=>b.addEventListener('click',()=>{if(!ready)return;all('button[data-view]').forEach(p=>p.classList.toggle('active',p===b));viewer.fit(b.dataset.view);}));
$('#reset-view').addEventListener('click',()=>{if(ready){viewer.fit('overview');all('button[data-view]').forEach(p=>p.classList.toggle('active',p.dataset.view==='overview'));}});
for(const name of ['mesh','path','grid','wire'])$(`#${name}-toggle`).addEventListener('change',e=>viewer?.setLayer(name,e.target.checked));
$('#cloud-toggle').addEventListener('change',async e=>{const el=e.target;if(!ready){el.checked=false;return;}el.disabled=true;try{await viewer.toggleCloud(el.checked);viewer.setPointSize($('#point-size').value);}catch(error){el.checked=false;toast('Point cloud could not be opened: '+error.message);}finally{el.disabled=false;}});
$('#point-size').addEventListener('input',e=>{$('#point-size-value').value=e.target.value;viewer?.setPointSize(e.target.value);});
function measurement(value){
  $('#measure-label').hidden=true;$('#clear-measure').hidden=!value;
  $('#measure-button').classList.toggle('active',!!value?.pending);
  $('#measure-button').setAttribute('aria-pressed',String(!!value?.pending));
  if(!value){$('#measure-result').textContent='Pick two locations on the mesh.';return;}
  if(value.pending){$('#measure-result').textContent='First point set. Select the second.';return;}
  const text=`${value.distance.toFixed(2)} ${imported?'model units':'m'}`;
  $('#measure-result').textContent=text;$('#measure-label').textContent=text;$('#measure-label').hidden=false;
  status(imported?'Measured in file units; physical scale is unknown.':'Distance in metres. Model shape is about 0.7 m median against survey LiDAR; absolute placement follows GNSS.');
}
$('#measure-button').addEventListener('click',()=>{if(!ready)return toast('Wait for the 3D model to load.');if(!viewer.stats().triangles)return toast('Point picking requires a surface mesh.');setMode('geometry');const active=!viewer.measuring;viewer.setMeasuring(active);$('#measure-button').classList.toggle('active',active);$('#measure-button').setAttribute('aria-pressed',String(active));$('#measure-result').textContent=active?'Select the first point on the mesh.':'Pick two locations on the mesh.';});
$('#clear-measure').addEventListener('click',()=>{viewer?.setMeasuring(false);viewer?.clearMeasurement();});
$('#fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();else toast('Full screen is not supported by this browser.');}catch{toast('Full screen was not available.');}});
$('#snapshot').addEventListener('click',()=>{if(mode==='appearance'){const a=document.createElement('a');a.href=`assets/photoreal_preview_${run.photos[windowIndex].id}.png`;a.download='observed-and-reconstructed.png';a.click();toast('Saved the observed / reconstructed comparison.');}else if(ready){const a=document.createElement('a');a.href=viewer.snapshot();a.download='reconstruction-view.png';a.click();}else toast('Wait for the 3D model to load.');});
all('[data-export]').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;status('Preparing geometry export…');try{if(!await ensureViewer())throw new Error('3D export is unavailable on this device. Download the original GLB or PLY instead.');const result=await viewer.export(b.dataset.export);saveFile(result.data,result.name,result.type);toast('Geometry export ready.');}catch(error){toast(error.message);}finally{b.disabled=false;status(imported?'Local model ready':'Saved reconstruction ready');}}));
$('#restore-scene').addEventListener('click',()=>location.reload());
$('#model-file').addEventListener('change',async e=>{
  const file=e.target.files?.[0];if(!file)return;
  if(!run){$('#import-feedback').textContent='The workspace is still opening. Please try again in a moment.';return;}
  if(file.size>100*1024*1024||!(/\.(glb|ply)$/i.test(file.name))){$('#import-feedback').textContent='Choose a GLB or PLY file under 100 MB.';return;}
  $('#import-feedback').textContent='Opening your local model…';e.target.disabled=true;
  try{
    if(!await ensureViewer())throw new Error('3D rendering is unavailable on this device.');
    const stats=await viewer.importFile(file);imported=true;stop();setMode('geometry');
    $('#scene-title').textContent=file.name;$('#stage-scene-name').textContent=file.name;$('#scene-subtitle').textContent='Local file · not uploaded';
    $('#quality-tag').textContent='Unvalidated import';$('#view-tag-text').textContent='LOCAL MODEL';$('#coordinates').textContent='FILE COORDINATES · UNITS UNSPECIFIED';
    $('#scale-bar b').textContent='10 model units at focus';$('#rmse').textContent='—';$('#point-count').textContent=stats.vertices.toLocaleString();$('#triangle-count').textContent=stats.triangles.toLocaleString();$('#crs').textContent='Unspecified';$('#runtime').textContent='—';
    $('.scene-meta').hidden=true;$('.status-chip').textContent='Local model';$('.metric-tag').textContent='No evaluation for this file';$('.inspector-metric .micro').textContent='Saved-run scores do not apply to imported geometry.';
    $('#runtime').closest('.inspector-section').querySelector('.micro').textContent='Coordinates and units are taken from the file; georeferencing is not verified.';
    $('#crs').parentElement.nextElementSibling.querySelector('dd').textContent='Not available';
    $('.timeline').hidden=true;$('#coverage-map').closest('section').hidden=true;$('#source-preview').closest('section').hidden=true;
    $('#path-toggle').checked=false;$('#path-toggle').disabled=true;$('#mesh-toggle').checked=stats.triangles>0;$('#cloud-toggle').checked=stats.triangles===0;$('#wire-toggle').checked=false;
    all('[data-mode="appearance"]').forEach(b=>b.disabled=true);
    all('#export-grid a').forEach(a=>a.hidden=true);$('#export-note').textContent='Exporting your imported model. Coordinates and units are unchanged; saved-run GIS products do not apply.';
    $('#export-dialog .dialog-intro').textContent='Save the geometry currently open on this device.';
    $('#export-dialog .dialog-footer').hidden=true;$('#import-dialog').close();$('#import-feedback').textContent='';
    page('workspace');location.hash='workspace';status('Local model ready. Coordinates and accuracy are unverified.');
  }catch(error){$('#import-feedback').textContent='Could not open model: '+error.message;}finally{e.target.disabled=false;e.target.value='';}
});

async function json(path){const r=await fetch(`assets/${path}.json`);if(!r.ok)throw new Error(`Missing ${path} (${r.status})`);return r.json();}
async function ensureViewer(){
  if(ready)return true;if(viewerPromise)return viewerPromise;if(!run)return false;
  viewerPromise=(async()=>{try{
    const {SceneViewer}=await import('./scene.js');
    viewer=new SceneViewer($('#scene'),{onStatus:message=>{status(message);$('#loading-detail').textContent=message;},onPick:measurement,onHover:p=>{$('#coordinates').textContent=`X ${p[0].toFixed(2)} · Y ${p[1].toFixed(2)} · Z ${p[2].toFixed(2)} ${imported?'units':'m'}`;}});
    await viewer.loadReconstruction('assets/latest/');viewer.setTrajectory(run.trajectory);viewer.fit('facade');ready=true;$('#loading').hidden=true;
    all('button[data-view]').forEach(button=>button.classList.toggle('active',button.dataset.view==='facade'));
    for(const name of ['mesh','path','grid','wire'])viewer.setLayer(name,$(`#${name}-toggle`).checked);
    viewer.setFrame(run.scene.thumbnail_frames[frameIndex]);$('#reset-view').disabled=mode!=='geometry';return true;
  }catch(error){$('#loading').innerHTML='<strong>3D view unavailable</strong><span>Use Compare images to inspect saved renders, or Download for the original models.</span>';status('3D view unavailable on this device. Saved images and downloads remain available.');console.error('3D initialization failed',error);return false;}})();
  return viewerPromise;
}
async function init(){
  const archived=await json('scene');
  const photos=await Promise.all(archived.windows.map(async w=>({...w,...await json(`photoreal_metrics_${w.id}`)})));
  run=await loadLatestRun(photos);const {scene}=run;populateLatestReports(run);
  $('#frame-strip').innerHTML=scene.thumbnail_frames.map(f=>`<button class="frame-thumb" data-frame="${f}" aria-label="Show observed frame ${f}"><img src="${imagePath(f)}" alt="" loading="lazy" width="96" height="72"><span>${String(f).padStart(3,'0')}</span></button>`).join('');
  all('.frame-thumb').forEach((b,i)=>b.addEventListener('click',()=>{stop();selectFrame(i);$('#flight-video').currentTime=scene.frame_times[Number(b.dataset.frame)];}));
  $('#window-list').innerHTML=photos.map((w,i)=>`<button class="window-item" data-window="${i}"><span>${escapeHTML(w.name)}</span><small>${w.id.toUpperCase()} · Frames ${w.range.join('–')}</small></button>`).join('');
  all('[data-window]').forEach(b=>b.addEventListener('click',()=>selectWindow(Number(b.dataset.window))));selectWindow(0);selectFrame(0);
  setMode(mode);status('Watch the real flight, then explore its reconstructed surface.');
}
init().catch(error=>{$('#loading').innerHTML='<strong>Saved assets could not be loaded</strong><span>Check your connection and reload the page.</span>';status(error.message);console.error(error);});
