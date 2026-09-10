import './pixel.js';
import {engineMessage} from './engine-copy.js';
import {validateVideoFile} from './video-input.js';
import {escapeHTML,formatTime} from './core.js';
import {sessionHeaders} from './session-bridge.js';
const $=s=>document.querySelector(s),all=s=>[...document.querySelectorAll(s)];
const hosted=document.querySelector('meta[name="hosting-platform"]')?.content==='firebase';
const local=!hosted&&['127.0.0.1','localhost'].includes(location.hostname);
let apiOrigin='',signedIn=false,connecting=false;const privateBlobs=[];
let session='',health,activeJob=null,loadedJob=null,viewer,polling=false,submitting=false,loading=false;
const terminal=new Set(['complete','failed','cancelled','interrupted']);
const progressRunner=document.createElement('div');progressRunner.className='runner';progressRunner.setAttribute('aria-hidden','true');progressRunner.hidden=true;$('#engine-progress').prepend(progressRunner);
const resultRunner=document.createElement('div');resultRunner.className='runner';resultRunner.setAttribute('aria-hidden','true');resultRunner.hidden=true;$('#engine-empty').prepend(resultRunner);
const status=text=>$('#engine-status').textContent=text;
function flow(step){document.body.dataset.flow=step;all('[data-step]').forEach(item=>{if(item.dataset.step===step)item.setAttribute('aria-current','step');else item.removeAttribute('aria-current');});}
$('#engine-video').setAttribute('aria-describedby','engine-feedback engine-privacy');
for(const select of all('.engine-options select'))select.addEventListener('change',()=>{
  $('.advanced-processing summary span').textContent=`${$('select[name="max_frames"]').value} views · ${$('select[name="size"]').value} px`;
});
function processingView(job){
  const stopped=terminal.has(job.state);if(job.state==='complete')return;
  $('#engine-empty').hidden=false;$('#engine-result').hidden=true;$('#engine-layers').hidden=true;
  $('#sample-contact-sheet').hidden=true;$('#sample-caption').hidden=true;$('#open-mac-sample').hidden=!stopped;
  resultRunner.hidden=stopped;$('#empty-eyebrow').textContent=stopped?'RUN ENDED':'RECONSTRUCTION IN PROGRESS';
  $('#engine-empty h2').textContent=stopped?'This run did not complete.':'Reconstructing your flight.';
  $('#engine-empty p').textContent=engineMessage(job.progress?.detail??job.message??(stopped?'Choose another video or explore the saved sample.':'The connected engine is preparing your footage.'));
  $('#result-title').textContent=job.name??'Current reconstruction';
}
async function api(path,options={}){const headers=hosted?(path==='health'?{}:await sessionHeaders()):(options.method?{'X-SIH3D-Session':session}:{});const response=await fetch(`${apiOrigin}/api/${path}`,{...options,credentials:hosted?'omit':'same-origin',signal:AbortSignal.timeout(options.method?180000:12000),headers:{...options.headers,...headers}});if(!response.ok){let message=`Request failed (${response.status})`;try{message=(await response.json()).detail??message;}catch{}throw new Error(engineMessage(message));}return response.json();}
function inputError(){const file=$('#engine-video').files?.[0];return hosted&&file?.size>60*1024**2?'Choose a video smaller than 60 MiB.':validateVideoFile(file);}
function enabled(){const unavailable=!health?.ready||(hosted?!signedIn||!health.queue_available:health.busy)||submitting;$('#engine-start').disabled=unavailable||!!inputError();$('#engine-start').textContent=hosted&&!signedIn?'Sign in to reconstruct':health?.ready?(submitting?'Uploading footage…':health.busy?(hosted&&health.queue_available?'Join processing queue':'Reconstruction in progress'):'Start reconstruction'):'Processing unavailable';all('#reconstruction-form input,#reconstruction-form select').forEach(el=>el.disabled=submitting);}
$('#engine-video').addEventListener('change',()=>{const file=$('#engine-video').files?.[0];const error=inputError();$('#engine-feedback').textContent=file?(error||`${file.name} selected. ${hosted?'Submit when the processing service is ready.':local?'Ready to reconstruct when the engine is connected.':'Open the processing workspace to start a run.'}`):'';$('#engine-video').setAttribute('aria-invalid',String(!!file&&!!error));enabled();});
async function connect(){
  if(!local&&!hosted)return offline('Engine connection required');
  if(connecting)return;connecting=true;
  try{
    if(hosted){const settings=await fetch('/auth-settings.json',{cache:'no-store'}).then(r=>r.json());if(!settings.apiOrigin||new URL(settings.apiOrigin).protocol!=='https:')throw new Error('Processing not configured');apiOrigin=new URL(settings.apiOrigin).origin;try{await sessionHeaders();signedIn=true;}catch{signedIn=false;}$('#engine-signin').hidden=signedIn;}
    health=await api('health');session=health.session;$('#engine-health').textContent=health.ready?(health.busy?'Processing service · queue active':'Processing service · ready'):'Processing service · offline';$('#engine-health').dataset.state=health.ready?'ready':'offline';$('#engine-offline').hidden=health.ready;enabled();if(local||signedIn)await jobs();
  }catch(error){offline('Processing unavailable');$('#engine-feedback').textContent='The processing service could not be reached. Saved samples are still available.';}finally{connecting=false;}
}
function offline(message){health=null;$('#engine-health').textContent=hosted?'Processing service · offline':local?message:'Public demo · processing not connected';$('#engine-health').dataset.state='offline';$('#engine-offline').hidden=false;$('#engine-start').disabled=true;$('#engine-start').textContent='Processing unavailable';if(local){$('#engine-offline h2').textContent='This device’s engine is offline.';$('#engine-offline>p').textContent='Start or reconnect the installed engine to process a video. You can still explore saved samples without it.';}}
async function jobs(){
  const list=await api('jobs');$('#job-list').innerHTML=list.length?list.map(j=>`<button class="engine-job ${j.id===loadedJob?'active':''}" data-job="${escapeHTML(j.id)}"><strong>${escapeHTML(j.name)}</strong><span>${escapeHTML(j.state)} · ${j.resolution} px · ${j.max_frames} sampled frames</span></button>`).join(''):'<p class="micro">No runs yet. Start with a short overlapping video.</p>';
  all('[data-job]').forEach(button=>button.addEventListener('click',()=>selectJob(button.dataset.job)));
  if(!activeJob){const running=list.find(j=>['queued','running','cancelling','uploading'].includes(j.state));if(running){activeJob=running.id;showProgress(running);}}
  return list;
}
function showProgress(job){flow(job.state==='complete'?'explore':terminal.has(job.state)?'prepare':'process');processingView(job);$('#engine-progress').hidden=false;progressRunner.hidden=terminal.has(job.state);$('#progress-stage').textContent=job.state==='queued'?`Queued · position ${job.queue_position||1}`:job.progress?.stage??job.state;if(Number.isFinite(job.progress?.progress))$('#progress-bar').value=job.progress.progress;else $('#progress-bar').removeAttribute('value');$('#progress-detail').textContent=job.state==='queued'?'Your upload is saved. Processing starts when the previous run finishes.':engineMessage(job.progress?.detail??job.message??'Preparing processing engine');$('#cancel-run').hidden=terminal.has(job.state);$('#cancel-run').disabled=job.state==='cancelling';}
async function selectJob(id){try{const job=await api(`jobs/${id}`);if(job.state==='complete')await showResult(job);else{showProgress(job);if(!terminal.has(job.state))activeJob=id;else $('#engine-feedback').textContent=engineMessage(job.message??job.state);}}catch(error){$('#engine-feedback').textContent=error.message;}}
$('#reconstruction-form').addEventListener('submit',async event=>{
  event.preventDefault();if(submitting||!health?.ready||(hosted?(!signedIn||!health.queue_available):health.busy))return;
  const data=new FormData(event.currentTarget),video=data.get('video');
  const invalid=validateVideoFile(video)||inputError();if(invalid)return $('#engine-feedback').textContent=invalid;
  for(const name of ['telemetry','calibration'])if(!data.get(name)?.size)data.delete(name);
  submitting=true;enabled();$('#engine-feedback').textContent='Uploading footage securely to the processing service…';
  try{const job=await api('jobs',{method:'POST',body:data});activeJob=job.id;health.busy=true;showProgress(job);$('#engine-feedback').textContent=hosted?'Your run is saved. You can return to it from Your reconstructions.':'Reconstruction started. Keep the processing workstation awake.';await jobs();}
  catch(error){$('#engine-feedback').textContent=error.message;}finally{submitting=false;enabled();}
});
$('#cancel-run').addEventListener('click',async()=>{if(!activeJob)return;try{await api(`jobs/${activeJob}/cancel`,{method:'POST'});$('#cancel-run').disabled=true;$('#engine-feedback').textContent='Cancelling reconstruction…';}catch(error){$('#engine-feedback').textContent=error.message;}});
$('#refresh-jobs').addEventListener('click',connect);
$('#open-mac-sample').addEventListener('click',()=>showResult({id:'mac-proof',name:'Zurich · sample reconstruction',staticSample:true}));
async function poll(){if((!local&&!hosted)||polling||submitting||document.hidden)return;if(!health&&!activeJob)return connect();polling=true;try{health=await api('health');session=health.session;enabled();$('#engine-health').textContent=health.busy?'Processing service · queue active':health.ready?'Processing service · ready':'Processing service · offline';$('#engine-health').dataset.state=health.ready?'ready':'offline';$('#engine-offline').hidden=health.ready;if(activeJob){const job=await api(`jobs/${activeJob}`);showProgress(job);if(terminal.has(job.state)){activeJob=null;await jobs();if(job.state==='complete')await showResult(job);else $('#engine-feedback').textContent=engineMessage(job.message??job.state);}}}catch(error){offline('Processing disconnected');$('#engine-feedback').textContent='Processing status unavailable. Your submitted run may still be running. Try Refresh.';}finally{polling=false;}}
async function showResult(job){
  if(loading)return;loading=true;resultRunner.hidden=false;$('#sample-contact-sheet').hidden=true;$('#open-mac-sample').hidden=true;$('#sample-caption').hidden=true;$('#empty-eyebrow').textContent='OPENING SAVED GEOMETRY';
  for(const url of privateBlobs)URL.revokeObjectURL(url);privateBlobs.length=0;
  const privateResult=hosted&&!job.staticSample;
  const base=job.staticSample?'assets/mac-proof/':`${apiOrigin}/api/jobs/${job.id}/assets/`;
  const resultHeaders=privateResult?sessionHeaders:async()=>({});
  status('Loading this run’s actual geometry…');$('#engine-empty').hidden=false;$('#engine-empty h2').textContent='Loading reconstruction…';$('#engine-empty p').textContent='Reading the mesh and source data for this run.';$('#engine-result').hidden=true;$('#engine-layers').hidden=true;
  try{
    const response=await fetch(`${base}report.json`,{headers:await resultHeaders()});if(!response.ok)throw new Error('Run report unavailable');const report=await response.json();
    if(!viewer){const {SceneViewer}=await import('./scene.js');viewer=new SceneViewer($('#engine-canvas'),{onStatus:status,onPick:value=>{$('#engine-distance').textContent=value?.pending?'Pick the second point':value?.distance!==undefined?`${value.distance.toFixed(2)} m · scale unverified`:'Scale unverified';}});}
    await viewer.loadReconstruction(base,{getHeaders:resultHeaders});viewer.setTrajectory(report.trajectory);loadedJob=job.id;
    $('#engine-empty').hidden=true;$('#engine-layers').hidden=false;$('#engine-result').hidden=false;flow('explore');
    $('#result-title').textContent=job.name;$('#engine-mesh').checked=true;$('#engine-cloud').checked=false;$('#engine-path').checked=true;viewer.setLayer('path',true);
    $('#result-metrics').innerHTML=[['Reconstructed points',report.points.toLocaleString()],['Surface triangles',report.triangles.toLocaleString()],['Actual elapsed',formatTime(report.runtime_s)],['Surface accuracy','Unmeasured']].map(([label,value])=>`<div><span>${label}</span><strong>${escapeHTML(value)}</strong></div>`).join('');
    $('#result-warnings').innerHTML=[report.coordinate_system,...report.warnings].map(w=>`<li>${escapeHTML(engineMessage(w))}</li>`).join('');
    const labels={'reconstruction_mesh.glb':'GLB mesh','pointcloud.ply':'PLY cloud','reconstruction.obj':'OBJ mesh','camera_poses.npz':'Camera poses','trajectory.csv':'Trajectory','reconstruction_utm.las':'LAS / UTM','surface_model.tif':'GeoTIFF DSM','report.json':'Run report'};
    $('#result-downloads').innerHTML=report.artifacts.filter(name=>Object.hasOwn(labels,name)).map(name=>`<a class="button quiet" href="${base}${name}" download>${labels[name]} ↓</a>`).join('');
    if(privateResult){
      $('#result-frames').replaceChildren();
      all('#result-downloads a').forEach(link=>link.addEventListener('click',async event=>{event.preventDefault();try{const r=await fetch(link.href,{headers:await resultHeaders()});if(!r.ok)throw new Error('Download unavailable. Sign in again and retry.');const url=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=url;a.download=new URL(link.href).pathname.split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);}catch(error){$('#engine-feedback').textContent=error.message;}}));
      // Four bounded requests keep private thumbnails responsive without flooding
      // the processing service. One missing frame does not discard a loaded mesh.
      const slots=report.frames.map(()=>{const a=document.createElement('a');a.hidden=true;a.target='_blank';a.rel='noopener';$('#result-frames').append(a);return a;});let nextFrame=0;
      await Promise.all(Array.from({length:Math.min(4,report.frames.length)},async()=>{
        while(nextFrame<report.frames.length){const index=nextFrame++,f=report.frames[index];try{
          const r=await fetch(`${base}frames/frame_${String(f.index).padStart(5,'0')}.jpg`,{headers:await resultHeaders(),signal:AbortSignal.timeout(12000)});if(!r.ok)continue;
          const url=URL.createObjectURL(await r.blob());privateBlobs.push(url);const img=document.createElement('img');img.loading='lazy';img.src=url;img.alt=`Actual sampled input frame ${f.index}`;slots[index].href=url;slots[index].append(img);slots[index].hidden=false;
        }catch{/* The geometry and other available frames remain usable. */}}
      }));
    }else $('#result-frames').innerHTML=report.frames.map(f=>`<a href="${base}frames/frame_${String(f.index).padStart(5,'0')}.jpg" target="_blank" rel="noopener"><img loading="lazy" src="${base}frames/frame_${String(f.index).padStart(5,'0')}.jpg" alt="Actual sampled input frame ${f.index}"></a>`).join('');
    all('[data-engine-view],#engine-snapshot').forEach(b=>b.disabled=false);
    $('#engine-feedback').textContent='Reconstruction loaded. Accuracy has not been independently measured.';status(`${report.keyframes} input views · ${report.windows} reconstruction windows`);if(local&&health)await jobs();
  }catch(error){$('#engine-empty h2').textContent='Result could not be displayed';$('#engine-empty p').textContent=error.message;$('#engine-feedback').textContent=error.message;$('#open-mac-sample').hidden=false;$('#open-mac-sample').textContent='Try the saved sample again';}finally{loading=false;resultRunner.hidden=true;}
}
all('[data-engine-view]').forEach(button=>button.addEventListener('click',()=>viewer?.fit(button.dataset.engineView)));
$('#engine-mesh').addEventListener('change',e=>viewer?.setLayer('mesh',e.target.checked));
$('#engine-path').addEventListener('change',e=>viewer?.setLayer('path',e.target.checked));
$('#engine-cloud').addEventListener('change',async e=>{e.target.disabled=true;try{await viewer?.toggleCloud(e.target.checked);}catch(error){e.target.checked=false;$('#engine-feedback').textContent=error.message;}finally{e.target.disabled=false;}});
$('#engine-measure').addEventListener('click',()=>{if(!viewer)return;viewer.setMeasuring(!viewer.measuring);$('#engine-distance').textContent=viewer.measuring?'Select two surface points':'Scale unverified';});
$('#engine-snapshot').addEventListener('click',()=>{if(!viewer)return;const a=document.createElement('a');a.href=viewer.snapshot();a.download='voxelflight-reconstruction.png';a.click();});
window.addEventListener('voxelflight:authchange',event=>{signedIn=event.detail.signedIn;if(!signedIn&&loadedJob&&loadedJob!=='mac-proof'){location.reload();return;}connect();});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)connect();});
await connect();
async function schedulePoll(){await poll();setTimeout(schedulePoll,document.hidden?30000:activeJob?2500:health?15000:30000);}
setTimeout(schedulePoll,3000);
