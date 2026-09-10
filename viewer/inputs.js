import {validateTelemetry,escapeHTML,saveFile,formatTime} from './core.js';
import {$,toast} from './ui.js';
import {validateVideoFile} from './video-input.js';

export function setupInputs(){
  const state={video:null,telemetry:null},errors={},revision={video:0,telemetry:0};
  function render(){
    const rows=[];
    for(const key of ['video','telemetry']){
      const value=state[key];let detail='No file selected';
      if(errors[key])detail=errors[key];
      else if(value)detail=key==='video'?`${value.name} · ${value.width} × ${value.height} · ${formatTime(value.duration_s)}`:`${value.name} · ${value.records.toLocaleString()} GPS records · ${value.format}`;
      rows.push(`<div class="input-check"><span>${key==='video'?'Video':'Telemetry'}</span><strong class="${errors[key]?'bad':value?'good':''}">${escapeHTML(detail)}</strong></div>`);
    }
    $('#input-results').innerHTML=rows.join('')+'<p class="micro">Local compatibility checks only. Camera calibration, time synchronization, coordinate datum and complete flight metadata still need worker-side validation. The manifest does not contain your files.</p>';
    $('#manifest-download').disabled=!(state.video&&state.telemetry);
  }
  $('#video-file').addEventListener('change',async e=>{
    const token=++revision.video,file=e.target.files?.[0];state.video=null;delete errors.video;if(!file){render();return;}
    errors.video='Reading video metadata…';render();
    try{
      const inputError=validateVideoFile(file);if(inputError)throw new Error(inputError);
      const metadata=await new Promise((resolve,reject)=>{
        const video=document.createElement('video'),url=URL.createObjectURL(file);let finished=false;
        const finish=(error)=>{if(finished)return;finished=true;clearTimeout(timeout);const result={width:video.videoWidth,height:video.videoHeight,duration_s:video.duration};video.onloadedmetadata=null;video.onerror=null;video.removeAttribute('src');video.load();URL.revokeObjectURL(url);error?reject(error):resolve(result);};
        const timeout=setTimeout(()=>finish(new Error('Metadata timed out. Try an H.264 MP4.')),15000);
        video.onloadedmetadata=()=>finish();video.onerror=()=>finish(new Error('This browser cannot decode the video. Try an H.264 MP4.'));
        video.preload='metadata';video.src=url;
      });
      if(!Number.isFinite(metadata.duration_s)||metadata.duration_s<=0||!metadata.width)throw new Error('Video duration or dimensions could not be read.');
      if(token!==revision.video)return;
      state.video={name:file.name,size_bytes:file.size,...metadata};delete errors.video;
      if(Math.min(metadata.width,metadata.height)<1080)toast('Video is below the stated 1080p input resolution.');
    }catch(error){if(token===revision.video)errors.video=error.message;}render();
  });
  $('#telemetry-file').addEventListener('change',async e=>{
    const token=++revision.telemetry,file=e.target.files?.[0];state.telemetry=null;delete errors.telemetry;if(!file){render();return;}
    errors.telemetry='Checking GPS fields…';render();
    try{if(file.size>10*1024*1024)throw new Error('Telemetry must be under 10 MB.');const result=validateTelemetry(await file.text(),file.name.split('.').pop().toLowerCase());if(token!==revision.telemetry)return;state.telemetry={name:file.name,size_bytes:file.size,...result};delete errors.telemetry;}catch(error){if(token===revision.telemetry)errors.telemetry=error.message;}render();
  });
  $('#manifest-download').addEventListener('click',()=>{
    if(!state.video||!state.telemetry)return;
    saveFile(JSON.stringify({schema:'sih3d.input-request.v1',created_at:new Date().toISOString(),status:'prepared_not_processed',worker_required:true,files_uploaded:false,video:state.video,telemetry:state.telemetry,calibration:null,pending_validation:['Camera calibration','Flight metadata completeness','Timestamp alignment and units','Coordinate reference and altitude datum','Reconstruction worker compatibility']},null,2),'reconstruction-input.json','application/json');
    toast('Input manifest saved. Keep it with your video and telemetry.');
  });
}
