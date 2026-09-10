
export function escapeHTML(value) { return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
export function formatTime(seconds) { const total=Math.round(seconds); return `${Math.floor(total/60)}m ${String(total%60).padStart(2,'0')}s`; }
export function parseCSV(text) {
  const rows=[];let row=[],cell='',quoted=false;
  for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
  else if(c===','&&!quoted){row.push(cell.trim());cell='';}
  else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell.trim());if(row.some(Boolean))rows.push(row);row=[];cell='';}
  else cell+=c;}
  if(quoted)throw new Error('Unclosed quote in CSV.');
  row.push(cell.trim());if(row.some(Boolean))rows.push(row);
  if(rows.length<2)throw new Error('The CSV needs headers and at least one data row.');
  const headers=rows.shift().map(h=>h.replace(/^\uFEFF/,''));
  if(new Set(headers).size!==headers.length)throw new Error('CSV column names must be unique.');
  if(headers.some(h=>!h))throw new Error('CSV column names cannot be empty.');
  if(rows.some(values=>values.length!==headers.length))throw new Error('CSV rows must match the number of columns.');
  return rows.map((values)=>Object.fromEntries(headers.map((key,i)=>[key,values[i]??''])));
}
const norm=s=>s.toLowerCase().replace(/[^a-z0-9]/g,'');
export function validateTelemetry(text, extension){
  if(text.length>10*1024*1024)throw new Error('Telemetry must be under 10 MB.');
  if(extension==='srt'){
    const entries=text.split(/\r?\n\s*\r?\n/).filter(v=>/-->/.test(v));
    const valid=entries.filter(s=>/latitude\s*[:=]\s*-?\d/i.test(s)&&/long(?:itude|titude)\s*[:=]\s*-?\d/i.test(s));
    if(!valid.length)throw new Error('No timestamped latitude / longitude records found. Export GPS as CSV or JSON.');
    return {format:'SRT',records:valid.length,valid_coordinates:valid.length,status:'header-check',note:'GPS fields detected. Full time alignment is checked by the reconstruction worker.'};
  }
  let rows;
  if(extension==='csv')rows=parseCSV(text);
  else if(extension==='json'){const value=JSON.parse(text);rows=Array.isArray(value)?value:(value.records??value.telemetry??value.data);if(!Array.isArray(rows)||!rows.length)throw new Error('JSON needs a nonempty array of telemetry records.');}
  else throw new Error('Use CSV, JSON or SRT telemetry.');
  const first=rows[0];if(!first||typeof first!=='object')throw new Error('Telemetry records must be objects.');
  const find=(aliases)=>Object.keys(first).find(k=>aliases.includes(norm(k)));
  const lat=find(['lat','latitude','gpslatitude','gpslat']),lon=find(['lon','lng','longitude','gpslongitude','gpslon','longtitude']);
  const time=find(['time','timestamp','timestamps','timestampms','timems','times','timeus','timpstemp','timestampus','sourceframe','frame','frameid','imgid']);
  if(!lat||!lon)throw new Error('Latitude and longitude columns are required.');
  if(!time)throw new Error('Add a timestamp or frame-index column to align GPS with the video.');
  const number=v=>typeof v==='number'?v:(String(v??'').trim()?Number(v):NaN);
  const valid=rows.filter(r=>r&&Number.isFinite(number(r[lat]))&&Number.isFinite(number(r[lon]))&&Math.abs(number(r[lat]))<=90&&Math.abs(number(r[lon]))<=180);
  if(valid.length!==rows.length)throw new Error(`${rows.length-valid.length} record(s) contain missing or invalid GPS coordinates.`);
  if(rows.some(r=>r[time]===undefined||String(r[time]).trim()===''))throw new Error('One or more telemetry records have no timestamp / frame index.');
  return {format:extension.toUpperCase(),records:rows.length,valid_coordinates:valid.length,columns:{latitude:lat,longitude:lon,time},status:'header-check',note:'Coordinates checked. Units, timestamp alignment and altitude datum are verified by the worker.'};
}
export function parseTrajectory(text){
  return parseCSV(text).map(row=>{const p=['camera_x','camera_y','camera_z'].map(k=>row[k]?.trim()?Number(row[k]):NaN);const frame=row.frame_id?.trim()?Number(row.frame_id):NaN;if(p.some(v=>!Number.isFinite(v))||!Number.isInteger(frame)||frame<0)throw new Error('Invalid trajectory coordinate or frame index.');return {frame,position:p};});
}
export function fileSize(bytes){return bytes>=1e6?`${(bytes/1e6).toFixed(1)} MB`:`${Math.ceil(bytes/1000)} KB`;}
export function saveFile(data,name,type='application/octet-stream'){
  const blob=data instanceof Blob?data:new Blob([data],{type});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
