import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {engineMessage} from '../engine-copy.js';
const read=file=>readFile(new URL('../'+file,import.meta.url),'utf8');

test('customer-facing labels describe reconstruction rather than development hardware',async()=>{
  for(const file of ['engine.html','index.html','account.html','login.html']){
    const visible=(await read(file)).replace(/<[^>]*>/g,' ');
    assert.doesNotMatch(visible,/\b(?:Mac|Apple|MPS)\b/i,file);
  }
  const html=await read('engine.html');
  assert.match(html,/<title>VoxelFlight — New reconstruction<\/title>/);
  assert.match(html,/>Start reconstruction<\/button>/);
  assert.match(html,/>Recent runs /);
});
test('connection and accuracy limitations remain explicit after copy cleanup',async()=>{
  const html=await read('engine.html');
  assert.match(html,/a cloud engine is not connected/);
  assert.match(html,/does not connect visitors to the site owner’s computer/);
  assert.match(html,/four source frames · accuracy unmeasured/);
  assert.match(html,/accuracy validation pending/);
  assert.match(html,/id="engine-start"[^>]*disabled/);
  assert.match(html,/href="http:\/\/127.0.0.1:8133\/engine.html"/);
});
test('sample display preserves all warnings and does not alter original evidence',async()=>{
  const report=JSON.parse(await read('assets/mac-proof/report.json'));
  const warnings=report.warnings.map(engineMessage);
  assert.equal(warnings.length,report.warnings.length);
  assert.match(warnings[0],/surface and trajectory accuracy have not been independently measured/);
  assert.match(warnings[2],/does not include semantic segmentation, inertial fusion, bundle adjustment or Gaussian-splat training/);
  assert.equal(warnings[1],report.warnings[1]);
  // Public reports omit operator hardware while preserving model provenance.
  if(report.device!==undefined)assert.equal(report.device,'Apple MPS');
  assert.match(report.model,/map-anything/);
  assert.equal(report.surface_rmse_m,null);assert.equal(report.sih_requirements_verified,false);
});
test('unrecognized status messages pass through without suppressing diagnostics',()=>{
  assert.equal(engineMessage('Insufficient image overlap'),'Insufficient image overlap');
  assert.match(engineMessage('The Mac is processing another run. Wait or cancel that run.'),/Another reconstruction is in progress/);
  assert.match(engineMessage('Processing cancelled. Uploaded files remain on this Mac.'),/Uploaded files remain on the processing workstation/);
});
