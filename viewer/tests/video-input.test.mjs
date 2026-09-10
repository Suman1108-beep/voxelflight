import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateVideoFile,SINGLE_IMAGE_MESSAGE} from '../video-input.js';
const file=(name,size=1024,type='')=>({name,size,type});
test('rejects single photos before starting reconstruction, including misleading extensions',()=>{
  for(const input of [file('frame.png'),file('FRAME.JPG'),file('photo.heic'),file('frame.mp4',1024,'image/jpeg')])assert.equal(validateVideoFile(input),SINGLE_IMAGE_MESSAGE);
});
test('accepts supported video names and rejects empty, unsupported and oversized input',()=>{
  assert.equal(validateVideoFile(file('flight.MP4')),'');assert.equal(validateVideoFile(file('flight.mov')),'');
  assert.ok(validateVideoFile(null));assert.ok(validateVideoFile(file('empty.mp4',0)));
  assert.ok(validateVideoFile(file('flight.webm')));assert.ok(validateVideoFile(file('large.mp4',1024**3+1)));
});
test('all screens carry the VoxelFlight identity while existing engine contracts remain compatible',async()=>{
  for(const name of ['index.html','engine.html','account.html','logout.html','login.html']){
    const html=await readFile(new URL('../'+name,import.meta.url),'utf8');assert.match(html,/<title>VoxelFlight — /);assert.doesNotMatch(html,/SIH<span>3D<\/span>|SIH3D/);
  }
  const engine=await readFile(new URL('../engine.js',import.meta.url),'utf8');
  assert.match(engine,/X-SIH3D-Session/);assert.match(engine,/validateVideoFile\(video\)/);
});
test('engine provides direct routes to workspace, validation, pipeline and account',async()=>{
  const html=await readFile(new URL('../engine.html',import.meta.url),'utf8');
  for(const route of ['/workspace','/workspace#validation','/workspace#pipeline','/account'])assert.ok(html.includes(`href="${route}"`));
  assert.match(html,/href="engine.html" aria-current="page"/);
});
