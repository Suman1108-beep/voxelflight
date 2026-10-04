import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');

test('explorer and studio load the pixel scout transition styles',async()=>{
  for(const page of ['explore.html','studio.html'])assert.match(await read(page),/href="transition\.css"/,page);
  assert.match(await read('build.mjs'),/'transition\.css'/);
  const css=await read('transition.css');
  assert.match(css,/survey-robot-run\.png/);assert.match(css,/@keyframes survey-run/);assert.match(css,/prefers-reduced-motion/);
  // Every selector stays inside the overlay so the host page's design is untouched.
  const selectors=css.replace(/\/\*[\s\S]*?\*\//g,'').replace(/@keyframes[^{]+\{(?:[^{}]*\{[^}]*\})*\s*\}/g,'').replace(/@font-face\{[^}]*\}/g,'')
    .replace(/@media[^{]+\{/g,'').match(/[^{}]+(?=\{)/g).flatMap(s=>s.split(',')).map(s=>s.trim()).filter(Boolean);
  for(const s of selectors)assert.match(s,/^(\.motion-paused )?\.route-transition/,s);
});

test('page links and explorer views both show the running scout',async()=>{
  const pixel=await read('pixel.js');
  for(const route of ['STUDIO','EXPLORER','ACCOUNT','SIGN IN'])assert.match(pixel,new RegExp(`'${route}'`));
  for(const view of ['guide','evidence','how'])assert.match(pixel,new RegExp(`\\b${view}:\\{label:`));
  const explore=await read('explore.js');
  assert.match(explore,/from '\.\/pixel\.js'/);assert.match(explore,/showTransition\(/);assert.match(explore,/motionDisabled\(\)/);
  assert.match(explore,/openPanel\(hash,\{animate:false\}\)/);
  assert.match(await read('studio.js'),/import '\.\/pixel\.js'/);
});
