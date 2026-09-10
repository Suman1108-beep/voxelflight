import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createViewNavigation,normalizeView,views} from '../navigation.js';

function harness(options={}) {
  const events=[],pending=new Map(),cancelled=[];let seq=0;
  const nav=createViewNavigation({
    render:name=>events.push(['render',name]),
    begin:view=>events.push(['begin',view.id]),
    finish:name=>events.push(['finish',name]),
    schedule:(callback,delay)=>{const id=++seq;pending.set(id,{callback,delay});return id;},
    cancel:id=>{cancelled.push(id);pending.delete(id);},
    ...options
  });
  return {nav,events,pending,cancelled};
}
test('all three named destinations have distinct, honest transition copy',()=>{
  assert.deepEqual(Object.keys(views),['workspace','validation','pipeline']);
  for(const view of Object.values(views))assert.ok(view.title&&view.detail&&view.color);
  assert.equal(normalizeView('validation'),'validation');
  for(const invalid of ['',null,undefined,'constructor','__proto__','other'])assert.equal(normalizeView(invalid),'workspace');
});
test('initial deep link renders without a cosmetic delay',()=>{
  const {nav,events,pending}=harness();nav.go('validation',{animate:false});
  assert.equal(nav.current,'validation');assert.deepEqual(events,[['render','validation']]);assert.equal(pending.size,0);
});
test('tab change updates the view immediately and bounds the decorative overlay',()=>{
  const {nav,events,pending}=harness();nav.go('workspace',{animate:false});nav.go('pipeline');
  assert.equal(nav.current,'pipeline');assert.deepEqual(events.slice(-2),[['begin','pipeline'],['render','pipeline']]);
  const timer=[...pending.values()][0];assert.equal(timer.delay,480);timer.callback();
  assert.deepEqual(events.at(-1),['finish','pipeline']);assert.equal(pending.size,0);
});
test('selecting the current view does not restart transitions or scene rendering',()=>{
  const {nav,events,pending}=harness();nav.go('workspace',{animate:false});nav.go('validation');
  const length=events.length;assert.equal(nav.go('validation'),false);assert.equal(events.length,length);assert.equal(pending.size,1);
});
test('rapid navigation cancels older timers and stale callbacks cannot hide a newer overlay',()=>{
  const {nav,events,pending,cancelled}=harness();nav.go('workspace',{animate:false});nav.go('validation');
  const stale=[...pending.values()][0].callback;nav.go('pipeline');
  assert.equal(cancelled.length,1);assert.equal(pending.size,1);stale();
  assert.equal(events.some(event=>event[0]==='finish'),false);
  [...pending.values()][0].callback();assert.deepEqual(events.at(-1),['finish','pipeline']);
});
test('reduced motion navigates immediately without any animation timer',()=>{
  const {nav,events,pending}=harness({motionDisabled:()=>true});nav.go('workspace');nav.go('validation');
  assert.deepEqual(events,[['render','workspace'],['render','validation']]);assert.equal(pending.size,0);
});
test('dismiss, motion changes and backgrounding can settle exactly once',()=>{
  const {nav,events,pending}=harness();nav.go('workspace',{animate:false});nav.go('validation');
  const stale=[...pending.values()][0].callback;nav.settle();nav.settle();stale();
  assert.equal(pending.size,0);assert.deepEqual(events.filter(e=>e[0]==='finish'),[['finish','validation']]);
});
test('an unanimated navigation clears a preceding animated transition',()=>{
  const {nav,events,pending}=harness();nav.go('validation');nav.go('pipeline',{animate:false});
  assert.equal(pending.size,0);assert.equal(nav.current,'pipeline');assert.deepEqual(events.at(-1),['finish','pipeline']);
});
test('render errors dismiss the overlay and permit a retry',()=>{
  let fail=false;const {nav,events,pending}=harness({render:()=>{if(fail)throw Error('View unavailable');}});
  nav.go('workspace',{animate:false});fail=true;assert.throws(()=>nav.go('validation'),/View unavailable/);
  assert.equal(nav.current,'workspace');assert.equal(pending.size,0);assert.deepEqual(events.at(-1),['finish','workspace']);
  fail=false;assert.equal(nav.go('validation'),true);nav.settle();
});
test('shared theme loads after the base theme on every public-facing screen',async()=>{
  for(const file of ['index.html','login.html','account.html','logout.html','engine.html']){
    const html=await readFile(new URL('../'+file,import.meta.url),'utf8');
    assert.match(html,/href="\/?pixel\.css"[\s\S]*href="\/?studio\.css"/);
    assert.equal([...html.matchAll(/href="\/?studio\.css"/g)].length,1);
  }
});
test('navigation wiring retains one scene and handles history, motion and dismissal',async()=>{
  const code=await readFile(new URL('../app.js',import.meta.url),'utf8');
  assert.match(code,/hashchange/);assert.match(code,/history\.pushState/);
  assert.match(code,/sih3d:transition-dismissed/);assert.match(code,/sih3d:motionchange/);
  assert.match(code,/page\(location.hash.slice\(1\),\{animate:false\}\)/);
  assert.equal([...code.matchAll(/new SceneViewer\(/g)].length,1);
  const theme=await readFile(new URL('../studio.css',import.meta.url),'utf8');
  assert.match(theme,/prefers-reduced-motion/);assert.match(theme,/max-width:760px/);
  assert.doesNotMatch(theme,/#scene\s*\{[^}]*filter:/);
});
