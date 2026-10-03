import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=file=>readFile(new URL('../'+file,import.meta.url),'utf8');

test('entrance prioritizes sign-in and retains a separate public guest demo',async()=>{
  const html=await read('login.html');
  assert.match(html,/class="guest-button launch-demo" href="\/workspace"/);
  assert.ok(html.indexOf('launch-demo')>html.indexOf('provider-buttons'));
  assert.match(html,/Decorative illustration, not a reconstruction/);
  assert.match(html,/Other sign-in providers/);
});
test('real geometry is loaded on demand rather than on initial video view',async()=>{
  const app=await read('app.js'),engine=await read('engine.js');
  for(const code of [app,engine]){
    assert.doesNotMatch(code,/^import \{SceneViewer\}/m);
    assert.match(code,/await import\('\.\/scene\.js'\)/);
  }
  assert.match(app,/mode='video'/);
  assert.match(app,/if\(viewerPromise\)return viewerPromise/);
  assert.match(app,/if\(!await ensureViewer\(\)\)/);
});
test('camera controls cannot accidentally bind to the page-state attribute on body',async()=>{
  const app=await read('app.js');
  assert.match(app,/all\('button\[data-view\]'\)/);
  assert.doesNotMatch(app,/all\('\[data-view\]'\)/);
});
test('simplified workspace retains evidence and a keyboard-accessible details dialog',async()=>{
  const app=await read('app.js'),html=await read('index.html');
  assert.match(app,/document\.createElement\('dialog'\)/);
  assert.match(app,/inspectorDialog\.append\(\$\('\.inspector'\)\)/);
  assert.match(app,/if\(inspectorDialog.open\)inspectorDialog.close\(\)/);
  assert.match(html,/id="open-details"[^>]*aria-haspopup="dialog"/);
  assert.match(app,/shape is about 0\.7 m median against survey LiDAR/);
  assert.match(app,/image scores do not describe the latest 3D mesh/);
});
test('workflow and availability are based on real job state, not a decorative timer',async()=>{
  const html=await read('engine.html'),code=await read('engine.js');
  for(const step of ['prepare','process','explore'])assert.match(html,new RegExp('data-step="'+step+'"'));
  assert.match(code,/flow\(job.state==='complete'/);
  assert.match(code,/flow\('explore'\)/);
  assert.match(html,/Nothing is uploaded until a connected engine accepts your run/);
  assert.match(code,/if\(!local&&!hosted\)return offline/);
  assert.match(code,/await sessionHeaders\(\)/);
  assert.match(code,/queue_available/);
  assert.match(code,/Engine connection required/);
});
test('all three updated screens ship the focused visual system',async()=>{
  for(const file of ['index.html','engine.html','login.html'])assert.match(await read(file),/href="\/?flight\.css"/);
  assert.match(await read('build.mjs'),/'flight.css'/);
  const css=await read('flight.css');assert.match(css,/prefers-reduced-motion/);assert.match(css,/max-width:760px/);
});
