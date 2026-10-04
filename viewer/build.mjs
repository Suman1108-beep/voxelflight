import {cp,mkdir,readFile,readdir,stat,writeFile,rm} from 'node:fs/promises';
import {resolve,relative,dirname} from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {build} from 'vite';
const root=import.meta.dirname,firebaseTarget=process.argv.includes('--firebase');
const outputRoot=resolve(root,firebaseTarget?'dist-firebase':'dist'),dist=firebaseTarget?outputRoot:resolve(outputRoot,'client');
const sources=['index.html','styles.css','fonts.css','favicon.svg','app.js','scene.js','core.js','ui.js','inputs.js','video-input.js','engine.html','engine.css','engine.js','engine-copy.js','login.html','account.html','logout.html','entrance.css','entrance.js','motion.css','pixel.css','pixel.js','navigation.js','studio.css','flight.css'];
sources.push('session-bridge.js','experience.css','latest-run.js','explore.html','explore.js','explore.css','studio.html','studio.js','studio-page.css','pro.css','transition.css');
for(const file of sources.filter(f=>f.endsWith('.js')))execFileSync(process.execPath,['--check',resolve(root,file)]);
// Only disposable build output is cleared; source and private Mac results are untouched.
await rm(outputRoot,{recursive:true,force:true});
await mkdir(dist,{recursive:true});
for(const file of sources){
  const destination=file==='index.html'?'workspace.html':file==='account.html'?'_pages/account.html':file;
  await mkdir(dirname(resolve(dist,destination)),{recursive:true});
  await cp(resolve(root,file),resolve(dist,destination));
}
// Root is the public 3D explorer; sign-in lives at /login and the full workspace at /workspace.
await cp(resolve(root,'explore.html'),resolve(dist,'index.html'));
// Preserve the original high-precision cloud locally; publish the compact full-count copy.
await cp(resolve(root,'assets'),resolve(dist,'assets'),{recursive:true,filter:p=>!p.endsWith('reconstruction_clean.ply')&&!p.endsWith('.DS_Store')});
// Keep only the vendored modules reachable from the app, including lazy exporters.
const visited=new Set();
async function moduleGraph(file){
  const path=resolve(root,file);if(visited.has(path))return;visited.add(path);
  const source=(await readFile(path,'utf8')).replace(/\/\*[\s\S]*?\*\//g,'').replace(/^\s*\/\/.*$/gm,'');
  for(const m of source.matchAll(/(?:^\s*(?:import|export)\s+[\w$*,{}\s]+?\s+from\s*|^\s*import\s*|import\s*\(\s*)['"]([^'"]+)['"]/gm)){
    const spec=m[1];let target;
    if(spec==='three')target=resolve(root,'vendor/three.module.js');
    else if(spec.startsWith('three/addons/'))target=resolve(root,'vendor/addons',spec.slice(13));
    else if(spec.startsWith('.'))target=resolve(dirname(path),spec);
    else throw new Error('Unresolved dependency: '+spec);
    if(!target.startsWith(root+'/'))throw new Error('Dependency escapes the site');
    await moduleGraph(relative(root,target));
  }
  if(file.startsWith('vendor/')){await mkdir(dirname(resolve(dist,file)),{recursive:true});await cp(path,resolve(dist,file));}
}
await moduleGraph('app.js');
await moduleGraph('engine.js');
await moduleGraph('explore.js');
await moduleGraph('studio.js');
if(firebaseTarget){const {prepareFirebase}=await import('./firebase/build.mjs');await prepareFirebase(root,dist);}
const html=await readFile(resolve(dist,'workspace.html'),'utf8');
for(const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)){const value=match[1];if(/^(?:https?:|data:)/.test(value)||['/','/account','/workspace','/engine'].includes(value))continue;await stat(resolve(dist,value.replace(/^\//,'')));}
const files=[];
async function scan(dir){for(const entry of await readdir(dir,{withFileTypes:true})){const path=resolve(dir,entry.name);if(entry.name==='asset-manifest.json')continue;if(entry.isDirectory())await scan(path);else{const data=await readFile(path);if(data.length>25*1024*1024)throw new Error('Static asset exceeds 25 MiB: '+path);files.push({path:relative(dist,path),bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')});}}}
await scan(dist);
await writeFile(resolve(dist,'asset-manifest.json'),JSON.stringify({schema:1,files},null,2));
console.log(`Build complete: ${files.length} files, ${(files.reduce((sum,f)=>sum+f.bytes,0)/1e6).toFixed(1)} MB. Source modules and linked assets validated.`);
if(!firebaseTarget){
  await build({configFile:resolve(root,'vite.config.js')});
  await stat(resolve(root,'dist/server/index.js'));
  await stat(resolve(root,'dist/.openai/hosting.json'));
}
