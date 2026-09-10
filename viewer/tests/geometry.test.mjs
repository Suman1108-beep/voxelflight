import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
import {readFile} from 'node:fs/promises';
// Resolve the browser's local import map in Node for geometry-only unit tests.
registerHooks({resolve(specifier,context,next){
  if(specifier==='three')return {url:new URL('../vendor/three.module.js',import.meta.url).href,shortCircuit:true};
  if(specifier.startsWith('three/addons/'))return {url:new URL('../vendor/addons/'+specifier.slice(13),import.meta.url).href,shortCircuit:true};
  return next(specifier,context);
}});
const {GLTFLoader}=await import('../vendor/addons/loaders/GLTFLoader.js');
const {PLYLoader}=await import('../vendor/addons/loaders/PLYLoader.js');
const {OBJExporter}=await import('../vendor/addons/exporters/OBJExporter.js');
const {SceneViewer}=await import('../scene.js');
const THREE=await import('../vendor/three.module.js');
const buffer=async name=>{const b=await readFile(new URL('../assets/'+name,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
test('Mac proof is real inference with embedded source texture and no inherited accuracy',async()=>{
  const report=JSON.parse(await readFile(new URL('../assets/mac-proof/report.json',import.meta.url),'utf8'));
  // Public releases omit operator hardware; model provenance remains verifiable.
  if(report.device!==undefined)assert.equal(report.device,'Apple MPS');
  assert.match(report.model,/map-anything/);assert.equal(report.ground_truth_used,false);assert.equal(report.surface_rmse_m,null);assert.equal(report.trajectory_rmse_m,null);assert.equal(report.training_performed,false);
  const bytes=await buffer('mac-proof/reconstruction_mesh.glb'),v=new DataView(bytes);const json=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,20,v.getUint32(12,true))));
  assert.ok(json.meshes[0].primitives[0].attributes.TEXCOORD_0!==undefined);assert.equal(json.images.length,1);assert.ok(json.images[0].bufferView!==undefined);assert.equal(json.images[0].uri,undefined);
  const cloud=new PLYLoader().parse(await buffer('mac-proof/pointcloud.ply'));assert.equal(cloud.attributes.position.count,report.points);
});
test('mesh styling retains a real source-image texture',()=>{
  const texture=new THREE.Texture();const material=new THREE.MeshStandardMaterial({map:texture});const mesh=new THREE.Mesh(new THREE.BoxGeometry(),material);
  const viewer=Object.create(SceneViewer.prototype);viewer.root=new THREE.Group();viewer.root.add(mesh);viewer.styleMesh();assert.equal(mesh.material.map,texture);assert.equal(mesh.material.type,'MeshBasicMaterial');
});
test('actual vendored loader parses the complete saved GLB',async()=>{
  const gltf=await new GLTFLoader().parseAsync(await buffer('reconstruction_mesh.glb'),'');
  const box=new THREE.Box3().setFromObject(gltf.scene);assert.ok(!box.isEmpty());assert.ok(Number.isFinite(box.min.length()+box.max.length()));
  let triangles=0;gltf.scene.traverse(o=>{if(o.isMesh)triangles+=o.geometry.index.count/3;});assert.equal(triangles,450000);
});
test('facade camera stays outside the complete scene and keeps its centre in view',()=>{
  const viewer=Object.create(SceneViewer.prototype);
  viewer.box=new THREE.Box3(new THREE.Vector3(-24,-78,-20),new THREE.Vector3(20,12,24));
  viewer.camera=new THREE.PerspectiveCamera(45,1.7,.03,3000);viewer.camera.up.set(0,0,1);
  viewer.trajectory=[new THREE.Vector3(1,0,0),new THREE.Vector3(12,-50,2)];
  viewer.controls={target:new THREE.Vector3(),update(){viewer.camera.lookAt(this.target);viewer.camera.updateMatrixWorld();}};
  viewer.render=()=>{};viewer.fit('facade');
  assert.equal(viewer.box.containsPoint(viewer.camera.position),false);
  const clip=viewer.box.getCenter(new THREE.Vector3()).project(viewer.camera);
  assert.ok(Math.abs(clip.x)<1e-6&&Math.abs(clip.y)<1e-6);
  for(const x of [viewer.box.min.x,viewer.box.max.x])for(const y of [viewer.box.min.y,viewer.box.max.y])for(const z of [viewer.box.min.z,viewer.box.max.z]){
    const p=new THREE.Vector3(x,y,z).project(viewer.camera);assert.ok(Math.abs(p.x)<=1&&Math.abs(p.y)<=1&&Math.abs(p.z)<=1);
  }
});
test('actual PLY loader parses compact cloud and preserves bounds',async()=>{
  const cloud=new PLYLoader().parse(await buffer('pointcloud.ply'));assert.equal(cloud.attributes.position.count,518786);cloud.computeBoundingBox();
  assert.ok(Math.abs(cloud.boundingBox.getSize(new THREE.Vector3()).y-88.897644)<.00001);
});
test('OBJ exporter creates actual vertices and faces',()=>{
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(1,2,3),new THREE.MeshBasicMaterial());
  const text=new OBJExporter().parse(mesh);assert.equal(text.match(/^f /gm).length,12);assert.ok(text.includes('v '));
});
test('invalid local GLB fails before replacing the active model',async()=>{
  let replaced=false;const viewer=Object.create(SceneViewer.prototype);viewer.replaceRoot=()=>{replaced=true;};viewer.fit=()=>{};
  await assert.rejects(viewer.importFile({name:'bad.glb',arrayBuffer:async()=>new ArrayBuffer(4)}),/truncated/);
  assert.equal(replaced,false);
});
test('external resources in imported GLB are rejected before any fetch',async()=>{
  const json=new TextEncoder().encode(JSON.stringify({buffers:[{uri:'https://example.com/private.bin'}]}));const b=new ArrayBuffer(20+json.length);const d=new DataView(b);d.setUint32(0,0x46546c67,true);d.setUint32(12,json.length,true);d.setUint32(16,0x4e4f534a,true);new Uint8Array(b,20).set(json);
  const viewer=Object.create(SceneViewer.prototype);await assert.rejects(viewer.importFile({name:'remote.glb',arrayBuffer:async()=>b}),/self-contained/);
});
