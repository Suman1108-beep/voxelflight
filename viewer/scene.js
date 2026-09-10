
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';

export class SceneViewer {
  constructor(canvas,{onStatus=()=>{},onPick=()=>{},onHover=()=>{}}={}){
    this.canvas=canvas;this.onStatus=onStatus;this.onPick=onPick;this.onHover=onHover;this.selection=[];this.measurements=new THREE.Group();
    this.renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x0c151e);this.renderer.toneMapping=THREE.NoToneMapping;
    this.scene=new THREE.Scene();this.scene.add(this.measurements);
    const fill=new THREE.HemisphereLight(0xffffff,0x697789,2.1);fill.position.set(0,0,1);this.scene.add(fill);
    this.scene.add(new THREE.AmbientLight(0xffffff,.7));
    this.camera=new THREE.PerspectiveCamera(45,1,.03,3000);this.camera.up.set(0,0,1);
    this.controls=new OrbitControls(this.camera,canvas);this.controls.enableDamping=false;this.controls.screenSpacePanning=true;
    this.controls.addEventListener('change',()=>this.render());
    this.root=new THREE.Group();this.scene.add(this.root);
    this.pathGroup=new THREE.Group();this.scene.add(this.pathGroup);
    this.grid=new THREE.GridHelper(200,40,0x3d5664,0x203543);this.grid.rotation.x=Math.PI/2;
    this.grid.material.transparent=true;this.grid.material.opacity=.48;this.scene.add(this.grid);
    this.raycaster=new THREE.Raycaster();this.pointer=new THREE.Vector2();this.isImported=false;
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(canvas.parentElement);
    canvas.addEventListener('pointerdown',e=>{this.down=[e.clientX,e.clientY];});
    canvas.addEventListener('pointerup',e=>{if(!this.measuring||!this.down||Math.hypot(e.clientX-this.down[0],e.clientY-this.down[1])>5)return;const hit=this.hit(e);if(hit)this.addMeasurement(hit.point);});
    let nextHover=0;
    canvas.addEventListener('pointermove',e=>{if(performance.now()<nextHover)return;nextHover=performance.now()+100;if(!this.measuring)return;const hit=this.hit(e);if(hit)this.onHover(hit.point.toArray());});
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.onStatus('Graphics context lost. Reload this page to restore the 3D view.');});
    this.resize();
  }
  resize(){const r=this.canvas.parentElement.getBoundingClientRect();if(r.width<1||r.height<1)return;this.renderer.setSize(r.width,r.height,false);this.camera.aspect=r.width/r.height;this.camera.updateProjectionMatrix();this.render();}
  async loadDefault(){
    return this.loadReconstruction('assets/');
  }
  async loadReconstruction(base,{getHeaders=async()=>({})}={}){
    this.assetBase=base;
    this.getAssetHeaders=getHeaders;
    this.onStatus('Loading saved mesh…');
    const gltf=await new GLTFLoader().setRequestHeader(await getHeaders()).loadAsync(`${base}reconstruction_mesh.glb`);
    this.replaceRoot(gltf.scene,false);
    this.styleMesh();this.fit('overview');
    this.onStatus('Saved reconstruction ready');
  }
  styleMesh(){
    this.root.traverse(child=>{if(child.isMesh){const color=!!child.geometry.getAttribute('color');
      const map=child.material?.map??null;
      if(child.material?.dispose)child.material.dispose();
      child.material=new THREE.MeshBasicMaterial({map,vertexColors:color,color:color||map?0xffffff:0xa3b4c1,side:THREE.DoubleSide});
      child.material.polygonOffset=true;child.material.polygonOffsetFactor=1;child.material.polygonOffsetUnits=1;
    }});
  }
  replaceRoot(object, imported){
    this.setMeasuring(false);this.clearMeasurement();this.generation=(this.generation??0)+1;
    if(this.object){this.root.remove(this.object);this.object.traverse(c=>{c.geometry?.dispose();if(Array.isArray(c.material))c.material.forEach(m=>{m.map?.dispose();m.dispose();});else{c.material?.map?.dispose();c.material?.dispose();}});}
    if(this.cloud){this.root.remove(this.cloud);this.cloud.geometry.dispose();this.cloud.material.dispose();this.cloud=null;}this.cloudPromise=null;
    this.object=object;this.root.add(object);this.box=new THREE.Box3().setFromObject(object);
    if(this.box.isEmpty())throw new Error('The file contains no renderable geometry.');
    this.isImported=imported;this.grid.position.z=this.box.min.z-.4;this.pathGroup.visible=!imported;
  }
  setTrajectory(rows){
    this.trajectory=rows.map(r=>new THREE.Vector3(...r.position));
    this.pathGroup.traverse(c=>{c.geometry?.dispose();c.material?.dispose();});this.pathGroup.clear();
    const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(this.trajectory),new THREE.LineBasicMaterial({color:0xe3c68c,transparent:true,opacity:.9}));
    this.pathGroup.add(line);
    const dotGeo=new THREE.SphereGeometry(.11,7,5),dotMat=new THREE.MeshBasicMaterial({color:0xe3c68c});
    for(let i=0;i<this.trajectory.length;i+=10){const dot=new THREE.Mesh(dotGeo,dotMat);dot.position.copy(this.trajectory[i]);this.pathGroup.add(dot);}
    this.marker=new THREE.Mesh(new THREE.SphereGeometry(.23,12,8),new THREE.MeshBasicMaterial({color:0xe7fff1}));this.pathGroup.add(this.marker);this.setFrame(0);
  }
  setFrame(frame){if(this.marker&&this.trajectory?.[frame]){this.marker.position.copy(this.trajectory[frame]);this.render();}}
  fit(preset='overview'){
    if(!this.box)return;
    let box=this.box.clone(),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const maxSpan=Math.max(size.x,size.y,size.z);
    if(preset==='facade'&&!this.isImported&&this.trajectory?.length){
      // View from the observed flight side, outside the entire scene bounds.
      // Cropping the bounds to one tile could put the camera inside another tile.
      const viewpoint=this.trajectory.reduce((sum,p)=>sum.add(p),new THREE.Vector3()).divideScalar(this.trajectory.length);
      const observedSide=viewpoint.sub(center);
      const tangent=this.trajectory.at(-1).clone().sub(this.trajectory[0]);
      const direction=new THREE.Vector3(-tangent.y,tangent.x,0);
      if(direction.length()<1)direction.set(observedSide.x,observedSide.y,0);
      if(direction.length()<1)direction.set(1,-1,0);
      if(direction.dot(observedSide)<0)direction.negate();
      direction.normalize();direction.z=.12;direction.normalize();
      const tanY=Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2)),tanX=tanY*this.camera.aspect;
      const right=new THREE.Vector3().crossVectors(direction,this.camera.up).normalize(),up=new THREE.Vector3().crossVectors(right,direction).normalize();
      let distance=1;
      for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){
        const offset=new THREE.Vector3(x,y,z).sub(center),depth=offset.dot(direction);
        distance=Math.max(distance,depth+1.12*Math.abs(offset.dot(right))/tanX,depth+1.12*Math.abs(offset.dot(up))/tanY);
      }
      distance+=maxSpan*.04;
      this.camera.position.copy(center).addScaledVector(direction,distance);
    }else if(preset==='plan'){this.camera.position.copy(center).add(new THREE.Vector3(0,-.001,maxSpan*1.32));}
    else{const distance=Math.max(size.z,size.y*.85,size.x/Math.max(this.camera.aspect,.4))*1.35;this.camera.position.copy(center).add(new THREE.Vector3(distance*.7,-distance*.65,distance*.55));}
    this.camera.near=Math.max(maxSpan/5000,.005);this.camera.far=Math.max(maxSpan*35,100);
    this.camera.updateProjectionMatrix();this.controls.target.copy(center);this.controls.update();this.render();
  }
  async toggleCloud(visible){
    if(this.isImported){this.object?.traverse(c=>{if(c.isPoints)c.visible=visible;});this.render();return;}
    if(visible&&!this.cloud){
      this.onStatus('Loading reconstructed points…');
      if(!this.cloudPromise)this.cloudPromise=new PLYLoader().setRequestHeader(await this.getAssetHeaders?.()??{}).loadAsync(`${this.assetBase??'assets/'}pointcloud.ply`);
      const generation=this.generation;const geometry=await this.cloudPromise;
      if(generation!==this.generation){geometry.dispose();return;}
      if(!this.cloud){this.cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:.04,vertexColors:true,sizeAttenuation:true,transparent:false}));this.root.add(this.cloud);}
    }
    if(this.cloud)this.cloud.visible=visible;
    this.onStatus('Saved reconstruction ready');this.render();
  }
  setLayer(layer,visible){
    if(layer==='mesh')this.object?.traverse(c=>{if(c.isMesh)c.visible=visible;});
    if(layer==='path')this.pathGroup.visible=visible&&!this.isImported;
    if(layer==='grid')this.grid.visible=visible;
    if(layer==='wire')this.object?.traverse(c=>{if(c.isMesh)c.material.wireframe=visible;});
    this.render();
  }
  setPointSize(value){const size=Number(value)*.015;if(this.cloud)this.cloud.material.size=size;if(this.object?.isPoints)this.object.material.size=size;this.render();}
  hit(e){
    if(!this.object||!this.object.visible)return null;const r=this.canvas.getBoundingClientRect();this.pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);this.raycaster.setFromCamera(this.pointer,this.camera);
    return this.raycaster.intersectObject(this.object,true).find(h=>h.object.isMesh&&h.object.visible);
  }
  setMeasuring(active){this.measuring=active;if(active)this.clearMeasurement();this.canvas.style.cursor=active?'crosshair':'';}
  clearMeasurement(){this.selection=[];for(const child of [...this.measurements.children]){child.geometry?.dispose();child.material?.dispose();this.measurements.remove(child);}this.onPick(null);this.render();}
  addMeasurement(point){
    if(this.selection.length>=2)this.clearMeasurement();this.selection.push(point.clone());
    const radius=Math.max(this.box.getSize(new THREE.Vector3()).length()*.0012,.035);
    const dot=new THREE.Mesh(new THREE.SphereGeometry(radius,12,8),new THREE.MeshBasicMaterial({color:0xffd58c,depthTest:false}));dot.position.copy(point);dot.renderOrder=10;this.measurements.add(dot);
    if(this.selection.length===2){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(this.selection),new THREE.LineBasicMaterial({color:0xffd58c,depthTest:false}));line.renderOrder=10;this.measurements.add(line);this.measuring=false;this.canvas.style.cursor='';
      this.onPick({distance:this.selection[0].distanceTo(this.selection[1]),points:this.selection.map(v=>v.toArray())});}
    else this.onPick({pending:true,points:[point.toArray()]});this.render();
  }
  render(){
    if(!this.renderer||!this.camera)return;this.renderer.render(this.scene,this.camera);
    const bar=document.querySelector('#scale-bar>span');if(bar&&this.controls){const distance=this.camera.position.distanceTo(this.controls.target);const metresPerPixel=2*distance*Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2))/Math.max(this.canvas.clientHeight,1);const px=10/metresPerPixel;bar.style.width=Math.min(Math.max(px,8),180)+'px';bar.parentElement.hidden=px>180||px<8;}
    const label=document.querySelector('#measure-label');if(label&&this.selection.length===2){const middle=this.selection[0].clone().add(this.selection[1]).multiplyScalar(.5).project(this.camera);label.hidden=middle.z>1||middle.z< -1;label.style.left=((middle.x+1)/2*this.canvas.clientWidth)+'px';label.style.top=((1-middle.y)/2*this.canvas.clientHeight)+'px';}
  }
  async importFile(file){
    const array=await file.arrayBuffer();let object;
    if(array.byteLength<20)throw new Error('The model file is empty or truncated.');
    if(file.name.toLowerCase().endsWith('.glb')){
      if(new DataView(array).getUint32(0,true)!==0x46546c67)throw new Error('Invalid GLB header.');
      // Reject external buffers/textures; local import must not fetch external resources.
      const view=new DataView(array);const length=view.getUint32(12,true),kind=view.getUint32(16,true);
      if(kind!==0x4e4f534a||20+length>array.byteLength)throw new Error('Invalid GLB JSON chunk.');
      const json=JSON.parse(new TextDecoder().decode(new Uint8Array(array,20,length)));
      if([...(json.buffers??[]),...(json.images??[])].some(x=>x.uri&&!x.uri.startsWith('data:')))throw new Error('Use a self-contained GLB with embedded images and buffers.');
      const gltf=await new GLTFLoader().parseAsync(array,'');object=gltf.scene;
    }else{const geometry=new PLYLoader().parse(array);const position=geometry.getAttribute('position');if(!position?.count||position.count>5000000)throw new Error('PLY must contain between 1 and 5 million vertices.');
      object=geometry.index?new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({vertexColors:!!geometry.attributes.color,color:geometry.attributes.color?0xffffff:0xaacabd,side:THREE.DoubleSide})):new THREE.Points(geometry,new THREE.PointsMaterial({size:.04,vertexColors:!!geometry.attributes.color,color:geometry.attributes.color?0xffffff:0xaacabd}));}
    const box=new THREE.Box3().setFromObject(object);if(box.isEmpty()||!Number.isFinite(box.min.length()+box.max.length()))throw new Error('The model has empty or invalid bounds.');
    this.replaceRoot(object,true);this.fit('overview');return this.stats();
  }
  stats(){let points=0,triangles=0;this.object?.traverse(c=>{if(c.geometry){points+=c.geometry.attributes.position?.count??0;if(c.isMesh)triangles+=(c.geometry.index?.count??c.geometry.attributes.position?.count??0)/3;}});
    return {vertices:points,triangles:Math.round(triangles),extent:this.box.getSize(new THREE.Vector3()).toArray()};}
  snapshot(){this.render();return this.canvas.toDataURL('image/png');}
  async export(format){
    if(!this.object)throw new Error('Wait for the model to finish loading.');
    const object=this.object.clone(true);object.traverse(c=>{c.visible=true;});
    if(format==='obj'){const {OBJExporter}=await import('three/addons/exporters/OBJExporter.js');return {data:new OBJExporter().parse(object),type:'text/plain',name:'reconstruction.obj'};}
    if(format==='gltf'){const {GLTFExporter}=await import('three/addons/exporters/GLTFExporter.js');const result=await new GLTFExporter().parseAsync(object,{binary:false,onlyVisible:false});return {data:JSON.stringify(result),type:'model/gltf+json',name:'reconstruction.gltf'};}
    throw new Error('Unsupported export format.');
  }
}
