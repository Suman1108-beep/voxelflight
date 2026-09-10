export const views=Object.freeze({
  workspace:{label:'Workspace',title:'Back to the workspace',detail:'Your scene and camera view stay right where you left them.',color:'mint'},
  validation:{label:'Validation',title:'Opening validation',detail:'Inspect the saved accuracy and appearance evidence.',color:'violet'},
  pipeline:{label:'Pipeline',title:'Opening the pipeline',detail:'Explore the processing stages and recorded run timings.',color:'blue'}
});
export const normalizeView=name=>Object.hasOwn(views,name)?name:'workspace';

// This is a brief visual transition, not model processing or invented load progress.
// Commit the new view immediately; only the decorative overlay has a duration.
export function createViewNavigation({render,begin,finish,motionDisabled=()=>false,schedule=setTimeout,cancel=clearTimeout,duration=480}){
  let current=null,timer=null,visible=false,revision=0;
  function settle(){revision++;if(timer!==null)cancel(timer);timer=null;if(visible){visible=false;finish(current);}}
  function go(requested,{animate=true}={}){
    const next=normalizeView(requested);if(next===current)return false;
    if(timer!==null)cancel(timer);timer=null;const ticket=++revision;
    const moving=animate&&!motionDisabled();
    if(moving){begin({...views[next],id:next});visible=true;}
    const previous=current;current=next;
    try{render(next);}catch(error){current=previous;settle();throw error;}
    if(moving)timer=schedule(()=>{if(ticket===revision)settle();},duration);
    else settle();
    return true;
  }
  return {go,settle,get current(){return current;}};
}
