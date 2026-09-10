// Presentation wording only. Preserve the original downloadable run reports.
const messages=new Map([
  ['Loading cached pretrained MapAnything weights onto Apple GPU','Loading cached reconstruction models onto the processing device'],
  ['Experimental Mac inference: no independent surface or trajectory accuracy measured.','Preview reconstruction: surface and trajectory accuracy have not been independently measured.'],
  ['Experimental reconstruction: no independent surface or trajectory accuracy measured.','Preview reconstruction: surface and trajectory accuracy have not been independently measured.'],
  ['No semantic segmentation, inertial fusion, bundle adjustment or Gaussian-splat training in this Mac mode.','This preview workflow does not include semantic segmentation, inertial fusion, bundle adjustment or Gaussian-splat training.'],
  ['No semantic segmentation, inertial fusion, bundle adjustment or Gaussian-splat training in this processing mode.','This preview workflow does not include semantic segmentation, inertial fusion, bundle adjustment or Gaussian-splat training.'],
  ['Processing cancelled. Uploaded files remain on this Mac.','Processing cancelled. Uploaded files remain on the processing workstation.'],
  ['The Mac is processing another run. Wait or cancel that run.','Another reconstruction is in progress. Wait for it to finish or cancel the active run.'],
  ['Model cache or Apple GPU is not ready yet','The processing engine is not ready. Check model availability and the workstation configuration.']
]);
export const engineMessage=value=>messages.get(value)??value;
