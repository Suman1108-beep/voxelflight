export const SINGLE_IMAGE_MESSAGE='A single image cannot establish a complete, metrically accurate 3D scene. Choose an MP4 or MOV video with overlapping views.';
export function validateVideoFile(file){
  if(!file||!file.size)return 'Choose a non-empty MP4 or MOV video.';
  if(file.type?.startsWith('image/')||/\.(png|jpe?g|webp|gif|heic|heif|tiff?|bmp|avif)$/i.test(file.name))return SINGLE_IMAGE_MESSAGE;
  if(!/\.(mp4|mov)$/i.test(file.name))return 'Choose an MP4 or MOV video.';
  if(file.size>1024**3)return 'The video must be no larger than 1 GiB.';
  return '';
}
