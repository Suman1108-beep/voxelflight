"""Download an exact original Zurich MAV segment from the official ZIP via HTTP ranges.

No full-archive download, ground-truth reconstruction input, or unsafe extractall.
Every selected member is verified against its ZIP CRC and recorded with SHA-256.
Existing matching files are retained. Conflicting existing files cause an error.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import threading
import time
import zipfile
import zlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL='https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ.zip'
SOURCE='https://rpg.ifi.uzh.ch/zurichmavdataset.html'


def session():
    s=requests.Session()
    retry=Retry(total=5,connect=5,read=5,backoff_factor=.5,status_forcelist=[429,500,502,503,504])
    s.mount('https://',HTTPAdapter(max_retries=retry))
    s.headers.update({'Accept-Encoding':'identity','User-Agent':'VoxelFlight-research-dataset-recovery/1.0'})
    return s


class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.session=session();self.position=0;self.network_bytes=0
        response=self.session.get(url,headers={'Range':'bytes=-65557'},timeout=30)
        response.raise_for_status()
        if response.status_code!=206:raise RuntimeError('Server must honor HTTP Range; refusing full archive.')
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers['Content-Range'])
        if not match:raise RuntimeError('Invalid range response')
        start,end,self.size=map(int,match.groups());self.etag=response.headers.get('ETag')
        self.cache_start=start;self.cache=response.content;self.network_bytes+=len(response.content)

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else (self.position+offset if whence==1 else self.size+offset)
        if self.position<0:raise ValueError('Invalid seek')
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if n<=0:return b''
        if n>32*1024**2:raise RuntimeError('Unexpected large ZIP index read; refusing full archive.')
        start=self.position;self.position+=n
        if self.cache_start<=start and start+n<=self.cache_start+len(self.cache):
            return self.cache[start-self.cache_start:start-self.cache_start+n]
        end=min(self.size-1,start+max(n,65536)-1)
        headers={'Range':f'bytes={start}-{end}'}
        if self.etag:headers['If-Match']=self.etag
        r=self.session.get(self.url,headers=headers,timeout=60);r.raise_for_status()
        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
            raise RuntimeError('Unexpected archive range or archive changed')
        self.cache_start=start;self.cache=r.content;self.network_bytes+=len(self.cache)
        return self.cache[:n]


def select_member(info,first,last,stride):
    p=PurePosixPath(info.filename)
    if info.is_dir() or '..' in p.parts or p.is_absolute():return False
    if 'MAV Images' in p.parts and p.suffix.lower()=='.jpg' and p.stem.isdigit():
        index=int(p.stem)
        return first<=index<=last and (index-first)%stride==0
    if 'Log Files' in p.parts and p.suffix.lower()=='.csv':return True
    return p.name in {'calibration_data.npz','readme.txt'}


def relative_target(name):
    parts=PurePosixPath(name).parts
    for marker in ['MAV Images','Log Files']:
        if marker in parts:
            return Path(*parts[parts.index(marker):])
    return Path(parts[-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first',type=int,default=61201);p.add_argument('--last',type=int,default=63000)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--list-only',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=8 or a.stride<1 or a.first>a.last:raise ValueError('Invalid bounded download settings')
    remote=RemoteZip(URL)
    with zipfile.ZipFile(remote) as archive:
        chosen=[i for i in archive.infolist() if select_member(i,a.first,a.last,a.stride)]
    compressed=sum(i.compress_size for i in chosen);uncompressed=sum(i.file_size for i in chosen)
    frames=sum('MAV Images' in PurePosixPath(i.filename).parts for i in chosen)
    print(json.dumps(dict(archive_bytes=remote.size,selected_files=len(chosen),frames=frames,
        compressed_bytes=compressed,extracted_bytes=uncompressed,index_download_bytes=remote.network_bytes,
        first_names=[i.filename for i in chosen[:5]],metadata=[i.filename for i in chosen if 'MAV Images' not in PurePosixPath(i.filename).parts]),indent=2),flush=True)
    if a.list_only:return
    if not frames:raise RuntimeError('Requested flight segment not found in archive')
    a.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.output).free<uncompressed+2*1024**3:raise RuntimeError('Insufficient free space with safety margin')
    local=threading.local();started=time.monotonic();records=[];downloaded=0
    def fetch(info):
        target=a.output/relative_target(info.filename)
        if target.exists():
            contents=target.read_bytes()
            if len(contents)!=info.file_size or zlib.crc32(contents)&0xffffffff!=info.CRC:
                raise RuntimeError(f'Existing file conflicts with official archive: {target}')
            return dict(path=str(target.relative_to(a.output)),bytes=len(contents),sha256=hashlib.sha256(contents).hexdigest(),zip_crc32=f'{info.CRC:08x}',resumed=True),0
        if not hasattr(local,'session'):local.session=session()
        start=info.header_offset
        # Local ZIP extra fields can exceed those in the central directory. Fetch
        # a generous header prefix, then only the remaining compressed bytes.
        end=min(remote.size-1,start+info.compress_size+65536-1)
        headers={'Range':f'bytes={start}-{end}'}
        if remote.etag:headers['If-Match']=remote.etag
        response=local.session.get(URL,headers=headers,timeout=(15,90));response.raise_for_status()
        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{remote.size}':raise RuntimeError('Invalid member range response')
        blob=response.content
        signature,version,flags,method,mtime,mdate,crc,csize,usize,nlen,xlen=struct.unpack('<IHHHHHIIIHH',blob[:30])
        if signature!=0x04034b50 or flags&1:raise RuntimeError('Invalid or encrypted ZIP entry')
        offset=30+nlen+xlen
        compressed_data=blob[offset:offset+info.compress_size]
        if len(compressed_data)!=info.compress_size:raise RuntimeError('Incomplete compressed entry')
        if method==0:contents=compressed_data
        elif method==8:contents=zlib.decompress(compressed_data,-15)
        else:raise RuntimeError(f'Unsupported ZIP method {method}')
        if len(contents)!=info.file_size or zlib.crc32(contents)&0xffffffff!=info.CRC:raise RuntimeError(f'ZIP integrity failure: {info.filename}')
        target.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive create protects existing results and concurrent download runs.
        with target.open('xb') as f:f.write(contents)
        return dict(path=str(target.relative_to(a.output)),bytes=len(contents),sha256=hashlib.sha256(contents).hexdigest(),zip_crc32=f'{info.CRC:08x}',resumed=False),len(blob)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(fetch,i) for i in chosen]
        for count,future in enumerate(as_completed(futures),1):
            record,size=future.result();records.append(record);downloaded+=size
            if count%50==0 or count==len(chosen):
                print(f'Downloaded and verified {count}/{len(chosen)} files · {downloaded/1024**2:.1f} MiB transferred · {time.monotonic()-started:.0f}s',flush=True)
    manifest=dict(source=SOURCE,archive=URL,archive_etag=remote.etag,archive_bytes=remote.size,
        first_image_id=a.first,last_image_id=a.last,stride=a.stride,frames=frames,
        image_resolution=[1920,1080],sensor_data='Original synchronized logs, not synthesized',
        ground_truth_policy='GroundTruth*.csv is evaluation-only; never reconstruction input.',
        transfer_bytes=downloaded+remote.network_bytes,extracted_bytes=sum(r['bytes'] for r in records),
        download_runtime_s=time.monotonic()-started,files=sorted(records,key=lambda r:r['path']))
    (a.output/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified dataset:',a.output.resolve(),flush=True)


if __name__=='__main__':main()
