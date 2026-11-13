from pathlib import Path
import os,time,sys,json,subprocess,shutil
from snapshot import snapshot,ROOT
LAUNCH_START=time.monotonic();LAUNCH_UNIX=time.time();prep={}
for sub in ['runs','audit','backups','generated','cache']:(ROOT/sub).mkdir(exist_ok=True)
name=sys.argv[1];args=sys.argv[2:];assert '/' not in name and not (ROOT/'runs'/name).exists()
t=time.monotonic();manifest=snapshot('run_'+name);prep['backup_seconds']=time.monotonic()-t
t=time.monotonic()
frozen=ROOT/'audit/sources'/name;frozen.mkdir(parents=True,exist_ok=False)
for sub in ['src','runtime','tools','configs']:shutil.copytree(ROOT/sub,frozen/sub,ignore=shutil.ignore_patterns('__pycache__'))
prep['source_freeze_seconds']=time.monotonic()-t
env=os.environ.copy();env.update(DEX_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(frozen/'src')+':'+str(frozen/'runtime'),DEX_CAPTURE_ARENA='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cache=ROOT/'cache'/name
t=time.monotonic()
cache_source=os.environ.get('DEX_REUSE_CACHE_FROM')
cache_state='new empty run-specific torch/triton/warp cache'
if cache_source:
 assert '/' not in cache_source and cache_source not in ('.','..')
 source=ROOT/'cache'/cache_source;assert source.is_dir() and not cache.exists()
 shutil.copytree(source,cache)
 cache_state='independent copy of compile caches from '+cache_source+'; no learned state read'
prep['cache_prepare_seconds']=time.monotonic()-t
for key,sub in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('DEX_WARP_CACHE','warp')]:env[key]=str(cache/sub)
cmd=[os.environ.get('DEX_PYTHON',sys.executable),'-B','-u','-m','dexnewrl.train','--out',str(ROOT/'runs'/name),*args]
env['DEX_PROCESS_MONOTONIC']=str(time.monotonic());begin=time.time()
job=ROOT/'audit'/(name+'_JOB.json')
prep['total_before_child_seconds']=time.monotonic()-LAUNCH_START
record={'preparation':dict(prep,launch_unix=LAUNCH_UNIX),'command':cmd,'source_manifest':manifest,'cache_state':cache_state,'log':str(ROOT/'runs'/(name+'.log')),'environment':{k:v for k,v in env.items() if k in ['PATH','HOME','USER','LANG','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','PYTHONPATH','DEX_PROJECT_ROOT','DEX_CAPTURE_ARENA','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','DEX_WARP_CACHE','DEX_PROCESS_MONOTONIC']}}
job.write_text(json.dumps(record,indent=2))
p=subprocess.Popen([sys.executable,str(ROOT/'tools/job.py'),str(job)],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
(ROOT/'audit'/(name+'_LAUNCH.json')).write_text(json.dumps({'supervisor_pid':p.pid,'job':str(job),'start_unix':begin},indent=2));print(json.dumps({'supervisor_pid':p.pid,'command':cmd,'log':record['log']}))
