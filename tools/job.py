from pathlib import Path
import os,sys,json,subprocess,time,traceback,xml.etree.ElementTree as ET
job=Path(sys.argv[1]);cfg=json.loads(job.read_text());root=job.parents[1];begin=time.monotonic();start=time.time()
with open(cfg['log'],'x') as out:
 p=subprocess.Popen(cfg['command'],cwd=root,env=cfg['environment'],stdout=out,stderr=subprocess.STDOUT)
 cfg['child_pid']=p.pid;cfg['start_unix']=start;job.write_text(json.dumps(cfg,indent=2))
 with open(str(job)+'.resources.jsonl','x') as res:
  while p.poll() is None:
   try:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    processes=[{key:node.findtext(key) for key in ['pid','process_name','type','used_memory']} for node in xml.findall('.//process_info')]
    res.write(json.dumps({'unix':time.time(),'elapsed':time.monotonic()-begin,'gpu':gpu,'gpu_processes':processes})+'\n');res.flush()
   except Exception as e:res.write(json.dumps({'error':str(e)})+'\n')
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:pass
 record={'returncode':p.returncode,'process_wall_seconds':time.monotonic()-begin,'start_unix':start,'end_unix':time.time(),'pid':p.pid}
 job.with_name(job.stem+'_RESULT.json').write_text(json.dumps(record,indent=2))
