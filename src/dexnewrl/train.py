"""Cold-only command learner. Training clock starts after verified prewarm reset; process cost is recorded separately."""
import time,os
START=float(os.environ.get('DEX_PROCESS_MONOTONIC',time.monotonic()))
import argparse,json,hashlib,random,traceback
from pathlib import Path
import numpy as np
import torch
import warp as wp
if os.environ.get('DEX_WARP_CACHE'):wp.config.kernel_cache_dir=os.environ['DEX_WARP_CACHE']
READS=[]
def forbidden_load(*args,**kwargs):
 READS.append(str(args[0]) if args else '<unknown>')
 raise RuntimeError('Cold training forbids checkpoint reads: '+str(READS[-1]))
torch.load=forbidden_load
torch.jit.load=forbidden_load
from .robot import ROOT,NAMES,MODEL_NAMES
from .task import Task
from .env import Env
from .learner import Learner
from .cold_start import ColdStart
from blinkmimic.buffer import ReplayBuffer

def atomic_json(path,obj):
 p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(obj,indent=2));p.replace(path)
def tensor_hashes(module):
 return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in module.state_dict().items()}
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=900);p.add_argument('--envs',type=int,default=2048)
 p.add_argument('--batch',type=int,default=128);p.add_argument('--horizon',type=int,default=96);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--iterations',type=int,default=10000000);p.add_argument('--wm-only',type=int,default=100)
 p.add_argument('--no-graphs',action='store_true');p.add_argument('--no-compile',action='store_true')
 p.add_argument('--wm-inputs',choices=['standard','grounded'],default='standard')
 p.add_argument('--wm-hidden',type=int,default=1024);p.add_argument('--wm-window',type=int,default=16);p.add_argument('--wm-batch',type=int,default=256)
 p.add_argument('--actor-lr',type=float,default=.0003);p.add_argument('--wm-lr',type=float,default=.0004)
 p.add_argument('--nconmax',type=int,default=256);p.add_argument('--njmax',type=int,default=1000)
 p.add_argument('--feet-only-fk',action='store_true')
 p.add_argument('--stand-phase',action='store_true')
 p.add_argument('--collect-graph',action='store_true')
 p.add_argument('--imag-unroll',type=int,choices=[1,2,4],default=1)
 p.add_argument('--actor-arch',choices=['standard','xsoma'],default='standard')
 p.add_argument('--checkpoint-interval',type=int,default=0,help='Wall-training seconds between checkpoints;0 keeps the control-recipe schedule')
 p.add_argument('--control-dt',type=float,choices=[.01,.02],default=.01);p.add_argument('--actor-opt',choices=['muon','adam'],default='muon');p.add_argument('--imag-bf16',action='store_true');p.add_argument('--ema-decay',type=float,default=None);p.add_argument('--hard-termination',action='store_true')
 p.add_argument('--profile-prewarm',action='store_true',help='Measure collect/world-model/policy replay cost before discarding prewarm state')
 p.add_argument('--skip-prewarm',action='store_true',help='Diagnostic only; default compiles then resets all learned state')
 p.add_argument('--warmup',type=int,default=64);p.add_argument('--recipe',default='r5_control')
 a=p.parse_args();assert not (a.profile_prewarm and a.no_graphs),'prewarm profiling requires graphs';out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 atomic_json(out/'args.json',vars(a));stages=[]
 def stamp(name):
  torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();stages.append({'stage':name,'wall_seconds':time.monotonic()-START,'device_free_bytes':free,'device_total_bytes':total});atomic_json(out/'timing.json',stages)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=True;torch.set_num_threads(4)
 stamp('imports_ready')
 task=Task('cuda',dt=a.control_dt,stand_phase=a.stand_phase,feet_only_fk=a.feet_only_fk);task.nconmax=a.nconmax;task.njmax=a.njmax;stamp('task_ready');env=Env(a.envs,task);stamp('physics_ready')
 buf=ReplayBuffer(256,a.envs,'cuda',23,6);buf.use_device_ptr(device_size=True)
 learner=Learner(task,buf,B=a.batch,H=a.horizon,warm=None,graphs=not a.no_graphs,compile_steps=not a.no_compile,W=a.wm_window,wm_hidden=a.wm_hidden,wm_batch=a.wm_batch,actor_lr=a.actor_lr,wm_lr=a.wm_lr,actor_opt=a.actor_opt,imag_bf16=a.imag_bf16,ema_decay=a.ema_decay,hard_termination=a.hard_termination,actor_arch=a.actor_arch,imag_unroll=a.imag_unroll,wm_inputs=a.wm_inputs)
 stamp('random_models_ready')
 initial={'seed':a.seed,'actor':tensor_hashes(learner.actor),'ema':tensor_hashes(learner.ema),'world_model':tensor_hashes(learner.wm),'checkpoint_reads':READS,'critic':None,'amp_discriminator':None,'data_prior':None,'reference_skip':False,'fixed_standing_controller':False,'sample_steps':0,'optimizer_update_counts':learner.update_counts.cpu().tolist(),'initialization':'fresh torch seed; fresh optimizer states and normalizer statistics'}
 atomic_json(out/'INITIALIZATION.json',initial)
 import shutil
 shutil.copy2(ROOT/'generated/physics_scene.mjb',out/'physics_scene.mjb')
 manifest={'NAMES':list(NAMES),'MODEL_NAMES':list(MODEL_NAMES),'obs_dim':840,'actions':23,'control_dt':task.dt,'physics_dt':.0025,'physics_substeps':task.substeps,'history_seconds':10*task.dt,'horizon_s':a.horizon*task.dt,'world_model_window_seconds':a.wm_window*task.dt,'world_model_inputs':a.wm_inputs,'full_collection_graph':a.collect_graph,'standing_phase_encoding':a.stand_phase,'feet_only_forward_kinematics':a.feet_only_fk,'actor_optimizer':a.actor_opt,'actor_arch':a.actor_arch,'imagination_mlp_bf16':a.imag_bf16,'actor_random':True,'world_model_source':None,'policy_future_reference':False,'policy_privileged_velocity':False,'physics_sha256':hashlib.sha256((out/'physics_scene.mjb').read_bytes()).hexdigest(),'cache':{k:os.environ.get(k) for k in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','DEX_WARP_CACHE']}}
 atomic_json(out/'CONTRACT.json',manifest)
 it=0;steps=0;metrics={};lossmean=None;nonfinite=0;nextsave=0;training_start=None
 def training_elapsed():return 0. if training_start is None else time.monotonic()-training_start
 deadlines=[60,120,180,240,295,600,900]
 if a.checkpoint_interval>0:
  deadlines=sorted(set(list(range(a.checkpoint_interval,int(a.seconds),a.checkpoint_interval))+([int(a.seconds)-5] if a.seconds>=10 else [])))
 def save(label):
  torch.cuda.synchronize();wall=time.monotonic()-START;counts=learner.update_counts.cpu().tolist()
  ck={'actor':learner.ema.state_dict(),'actor_live':learner.actor.state_dict(),'wm':learner.wm.state_dict(),'iteration':it,'wall_seconds':wall,'training_seconds':training_elapsed(),'prewarm_seconds':training_start-START if training_start is not None else None,'sim_steps':steps,'args':vars(a),'contract':manifest,'metrics':metrics,'opt_actor':learner.oa.state_dict(),'opt_wm':learner.ow.state_dict(),'update_counts':counts,'checkpoint_reads':list(READS)}
  tmp=out/(label+'.tmp');torch.save(ck,tmp);tmp.replace(out/label)
  row={'checkpoint':label,'iteration':it,'wall_seconds':wall,'training_seconds':training_elapsed(),'save_completed_training_seconds':None,'save_completed_wall_seconds':time.monotonic()-START,'sim_steps':steps,'update_counts':counts,'metrics':metrics}
  row['save_completed_training_seconds']=training_elapsed()
  with (out/'checkpoints.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  atomic_json(out/'status.json',row)
 from .collector import Collector
 collect=Collector(env,learner,buf,graph=a.collect_graph)
 save('initial.pt');stamp('initial_saved')
 if not a.skip_prewarm:
  cold=ColdStart(learner)
  for _ in range(a.warmup):collect()
  for _ in range(4):learner.update(allow_policy=True)
  stamp('compilation_and_capture_finished')
  with torch.no_grad():
   checked=learner._imag(tuple(v[:2] for v in env.state()),env.hist[:2],env.a_prev[:2],env.cmd[:2],env.base_cmd[:2],env.step_count[:2])
   outputs=list(checked[0])+list(checked[1:]);dtypes=[str(v.dtype) for v in outputs]
   assert all(v.dtype==torch.float32 for v in outputs),'integration/reward/observation must stay FP32'
   atomic_json(out/'PRECISION_CHECK.json',{'imagined_mlp_bf16':a.imag_bf16,'state_action_history_reward_done_command_dtypes':dtypes,'passed':True,'scope':'outer integration, geometry/reward/observation outputs; inner MLP autocast only'})
  if a.profile_prewarm:
   samples=[]
   for i in range(8):
    row={}
    for name,fn in [('collect',collect),('world_model_updates',lambda:[learner.wmg.replay() for _ in range(3)]),('policy_updates',lambda:[learner.pig.replay() for _ in range(2)])]:
     torch.cuda.synchronize();e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);t=time.monotonic();e0.record();fn();enqueue=time.monotonic()-t;e1.record();torch.cuda.synchronize()
     row[name]={'host_enqueue_seconds':enqueue,'completed_seconds':time.monotonic()-t,'cuda_elapsed_ms':e0.elapsed_time(e1)}
    samples.append(row)
   atomic_json(out/'PREWARM_PROFILE.json',{'samples':samples,'excluded_from_training':True,'all_resulting_learning_state_discarded_before_clock':True})
  receipt=cold.restore(learner,buf,env,a.seed+100000)
  atomic_json(out/'PREWARM_RESET.json',receipt);stamp('cold_state_restored')
 training_start=time.monotonic()
 save('cold_start.pt')
 for _ in range(a.warmup):tr,term=collect();steps+=a.envs
 save('prepared.pt');stamp('initial_collection_ready')
 log=(out/'train.jsonl').open('w');failure=None
 try:
  while it<a.iterations and training_elapsed()<a.seconds and not (out/'STOP').exists() and not (ROOT/'STOP').exists():
   t0=time.monotonic();tr,term=collect();steps+=a.envs
   allow=it>=a.wm_only and (lossmean is None or metrics.get('wm_loss',0)<2*lossmean)
   w,pi=learner.update(allow_policy=allow);it+=1
   if it%10==0 or it==1 or (pi is not None and 'first_policy_update' not in [x['stage'] for x in stages]):
    torch.cuda.synchronize();wl=float(w[0]);lossmean=wl if lossmean is None else .95*lossmean+.05*wl
    if not np.isfinite(wl) or not bool(w[1]):nonfinite+=1
    if pi is not None and not np.isfinite(float(pi[0])):nonfinite+=1
    S=env.state();vb=torch.einsum('nji,nj->ni',S[1],S[2])
    metrics={'wm_loss':wl,'actor_loss':float(pi[0]) if pi else None,'imag_survival':float(pi[1]) if pi else None,'actor_grad_norm':float(pi[2]) if pi else None,'actor_update_ok':bool(pi[3]) if pi else None,'policy_allowed':bool(allow),'nonfinite':nonfinite,'mean_height':float(S[0][:,2].mean()),'mean_abs_vx_error':float((vb[:,0]-env.cmd[:,0]).abs().mean()),'mean_vx':float(vb[:,0].mean()),'mean_cmd_vx':float(env.cmd[:,0].mean()),'sim_reward':float(tr['r'].mean()),'termination_fraction':float(term.float().mean()),'iteration_seconds':time.monotonic()-t0,'learning_graphs':learner.pig is not None,'physics_graph':env.physics_graph is not None or collect.graph is not None,'full_collection_graph':collect.graph is not None,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'update_counts':learner.update_counts.cpu().tolist()}
    data=env.sim.wp_data
    metrics['constraints_max']=int(wp.to_torch(data.nefc).max());metrics['contacts_total']=int(wp.to_torch(data.nacon)[0]);metrics['contacts_capacity']=int(data.naconmax)
    if metrics['constraints_max']>=data.njmax or metrics['contacts_total']>=data.naconmax:raise RuntimeError('Physics buffer capacity reached; reject run instead of accepting truncated contacts')
    row={'iteration':it,'wall_seconds':time.monotonic()-START,'training_seconds':training_elapsed(),'sim_steps':steps,**metrics};log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    if pi is not None and 'first_policy_update' not in [x['stage'] for x in stages]:stamp('first_policy_update')
    if nonfinite>=3:raise RuntimeError('repeated nonfinite metrics')
   if nextsave<len(deadlines) and training_elapsed()>=deadlines[nextsave]:
    due=deadlines[nextsave];save(f'train_{due:04d}.pt')
    while nextsave<len(deadlines) and deadlines[nextsave]<=training_elapsed():nextsave+=1
 except BaseException as e:
  failure=repr(e);atomic_json(out/'FAILURE.json',{'error':failure,'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-START});raise
 finally:
  save('last.pt');log.close();atomic_json(out/'FINISHED.json',{'failure':failure,'iteration':it,'sim_steps':steps,'wall_seconds':time.monotonic()-START,'training_seconds':training_elapsed(),'prewarm_seconds':training_start-START,'checkpoint_reads':READS,'update_counts':learner.update_counts.cpu().tolist(),'stopped':(out/'STOP').exists() or (ROOT/'STOP').exists()})
if __name__=='__main__':main()
