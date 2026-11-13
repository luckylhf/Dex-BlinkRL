from pathlib import Path
import argparse,json,time,sys,hashlib,os
import numpy as np,torch,mujoco
from dexnewrl.task import Task
from dexnewrl.learner import Actor
from dexnewrl.robot import NAMES
from blinkmimic.rotations import quat_to_mat
torch.set_num_threads(1)
D=Path(__file__).resolve().parents[1]
def command(tm,speed=.4):
 if tm<10:return 0.
 if tm<12:return speed*(tm-10)/2
 if tm<40:return speed
 if tm<42:return speed*(42-tm)/2
 return 0.
def score(arr,failed,dt=.01):
 phases={}
 for name,lo,hi in [("stand",0,10),("walk",12,40),("stop",42,60)]:
  x=arr[(arr[:,0]>=lo)&(arr[:,0]<hi)]
  phases[name]=dict(samples=len(x),mean_vx=float(x[:,3].mean()) if len(x) else None,speed_MAE_xy=float(np.linalg.norm(x[:,3:5]-np.stack((x[:,5],np.zeros(len(x))),-1),axis=-1).mean()) if len(x) else None,XY_drift=float(np.linalg.norm(x[-1,6:8]-x[0,6:8])) if len(x) else None)
 walk=arr[(arr[:,0]>=12)&(arr[:,0]<40)];lifts=[]
 for j in [8,9]:
  count=0;armed=True
  for z in walk[:,j]:
   if z<.005:armed=True
   if armed and z>.01:count+=1;armed=False
  lifts.append(count)
 complete=not failed and len(arr)==round(60/dt)
 basic=complete and phases["stand"]["speed_MAE_xy"]<.1 and phases["stand"]["XY_drift"]<.25 and phases["walk"]["speed_MAE_xy"]<.15 and phases["stop"]["speed_MAE_xy"]<.1 and phases["stop"]["XY_drift"]<.25
 passed=basic and min(lifts)>=4
 before40=arr[arr[:,0]<40]
 before10=arr[arr[:,0]<10]
 walk_alive=np.isfinite(before40).all() and (before40[:,1]>=.65).all() and (before40[:,2]>=.6).all()
 stand_alive=np.isfinite(before10).all() and (before10[:,1]>=.65).all() and (before10[:,2]>=.6).all()
 walk_pass=bool(walk_alive) and len(walk)==round(28/dt) and phases["walk"]["speed_MAE_xy"]<.15 and min(lifts)>=4
 stand_pass=bool(stand_alive) and phases["stand"]["samples"]==round(10/dt) and phases["stand"]["speed_MAE_xy"]<.1 and phases["stand"]["XY_drift"]<.25
 return dict(standing_initial10_pass=bool(stand_pass),walking28_pass=bool(walk_pass),completed=bool(complete),basic_speed_and_stability=bool(basic),passed=bool(passed),duration=len(arr)*dt,min_height=float(arr[:,1].min()),min_upright=float(arr[:,2].min()),whole_foot_lifts_steady_walk=lifts,phases=phases)
def evaluate(path,variant="ema",scenario="primary",speed=.4,seeds=(1000,1001,1002)):
 N=len(seeds)
 assert scenario in ("primary","stand_only","static"), scenario
 ck=torch.load(path,map_location="cpu",weights_only=False);dt=ck.get("contract",{}).get("control_dt",.01);task=Task("cpu",dt=dt,stand_phase=ck.get("args",{}).get("stand_phase",False),feet_only_fk=ck.get("args",{}).get("feet_only_fk",False));pi=Actor(task,kind=ck.get("args",{}).get("actor_arch","standard")).eval();pi.load_state_dict(ck["actor_live" if variant=="live" else "actor"])
 assert all(bool(torch.isfinite(v).all()) for v in pi.state_dict().values())
 physics=Path(os.environ.get("DEX_PHYSICS_MODEL",str(path.parent/"physics_scene.mjb"))).resolve();m=mujoco.MjModel.from_binary_path(str(physics));qids=[m.joint("robot/"+n).qposadr[0] for n in NAMES];vids=[m.joint("robot/"+n).dofadr[0] for n in NAMES];aids=[next(i for i in range(m.nu) if m.actuator_trnid[i,0]==m.joint("robot/"+n).id) for n in NAMES]
 geoms=[np.where((m.geom_bodyid==m.body("robot/"+n).id)&(m.geom_type==mujoco.mjtGeom.mjGEOM_CYLINDER))[0] for n in ["ankle_roll_l_link","ankle_roll_r_link"]]
 ds=[];traces=[[] for _ in range(N)];alive=np.ones(N,bool);failed=np.zeros(N,bool)
 for seed in range(N):
  d=mujoco.MjData(m);d.qpos[:7]=[0,0,.995,1,0,0,0];d.qpos[qids]=task.q0.numpy()+np.random.default_rng(seeds[seed]).normal(0,.01,23);mujoco.mj_forward(m,d);ds.append(d)
 ap=torch.zeros(N,23);hist=None;begin=time.monotonic();actions=[];qposes=[]
 with torch.inference_mode():
  for k in range(round(60/dt)):
   tm=k*dt;target_vx=command(tm,speed) if scenario=="primary" else 0.;cmd=torch.tensor([[target_vx,0.,0.]]*N)
   R=quat_to_mat(torch.tensor(np.stack([d.qpos[3:7] for d in ds]),dtype=torch.float32))
   v=torch.tensor(np.stack([d.qvel[:3] for d in ds]),dtype=torch.float32);w=torch.einsum("nij,nj->ni",R,torch.tensor(np.stack([d.qvel[3:6] for d in ds]),dtype=torch.float32))
   S=(torch.tensor(np.stack([d.qpos[:3] for d in ds]),dtype=torch.float32),R,v,w,torch.tensor(np.stack([d.qpos[qids] for d in ds]),dtype=torch.float32),torch.tensor(np.stack([d.qvel[vids] for d in ds]),dtype=torch.float32))
   o=task.obs(S,ap,cmd,torch.full((N,),k));hist=o[:,None].repeat(1,10,1) if hist is None else torch.cat((hist[:,1:],o[:,None]),1)
   a=pi(hist);target=(task.q0+.25*a).numpy();actions.append(a.numpy());qposes.append(np.stack([d.qpos for d in ds]).astype(np.float32))
   for i,d in enumerate(ds):
    if not alive[i]:continue
    d.ctrl[aids]=target[i];mujoco.mj_step(m,d,nstep=task.substeps);mujoco.mj_forward(m,d)
    rot=quat_to_mat(torch.tensor(d.qpos[3:7],dtype=torch.float32)).numpy();vb=rot.T@d.qvel[:3]
    zz=[]
    for ids in geoms:
     axisz=d.geom_xmat[ids].reshape(-1,3,3)[:,2,2].clip(-1,1);extent=m.geom_size[ids,0]*np.sqrt(np.maximum(1-axisz**2,0))+m.geom_size[ids,1]*abs(axisz);zz.append(float(np.min(d.geom_xpos[ids,2]-extent)))
    traces[i].append([tm,float(d.qpos[2]),float(rot[2,2]),float(vb[0]),float(vb[1]),target_vx,float(d.qpos[0]),float(d.qpos[1]),*zz])
    bad=d.qpos[2]<.65 or rot[2,2]<.6 or not np.isfinite(d.qpos).all()
    failed[i]|=bad;alive[i]=not bad
   ap=a
   if not alive.any():break
 results=[dict(seed=seeds[i],**score(np.asarray(traces[i]),failed[i],dt=dt)) for i in range(N)]
 if scenario in ("stand_only","static"):
  for r,trace in zip(results,traces):
   ar=np.asarray(trace);r["static_MAE_xy"]=float(np.linalg.norm(ar[:,3:5],axis=-1).mean());r["static_XY_drift"]=float(np.linalg.norm(ar[-1,6:8]-ar[0,6:8]));r["passed"]=bool(r["completed"] and r["static_MAE_xy"]<.1 and r["static_XY_drift"]<.25)
 for rr,trace in zip(results,traces):
  ar=np.asarray(trace);rr["forward_progress_m"]=float(ar[-1,6]-ar[0,6]);rr["max_lateral_displacement_m"]=float(abs(ar[:,7]-ar[0,7]).max());rr["max_foot_clearance_m"]=ar[:,8:10].max(0).tolist()
  if scenario!="primary":rr["walking28_pass"]=None
 out=dict(physics_model=str(physics),protocol_version=4,control_dt=dt,standing_phase_encoding=task.stand_phase,feet_only_forward_kinematics=task.feet_only_fk,evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command_speed=speed,scenario=scenario,checkpoint=str(path),variant=variant,training_seconds=ck.get("training_seconds",ck["wall_seconds"]),process_wall_seconds=ck["wall_seconds"],prewarm_seconds=ck.get("prewarm_seconds"),iteration=ck["iteration"],sim_steps=ck["sim_steps"],backend="nativeMuJoCo3.11,FP32actor,independentnominalstart",evaluation_seconds=time.monotonic()-begin,completed=sum(r["completed"] for r in results),passed=sum(r["passed"] for r in results),results=results)
 dest=Path(os.environ.get("DEX_EVAL_OUTPUT",str(path.parent/"evaluations_cpu")))/f"v{speed:g}";dest.mkdir(parents=True,exist_ok=True);stem=path.stem+"_"+variant+"_"+scenario+"_seeds"+"-".join(map(str,seeds))
 np.savez_compressed(dest/(stem+".npz"),**{f"trajectory_{i}":np.asarray(x,np.float32) for i,x in enumerate(traces)},action=np.asarray(actions,np.float32),qpos=np.asarray(qposes,np.float32))
 (dest/(stem+".json")).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);return out
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument('checkpoints',nargs='+');p.add_argument('--variant',choices=['ema','live','both'],default='both');p.add_argument('--scenario',choices=['primary','stand_only','static','both'],default='both');p.add_argument('--speed',type=float,default=.4);p.add_argument('--seeds',type=int,nargs='+',default=[1000,1001,1002]);a=p.parse_args()
 for name in a.checkpoints:
  for variant in (['live','ema'] if a.variant=='both' else [a.variant]):
   for scenario in (['stand_only','primary'] if a.scenario=='both' else [a.scenario]):evaluate(Path(name),variant,scenario,a.speed,a.seeds)
