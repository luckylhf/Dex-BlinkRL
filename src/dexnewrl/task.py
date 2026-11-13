"""Reference-free Dex command task. Actor contract is original84x10, allscales1."""
import torch,numpy as np,mujoco
from blinkmimic.kinematics import Kinematics
from blinkmimic.rotations import mv3,mm3,axis_angle_to_mat,quat_to_mat
from .robot import SPEC,NAMES,DEFAULT
DT=.01
class Task:
 def __init__(self,device,dt=.01,stand_phase=False,feet_only_fk=False):
  assert dt in (.01,.02)
  self.dt=float(dt);self.substeps=round(dt/.0025);self.stand_phase=stand_phase;self.feet_only_fk=feet_only_fk
  self.action_rate_scale=(.01/dt)**2
  self.device=device
  self.mirror_perm=torch.tensor([NAMES.index(n.replace("_l_","_SIDE_").replace("_r_","_l_").replace("_SIDE_","_r_")) for n in NAMES],device=device)
  self.mirror_sign=torch.tensor([-1. if "_roll_" in n or "_yaw_" in n else 1. for n in NAMES],device=device)
  self.reflection=torch.tensor([1.,-1.,1.],device=device)
  self.ramp_limit=torch.tensor([.005,.005,.01],device=device)*(dt/.01)
  self.kin=Kinematics(SPEC,device);self.q0=torch.tensor([DEFAULT.get(n,0.) for n in NAMES],device=device)
  self.qlo,self.qhi=self.kin.q_lo,self.kin.q_hi
  self.foot_body_indices=tuple(self.kin.body_names.index(n) for n in SPEC.feet)
  chain=set()
  for i in self.foot_body_indices:
   while i>=0:chain.add(i);i=self.kin.parent[i]
  self.foot_chain=tuple(sorted(chain))
  self.feet=torch.tensor([1,2],device=device);self.phase_offset=torch.tensor([.38,.88],device=device)
  self.posew=torch.tensor([50. if any(t in n for t in ["waist","shoulder","elbow"]) else (.01 if "hip_pitch" in n or "knee" in n else (5. if "ankle" in n or "hip_yaw" in n else 1.)) for n in NAMES],device=device)
  m=mujoco.MjModel.from_xml_path(SPEC.mjcf_path);ids=[]
  for n in SPEC.feet:
   ii=np.where(m.geom_bodyid==m.body(n).id)[0];ii=ii[m.geom_type[ii]==mujoco.mjtGeom.mjGEOM_CYLINDER];assert len(ii)==7,(n,ii);ids.append(ii)
  ids=np.stack(ids)
  self.gpos=torch.tensor(m.geom_pos[ids],device=device,dtype=torch.float32)
  self.grot=quat_to_mat(torch.tensor(m.geom_quat[ids],device=device,dtype=torch.float32))
  self.radius=torch.tensor(m.geom_size[ids,0],device=device,dtype=torch.float32);self.half=torch.tensor(m.geom_size[ids,1],device=device,dtype=torch.float32)
 def phase(self,step):return (step.float().unsqueeze(-1)*self.dt/.85+self.phase_offset)%1
 def obs(self,S,a_prev,cmd,step):
  p,R,v,w,q,qd=S;phase=self.phase(step)
  if self.stand_phase:
   standing=(cmd[...,:2].norm(dim=-1)<.01)&(cmd[...,2].abs()<.01)
   phase=torch.where(standing[...,None],.5,phase)
  return torch.cat((mv3(R.transpose(-1,-2),w),-R[...,2,:],cmd,q-self.q0,qd,a_prev,torch.sin(2*torch.pi*phase),torch.cos(2*torch.pi*phase),torch.ones_like(phase)*.38),-1).clamp(-100,100)
 def features(self,S,ap,cmd,f):
  return torch.cat((self.obs(S,ap,cmd,f),mv3(S[1].transpose(-1,-2),S[2]),self.feet_geometry(S)[2]),-1)
 def sample_commands(self,N):
  c=torch.zeros(N,3,device=self.q0.device);c[:,0]=.05+.55*torch.rand(N,device=c.device)
  return torch.where((torch.rand(N,device=c.device)<.35)[:,None],0.,c)
 def ramp(self,c,target):
  return c+(target-c).clamp(-self.ramp_limit,self.ramp_limit)
 def foot_poses(self,S):
  # Same MJCF transforms and arithmetic as Kinematics.forward, only the two ancestor chains.
  p,R,_,_,q,_=S;kin=self.kin;P={};Rs={}
  for i in self.foot_chain:
   par=kin.parent[i]
   if par<0:P[i]=p;Rs[i]=R
   else:
    P[i]=P[par]+mv3(Rs[par],kin.off_p[i])
    Rs[i]=mm3(mm3(Rs[par],kin.off_R[i]),axis_angle_to_mat(kin.axes[i],q[...,kin.joint_idx[i]]))
  return torch.stack([P[i] for i in self.foot_body_indices],-2),torch.stack([Rs[i] for i in self.foot_body_indices],-3)
 def sole_geometry(self,S):
  if self.feet_only_fk:fp,fr=self.foot_poses(S)
  else:
   bp,bR,bv,bw=self.kin.forward(*S);fp,fr=bp[...,self.feet,:],bR[...,self.feet,:,:]
  center=fp.unsqueeze(-2)+(fr.unsqueeze(-3)@self.gpos.unsqueeze(-1)).squeeze(-1)
  axes=fr.unsqueeze(-3)@self.grot;zz=axes[...,2,2].clamp(-1,1)
  extent=self.radius*torch.sqrt((1-zz.square()).clamp(min=1e-9))+self.half*zz.abs()
  return fp,fr,center[...,2]-extent
 def feet_geometry(self,S):
  fp,fr,z=self.sole_geometry(S)
  return fp,fr,z.min(-1).values
 def reward(self,S,a,ap,cmd,step):
  p,R,v,w,q,qd=S;vb=mv3(R.transpose(-1,-2),v);wb=mv3(R.transpose(-1,-2),w)
  fp,fr,z=self.feet_geometry(S)
  phase=self.phase(step);swing=torch.where(phase<.38,.065*torch.sin(torch.pi*phase/.38),0.)
  swing=swing*(cmd[...,:2].norm(dim=-1,keepdim=True)>.05)
  lateral=((fp[...,0,:2]-fp[...,1,:2])*R[...,:2,1]).sum(-1).abs()
  # All geometry and terrain(planez0) are analytic. No learnedreward orcritic.
  track=2*torch.exp(-(vb[...,:2]-cmd[...,:2]).square().sum(-1)/.25)+1.5*torch.exp(-(wb[...,2]-cmd[...,2]).square()/.25)
  phase_reward=5*torch.exp(-(z-swing).square().sum(-1)/.008)
  penalty=(wb[...,:2].square().sum(-1)+10*R[...,2,:2].square().sum(-1)+2*self.action_rate_scale*(a-ap).square().sum(-1)+.5*((q-self.q0).square()*self.posew).sum(-1)+10*torch.relu(.14-lateral).square()+5*fr[...,2,:2].square().sum((-1,-2)))
  penetration=20*torch.relu(-z-.003).square().sum(-1)
  softlimit=2*(torch.relu(self.qlo+.05-q)+torch.relu(q-self.qhi+.05)).square().sum(-1)
  return self.dt*(track+phase_reward+10-.5*penalty-penetration-softlimit-vb[...,2].square())
 def soft_done(self,S):
  p,R=S[:2];return 1-torch.sigmoid((p[...,2]-.65)/.025)*torch.sigmoid((R[...,2,2]-.6)/.04)
 def done(self,S):
  return (S[0][...,2]<.65)|(S[1][...,2,2]<.6)|~torch.isfinite(torch.cat((S[0],S[2],S[3],S[4],S[5]),-1)).all(-1)
