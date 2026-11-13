"""XSoma-inspired command MBRL: analytic 0.96s return, no critic, online dynamics."""
import copy,torch
from torch import nn
from blinkmimic.networks import WorldModel,RunningNorm,mlp
from blinkmimic.optim import MuonAdam
from blinkmimic.mdp import unflatten_state,flatten_state
from .robot import NAMES,MODEL_NAMES
class Actor(nn.Module):
 def __init__(self,task,kind="standard"):
  super().__init__();self.kind=kind;self.norm=RunningNorm(84);self.net=mlp(840,[512,256,128],23)
  if kind=="xsoma":
   self.net=nn.Sequential(self.net[0],nn.LayerNorm(512,eps=1e-6),*list(self.net)[1:])
   # Keep later world-model RNG draws identical to the standard actor comparison.
   with torch.random.fork_rng(devices=[]):
    linear=[m for m in self.net if isinstance(m,nn.Linear)]
    for m in linear:nn.init.orthogonal_(m.weight,gain=2**.5);nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linear[-1].weight,gain=1.)
  elif kind!="standard":raise ValueError(kind)
  self.register_buffer("alo",(task.qlo-task.q0)/.25);self.register_buffer("ahi",(task.qhi-task.q0)/.25)
  with torch.no_grad():self.net[-1].weight.mul_(.01);self.net[-1].bias.zero_()
 def forward(self,h,amp_dtype=None):
  x=self.norm(h).flatten(-2)
  if amp_dtype is None:y=self.net(x)
  else:
   with torch.autocast("cuda",dtype=amp_dtype):y=self.net(x).float()
  return y.clamp(self.alo,self.ahi)
class Dynamics(nn.Module):
 """Fresh dynamics preserving the model joint order and normalized xi contract."""
 def __init__(self,device,warm=None,hidden=1024,dt=.01,substeps=4,inputs="standard",task=None):
  super().__init__()
  if inputs=="standard":self.model=WorldModel(23,27,hidden=(hidden,)*3,structured=True,dt=dt).to(device)
  elif inputs=="grounded":
   from .grounded import GroundedWorldModel
   assert task is not None
   self.model=GroundedWorldModel(task,hidden,dt).to(device)
  else:raise ValueError(inputs)
  self.model.DRIFT_W=(substeps+1)/(2*substeps)
  self.register_buffer("p2m",torch.tensor([NAMES.index(n) for n in MODEL_NAMES],device=device))
  self.register_buffer("m2p",torch.argsort(self.p2m))
  from blinkmimic.robot import get_robot
  from blinkmimic.dr import XiLayout
  layout=XiLayout(get_robot("dexrl"),device)
  assert layout.dim==27
  raw=torch.zeros(1,layout.dim,device=device);raw[:,0]=1.
  self.register_buffer("xi_n",layout.normalize(raw)[0].clone())
  if warm:
   raise ValueError("Cold training forbids pretrained weights")
   ck=torch.load(warm,map_location=device,weights_only=False)
   self.model.load_state_dict(ck["wm"],strict=True)
 @property
 def net(self):return self.model.net
 def ordered(self,S):return (*S[:4],S[4][...,self.p2m],S[5][...,self.p2m])
 def native(self,S):return (*S[:4],S[4][...,self.m2p],S[5][...,self.m2p])
 def context(self,a):return self.xi_n.expand(a.shape[:-1]+(27,))
 def step(self,S,a,ap,unused=None,amp_dtype=None):
  return self.native(self.model.step(self.ordered(S),a[...,self.p2m],ap[...,self.p2m],self.context(a),amp_dtype=amp_dtype))
 @torch.no_grad()
 def observe(self,S,S1,a,ap):
  A,B=self.ordered(S),self.ordered(S1)
  self.model.in_norm.update(self.model.inputs(A,a[...,self.p2m],ap[...,self.p2m],self.context(a)))
  self.model.out_norm.update(self.model.targets(A,B));self.model.inc_norm.update(self.model.increments(A,B))
class Learner:
 def __init__(self,task,buf,device="cuda",B=128,H=96,W=16,warm=None,graphs=True,compile_steps=True,wm_hidden=1024,wm_batch=256,actor_lr=.0003,wm_lr=.0004,actor_opt="muon",imag_bf16=False,ema_decay=None,hard_termination=False,actor_arch="standard",imag_unroll=1,wm_inputs="standard"):
  RunningNorm.no_sync=True
  self.task,self.buf,self.B,self.H,self.W=task,buf,B,H,W;self.graphs=graphs
  self.U=imag_unroll;assert H%self.U==0
  self.actor=Actor(task,kind=actor_arch).to(device);self.ema=copy.deepcopy(self.actor)
  self.actor_opt=actor_opt;self.imag_dtype=torch.bfloat16 if imag_bf16 else None
  self.ema_decay=.995**(task.dt/.02) if ema_decay is None else ema_decay;self.hard_termination=hard_termination
  self.wm=Dynamics(device,warm,hidden=wm_hidden,dt=task.dt,substeps=task.substeps,inputs=wm_inputs,task=task)
  self.wm_batch=wm_batch
  self.oa=MuonAdam(self.actor,[self.actor.net],lr=actor_lr,capturable=graphs) if actor_opt=="muon" else GatedAdam(self.actor,actor_lr,graphs)
  self.ow=MuonAdam(self.wm,[self.wm.net],lr=wm_lr,capturable=graphs)
  self.err_floor=torch.tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*23+[.05]*23,device=device)
  self.update_counts=torch.zeros(4,device=device,dtype=torch.long)
  self.wmg=self.pig=None;self.calls=0;self.gn_ema=torch.ones((),device=device)
  self.imag=self._imag;self.block=self._imag_block;self.loss_fn=self._wm_loss
  if compile_steps:
   import blinkmimic.rotations as rotations
   rotations.ELEMENTWISE=True
   if self.U==1:self.imag=torch.compile(self._imag,dynamic=False,options={"triton.cudagraphs":False})
   else:self.block=torch.compile(self._imag_block,dynamic=False,options={"triton.cudagraphs":False})
   self.loss_fn=torch.compile(self._wm_loss,dynamic=False,options={"triton.cudagraphs":False})
 def _imag(self,S,hist,ap,cmd,target,f):
  a=self.actor(hist,amp_dtype=self.imag_dtype);Sn=self.wm.step(S,a,ap,amp_dtype=self.imag_dtype)
  r=self.task.reward(Sn,a,ap,cmd,f);d=self.task.soft_done(Sn)
  if self.hard_termination:d=self.task.done(Sn).float()+d-d.detach()
  nextcmd=self.task.ramp(cmd,target)
  obs=self.task.obs(Sn,a,nextcmd,f+1);hist=torch.cat((hist[:,1:],obs[:,None]),1)
  return Sn,a,hist,r,d,nextcmd
 def _imag_block(self,S,hist,ap,cmd,target,f):
  J=torch.zeros_like(cmd[:,0]);surv=torch.ones_like(J);gamma=.99**(self.task.dt/.02)
  for j in range(self.U):
   S,a,hist,r,d,cmd=self._imag(S,hist,ap,cmd,target,f)
   J=J+(gamma**j)*surv*r;surv=surv*(1-d);ap=a;f=f+1
  return S,a,hist,J,surv,cmd
 @torch.no_grad()
 def observe(self,tr,hist):
  self.actor.norm.update(hist[:,-1]);self.wm.observe(unflatten_state(tr["s"]),unflatten_state(tr["s1"]),tr["a"],tr["a_prev"])
 def _wm_loss(self,w):
  model=self.wm.model;S=self.wm.ordered(unflatten_state(w["s"][:,0]));std=torch.maximum(model.inc_norm.std(),self.err_floor);loss=0.
  for j in range(self.W):
   a=w["a"][:,j,...,self.wm.p2m];ap=w["a_prev"][:,j,...,self.wm.p2m]
   S=model.step(S,a,ap,self.wm.context(a))
   e=(model.state_error(S,self.wm.ordered(unflatten_state(w["s1"][:,j])))/std).square().mean(-1)
   loss=loss+(e*w["mask"][:,j]).mean()
  return loss/self.W
 def wm_update(self):
  w=self.buf.sample_windows(self.wm_batch,self.W)
  # 50% sagittal replay augmentation, validjointlimits only; command is not a WM input.
  if True:
   from .symmetry import mirror_windows
   w=mirror_windows(w,self.task)
  loss=self.loss_fn(w);self.ow.zero_grad();loss.backward()
  gn=nn.utils.clip_grad_norm_(self.wm.parameters(),1.);good=torch.isfinite(gn)&torch.isfinite(loss)
  for p in self.wm.parameters():
   if p.grad is not None:p.grad.copy_(torch.where(good,p.grad,0.))
  self.ow.step(ok=good);self.update_counts[0].add_(1);self.update_counts[1].add_(good.long());return loss.detach(),good
 def pi_update(self):
  s,h=self.buf.sample_starts(self.B,10);S=unflatten_state(s["s"]);cmd=s["xi"][:,:3];target=s["xi"][:,3:];f=s["f"];ap=s["a_prev"]
  # Fresh goals ramp from the replayed command, matching physical command evolution.
  fresh=self.task.sample_commands(self.B);target=torch.where((torch.rand(self.B,device=cmd.device)<.5)[:,None],fresh,target)
  with torch.no_grad():hist=self.task.obs(unflatten_state(h["s"]),h["a_prev"],h["xi"][...,:3],h["f"])
  hist=torch.cat((hist[:,:-1],self.task.obs(S,ap,cmd,f)[:,None]),1)
  for p in self.wm.parameters():p.requires_grad_(False)
  J=torch.zeros(self.B,device=cmd.device);surv=torch.ones_like(J);gamma=.99**(self.task.dt/.02)
  if self.U==1:
   for t in range(self.H):
    S,a,hist,r,d,cmd=self.imag(S,hist,ap,cmd,target,f)
    J=J+(gamma**t)*surv*r
    surv=surv*(1-d);ap=a;f=f+1
  else:
   for t in range(0,self.H,self.U):
    S,a,hist,j,s,cmd=self.block(S,hist,ap,cmd,target,f)
    J=J+(gamma**t)*surv*j;surv=surv*s;ap=a;f=f+self.U
  loss=-J.mean();self.oa.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
  good=torch.isfinite(gn)&torch.isfinite(loss)&((gn<10*self.gn_ema.clamp(min=.01))|((self.update_counts[3]==0)&(self.actor_opt=="adam")))
  if self.actor_opt=="adam":
   average=torch.where(self.update_counts[3]==0,gn,.98*self.gn_ema+.02*gn)
   self.gn_ema.copy_(torch.where(good,average,1.01*self.gn_ema))
  else:self.gn_ema.lerp_(torch.where(torch.isfinite(gn),gn,self.gn_ema),.01)
  for p in self.actor.parameters():
   if p.grad is not None:p.grad.copy_(torch.where(good,p.grad,0.))
  self.oa.step(ok=good)
  for p in self.wm.parameters():p.requires_grad_(True)
  with torch.no_grad():
   for e,p in zip(self.ema.parameters(),self.actor.parameters()):
    if self.actor_opt=="adam":e.copy_(torch.where(self.update_counts[3]==0,p,e.lerp(p,1-self.ema_decay)))
    else:e.lerp_(p,1-self.ema_decay)
   for e,p in zip(self.ema.buffers(),self.actor.buffers()):e.copy_(p)
  self.update_counts[2].add_(1);self.update_counts[3].add_(good.long())
  return loss.detach(),surv.mean().detach(),gn.detach(),good
 def update(self,allow_policy=True):
  self.calls+=1
  if not self.graphs or self.calls<=3:
   for _ in range(3):w=self.wm_update()
   p=self.pi_update() if allow_policy else None
   if allow_policy and p is not None:p=self.pi_update()
  else:
   if self.wmg is None:
    torch.cuda.synchronize();side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):self.wm_update()
    torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
    self.wmg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.wmg):self.wout=self.wm_update()
   for _ in range(3):self.wmg.replay()
   w=self.wout;p=None
   if allow_policy:
    if self.pig is None:
     side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(side):self.pi_update()
     torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
     self.pig=torch.cuda.CUDAGraph()
     with torch.cuda.graph(self.pig):self.pout=self.pi_update()
    for _ in range(2):self.pig.replay()
    p=self.pout
  return w,p

class GatedAdam:
 """XSoma uses Adam for every policy parameter; gates leave moments and step count unchanged."""
 def __init__(self,module,lr,capturable):
  self.adam=torch.optim.Adam(module.parameters(),lr=torch.tensor(lr,device=next(module.parameters()).device) if capturable else lr,betas=(.9,.95),fused=True,capturable=capturable)
 @property
 def param_groups(self):return self.adam.param_groups
 def zero_grad(self):self.adam.zero_grad(set_to_none=True)
 def step(self,ok=None):
  if ok is not None:self.adam.found_inf=(~ok).float()
  try:self.adam.step()
  finally:self.adam.found_inf=None
 def state_dict(self):return {"adam":self.adam.state_dict()}
