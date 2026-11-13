import torch
from blinkmimic.mdp import flatten_state,unflatten_state
from .robot import NAMES
def mirror_windows(w,task):
 dev=w["s"].device
 perm,sign,F=task.mirror_perm,task.mirror_sign,task.reflection
 def flip(x):
  p,R,v,ang,q,qd=unflatten_state(x)
  return p*F,R*F[:,None]*F[None,:],v*F,-ang*F,q[...,perm]*sign,qd[...,perm]*sign
 A,B=flip(w["s"]),flip(w["s1"])
 valid=((A[4]>=task.qlo)&(A[4]<=task.qhi)&(B[4]>=task.qlo)&(B[4]<=task.qhi)).all(-1).all(-1)
 valid&=torch.arange(len(valid),device=dev)<len(valid)//2
 for k,x in [("s",flatten_state(A)),("s1",flatten_state(B)),("a",w["a"][...,perm]*sign),("a_prev",w["a_prev"][...,perm]*sign)]:
  w[k]=torch.where(valid[:,None,None],x,w[k])
 return w
