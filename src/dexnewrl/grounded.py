"""Explicit geometry and PD-error inputs, following XSoma's published world model."""
import torch
from blinkmimic.networks import WorldModel
from blinkmimic.mdp import invariant_features
from .robot import NAMES,MODEL_NAMES

class GroundedWorldModel(WorldModel):
 """Model-order state/action; geometry is evaluated in policy joint order on the physical flat plane."""
 def __init__(self,task,hidden,dt):
  super().__init__(23,27+14,hidden=(hidden,)*3,structured=True,dt=dt)
  self.task=task
  p2m=torch.tensor([NAMES.index(n) for n in MODEL_NAMES],device=task.device)
  self.register_buffer("model_to_policy",torch.argsort(p2m))
  self.register_buffer("target_zero_model",task.q0[p2m].clone())
 def inputs(self,S,a,a_prev,xi_n):
  native=(*S[:4],S[4][...,self.model_to_policy],S[5][...,self.model_to_policy])
  clearances=self.task.sole_geometry(native)[2].flatten(-2)
  pd_error=self.target_zero_model+.25*a-S[4]
  return torch.cat((invariant_features(S),clearances,pd_error,xi_n),-1)
