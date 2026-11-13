"""Discard every learned warmup state while preserving CUDA graph tensor addresses."""
import hashlib,random
import numpy as np
import torch

def hashes(module):
 return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in module.state_dict().items()}

class ColdStart:
 def __init__(self,learner):
  self.models={'actor':learner.actor,'ema':learner.ema,'world_model':learner.wm}
  self.initial={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in self.models.items()}
  self.initial_hashes={n:hashes(m) for n,m in self.models.items()}
 @torch.no_grad()
 def restore(self,learner,buf,env,seed):
  before=learner.update_counts.cpu().tolist()
  for name,module in self.models.items():
   # load_state_dict(assign=False) copies into existing storage captured by the graphs.
   module.load_state_dict(self.initial[name],strict=True,assign=False)
   for p in module.parameters():
    if p.grad is not None:p.grad.zero_()
  optimizer_tensors=[]
  for optimizer in (learner.oa,learner.ow):
   for sub in ((optimizer.muon,optimizer.adam) if hasattr(optimizer,"muon") else (optimizer.adam,)):
    for state in sub.state.values():
     for key,value in state.items():
      if torch.is_tensor(value):value.zero_();optimizer_tensors.append(value)
      elif isinstance(value,(int,float)):state[key]=type(value)(0)
      else:raise TypeError('Unknown optimizer state: '+key)
  learner.update_counts.zero_();learner.gn_ema.fill_(1.)
  # calls is execution/capture bookkeeping, not a learned state; graphs remain reusable.
  for value in buf.data.values():value.zero_()
  buf.ptr=buf.size=0;buf.ptr_t.zero_();buf.size_t.zero_()
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  env.reset(torch.arange(env.N,device=env.device))
  # Define a fresh, deterministic training environment. Reset observations are newly generated.
  after={n:hashes(m) for n,m in self.models.items()}
  exact=after==self.initial_hashes
  opt_zero=all(int(torch.count_nonzero(v))==0 for v in optimizer_tensors)
  replay_zero=all(int(torch.count_nonzero(v))==0 for v in buf.data.values())
  assert exact and opt_zero and replay_zero
  assert learner.update_counts.cpu().tolist()==[0,0,0,0]
  return {'restored_random_parameter_and_normalizer_hashes_exact':exact,'optimizer_tensors_zero':opt_zero,'optimizer_tensor_count':len(optimizer_tensors),'replay_tensors_zero':replay_zero,'replay_size':buf.size,'update_counts_after':[0,0,0,0],'discarded_prewarm_update_counts':before,'training_rng_seed':seed,'initial_hashes':self.initial_hashes,'after_restore_hashes':after,'graph_addresses_preserved':True,'learning_statistics_reset':['actor/world model normalizers','optimizer momentum, moments and steps','EMA','gradient moving average','update counters','replay data and pointers'],'preserved_nonlearned_state':['compiled code','CUDA graphs','allocation/capture bookkeeping']}
