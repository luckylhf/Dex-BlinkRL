"""Full collection graph with fixed-shape masked resets, adapted from BlinkMimic."""
import torch
class Collector:
 def __init__(self,env,learner,buffer,graph=False):
  self.env,self.learner,self.buffer=env,learner,buffer;self.enabled=graph;self.calls=0;self.graph=None
  if graph:env.enable_physics_graph=False
  self.noisy=(torch.arange(env.N,device=env.device)%2).float()[:,None]
 @torch.no_grad()
 def core(self):
  e,l,b=self.env,self.learner,self.buffer
  mean=l.actor(e.hist);action=(mean+.2*self.noisy*torch.randn_like(mean)).clamp(l.actor.alo,l.actor.ahi)
  tr,term=e.step(action,reset=not self.enabled)
  if self.enabled:e.reset_masked(tr["done"])
  if self.enabled:b.add_device(tr)
  else:b.add(tr)
  l.observe(tr,e.hist)
  return tr,term
 @torch.no_grad()
 def __call__(self):
  if not self.enabled:return self.core()
  self.calls+=1
  if self.calls<=3:out=self.core()
  else:
   if self.graph is None:
    import warp as wp
    from mjlab.sim.sim_data import TorchArray
    torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph();old=TorchArray.__setitem__;old_use=self.env.sim.use_cuda_graph
    TorchArray.__setitem__=lambda obj,idx,value:obj._tensor.__setitem__(idx,value);self.env.sim.use_cuda_graph=False
    try:
     with torch.cuda.graph(self.graph,capture_error_mode="thread_local"):
      ws=wp.stream_from_torch(torch.cuda.current_stream())
      with wp.ScopedStream(ws,sync_enter=False,sync_exit=False):
       wp.capture_begin(stream=ws,external=True)
       try:self.output=self.core()
       finally:self.wp_graph=wp.capture_end(stream=ws)
    finally:TorchArray.__setitem__=old;self.env.sim.use_cuda_graph=old_use
    print("Captured full collection and masked-reset graph",flush=True)
   self.graph.replay();out=self.output;self.env.scene.update(self.env.dt)
  self.buffer.advance_host()
  return out
