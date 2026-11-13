
"""Batched physics environment with pre-reset terminal transitions."""
import torch
# Reuse upstream reset/capture preallocation fixes, without constructing a tracking task.
import blinkmimic.env
from blinkmimic.rotations import quat_to_mat,mv3
from mjlab.scene import Scene,SceneCfg
from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from .robot import SPEC,NAMES
from .task import DT
class Env:
    def __init__(self,N,task,device='cuda',eval_mode=False):
        self.N=N;self.task=task;self.device=device;self.eval_mode=eval_mode
        self.dt=task.dt;self.substeps=task.substeps;self.episode_steps=round(10/self.dt);self.resample_steps=round(4/self.dt)
        self.scene=Scene(SceneCfg(num_envs=N,env_spacing=2.5,terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':SPEC.robot_cfg()}),device=device)
        cfg=SimulationCfg(njmax=getattr(task,"njmax",1000),nconmax=getattr(task,"nconmax",256),mujoco=MujocoCfg(timestep=self.dt/self.substeps,integrator='implicitfast',
              cone='pyramidal',solver='newton',iterations=100,ls_iterations=50,tolerance=1e-6,ccd_iterations=35))
        self.sim=Simulation(num_envs=N,cfg=cfg,model=self.scene.compile(),device=device)
        self.scene.initialize(mj_model=self.sim.mj_model,model=self.sim.model,data=self.sim.data)
        import mujoco
        from .robot import ROOT
        import os
        model_path=ROOT/'generated/physics_scene.mjb';model_tmp=model_path.with_name('physics_scene.'+str(os.getpid())+'.mjb')
        mujoco.mj_saveModel(self.sim.mj_model,str(model_tmp),None);os.replace(model_tmp,model_path)
        self.robot=self.scene['robot'];self.names=list(self.robot.joint_names)
        self.p2m=torch.tensor([self.names.index(n) for n in NAMES],device=device)
        self.m2p=torch.argsort(self.p2m)
        self.step_count=torch.zeros(N,dtype=torch.long,device=device)
        self.a_prev=torch.zeros(N,23,device=device);self.cmd=torch.zeros(N,3,device=device)
        self.base_cmd=torch.zeros_like(self.cmd);self.command_scale=1.
        self.hist=torch.zeros(N,9*getattr(task,'history_stride',1)+1,84,device=device)
        self.elapsed=torch.zeros(N,dtype=torch.long,device=device)
        self.ep_error=torch.zeros(N,device=device);self.ep_vx=torch.zeros(N,device=device);self.ep_cmd=torch.zeros(N,device=device)
        self.start_xy=torch.zeros(N,2,device=device);self.ep_max_cmd=torch.zeros(N,device=device)
        self.stage=0;self.speed_max=0.;self.recent_episodes=[];self.phase_fail=torch.ones(12,device=device)
        self.stage_events=[]
        self.physics_calls=0;self.physics_graph=None;self.enable_physics_graph=True
        self.reset(torch.arange(N,device=device))
    def state(self):
        ix=self.robot.indexing;q=self.sim.data.qpos[:,ix.free_joint_q_adr];v=self.sim.data.qvel[:,ix.free_joint_v_adr]
        R=quat_to_mat(q[:,3:7])
        return q[:,:3]-self.scene.env_origins,R,v[:,:3].clone(),mv3(R,v[:,3:6]),self.sim.data.qpos[:,ix.joint_q_adr][:,self.p2m].clone(),self.sim.data.qvel[:,ix.joint_v_adr][:,self.p2m].clone()
    def reset(self,ids,q=None,cmd=None):
        n=len(ids)
        if not n:return
        self.sim.reset(ids);self.scene.reset(ids)
        pose=torch.zeros(n,7,device=self.device);pose[:,:3]=self.scene.env_origins[ids];pose[:,2]+=.995;pose[:,3]=1.
        if q is None:
            q=self.task.q0.expand(n,-1).clone()
            if not self.eval_mode:q+=torch.randn_like(q)*.015
        self.robot.write_root_link_pose_to_sim(pose,env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(torch.zeros(n,6,device=self.device),env_ids=ids)
        self.robot.write_joint_position_to_sim(q[:,self.m2p],env_ids=ids)
        self.robot.write_joint_velocity_to_sim(torch.zeros_like(q),env_ids=ids)
        self.robot.set_joint_position_target(q[:,self.m2p],env_ids=ids)
        self.step_count[ids]=0;self.elapsed[ids]=0;self.a_prev[ids]=0.
        if cmd is None:
            cmd=torch.zeros(n,3,device=self.device)
            if not self.eval_mode:cmd=self.task.sample_commands(n)
        self.base_cmd[ids]=cmd;self.cmd[ids]=0.
        self.sim.forward()
        o=self.task.obs(self.state(),self.a_prev,self.cmd,self.step_count)
        self.hist[ids]=o[ids,None,:].expand(-1,10,-1)
    def reset_masked(self,mask,q=None,cmd=None):
        """BlinkMimic-style fixed-shape reset. Explicit q/cmd support exact parity checks."""
        import warp as wp
        import mujoco_warp as mjwarp
        d=mask;d1=d[:,None];n=self.N
        if q is None:
            q=self.task.q0.expand(n,-1).clone()
            if not self.eval_mode:q=q+torch.randn_like(q)*.015
        if cmd is None:cmd=torch.zeros(n,3,device=self.device) if self.eval_mode else self.task.sample_commands(n)
        self.sim._reset_mask.copy_(d)
        with wp.ScopedDevice(self.sim.wp_device):
            if self.sim.use_cuda_graph and self.sim.reset_graph is not None:wp.capture_launch(self.sim.reset_graph)
            else:mjwarp.reset_data(self.sim.wp_model,self.sim.wp_data,reset=self.sim._reset_mask_wp)
        ed=self.robot.data
        for target in (ed.joint_pos_target,ed.joint_vel_target,ed.joint_effort_target,ed.tendon_len_target,ed.tendon_vel_target,ed.tendon_effort_target,ed.site_effort_target):
            target.masked_fill_(d.view(-1,*([1]*(target.dim()-1))),0.)
        ix=self.robot.indexing;pose=torch.zeros(n,7,device=self.device);pose[:,:3]=self.scene.env_origins;pose[:,2]+=.995;pose[:,3]=1.
        qp,qv=self.sim.data.qpos,self.sim.data.qvel
        qp[:,ix.free_joint_q_adr]=torch.where(d1,pose,qp[:,ix.free_joint_q_adr])
        qv[:,ix.free_joint_v_adr]=torch.where(d1,0.,qv[:,ix.free_joint_v_adr])
        qp[:,ix.joint_q_adr]=torch.where(d1,q[:,self.m2p],qp[:,ix.joint_q_adr])
        qv[:,ix.joint_v_adr]=torch.where(d1,0.,qv[:,ix.joint_v_adr])
        ed.joint_pos_target.copy_(torch.where(d1,q[:,self.m2p],ed.joint_pos_target))
        self.step_count.masked_fill_(d,0);self.elapsed.masked_fill_(d,0);self.a_prev.masked_fill_(d1,0.)
        self.base_cmd.copy_(torch.where(d1,cmd,self.base_cmd));self.cmd.masked_fill_(d1,0.)
        # Keep Dex's original reset-forward semantics, including fresh kinematics.
        self.sim.forward()
        o=self.task.obs(self.state(),self.a_prev,self.cmd,self.step_count)
        self.hist.copy_(torch.where(d[:,None,None],o[:,None,:].expand_as(self.hist),self.hist))
    def _physics_core(self):
        for _ in range(self.substeps):
            self.scene.write_data_to_sim();self.sim.step();self.scene.update(self.dt/self.substeps)
    def advance(self):
        self.physics_calls+=1
        if not self.enable_physics_graph or self.physics_calls<=3:
            self._physics_core();return
        if self.physics_graph is None:
            import warp as wp
            from mjlab.sim.sim_data import TorchArray
            torch.cuda.synchronize();self.physics_graph=torch.cuda.CUDAGraph()
            old=TorchArray.__setitem__;old_use=self.sim.use_cuda_graph
            TorchArray.__setitem__=lambda obj,idx,value:obj._tensor.__setitem__(idx,value)
            self.sim.use_cuda_graph=False
            try:
                with torch.cuda.graph(self.physics_graph,capture_error_mode='thread_local'):
                    ws=wp.stream_from_torch(torch.cuda.current_stream())
                    with wp.ScopedStream(ws,sync_enter=False,sync_exit=False):
                        wp.capture_begin(stream=ws,external=True)
                        try:self._physics_core()
                        finally:self.wp_physics_graph=wp.capture_end(stream=ws)
            finally:
                TorchArray.__setitem__=old;self.sim.use_cuda_graph=old_use
            print(f'Captured {self.substeps}-substep physics graph',flush=True)
        self.physics_graph.replay()
        # Graph replay does not execute Python cache invalidation.
        self.scene.update(self.dt)
    def step(self,a,reset=True):
        S=tuple(x.clone() for x in self.state());ap=self.a_prev.clone();cmd=self.cmd.clone();f=self.step_count.clone();elapsed=self.elapsed.clone()
        target=self.task.q0+.25*a
        if getattr(self.task,'target_q_clamp',True):target=torch.clamp(target,self.task.qlo,self.task.qhi)
        self.robot.set_joint_position_target(target[:,self.m2p])
        self.advance()
        S1=self.state();self.step_count+=1;self.elapsed+=1
        term=self.task.done(S1);done=term|((self.elapsed if getattr(self.task,'guided',False) else self.elapsed)>=self.episode_steps)
        reward=self.task.reward(S1,a,ap,cmd,f)
        from blinkmimic.mdp import flatten_state
        tr={'s':flatten_state(S),'s1':flatten_state(S1),'a':a.clone(),'a_prev':ap,'xi':torch.cat((cmd,self.base_cmd.clone()),-1),'r':reward,
           'f':f,'ep_step':(elapsed if getattr(self.task,'guided',False) else f).to(torch.int32),'done':done,'cut':done}
        self.a_prev.copy_(a)
        if not self.eval_mode:
            change=(self.elapsed%self.resample_steps)==0
            self.base_cmd.copy_(torch.where(change[:,None],self.task.sample_commands(self.N),self.base_cmd))
            self.cmd.copy_(self.task.ramp(self.cmd,self.base_cmd))
        o=self.task.obs(S1,a,self.cmd,self.step_count)
        self.hist.copy_(torch.cat((self.hist[:,1:],o[:,None,:]),1))
        if reset:self.reset(done.nonzero().flatten())
        return tr,term
