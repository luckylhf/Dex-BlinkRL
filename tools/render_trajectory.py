"""Render saved CPU evaluation qpos without re-simulating policy behavior."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import argparse,json
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from evidence_paths import recorded_path
p=argparse.ArgumentParser();p.add_argument("evaluation");p.add_argument("--out",required=True);p.add_argument("--index",type=int,default=0);p.add_argument("--fps",type=int,default=20);a=p.parse_args()
src=Path(a.evaluation);dest=Path(a.out);assert not dest.exists();dest.mkdir(parents=True)
report=json.loads(src.read_text());raw=np.load(src.with_suffix(".npz"));dt=report["control_dt"];i=a.index
qpos=raw["qpos"][:,i];trace=raw[f"trajectory_{i}"];duration=min(len(qpos),len(trace))*dt
m=mujoco.MjModel.from_binary_path(str(recorded_path(report.get("physics_model",str(Path(report["checkpoint"]).parent/"physics_scene.mjb")))));m.vis.global_.offwidth=960;m.vis.global_.offheight=540
d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=540,width=960)
cam=mujoco.MjvCamera();cam.distance=3.2;cam.azimuth=120;cam.elevation=-12
opt=mujoco.MjvOption();opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
writer=imageio.get_writer(str(dest/"replay.mp4"),fps=a.fps,codec="libx264",quality=8)
stills=[];target_times=[1,10,13,20,30,40,45,59]
for n in range(int(duration*a.fps)):
 tm=n/a.fps;k=min(round(tm/dt),len(trace)-1,len(qpos)-1);d.qpos[:]=qpos[k];mujoco.mj_forward(m,d)
 cam.lookat[:]=[d.qpos[0],d.qpos[1],.85];renderer.update_scene(d,camera=cam,scene_option=opt)
 im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,960,51),fill=(245,245,245))
 dr.text((10,5),f"{Path(report['checkpoint']).parent.name} {report['variant']} | {report['scenario']} | t={tm:.2f}s",fill=(0,0,0))
 tr=trace[k];dr.text((10,25),f"command={tr[5]:.2f}m/s body_vx={tr[3]:.2f}m/s | recorded qpos, 1x speed | CPU MuJoCo",fill=(0,0,0))
 writer.append_data(np.asarray(im))
 if any(abs(tm-t)<.5/a.fps for t in target_times):
  im.save(dest/f"frame_{tm:05.1f}.png");stills.append(im.resize((480,270)))
writer.close();renderer.close()
if stills:
 sheet=Image.new("RGB",(960,270*((len(stills)+1)//2)),(255,255,255))
 for k,im in enumerate(stills):sheet.paste(im,((k%2)*480,(k//2)*270))
 sheet.save(dest/"contact_sheet.png")
(dest/"RECEIPT.json").write_text(json.dumps({"source_evaluation":str(src),"raw_trajectory":str(src.with_suffix(".npz")),"initial_seed":report["results"][i]["seed"],"fps":a.fps,"duration":duration,"playback_speed":1,"camera":"pelvis-following; path drift must be assessed from raw world positions","method":"direct qpos replay; no new policy inference or dynamical simulation"},indent=2))
print(json.dumps({"out":str(dest),"duration":duration}))
