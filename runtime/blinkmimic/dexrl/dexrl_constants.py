from dataclasses import replace
from pathlib import Path
import hashlib,json,numpy as np
from dexnewrl.robot import SPEC as BASE,MODEL_NAMES
from blinkmimic.dr import DomainRandCfg
NAMES=tuple(MODEL_NAMES)
TRACKED=('pelvis','hip_roll_l_link','knee_pitch_l_link','ankle_roll_l_link','hip_roll_r_link','knee_pitch_r_link','ankle_roll_r_link','waist_pitch_link','shoulder_roll_l_link','elbow_pitch_l_link','shoulder_roll_r_link','elbow_pitch_r_link')
def load(path):
 z=np.load(path,allow_pickle=False)
 assert float(z['fps'])==50 and str(z['quat_order'])=='wxyz'
 source_names=z['joint_names'].tolist()
 assert str(z['model_sha256'])=='17573c55d1c9d9d43b2e44a5e3ebc237027bfcfb6f5b5d928aa0205792548194'
 return z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos'][:,[source_names.index(n) for n in NAMES]],float(z['fps'])
mirror=tuple((n.replace('_l_','_X_').replace('_r_','_l_').replace('_X_','_r_'),-1. if '_roll_' in n or '_yaw_' in n else 1.) for n in NAMES)
SPEC=replace(BASE,name='dexrl',joint_names=NAMES,tracked_bodies=TRACKED,motion_glob='*.npz',load_motion=load,mirror_joints=mirror,
 dr=DomainRandCfg(friction_range=(.7,1.2),joint_offset={'.*':0.},reset_pos=(.01,.01,.005),reset_rot=(.02,.02,.03),
 reset_lin_vel=(.05,.05,.02),reset_ang_vel=(.05,.05,.05),reset_joint_pos=.02,reset_joint_vel=0.))
