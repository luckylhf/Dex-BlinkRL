
"""Local-authoritative Dex-RL robot contract. All generated files stay in this new repository."""
from pathlib import Path
import ast, json, hashlib, re, os
import xml.etree.ElementTree as ET
import mujoco, yaml, numpy as np
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from blinkmimic.robot import RobotSpec
ROOT=Path(os.environ.get("DEX_PROJECT_ROOT",str(Path(__file__).resolve().parents[2])))
SOURCE=ROOT/'xSIM_MUJOCO-main/resources/dex_v3/urdf/dex_v3.xml'
CFG=yaml.safe_load((ROOT/'Deploy_Tienkung/policy/walk_amp/config/walk_amp.yaml').read_text())
NAMES=tuple(CFG['joint_names'])
tree=ast.parse((ROOT/'TienKung-Lab/legged_lab/assets/Tienkung_Dex/walk.py').read_text())
DEFAULT=next(ast.literal_eval(k.value) for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='InitialStateCfg' for k in x.keywords if k.arg=='joint_pos')
def training_parameter(name,key):
    for x in ast.walk(tree):
        if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='ImplicitActuatorCfg':
            kw={k.arg:ast.literal_eval(k.value) for k in x.keywords}
            if any(re.fullmatch(p,name) for p in kw['joint_names_expr']):
                value=kw[key]
                if not isinstance(value,dict):return float(value)
                matches=[float(v) for pat,v in value.items() if re.fullmatch(pat,name)]
                assert len(matches)==1,(name,key,matches)
                return matches[0]
    raise ValueError((name,key))
KP={n:training_parameter(n,'stiffness') for n in NAMES}
KD={n:training_parameter(n,'damping') for n in NAMES}
# The local Isaac training config supplies no rotor armature; the source MJCF defaults to zero.
ARM={n:0. for n in NAMES}
XML=ROOT/'generated/dex_rl_23.xml'
def build_asset():
    XML.parent.mkdir(exist_ok=True)
    t=ET.parse(SOURCE);r=t.getroot();compiler=r.find('compiler')
    meshbase=(SOURCE.parent/compiler.get('meshdir','.')).resolve()
    for mesh in r.findall('./asset/mesh'):
        mesh.set('file',str((meshbase/mesh.get('file')).resolve()))
    compiler.set('meshdir',str(meshbase))
    locked=[]
    for body in r.findall('.//body'):
        for joint in list(body.findall('joint')):
            if joint.get('type')!='free' and joint.get('name') not in NAMES:
                locked.append(joint.get('name'));body.remove(joint)
    for tag in ('actuator','sensor','keyframe'):
        element=r.find(tag)
        if element is not None:r.remove(element)
    for geom in list(r.find('worldbody').findall('geom')):r.find('worldbody').remove(geom)
    xml_tmp=XML.with_name(XML.name+'.'+str(os.getpid())+'.tmp')
    t.write(xml_tmp,encoding='unicode');os.replace(xml_tmp,XML)
    m=mujoco.MjModel.from_xml_path(str(XML))
    assert m.nq==30 and m.nv==29,(m.nq,m.nv)
    actual=tuple(m.joint(i).name for i in range(m.njnt) if m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE)
    assert set(actual)==set(NAMES)
    contract={'policy_joints':NAMES,'model_joints':actual,'locked_at_zero':locked,'control_dt':.01,'physics_dt':.0025,
      'obs_dim':840,'action_dim':23,'default_joint_pos':DEFAULT,'kp':KP,'kd':KD,'armature':ARM,'action_scale':.25,
      'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(XML.read_bytes()).hexdigest(),
      'scope':'MuJoCo-Warp flat-ground algorithm experiment; not Isaac/real-robot acceptance'}
    contract_path=ROOT/'generated/contract.json'
    contract_tmp=contract_path.with_name(contract_path.name+'.'+str(os.getpid())+'.tmp')
    contract_tmp.write_text(json.dumps(contract,indent=2));os.replace(contract_tmp,contract_path)
    return actual
MODEL_NAMES=build_asset()
def get_cfg():
    def spec_fn():
        s=mujoco.MjSpec.from_file(str(XML))
        return s
    model=mujoco.MjModel.from_xml_path(str(XML))
    acts=tuple(BuiltinPositionActuatorCfg(target_names_expr=(n,),stiffness=KP[n],damping=KD[n],armature=ARM[n],
        frictionloss=0.,viscous_damping=0.,effort_limit=float(model.jnt_actfrcrange[model.joint(n).id,1])) for n in MODEL_NAMES)
    return EntityCfg(spec_fn=spec_fn,sort_actuators=True,
       init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.995),joint_pos=DEFAULT,joint_vel={'.*':0.}),
       articulation=EntityArticulationInfoCfg(actuators=acts,soft_joint_pos_limit_factor=.9))
SPEC=RobotSpec(name='dex_rl',mjcf_path=str(XML),joint_names=NAMES,
 tracked_bodies=('pelvis','ankle_roll_l_link','ankle_roll_r_link','waist_pitch_link','elbow_pitch_l_link','elbow_pitch_r_link'),
 default_joint_pos=DEFAULT,action_scale={n:.25 for n in NAMES},com_body='waist_pitch_link',
 com_range=((0.,0.),)*3,robot_cfg=get_cfg,motion_dir='',motion_glob='',load_motion=lambda _:None,
 feet=('ankle_roll_l_link','ankle_roll_r_link'),contact_z=.06,njmax=1000,nconmax=256)
