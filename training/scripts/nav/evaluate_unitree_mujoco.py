#!/usr/bin/env python3
"""Frozen Transformer-SRU -> Unitree velocity policy -> official G1 MuJoCo asset.

Nominal sim2sim evaluation, without training noise/delay or any hardware I/O.
Uses the project's actual action mapping and low-level observation methods.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import glfw
import imageio.v2 as imageio
import mujoco as mj
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'training/source/engineai_rl_lab/engineai_rl_lab/tasks/navigation'
DEFAULT_XML = ROOT / 'third_party/unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
sys.path.insert(0, str(ROOT / 'training/sru-navigation-learning'))
from rsl_rl.modules import ActorCriticSRURangeImage

# Populated from the explicit scene JSON before constructing a simulation.
BOXES = []
GOAL = np.zeros(3)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bridge():
    path = SRC / 'mdp/navigation/actions/navigation_se2_actions_unitree.py'
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UnitreeNavigationSE2Action')
    names = ('_current_terms', '_build_policy_observations', 'process_actions')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    ns = dict(torch=torch, UNITREE_POLICY_HISTORY_LENGTH=5, UNITREE_POLICY_INPUT_DIM=480)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)
    bridge = type('ProjectControlBridge', (), {n.name: ns[n.name] for n in methods})()
    return bridge, constants


class Simulation:
    def __init__(self, checkpoint, xml=DEFAULT_XML):
        self.bridge, constants = load_bridge()
        b = self.bridge
        self.order = np.array(constants['UNITREE_POLICY_TO_SDK'])
        self.q0 = np.array(constants['UNITREE_POLICY_DEFAULT_JOINT_POS'])
        deploy = SRC / 'assets/data/Policies/unitree_g1_velocity/deploy.yaml'
        cfg = yaml.safe_load(deploy.read_text())
        assert self.order.tolist() == cfg['joint_ids_map']
        assert np.allclose(self.q0, cfg['default_joint_pos'])
        self.xml = Path(xml)
        if not self.xml.is_file():
            raise FileNotFoundError(f'Missing MuJoCo scene: {self.xml}. See deploy/sim2sim/README.md.')
        spec = mj.MjSpec.from_file(str(self.xml))
        for i, (x, y, sx, sy, h) in enumerate(BOXES):
            spec.worldbody.add_geom(name=f'obstacle_{i}', type=mj.mjtGeom.mjGEOM_BOX,
                pos=(x, y, h/2), size=(sx/2, sy/2, h/2), rgba=(.72, .36, .12, 1),
                contype=1, conaffinity=1)
        spec.worldbody.add_site(name='goal', type=mj.mjtGeom.mjGEOM_CYLINDER,
            pos=(*GOAL[:2], .015), size=(.5, .015), rgba=(.1, .9, .2, .75))
        self.model = m = spec.compile()
        m.opt.timestep = .002
        self.data = d = mj.MjData(m)
        self.body = m.body('pelvis').id
        self.torso = m.body('torso_link').id
        self.obstacle_ids = {m.geom(f'obstacle_{i}').id for i in range(len(BOXES))}
        joints = np.array([m.joint(name).id for name in constants['UNITREE_SDK_JOINT_NAMES']])
        self.qadr = m.jnt_qposadr[joints]
        self.vadr = m.jnt_dofadr[joints]
        self.act = np.array([next(i for i in range(m.nu) if m.actuator_trnid[i, 0] == j) for j in joints])
        self.kp, self.kd = np.array(cfg['stiffness']), np.array(cfg['damping'])
        self.default = np.zeros(29)
        self.default[self.order] = self.q0
        self.target = self.default.copy()
        module_path = SRC / 'g1/unitree_velocity_policy.py'
        module_spec = importlib.util.spec_from_file_location('unitree_loader', module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.low_path = module.get_unitree_velocity_policy_path()
        self.low = module.UnitreeVelocityPolicyBatch(self.low_path, 'cpu').eval()
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.iteration = int(ckpt['iter'])
        self.high = ActorCriticSRURangeImage(1984, 5121, 3,
            actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128],
            activation='elu', init_noise_std=1., rnn_hidden_size=512, rnn_type='lstm_sru',
            rnn_num_layers=1, dropout=.2, num_points=656, point_dim=3,
            height_input_dims=(64,7,7), range_image_shape=(16,41),
            range_image_feature_dim=128, range_image_batch_size=256,
            range_image_max_distance=5., transformer_num_layers=2,
            transformer_num_heads=4, transformer_feedforward_dim=512, transformer_dropout=.1)
        self.high.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.high.eval().requires_grad_(False)
        self.high.memory_a.hidden_states = None
        b._joint_ids = torch.tensor(self.order)
        b._TERM_DIMS = dict(base_ang_vel=3, projected_gravity=3, velocity_commands=3,
                            joint_pos_rel=29, joint_vel_rel=29, last_action=29)
        b._asset = SimpleNamespace(data=SimpleNamespace())
        b._velocity_commands = torch.zeros(1,3)
        b._raw_navigation_actions = torch.zeros(1,3)
        b._last_policy_action = torch.zeros(1,29)
        b._history_initialized = torch.zeros(1,dtype=torch.bool)
        b._term_history = {k:torch.zeros(1,5,v) for k,v in b._TERM_DIMS.items()}
        b._policy_bias = torch.zeros(1,3)
        b._policy_scaling = torch.ones(1,3)
        b._command_min = torch.tensor([[-.5,-.3,-1.]])
        b._command_max = torch.tensor([[1.,.3,1.]])
        b._per_env_per_dim_low_pass_alpha = torch.tensor([[.8819113783,.8819113783,0.]])
        b.cfg = SimpleNamespace(enable_low_pass_filter=True)
        b._asset.data.default_joint_pos = torch.tensor(self.default,dtype=torch.float32)[None]
        self.ids = torch.tensor([0])
        d.qpos[:3] = [0,0,.8]
        d.qpos[3:7] = [1,0,0,0]
        d.qpos[self.qadr] = self.default
        mj.mj_forward(m,d)
        elevation, azimuth = np.meshgrid(np.deg2rad(np.linspace(-45,0,16)),
                                        np.deg2rad(np.linspace(-60,60,41)), indexing='ij')
        self.rays = np.stack((np.cos(elevation)*np.cos(azimuth),
                            np.cos(elevation)*np.sin(azimuth), np.sin(elevation)),axis=-1).reshape(-1,3)
        angle = .04014257279586953
        self.mount = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        for i in range(m.ngeom):
            m.geom_group[i] = 0 if m.geom_bodyid[i] == 0 else 1
        self.groups = np.array([1,0,0,0,0,0],dtype=np.uint8)
        self.last_obs = None
        self.hits = self.obstacle_hits = 0

    def state(self):
        velocity = np.zeros(6)
        # BODY local orientation can be the inertial frame. Training uses the
        # root link axes, so explicitly rotate world COM velocity into them.
        mj.mj_objectVelocity(self.model,self.data,mj.mjtObj.mjOBJ_BODY,self.body,velocity,0)
        rotation = self.data.xmat[self.body].reshape(3,3)
        velocity[:3] = rotation.T @ velocity[:3]
        velocity[3:] = rotation.T @ velocity[3:]
        return velocity, rotation

    def lidar(self):
        d = self.data
        torso_r = d.xmat[self.torso].reshape(3,3)
        return self.lidar_at_pose(d.xpos[self.torso], torso_r)

    def lidar_at_pose(self, torso_position, torso_r):
        """Cast from an explicit link pose; also used for recorded-state audits."""
        d = self.data
        sensor_r = torso_r @ self.mount
        origin = torso_position + torso_r @ np.array([.0002835,.00003,.40618])
        rays_w = self.rays @ sensor_r.T
        ids = np.empty(656,dtype=np.int32)
        distances = np.empty(656)
        # cutoff culls candidate geometry; it is NOT ray length. A 5m cutoff
        # incorrectly drops the infinite floor once its reference origin is
        # far from the sensor. Apply the actual 5m sensor range to ray hits.
        mj.mj_multiRay(self.model,d,origin,rays_w.reshape(-1),self.groups,True,-1,ids,distances,None,656,mj.mjMAXVAL)
        valid = (distances >= 0) & (distances <= 5.)
        yaw = math.atan2(sensor_r[1,0],sensor_r[0,0])
        c,s = math.cos(yaw),math.sin(yaw)
        yaw_r = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        points = distances[:,None]*rays_w @ yaw_r / 5.
        points[~valid] = 0
        self.hits = int(valid.sum())
        self.obstacle_hits = int((valid & np.isin(ids,list(self.obstacle_ids))).sum())
        return points.astype(np.float32).reshape(-1)

    @torch.inference_mode()
    def navigate(self):
        vel,rotation = self.state()
        goal_b = rotation.T @ (GOAL-self.data.xpos[self.body])
        distance = np.linalg.norm(goal_b)
        goal = np.r_[goal_b/max(distance,1.e-6), np.log1p(distance)]
        obs = np.r_[vel[3:], vel[:3], rotation.T @ [0,0,-1],
                    self.bridge._raw_navigation_actions[0].numpy(), goal, self.lidar()].astype(np.float32)
        assert obs.shape == (1984,) and np.isfinite(obs).all()
        self.last_obs = obs
        raw = self.high.act_inference(torch.from_numpy(obs)[None])
        if not torch.isfinite(raw).all():
            raise RuntimeError('Nonfinite navigation output')
        self.bridge.process_actions(raw)

    @torch.inference_mode()
    def low_update(self):
        b,d = self.bridge,self.data
        vel,rotation = self.state()
        root = b._asset.data
        root.root_ang_vel_b = torch.tensor(vel[:3],dtype=torch.float32)[None]
        root.projected_gravity_b = torch.tensor(rotation.T @ [0,0,-1],dtype=torch.float32)[None]
        root.joint_pos = torch.tensor(d.qpos[self.qadr],dtype=torch.float32)[None]
        root.joint_vel = torch.tensor(d.qvel[self.vadr],dtype=torch.float32)[None]
        output = self.low(b._build_policy_observations(self.ids))
        assert torch.isfinite(output).all()
        b._last_policy_action.copy_(output)
        self.target[self.order] = self.q0 + .25*output[0].numpy()

    def physics(self):
        m,d = self.model,self.data
        torque = self.kp*(self.target-d.qpos[self.qadr])-self.kd*d.qvel[self.vadr]
        d.ctrl[self.act] = np.clip(torque,m.actuator_ctrlrange[self.act,0],m.actuator_ctrlrange[self.act,1])
        mj.mj_step(m,d)
        # Refresh body kinematics and contacts at the new state before observations.
        mj.mj_forward(m,d)

    def obstacle_contact(self):
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.obstacle_ids or contact.geom2 in self.obstacle_ids:
                mj.mj_contactForce(self.model,self.data,i,force)
                if np.linalg.norm(force[:3]) > 1.:
                    return True
        return False


def main():
    global BOXES, GOAL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/model_24999.pt')
    p.add_argument('--output',type=Path,default=ROOT/'experiments/sim2sim_obstacles'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--duration',type=float,default=90.)
    p.add_argument('--start-delay',type=float,default=5.)
    p.add_argument('--hold',type=float,default=20.)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--scene-json',type=Path,default=ROOT/'deploy/sim2sim/scene.json',help='Fixed evaluation scene with boxes (x,y,sx,sy,height) and goal (x,y,z).')
    p.add_argument('--xml',type=Path,default=DEFAULT_XML,help='Unitree G1 scene_29dof.xml')
    p.add_argument('--camera-distance',type=float,default=18.)
    p.add_argument('--camera-lookat',type=float,nargs=3,default=[6.,0.,.4])
    args = p.parse_args()
    if args.scene_json:
        scene_config = json.loads(args.scene_json.read_text())
        boxes = np.asarray(scene_config['boxes'],dtype=float)
        goal = np.asarray(scene_config['goal'],dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 5 or len(boxes)==0 or goal.shape != (3,):
            p.error('scene boxes must be nonempty Nx5 and goal must be length 3')
        if not np.isfinite(boxes).all() or not np.isfinite(goal).all() or (boxes[:,2:]<=0).any():
            p.error('scene values must be finite and obstacle sizes positive')
        BOXES, GOAL = boxes.tolist(), goal
    if not np.isfinite([args.duration,args.start_delay,args.hold]).all() or args.duration <= 0 or min(args.start_delay,args.hold) < 0:
        p.error('duration must be positive and delays nonnegative')
    torch.set_num_threads(1)
    torch.manual_seed(60)
    args.output.mkdir(parents=True,exist_ok=False)
    sim = Simulation(args.checkpoint, args.xml)
    manifest = dict(checkpoint=str(args.checkpoint.resolve()),sha256=sha256(args.checkpoint),
        iteration=sim.iteration, simulator='MuJoCo',mujoco_version=mj.__version__,
        python_version=sys.version.split()[0],torch_version=torch.__version__,numpy_version=np.__version__,
        scene_json=str(args.scene_json.resolve()),scene_sha256=sha256(args.scene_json),
        low_level=str(sim.low_path),low_level_sha256=sha256(sim.low_path),xml=str(sim.xml),
        source_sha256=sha256(__file__),boxes=BOXES,goal=GOAL.tolist(),start=[0,0,.8],
        actor_obs=1984,lidar_shape=[16,41],physics_hz=500,low_level_hz=50,navigation_hz=5,
        inference='deterministic, frozen weights; no artificial observation noise/delay',
        physics='official MuJoCo defaults including contact friction; explicit torque PD',
        success='XY distance < 0.5m continuously for 1s, checked at every physics tick',
        duration=args.duration)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sim.lidar()
    assert sim.obstacle_hits > 0, 'Obstacles must appear in LiDAR at start'
    print('[sim2sim] READY',json.dumps(manifest), 'initial obstacle rays',sim.obstacle_hits,flush=True)
    window = video = context = None
    if not args.headless:
        if not glfw.init():
            raise RuntimeError('GLFW init failed')
        window = glfw.create_window(1200,800,f'Model {sim.iteration} | MuJoCo navigation | ESC close',None,None)
        if not window:
            raise RuntimeError('GLFW window failed')
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        glfw.set_key_callback(window,lambda w,k,s,a,m:glfw.set_window_should_close(w,True) if k==glfw.KEY_ESCAPE and a==glfw.PRESS else None)
        camera = mj.MjvCamera()
        mj.mjv_defaultCamera(camera)
        camera.lookat[:] = args.camera_lookat
        camera.distance, camera.azimuth, camera.elevation = args.camera_distance,-115,-48
        option = mj.MjvOption()
        scene = mj.MjvScene(sim.model,maxgeom=10000)
        context = mj.MjrContext(sim.model,mj.mjtFontScale.mjFONTSCALE_150)
        video = imageio.get_writer(args.output/'evaluation.mp4',fps=10,codec='libx264',macro_block_size=1)

    def draw(status,record=False,snapshot=None):
        if window is None:
            return
        glfw.poll_events()
        rect = mj.MjrRect(0,0,*glfw.get_framebuffer_size(window))
        mj.mjv_updateScene(sim.model,sim.data,option,None,camera,mj.mjtCatBit.mjCAT_ALL,scene)
        mj.mjr_render(rect,scene,context)
        mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,mj.mjtGridPos.mjGRID_TOPLEFT,rect,
            f'Model {sim.iteration} | MuJoCo | {len(BOXES)} obstacles | GREEN = goal',status,context)
        if record or snapshot:
            rgb = np.empty((rect.height,rect.width,3),dtype=np.uint8)
            mj.mjr_readPixels(rgb,None,rect,context)
            rgb = rgb[::-1].copy()
            if record:
                video.append_data(rgb)
            if snapshot:
                imageio.imwrite(args.output/snapshot,rgb)
        glfw.swap_buffers(window)

    rows = []
    outcome, elapsed, dwell, min_distance, max_tilt = 'timeout',0.,0,float('inf'),0.
    started = time.monotonic()
    try:
        while window and time.monotonic()-started < args.start_delay:
            draw(f'Starting in {max(0,args.start_delay-(time.monotonic()-started)):.0f}s | ESC close')
            if glfw.window_should_close(window):
                outcome='closed_before_start'
                break
            time.sleep(.03)
        if outcome != 'closed_before_start':
            for tick in range(1500):  # 3s settle, standing command; navigation memory untouched.
                if tick % 10 == 0:
                    sim.low_update()
                sim.physics()
            draw(f'START | goal ({GOAL[0]:.2f}, {GOAL[1]:.2f}) | GREEN = goal',snapshot='initial.png')
            nav_start = sim.data.time
            wall_start = time.monotonic()
            for tick in range(round(args.duration/.002)):
                if window and glfw.window_should_close(window):
                    outcome='user_closed'
                    break
                if tick % 100 == 0:
                    sim.navigate()
                    if tick == 0:
                        np.save(args.output/'initial_observation.npy',sim.last_obs)
                if tick % 10 == 0:
                    sim.low_update()
                sim.physics()
                elapsed = sim.data.time-nav_start
                distance = float(np.linalg.norm(sim.data.qpos[:2]-GOAL[:2]))
                tilt = math.degrees(math.acos(np.clip(sim.data.xmat[sim.body].reshape(3,3)[2,2],-1,1)))
                min_distance,max_tilt = min(min_distance,distance),max(max_tilt,tilt)
                dwell = dwell+1 if distance < .5 else 0
                if tick % 50 == 0:
                    vel,rot = sim.state()
                    rows.append([elapsed,*sim.data.qpos[:3],math.atan2(rot[1,0],rot[0,0]),distance,tilt,
                        *sim.bridge._raw_navigation_actions[0].tolist(),*sim.bridge._velocity_commands[0].tolist(),
                        *vel[3:],vel[2],sim.hits,sim.obstacle_hits])
                    draw(f't={elapsed:.1f}s | distance={distance:.2f}m | tilt={tilt:.1f}deg | RUNNING',record=True)
                    if window:
                        time.sleep(max(0,elapsed-(time.monotonic()-wall_start)))
                if sim.data.qpos[2] < .45 or tilt > 40:
                    outcome='fall'
                    break
                if sim.obstacle_contact():
                    outcome='obstacle_contact'
                    break
                if dwell >= 500:
                    outcome='reached'
                    break
                if tick % 2500 == 0:
                    print(f'[sim2sim] t={elapsed:.1f}s distance={distance:.3f}m tilt={tilt:.2f}',flush=True)
        result = dict(outcome=outcome,reached=outcome=='reached',navigation_time_s=elapsed,
            final_distance_m=float(np.linalg.norm(sim.data.qpos[:2]-GOAL[:2])),
            min_distance_m=min_distance if np.isfinite(min_distance) else None,max_tilt_deg=max_tilt,
            final_position=sim.data.qpos[:3].tolist(),
            final_command=sim.bridge._velocity_commands[0].tolist(),
            note='Simulation freezes on result; reaching does not prove stable stopping or general success rate.')
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        with (args.output/'trajectory.csv').open('w') as f:
            writer=csv.writer(f)
            writer.writerow(['time','x','y','z','yaw','goal_distance','tilt','raw_x','raw_y','raw_yaw',
                'cmd_vx','cmd_vy','cmd_wz','vx','vy','vz','wz','lidar_hits','obstacle_hits'])
            writer.writerows(rows)
        draw(f'{outcome.upper()} | distance={result["final_distance_m"]:.2f}m | FROZEN | ESC close',record=True,snapshot='final.png')
        if video:
            video.close()
            video=None
        print('[sim2sim] RESULT',json.dumps(result),'output',args.output,flush=True)
        finish = time.monotonic()
        while window and not glfw.window_should_close(window) and time.monotonic()-finish < args.hold:
            draw(f'{outcome.upper()} | distance={result["final_distance_m"]:.2f}m | FROZEN | ESC close')
            time.sleep(.03)
    finally:
        if video:
            video.close()
        if context:
            context.free()
        if window:
            glfw.destroy_window(window)
            glfw.terminate()
    return 0 if outcome in ('reached', 'user_closed', 'closed_before_start') else 1


if __name__ == '__main__':
    sys.exit(main())
