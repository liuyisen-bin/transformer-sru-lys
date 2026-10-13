#!/usr/bin/env python3
"""Run the released ONNX hierarchy in local MuJoCo, with an ideal LiDAR sensor.

Physics 500 Hz, Unitree policy 50 Hz, navigation policy 5 Hz. No ROS or DDS.
"""
import argparse
import json
import os
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import yaml

from core import CONTRACT, ROOT, GoalTracker, Navigation, Unitree, relative_goal, rotation
from startup import Startup
from lidar_projection import LidarProjection

HERE = Path(__file__).resolve().parent
XML = Path(os.environ.get('SRU_MUJOCO_XML',
    str(ROOT/'third_party/unitree_mujoco/unitree_robots/g1/scene_29dof.xml')))


class Simulation:
    def __init__(self, scale, xml=XML, boxes=()):
        self.mode = "policy"
        if not Path(xml).is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {xml}. Clone unitree_mujoco or pass --xml.")
        self.boxes = [tuple(map(float, box)) for box in boxes]
        if any(len(box)!=5 or not np.isfinite(box).all() or min(box[2:])<=0 for box in self.boxes):
            raise ValueError('Boxes must contain finite x,y,size_x,size_y,height with positive sizes')
        spec = mj.MjSpec.from_file(str(xml))
        for i,(x,y,sx,sy,height) in enumerate(self.boxes):
            spec.worldbody.add_geom(name=f'nav_obstacle_{i}',type=mj.mjtGeom.mjGEOM_BOX,
                                    pos=(x,y,height/2),size=(sx/2,sy/2,height/2),
                                    rgba=(.72,.36,.12,1),contype=1,conaffinity=1)
        spec.worldbody.add_site(name='nav_goal', type=mj.mjtGeom.mjGEOM_CYLINDER,
                               pos=(0, 0, -1), size=(.5, .012), rgba=(.1, .9, .2, .7))
        self.model = m = spec.compile()
        m.opt.timestep = .002
        self.data = mj.MjData(m)
        self.obstacle_ids={m.geom(f'nav_obstacle_{i}').id for i in range(len(self.boxes))}
        self.robot_geom_ids={i for i in range(m.ngeom) if m.geom_bodyid[i]!=0}
        self.body, self.torso = m.body('pelvis').id, m.body('torso_link').id
        joints = np.array([m.joint(n).id for n in CONTRACT['joint_names_sdk']])
        self.qadr, self.vadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.act = np.array([next(i for i in range(m.nu) if m.actuator_trnid[i, 0] == j) for j in joints])
        feet = [m.body(n).id for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        self.feet = [i for i in range(m.ngeom) if m.geom_bodyid[i] in feet and m.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE]
        for i in range(m.ngeom):
            m.geom_group[i] = 0 if m.geom_bodyid[i] == 0 else 1
        self.groups = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        self.projection = LidarProjection()
        self.rays, self.mount = self.projection.rays, self.projection.mount
        self.low, self.nav = Unitree(action_scale=scale), Navigation()
        self.scale = scale
        self.reset()

    def reset(self):
        m, d = self.model, self.data
        mj.mj_resetData(m, d)
        self.low.reset(); self.nav.reset()
        self.startup, self.goals = Startup(), GoalTracker()
        d.qpos[:3] = [0, 0, .8]; d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = self.low.default_sdk
        mj.mj_forward(m, d)
        d.qpos[2] += .001 - min(d.geom_xpos[i, 2] - m.geom_size[i, 0] for i in self.feet)
        mj.mj_forward(m, d)
        self.support_pose = d.qpos[:7].copy()
        self.origin = d.qpos[:3].copy()
        self.started = None; self.next_nav = 0.; self.nav_steps = 0
        self.command = np.zeros(3); self.raw = np.zeros(3); self.goal_body = np.zeros(3)
        self.stopped = False; self.fallen = False
        self.lidar_points = np.zeros((656, 3)); self.hits = 0; self.trace = []; self.obs = []
        self.obstacle_hits=0;self.first_obstacle_contact=None;self.obstacle_contact_substeps=0
        m.site_pos[m.site('nav_goal').id] = [0, 0, -1]

    def state(self):
        d = self.data; r = d.xmat[self.body].reshape(3, 3); v = np.zeros(6)
        mj.mj_objectVelocity(self.model, d, mj.mjtObj.mjOBJ_BODY, self.body, v, 0)
        return dict(q=d.qpos[self.qadr].copy(), dq=d.qvel[self.vadr].copy(),
                    quat=d.qpos[3:7].copy(), gyro=r.T@v[:3], linear=r.T@v[3:])

    def prepare(self):
        if self.stopped or self.fallen: raise ValueError('Reset first')
        self.startup.prepare(self.state()['q'], self.data.time)

    def start_policy(self):
        if self.stopped or self.fallen: raise ValueError('Reset first')
        self.startup.start_policy(self.state(), self.data.time)
        self.low.reset(); self.nav.reset()

    def goal(self):
        if self.stopped or self.fallen or self.startup.phase != 'ZERO_VELOCITY':
            raise ValueError('G requires active zero-velocity policy')
        p = self.data.qpos[:3]
        goal = relative_goal(p, self.state()['quat'], 7., 0., .5)
        self.goals.set(goal, self.data.time); self.nav.reset()
        self.started = self.data.time; self.next_nav = self.data.time
        self.nav_steps = 0
        self.model.site_pos[self.model.site('nav_goal').id] = [*goal[:2], .012]

    def lidar(self):
        m, d = self.model, self.data
        torso_r = d.xmat[self.torso].reshape(3, 3); sensor_r = torso_r@self.mount
        self.lidar_origin = d.xpos[self.torso] + torso_r@self.projection.origin
        rw = self.rays@sensor_r.T; ids = np.empty(656, np.int32); distances = np.empty(656)
        mj.mj_multiRay(m, d, self.lidar_origin, rw.reshape(-1), self.groups, True, -1,
                       ids, distances, None, 656, mj.mjMAXVAL)
        valid = (distances > 1e-6) & (distances <= self.projection.max_distance)
        points, self.hits = self.projection.from_virtual_ranges(distances, d.xquat[self.torso])
        self.obstacle_hits=int((valid & np.isin(ids,list(self.obstacle_ids))).sum())
        self.lidar_points = self.lidar_origin + distances[valid, None]*rw[valid]
        return points

    @property
    def phase(self):
        if self.stopped: return 'STOPPED / LATCHED'
        if self.fallen: return 'FALL / PAUSED'
        return self.startup.phase if self.started is None else self.goals.state

    def step(self):
        if self.stopped or self.fallen: return
        m, d = self.model, self.data; st = self.state()
        self.startup.observe(st, d.time)
        position = self.startup.position_command(st, d.time)
        if position is None:
            if self.started is not None:
                self.goal_body = rotation(st['quat']).T@(self.goals.goal-d.qpos[:3])
                state = self.goals.update(d.qpos[:3], d.time)
                if state != 'NAVIGATING':
                    self.nav.reset(); self.command[:] = 0; self.raw[:] = 0
                elif d.time+1e-9 >= self.next_nav:
                    self.command = self.nav.step(st['linear'], st['gyro'], rotation(st['quat']).T@np.array([0., 0., -1.]), self.goal_body, self.lidar())
                    self.raw = self.nav.previous.copy(); self.nav_steps += 1; self.next_nav += .2
                    self.obs.append(self.nav.last_obs[0].copy())
            target = self.low.step(st['q'], st['dq'], st['gyro'], rotation(st['quat']).T@np.array([0., 0., -1.]), self.command)
            kp, kd = np.array(CONTRACT['kp_sdk']), np.array(CONTRACT['kd_sdk'])
        else:
            target, kp, kd = position
        self.trace.append(dict(t=float(d.time), goal_t=None if self.started is None else float(d.time-self.started),
                               phase=self.phase, raw_action=self.raw.tolist(), command=self.command.tolist(),
                               goal_body=self.goal_body.tolist(), position=d.qpos[:3].tolist(), quat=st['quat'].tolist(),
                               actual_body_velocity=st['linear'].tolist(), actual_yaw_rate=float(st['gyro'][2]), lidar_hits=self.hits))
        for _ in range(10):
            torque = kp*(target-d.qpos[self.qadr])-kd*d.qvel[self.vadr]
            d.ctrl[self.act] = np.clip(torque, m.actuator_ctrlrange[self.act, 0], m.actuator_ctrlrange[self.act, 1])
            mj.mj_step(m, d)
            if self.obstacle_ids:
                for contact in d.contact:
                    if ((contact.geom1 in self.obstacle_ids and contact.geom2 in self.robot_geom_ids) or
                        (contact.geom2 in self.obstacle_ids and contact.geom1 in self.robot_geom_ids)):
                        self.obstacle_contact_substeps+=1
                        if self.first_obstacle_contact is None:
                            self.first_obstacle_contact=dict(time=float(d.time),position=d.qpos[:3].tolist(),
                                geoms=[m.geom(contact.geom1).name,m.geom(contact.geom2).name])
                        break
            # Explicit virtual support only during Passive/FixStand, analogous
            # to pre-policy hoist support. Released completely at start-policy.
            if position is not None:
                d.qpos[:7] = self.support_pose; d.qvel[:6] = 0
            mj.mj_forward(m, d)
        self.fallen = bool(rotation(d.qpos[3:7])[2, 2] < np.cos(np.deg2rad(55)) or d.qpos[2] < .4)


def save(args, sims):
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(hardware_motor_output=False, ros_dds_io=False,
                   high_model=CONTRACT['high_model'], high_sha256=CONTRACT['high_sha256'], low_sha256=CONTRACT['low_sha256'],
                   physics_hz=500, low_hz=50, high_hz=5, boxes=[], command_min=CONTRACT['command_min'],
                   command_max=CONTRACT['command_max'], command_alpha=CONTRACT['command_alpha'],
                   scene_xml=str(args.xml), cases=[],
                   limitations=['Ideal 16x41 floor ray sensor; no real Livox sparsity, latency, IMU error or room geometry.',
                                'Nominal initial pose; root virtually supported before start-policy, then released.'])
    for s in sims:
        (args.output/(s.mode+'_trace.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in s.trace))
        np.savez_compressed(args.output/(s.mode+'_observations.npz'), obs=np.array(s.obs))
        summary['cases'].append(dict(mode=s.mode, phase=s.phase, nav_steps=s.nav_steps, fallen=s.fallen,
                                     low_action_scale=s.scale, sim_time=float(s.data.time),
                                     displacement_m=(s.data.qpos[:3]-s.origin).tolist(), final_command=s.command.tolist()))
    (args.output/'result.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['cases'], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', type=Path, default=XML, help='Official Unitree G1 scene_29dof.xml')
    p.add_argument('--output', type=Path, default=ROOT/'experiments/mujoco_navigation'/time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--check', action='store_true'); p.add_argument('--seconds', type=float, default=15.)
    p.add_argument('--auto', action='store_true', help='Automatically prepare, start-policy, then set forward goal')
    p.add_argument('--frames', type=int, default=0, help='Optional bounded GUI run')
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0: p.error('--seconds must be finite and positive')
    if args.frames < 0: p.error('--frames must be nonnegative')
    scale = yaml.safe_load((HERE/'config.yaml').read_text())['low_action_scale']
    sims = [Simulation(scale, xml=args.xml)]
    running = False
    def dispatch(action):
        nonlocal running
        if action == 'stop':
            for s in sims: s.stopped = True; s.command[:] = 0
            running = False
        elif action == 'reset':
            for s in sims: s.reset()
            running = False
        else:
            for s in sims:
                try: getattr(s, action)()
                except ValueError as exc: print(s.mode, 'REJECTED', str(exc), flush=True)
            running = True
        print('ACTION', action, [s.phase for s in sims], flush=True)

    def automatic():
        dispatch('prepare')
        for _ in range(160):
            for s in sims: s.step()
        dispatch('start_policy')
        for _ in range(100):
            for s in sims: s.step()
        dispatch('goal')

    if args.check:
        automatic()
        for _ in range(round(args.seconds/.02)):
            for s in sims: s.step()
        save(args, sims); return
    import glfw
    if not glfw.init(): raise RuntimeError('GLFW display unavailable')
    win = glfw.create_window(1560, 920, 'G1 navigation ONNX | F prepare P policy G goal', None, None)
    if not win: raise RuntimeError('Cannot create GLFW window')
    glfw.make_context_current(win); glfw.swap_interval(1)
    def key(w, k, scancode, action, mods):
        nonlocal running
        if action != glfw.PRESS: return
        if k == glfw.KEY_F: dispatch('prepare')
        elif k == glfw.KEY_P: dispatch('start_policy')
        elif k == glfw.KEY_G: dispatch('goal')
        elif k == glfw.KEY_BACKSPACE: dispatch('stop')
        elif k == glfw.KEY_R: dispatch('reset')
        elif k == glfw.KEY_SPACE: running = not running
        elif k == glfw.KEY_ESCAPE: glfw.set_window_should_close(w, True)
    glfw.set_key_callback(win, key)
    scenes = [mj.MjvScene(s.model, maxgeom=12000) for s in sims]
    contexts = [mj.MjrContext(s.model, mj.mjtFontScale.mjFONTSCALE_150) for s in sims]
    cameras = [mj.MjvCamera() for _ in sims]; option = mj.MjvOption()
    for camera in cameras:
        mj.mjv_defaultCamera(camera); camera.distance = 8.; camera.azimuth = 140; camera.elevation = -45
    print('F prepare -> wait 3s -> P start-policy -> G forward 7m; Backspace stop (latched); Space pause; R reset; Esc exit', flush=True)
    print('Keyboard only. No ROS, DDS or real robot I/O.', flush=True)
    print('Ideal floor sensor and local physics. Output:', args.output, flush=True)
    if args.auto: automatic()
    frames = 0
    try:
        while not glfw.window_should_close(win):
            begin = time.monotonic(); glfw.poll_events()
            if running:
                for s in sims: s.step()
            w, h = glfw.get_framebuffer_size(win)
            for i, s in enumerate(sims):
                st = s.state(); camera = cameras[i]
                camera.lookat[:] = s.data.qpos[:3] + [1.5, 0, -.1]
                rect = mj.MjrRect(0, 0, w, h)
                mj.mjv_updateScene(s.model, s.data, option, None, camera, mj.mjtCatBit.mjCAT_ALL, scenes[i])
                if i == 0:
                    for point in s.lidar_points[::4]:
                        g = scenes[i].geoms[scenes[i].ngeom]
                        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([.012]*3), point, np.eye(3).reshape(-1), np.array([1., .7, .05, .8], np.float32))
                        scenes[i].ngeom += 1
                mj.mjr_render(rect, scenes[i], contexts[i])
                heading = 'G1 | 24999 ONNX CLOSED LOOP'
                text = f'{s.phase} | {"RUN" if running else "PAUSED"}\n'
                text += f'goal t={0 if s.started is None else s.data.time-s.started:.2f}s | nav steps={s.nav_steps}\n'
                text += 'raw action: ' + ' '.join(f'{v:+.3f}' for v in s.raw) + '\n'
                text += 'cmd vx vy wz: ' + ' '.join(f'{v:+.3f}' for v in s.command) + '\n'
                text += f'actual vx={st["linear"][0]:+.3f} wz={st["gyro"][2]:+.3f}\n'
                text += 'goal body: ' + ' '.join(f'{v:+.2f}' for v in s.goal_body) + '\n'
                text += f'low scale={s.scale} | lidar cells={s.hits}\n'
                text += 'IDEAL FLOOR SENSOR / EMPTY SCENE'
                mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_TOPLEFT, rect, heading+'\n'+text, '', contexts[i])
                mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_BOTTOMLEFT, rect,
                               'F prepare | P policy | G goal | Backspace stop | Space pause | R reset', '', contexts[i])
            glfw.swap_buffers(win); frames += 1
            if args.frames and frames >= args.frames: break
            time.sleep(max(0, .02-(time.monotonic()-begin)))
    finally:
        save(args, sims)
        for c in contexts: c.free()
        glfw.destroy_window(win); glfw.terminate()


if __name__ == '__main__': main()
