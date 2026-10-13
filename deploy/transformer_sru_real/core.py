"""Pure numerical navigation/Unitree controller; no ROS, SDK, or motor I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads(Path(__file__).with_name('contract.json').read_text())


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('invalid_' + name)
    return result


def rotation(q):
    q = vector(q, 4, 'quaternion').astype(np.float64)
    norm = np.linalg.norm(q)
    if abs(norm-1.) > .05:
        raise ValueError('quaternion_not_unit')
    w,x,y,z = q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def load_session(kind):
    path = ROOT / CONTRACT[kind+'_model']
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTRACT[kind+'_sha256']:
        raise ValueError(kind + '_model_hash_mismatch')
    opt = ort.SessionOptions(); opt.intra_op_num_threads=1; opt.inter_op_num_threads=1
    return ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])


def relative_goal(position, q_wxyz, forward, left, height):
    p = vector(position,3,'odom_position').astype(float)
    r = rotation(q_wxyz); yaw = np.arctan2(r[1,0],r[0,0])
    if not np.isfinite([forward,left,height]).all() or not .2 <= np.hypot(forward,left) <= 10:
        raise ValueError('goal_distance_must_be_0.2_to_10m')
    return np.array([p[0]+np.cos(yaw)*forward-np.sin(yaw)*left,
                     p[1]+np.sin(yaw)*forward+np.cos(yaw)*left, height])


def check_receipt_age(received_at, steady_now, limit, name):
    """Local stream liveness only; does not establish acquisition-time latency."""
    if not np.isfinite([received_at, steady_now, limit]).all() or not 0 <= steady_now-received_at <= limit:
        raise ValueError(name+'_receipt_stale')


def check_age(stamp_ns, received_at, ros_now_ns, steady_now, limit, name):
    if not isinstance(stamp_ns, (int,np.integer)) or stamp_ns <= 0:
        raise ValueError(name+'_invalid_stamp')
    age = (ros_now_ns-stamp_ns)/1e9
    if not -.05 <= age <= limit:
        raise ValueError(f'{name}_source_age:{age:.3f}s')
    check_receipt_age(received_at, steady_now, limit, name)


def tick_advanced(old, new):
    return old is None or 0 < ((new-old) & 0xffffffff) < 0x80000000


class Navigation:
    def __init__(self):
        self.session=load_session('high')
        ins={x.name:x.shape for x in self.session.get_inputs()}
        if ins != {'obs':[1,1984],'h_in':[1,1,512],'c_in':[1,1,512]}:
            raise ValueError('wrong_navigation_model_contract')
        self.reset()

    def reset(self):
        self.h=np.zeros((1,1,512),np.float32);self.c=self.h.copy()
        self.previous=np.zeros(3,np.float32);self.command=self.previous.copy()
        self.last_obs=None

    def step(self, linear_body, gyro_body, gravity_body, goal_body, lidar):
        goal=vector(goal_body,3,'goal_body');distance=float(np.linalg.norm(goal))
        obs=np.concatenate([vector(linear_body,3,'linear_velocity'),vector(gyro_body,3,'gyro'),
                            vector(gravity_body,3,'gravity'),self.previous,
                            goal/max(distance,1e-6),[np.log1p(distance)],
                            vector(lidar,1968,'lidar')]).astype(np.float32)[None]
        action,h,c=self.session.run(None,{'obs':obs,'h_in':self.h,'c_in':self.c})
        if action.shape!=(1,3) or h.shape!=(1,1,512) or c.shape!=(1,1,512) or not all(np.isfinite(a).all() for a in (action,h,c)):
            raise ValueError('invalid_navigation_output')
        self.previous=action[0].copy();self.h=h;self.c=c;self.last_obs=obs
        normalized=np.tanh(self.previous)
        mapped=normalized*np.where(normalized>=0,CONTRACT['command_max'],np.abs(CONTRACT['command_min']))
        alpha=np.array(CONTRACT['command_alpha'],np.float32)
        self.command=(alpha*self.command+(1-alpha)*mapped).astype(np.float32)
        return self.command.copy()


class Unitree:
    def __init__(self, action_scale=None):
        self.session=load_session('low')
        inputs=self.session.get_inputs()
        if len(inputs)!=1 or inputs[0].shape!=[1,480]:
            raise ValueError('wrong_unitree_input_contract')
        self.input_name=inputs[0].name
        # None preserves the official contract for reference/replay callers.
        if action_scale is not None and (not np.isfinite(action_scale) or action_scale<=0):
            raise ValueError('invalid_low_action_scale')
        self.action_scale=np.array(CONTRACT['action_scale'],np.float32) if action_scale is None else np.full(29,action_scale,np.float32)
        self.order=np.array(CONTRACT['policy_to_sdk'])
        self.default=np.array(CONTRACT['default_q_policy'],np.float32)
        self.default_sdk=np.empty(29,np.float32);self.default_sdk[self.order]=self.default
        self.reset()

    def reset(self):
        self.history=None;self.previous=np.zeros(29,np.float32);self.last_obs=None

    def observations(self,q_sdk,dq_sdk,gyro,gravity,commands):
        q=vector(q_sdk,29,'joint_positions')[self.order]
        dq=vector(dq_sdk,29,'joint_velocities')[self.order]
        terms=[vector(gyro,3,'gyro')*.2,vector(gravity,3,'gravity'),vector(commands,3,'commands'),
               q-self.default,dq*.05,self.previous]
        if self.history is None:
            self.history=[np.tile(t,(5,1)) for t in terms]
        else:
            self.history=[np.concatenate([h[1:],t[None]],axis=0) for h,t in zip(self.history,terms)]
        self.last_obs=np.concatenate([h.ravel() for h in self.history]).astype(np.float32)[None]
        return self.last_obs

    def step(self,q_sdk,dq_sdk,gyro,gravity,commands):
        obs=self.observations(q_sdk,dq_sdk,gyro,gravity,commands)
        action=self.session.run(None,{self.input_name:obs})[0]
        if action.shape!=(1,29) or not np.isfinite(action).all():
            raise ValueError('invalid_unitree_output')
        self.previous=action[0].copy()
        target=np.empty(29,np.float32)
        target[self.order]=self.default+self.action_scale*self.previous
        return target


def target_diagnostic(target,q,dq,kp=None,kd=None):
    # Diagnostic only: neither clips commands nor decides whether to publish.
    target=vector(target,29,'target');q=vector(q,29,'q');dq=vector(dq,29,'dq')
    bounds=np.array(CONTRACT['joint_limits_sdk'])
    kp=np.array(CONTRACT['kp_sdk'] if kp is None else kp)
    outside=np.flatnonzero(((target<bounds[:,0])|(target>bounds[:,1])) & (kp>0))
    if len(outside):return 'target_joint_limit:'+CONTRACT['joint_names_sdk'][outside[0]]
    kd=np.array(CONTRACT['kd_sdk'] if kd is None else kd)
    torque=kp*(target-q)-kd*dq
    bad=np.flatnonzero(abs(torque)>np.array(CONTRACT['effort_limits_sdk']))
    if len(bad):return 'estimated_pd_torque_limit:'+CONTRACT['joint_names_sdk'][bad[0]]
    return ''


class GoalTracker:
    """Fixed world-frame target; arrival/timeout explicitly clear the command."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.goal=None;self.started=None;self.inside_since=None;self.state='IDLE'

    def set(self,goal,now):
        self.goal=vector(goal,3,'world_goal').copy();self.started=now
        self.inside_since=None;self.state='NAVIGATING'

    def update(self,position,now):
        if self.goal is None:return self.state
        if self.state in ('ARRIVED','TIMEOUT'):return self.state
        if now-self.started>=60:
            self.state='TIMEOUT';return self.state
        distance=np.linalg.norm((self.goal-vector(position,3,'position'))[:2])
        if distance<.5:
            if self.inside_since is None:self.inside_since=now
            self.state='ARRIVED' if now-self.inside_since>=1. else 'SETTLING'
        else:
            self.inside_since=None;self.state='NAVIGATING'
        return self.state
