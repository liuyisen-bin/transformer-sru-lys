#!/usr/bin/env python3
"""24999 at 5 Hz -> official Unitree at 50 Hz -> guarded HG targets at 500 Hz.

Default is read-only. A LowCmd publisher and MotionSwitcher release are created
only by --enable-lowcmd, after fresh robot-state checks have passed. Motor mode
starts in PASSIVE. Navigation separately requires fresh, calibrated sensors.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger
from unitree_api.msg import Request, Response
from unitree_hg.msg import LowState, LowCmd
from transformer_sru_lidar_adapter.msg import LidarTensor
import yaml

from core import CONTRACT, ROOT, Navigation, Unitree, GoalTracker, rotation, vector, check_age, check_receipt_age, tick_advanced, target_diagnostic
from transport import motor_message
from startup import Startup, remote_transition, SelectDoubleClick, ButtonDoubleClick
from core import relative_goal
from navigation_trace import NavigationTrace


def stamp_ns(stamp):
    if stamp.sec<0 or not 0<=stamp.nanosec<10**9:raise ValueError('invalid_ROS_timestamp')
    return int(stamp.sec)*10**9+int(stamp.nanosec)


class Controller(Node):
    def __init__(self,args):
        super().__init__('transformer_sru_unitree_controller')
        self.args=args;self.cfg=yaml.safe_load(args.config.read_text())
        for name in ('lowstate_timeout_s','lidar_timeout_s','odom_timeout_s','max_tilt_deg',
                     'command_watchdog_s','writer_deadline_s','stop_damping_sdk','low_action_scale'):
            if not np.isfinite(self.cfg[name]) or self.cfg[name]<=0:raise ValueError('invalid_config:'+name)
        if self.cfg['writer_deadline_s']>=self.cfg['command_watchdog_s']:raise ValueError('writer deadline must be shorter than target watchdog')
        self.nav=Navigation();self.low=Unitree(action_scale=self.cfg['low_action_scale']);self.goals=GoalTracker()
        self.navigation_trace=NavigationTrace()
        self.startup=Startup();self.remote_buttons=None;self.last_transition_error=''
        self.select_stop=SelectDoubleClick()
        self.a_goal=ButtonDoubleClick(8)  # SDK A: wireless_remote[3] bit 0.
        self.lowstate=None;self.odom=None;self.lidar=None
        self.last_tick=None;self.low_updates=0;self.low_at=0.;self.odom_at=0.;self.lidar_at=0.
        self.phase='WAITING';self.reason='waiting_for_inputs';self.fault='';self.goal_stamp=0
        self.stop_event=threading.Event();self.writer=None;self.writer_fault=''
        self.lowcmd_pub=None;self.owns_control=False;self.suppress_stop_publish=False
        self.snapshot=None;self.damping=None;self.publish_count=0;self.max_publish_gap=0.;self.max_compute_s=0.
        self.last_publish_at=0.;self.last_low_step=0.;self.last_nav_step=0.;self.nav_count=0;self.low_count=0
        self.next_low=0.;self.next_nav=0.
        self.preview_target=None;self.target_diagnostic_text='';self.lock_file=None;self.ramp_start=None
        self.last_goal_error='';self.last_ros_ns=self.get_clock().now().nanoseconds
        self.last_status={};self.responses={};self.rpc_pub=None
        self.log_path=args.log or ROOT/'experiments/real_deploy_logs'/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
        self.log_path.parent.mkdir(parents=True,exist_ok=True);self.log_file=self.log_path.open('x')
        self.create_subscription(LowState,'/lowstate',self.on_low,qos_profile_sensor_data)
        self.create_subscription(Odometry,self.cfg['odom_topic'],self.on_odom,qos_profile_sensor_data)
        self.create_subscription(LidarTensor,self.cfg['lidar_topic'],self.on_lidar,qos_profile_sensor_data)
        self.create_subscription(PointStamped,'/transformer_sru/goal',self.on_goal,10)
        self.status_pub=self.create_publisher(String,'/transformer_sru/status',10)
        self.preview_pub=self.create_publisher(TwistStamped,'/transformer_sru/preview_command',10)
        self.create_service(Trigger,'/transformer_sru/stop',self.on_stop)
        for action in ('prepare','start_policy','damping'):
            self.create_service(Trigger,'/transformer_sru/'+action,
                lambda req,res,action=action:self.on_transition(action,res))
        self.event('start',control_enabled=args.enable_lowcmd,log=str(self.log_path),contract=CONTRACT['high_sha256'],low_action_scale=self.cfg['low_action_scale'])

    def event(self,kind,**values):
        record={'event':kind,'wall_time':time.time(),**values}
        self.log_file.write(json.dumps(record)+'\n');self.log_file.flush()
        if kind!='step':print(json.dumps(record),flush=True)

    def fail(self,reason):
        if not self.fault:self.event('fault',reason=reason)
        self.fault=reason;self.phase='FAULT';self.reason=reason
        self.nav.reset();self.goals.clear()
        if self.args.enable_lowcmd:self.stop_event.set()

    def on_low(self,msg):
        try:
            if int(msg.mode_machine)!=5 or int(msg.mode_pr)!=0:raise ValueError('requires_g1_29dof_PR_mode')
            q=vector([m.q for m in msg.motor_state[:29]],29,'lowstate_q')
            dq=vector([m.dq for m in msg.motor_state[:29]],29,'lowstate_dq')
            quat=vector(msg.imu_state.quaternion,4,'base_imu');rotation(quat)
            gyro=vector(msg.imu_state.gyroscope,3,'base_gyro')
            tick=int(msg.tick)
            if not tick_advanced(self.last_tick,tick):return
            self.last_tick=tick;self.low_at=time.monotonic();self.low_updates+=1
            self.lowstate={'q':q,'dq':dq,'quat':quat,'gyro':gyro,'tick':tick,'mode_machine':int(msg.mode_machine)}
            buttons=int(msg.wireless_remote[2]) | (int(msg.wireless_remote[3])<<8)
            action=remote_transition(self.remote_buttons,buttons);self.remote_buttons=buttons
            select_stop=self.select_stop.update(buttons,self.low_at)
            a_goal=self.a_goal.update(buttons,self.low_at)
            if select_stop or action=='stop':
                self.reason='remote_select_double_click_stop' if select_stop else 'remote_LT_B_stop'
                self.stop_event.set();self.nav.reset();self.goals.clear()
            elif action and self.owns_control:
                try:self.transition(action)
                except ValueError as error:
                    self.last_transition_error=str(error);self.event('transition_rejected',action=action,reason=str(error))
            elif a_goal:
                self.on_remote_goal()
        except Exception as error:self.fail(str(error))

    def on_odom(self,msg):
        try:
            if msg.header.frame_id!=self.cfg['odom_frame'] or msg.child_frame_id!=self.cfg['odom_child_frame']:
                raise ValueError('odom_frame_mismatch')
            stamp=stamp_ns(msg.header.stamp)
            if stamp<=0:raise ValueError('odom_invalid_stamp')
            if self.odom is not None and stamp<=self.odom['stamp']:return
            p=msg.pose.pose.position;q=msg.pose.pose.orientation;v=msg.twist.twist.linear
            quat=vector([q.w,q.x,q.y,q.z],4,'odom_quaternion');rotation(quat)
            position=vector([p.x,p.y,p.z],3,'odom_position')
            if self.odom is not None:
                dt=(stamp-self.odom['stamp'])/1e9
                if np.linalg.norm(position-self.odom['position'])>max(.3,3*dt):
                    raise ValueError('odometry_position_discontinuity')
                angle=2*np.arccos(np.clip(abs(float(quat@self.odom['quat']))/(np.linalg.norm(quat)*np.linalg.norm(self.odom['quat'])),0,1))
                if angle>max(.35,2*dt):raise ValueError('odometry_orientation_discontinuity')
            self.odom={'stamp':stamp,'position':position,
                       'quat':quat,'linear':vector([v.x,v.y,v.z],3,'odom_linear')}
            self.odom_at=time.monotonic()
        except Exception as error:self.fail(str(error))

    def on_lidar(self,msg):
        try:
            if not np.isfinite([msg.normalization_distance_m,msg.imu_cloud_offset_s]).all():raise ValueError('nonfinite_lidar_metadata')
            if msg.rows!=16 or msg.columns!=41 or abs(msg.normalization_distance_m-5.)>1e-6 or msg.header.frame_id!=self.cfg['lidar_frame']:
                raise ValueError('lidar_contract_mismatch')
            xyz=vector(msg.xyz,1968,'lidar');norm=np.linalg.norm(xyz.reshape(656,3),axis=1)
            if np.any(norm>1.000001) or np.count_nonzero(norm)>656 or np.count_nonzero(norm)==0:
                raise ValueError('lidar_range_or_empty')
            if int(msg.occupied_cells)!=np.count_nonzero(norm):raise ValueError('lidar_count_mismatch')
            if abs(float(msg.imu_cloud_offset_s))>.050001:raise ValueError('lidar_imu_time_mismatch')
            stamp=stamp_ns(msg.header.stamp)
            if self.lidar is not None and stamp<=self.lidar['stamp']:return
            self.lidar={'stamp':stamp,'xyz':xyz,'occupied':int(msg.occupied_cells),
                        'observed':int(msg.observed_cells),'floor_filled':int(msg.floor_filled_cells),
                        'floor_height':float(msg.floor_height_m),
                        'extrinsics_verified':bool(msg.extrinsics_verified)};self.lidar_at=time.monotonic()
        except Exception as error:self.fail(str(error))

    def input_error(self,full=True):
        now=time.monotonic();ros=self.get_clock().now().nanoseconds
        if self.lowstate is None or self.low_updates<2:return 'waiting_lowstate'
        if now-self.low_at>self.cfg['lowstate_timeout_s']:return 'lowstate_tick_stale'
        gravity=rotation(self.lowstate['quat']).T@np.array([0.,0.,-1.])
        tilt=np.rad2deg(np.arccos(np.clip(-gravity[2],-1,1)))
        if tilt>self.cfg['max_tilt_deg']:return f'base_tilt:{tilt:.1f}deg'
        if max(abs(self.lowstate['dq']))>35:return 'joint_velocity_exceeds_35rad_s'
        if not full:return ''
        problems=[]
        for name,record,at in [('odom',self.odom,self.odom_at),('lidar',self.lidar,self.lidar_at)]:
            if record is None:problems.append('waiting_'+name);continue
            try:
                if name=='odom':check_receipt_age(at,now,self.cfg['odom_timeout_s'],name)
                else:check_age(record['stamp'],at,ros,now,self.cfg['lidar_timeout_s'],name)
            except ValueError as error:problems.append(str(error))
        if self.args.enable_lowcmd and self.lidar is not None and not self.lidar['extrinsics_verified']:
            problems.append('lidar_extrinsics_unverified')
        return ';'.join(problems)

    def on_remote_goal(self):
        source='remote_A_double_click'
        if not self.owns_control or self.startup.phase!='ZERO_VELOCITY':
            self.last_goal_error='remote_goal_requires_active_policy'
            self.event('goal_rejected',source=source,reason=self.last_goal_error)
            return
        if self.odom is None:
            self.last_goal_error='waiting_odom'
            self.event('goal_rejected',source=source,reason=self.last_goal_error)
            return
        world=relative_goal(self.odom['position'],self.odom['quat'],7.,0.,.5)
        msg=PointStamped();msg.header.frame_id=self.cfg['odom_frame']
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.point.x,msg.point.y,msg.point.z=map(float,world)
        self.on_goal(msg,source=source)

    def on_goal(self,msg,source='topic'):
        try:
            if self.stop_event.is_set():raise ValueError('stop_or_fault_latched')
            if self.fault or self.phase not in ('READY','ZERO_VELOCITY','NAVIGATING','SETTLING','ARRIVED'):
                raise ValueError('controller_not_ready:'+self.phase)
            problem=self.input_error()
            if problem:raise ValueError(problem)
            if self.args.enable_lowcmd and self.count_publishers(self.cfg['lidar_topic'])!=1:
                raise ValueError('lidar_publisher_count_not_one')
            if msg.header.frame_id!=self.cfg['odom_frame']:raise ValueError('goal_frame_mismatch')
            stamp=stamp_ns(msg.header.stamp)
            check_age(stamp,time.monotonic(),self.get_clock().now().nanoseconds,time.monotonic(),2.,'goal')
            if stamp<=self.goal_stamp:raise ValueError('duplicate_or_old_goal')
            goal=vector([msg.point.x,msg.point.y,msg.point.z],3,'goal')
            delta=goal-self.odom['position']
            if not .2<=np.linalg.norm(delta[:2])<=10 or abs(delta[2])>2:raise ValueError('goal_out_of_bounds')
            self.goals.set(goal,time.monotonic());self.nav.reset();self.goal_stamp=stamp
            self.last_goal_error='';self.phase='NAVIGATING';self.last_nav_step=0.;self.next_nav=0.
            self.event('goal_accepted',source=source,goal_stamp_ns=stamp,world_xyz=goal.tolist())
        except Exception as error:
            self.last_goal_error=str(error);self.event('goal_rejected',source=source,reason=str(error))

    def on_stop(self,request,response):
        self.nav.reset();self.goals.clear();self.reason='stop_service';self.stop_event.set()
        response.success=True;response.message='Stop latched; no automatic restart'
        return response

    def transition(self,action):
        if not self.owns_control:raise ValueError('transition_requires_motor_control_mode')
        if self.fault or self.stop_event.is_set():raise ValueError('stop_or_fault_latched')
        problem=self.input_error(full=False)
        if problem:raise ValueError(problem)
        now=time.monotonic()
        if action=='prepare':self.startup.prepare(self.lowstate['q'],now)
        elif action=='start_policy':self.startup.start_policy(self.lowstate,now)
        elif action=='damping':self.startup.passive()
        else:raise ValueError('unknown_transition')
        self.nav.reset();self.low.reset();self.goals.clear()
        self.a_goal.reset()  # A clicks cannot span prepare/policy/damping transitions.
        self.phase=self.startup.phase;self.next_low=0.;self.next_nav=0.
        self.target_diagnostic_text='';self.last_transition_error=''
        if action=='damping':
            self.snapshot=(now,motor_message(5,self.lowstate['q'],np.zeros(29),self.startup.passive_kd))
        self.event('transition',action=action,phase=self.phase)

    def on_transition(self,action,response):
        try:
            self.transition(action);response.success=True;response.message=self.phase
        except ValueError as error:
            self.last_transition_error=str(error);response.success=False;response.message=str(error)
            self.event('transition_rejected',action=action,reason=str(error))
        return response

    def wait_until(self,predicate,timeout):
        end=time.monotonic()+timeout
        while time.monotonic()<end and not self.stop_event.is_set():
            rclpy.spin_once(self,timeout_sec=.005)
            if self.stop_event.is_set():return False
            if predicate():return True
        return False

    def rpc(self,api_id):
        if self.rpc_pub is None:
            self.rpc_pub=self.create_publisher(Request,'/api/motion_switcher/request',10)
            self.create_subscription(Response,'/api/motion_switcher/response',lambda m:self.responses.__setitem__(int(m.header.identity.id),m),10)
            if not self.wait_until(lambda:self.rpc_pub.get_subscription_count()>0,3):raise RuntimeError('motion_switcher_not_found')
        ident=time.monotonic_ns();req=Request();req.header.identity.id=ident
        req.header.identity.api_id=api_id;req.header.policy.noreply=False;req.parameter='{}'
        self.rpc_pub.publish(req)
        if not self.wait_until(lambda:ident in self.responses,5):raise RuntimeError('motion_switcher_timeout')
        resp=self.responses.pop(ident)
        if int(resp.header.identity.api_id)!=api_id or resp.header.status.code!=0:raise RuntimeError('motion_switcher_error')
        return json.loads(resp.data or '{}')

    def enable_control(self):
        # This method is unreachable in default read-only mode.
        if not self.wait_until(lambda:not self.input_error(full=False),8):raise RuntimeError('preflight:'+self.input_error(full=False))
        domain=os.environ.get('ROS_DOMAIN_ID','0')
        self.lock_file=open('/tmp/transformer_sru_lowcmd_domain_'+domain+'.lock','a')
        fcntl.flock(self.lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Discovery settling + stable freshness before ownership changes.
        end=time.monotonic()+1.
        while time.monotonic()<end:
            rclpy.spin_once(self,timeout_sec=.01)
            if self.input_error(full=False):raise RuntimeError('preflight:'+self.input_error(full=False))
            if self.count_publishers('/lowcmd')!=0:raise RuntimeError('existing_lowcmd_publisher')
        mode=self.rpc(1001)
        if self.count_publishers('/lowcmd')!=0:raise RuntimeError('existing_lowcmd_publisher')
        self.lowcmd_pub=self.create_publisher(LowCmd,'/lowcmd',qos_profile_sensor_data)
        if not self.wait_until(lambda:self.lowcmd_pub.get_subscription_count()>0,3):raise RuntimeError('no_robot_lowcmd_subscriber')
        if self.count_publishers('/lowcmd')!=1:raise RuntimeError('lowcmd_ownership_conflict')
        if self.input_error(full=False):raise RuntimeError('pre_release:'+self.input_error(full=False))
        if self.stop_event.is_set() or self.fault:raise RuntimeError('stop_latched_before_release')
        self.damping=motor_message(5,np.zeros(29),np.zeros(29),np.full(29,self.cfg['stop_damping_sdk']))
        if mode.get('name'):
            self.rpc(1003)
            self.owns_control=True
            self.event('motion_service_release_requested',previous_name=mode.get('name'))
        if self.rpc(1001).get('name'):raise RuntimeError('motion_service_still_active')
        self.owns_control=True
        if self.input_error(full=False):raise RuntimeError('post_release:'+self.input_error(full=False))
        if self.stop_event.is_set() or self.fault:raise RuntimeError('stop_latched_after_release')
        self.ramp_start=time.monotonic();self.startup.passive();self.phase='PASSIVE'
        self.nav.reset();self.low.reset()
        self.damping=motor_message(5,np.zeros(29),np.zeros(29),np.full(29,self.cfg['stop_damping_sdk']))
        self.snapshot=(time.monotonic(),motor_message(5,self.lowstate['q'],np.zeros(29),self.startup.passive_kd))
        self.writer=threading.Thread(target=self.write_loop,name='unitree_target_writer',daemon=True)
        self.writer.start();self.event('control_started',initial_state=self.phase)

    def write_loop(self):
        next_tick=time.monotonic()
        try:
            while not self.stop_event.is_set():
                now=time.monotonic();snapshot=self.snapshot
                if now-self.low_at>self.cfg['lowstate_timeout_s']:raise RuntimeError('writer_lowstate_stale')
                if snapshot is None:
                    if now-self.ramp_start>self.cfg['command_watchdog_s']:raise RuntimeError('writer_no_initial_target')
                else:
                    at,command=snapshot
                    if now-at>self.cfg['command_watchdog_s']:raise RuntimeError('writer_target_stale')
                    if self.last_publish_at and now-self.last_publish_at>self.cfg['writer_deadline_s']:
                        raise RuntimeError('writer_deadline_missed')
                    self.lowcmd_pub.publish(command)
                    if self.last_publish_at:self.max_publish_gap=max(self.max_publish_gap,now-self.last_publish_at)
                    self.last_publish_at=now;self.publish_count+=1
                next_tick+=.002
                if next_tick<time.monotonic():next_tick=time.monotonic()
                self.stop_event.wait(max(0,next_tick-time.monotonic()))
        except Exception as error:
            self.writer_fault=str(error)
            if self.damping is not None and not self.suppress_stop_publish:
                try:self.lowcmd_pub.publish(self.damping)
                except Exception:pass
            self.stop_event.set()

    def compute(self,now):
        problem=self.input_error(full=False)
        if problem:
            self.reason=problem
            if self.owns_control or self.goals.goal is not None:self.fail(problem)
            return
        st=self.lowstate;gravity=rotation(st['quat']).T@np.array([0.,0.,-1.])
        full_error=self.input_error()
        if full_error and self.goals.goal is not None:self.fail(full_error);return
        if self.fault:return
        if not self.owns_control:
            self.phase=self.goals.state if self.goals.goal is not None else ('WAITING' if full_error else 'READY')
            self.reason=full_error
        else:self.reason=''
        position_command=None
        if self.owns_control:
            self.startup.observe(st,now)
            position_command=self.startup.position_command(st,now)
            self.phase=self.startup.phase if self.goals.goal is None else self.goals.state
        if position_command is not None:
            target,kp,kd=position_command
        else:
            if self.goals.goal is not None:
                state=self.goals.update(self.odom['position'],now);self.phase=state
                if state=='TIMEOUT':self.fail('goal_timeout');return
                if state!='NAVIGATING':self.nav.reset()
                elif now>=self.next_nav:
                    goal_body=rotation(self.odom['quat']).T@(self.goals.goal-self.odom['position'])
                    h_in=self.nav.h.copy();c_in=self.nav.c.copy()
                    self.nav.step(self.odom['linear'],st['gyro'],gravity,goal_body,self.lidar['xyz'])
                    self.navigation_trace.append(self.nav,h_in,c_in,{
                        'wall_time':time.time(),'goal_stamp_ns':self.goal_stamp,'nav_step':self.nav_count+1,
                        'goal_world':self.goals.goal.tolist(),'goal_body':goal_body.tolist(),
                        'odom_position':self.odom['position'].tolist(),'odom_quat_wxyz':self.odom['quat'].tolist(),
                        'base_quat_wxyz':st['quat'].tolist(),'lowstate_tick':st['tick'],
                        'odom_stamp_ns':self.odom['stamp'],'lidar_stamp_ns':self.lidar['stamp'],
                        'lidar_occupied_cells':self.lidar['occupied'],
                        'lidar_observed_cells':self.lidar['observed'],
                        'lidar_floor_filled_cells':self.lidar['floor_filled'],
                        'lidar_floor_height_m':self.lidar['floor_height'],
                        'lidar_extrinsics_verified':self.lidar['extrinsics_verified'],
                        'odom_receipt_age_s':now-self.odom_at,'lowstate_receipt_age_s':now-self.low_at,
                        'lidar_receipt_age_s':now-self.lidar_at})
                    self.last_nav_step=now;self.nav_count+=1
                    self.next_nav=self.next_nav+.2 if 0<=now-self.next_nav<.2 else now+.2
            else:self.nav.reset()
            target=self.low.step(st['q'],st['dq'],st['gyro'],gravity,self.nav.command);self.low_count+=1
            kp=np.array(CONTRACT['kp_sdk']);kd=np.array(CONTRACT['kd_sdk'])
        self.preview_target=target;self.target_diagnostic_text=target_diagnostic(target,st['q'],st['dq'],kp,kd)
        if self.owns_control:
            msg=motor_message(st['mode_machine'],target,kp,kd)
            self.snapshot=(time.monotonic(),msg)
        self.event('step',phase=self.phase,raw_action=self.nav.previous.tolist(),command=self.nav.command.tolist(),
                   q_target=target.tolist(),low_action_scale=self.cfg['low_action_scale'],target_diagnostic=self.target_diagnostic_text,lowstate_tick=st['tick'])

    def status(self):
        now=self.get_clock().now().nanoseconds
        steady=time.monotonic()
        status={'mode':'control' if self.args.enable_lowcmd else 'readonly','phase':self.phase,
                'reason':self.writer_fault or self.fault or self.reason,
                'input_error':self.input_error(full=not self.owns_control or self.goals.goal is not None),
                'navigation_input_error':self.input_error(),
                'startup_phase':self.startup.phase if self.owns_control else None,
                'fixstand_max_joint_error_rad':self.startup.q_error if self.owns_control else None,
                'last_transition_error':self.last_transition_error,
                'goal_stamp_ns':self.goal_stamp,'goal_world':None if self.goals.goal is None else self.goals.goal.tolist(),
                'last_goal_error':self.last_goal_error,'command':self.nav.command.tolist(),'nav_steps':self.nav_count,
                'low_steps':self.low_count,'low_action_scale':self.cfg['low_action_scale'],'lowcmd_count':self.publish_count,'target_diagnostic':self.target_diagnostic_text,
                'odom_age_s':None if self.odom is None else steady-self.odom_at,
                'odom_time_basis':'monotonic_receipt_of_advancing_stamp',
                'odom_header_clock_difference_s':None if self.odom is None else (now-self.odom['stamp'])/1e9,
                'lidar_extrinsics_verified':None if self.lidar is None else self.lidar['extrinsics_verified'],
                'lidar_observed_cells':None if self.lidar is None else self.lidar['observed'],
                'lidar_floor_filled_cells':None if self.lidar is None else self.lidar['floor_filled'],
                'lidar_occupied_cells':None if self.lidar is None else self.lidar['occupied'],
                'lidar_floor_height_m':None if self.lidar is None else self.lidar['floor_height'],
                'lidar_age_s':None if self.lidar is None else (now-self.lidar['stamp'])/1e9,
                'max_publish_gap_ms':self.max_publish_gap*1000,'max_compute_ms':self.max_compute_s*1000}
        msg=String();msg.data=json.dumps(status);self.status_pub.publish(msg)
        preview=TwistStamped();preview.header.stamp=self.get_clock().now().to_msg();preview.header.frame_id=self.cfg['odom_child_frame']
        preview.twist.linear.x=float(self.nav.command[0]);preview.twist.linear.y=float(self.nav.command[1]);preview.twist.angular.z=float(self.nav.command[2])
        self.preview_pub.publish(preview);self.last_status=status;self.event('status',**status)

    def run(self):
        started=time.monotonic();last_status=0.;last_owner_check=0.
        if self.args.enable_lowcmd:self.enable_control()
        while rclpy.ok() and not self.stop_event.is_set():
            rclpy.spin_once(self,timeout_sec=.001)
            if self.stop_event.is_set():break
            now=time.monotonic();ros=self.get_clock().now().nanoseconds
            if ros<self.last_ros_ns:self.fail('ROS_clock_moved_backwards')
            self.last_ros_ns=ros
            if now>=self.next_low:
                begin=time.monotonic();self.compute(now);self.last_low_step=now
                self.next_low=self.next_low+.02 if 0<=now-self.next_low<.02 else now+.02
                self.max_compute_s=max(self.max_compute_s,time.monotonic()-begin)
            if now-last_owner_check>=.2 and self.owns_control:
                last_owner_check=now
                if self.count_publishers('/lowcmd')!=1:
                    self.suppress_stop_publish=True;self.fail('another_lowcmd_publisher_appeared')
                if self.goals.goal is not None and self.count_publishers(self.cfg['lidar_topic'])!=1:self.fail('lidar_publisher_count_not_one')
            if now-last_status>=.5:self.status();last_status=now
            if self.args.duration and now-started>=self.args.duration:
                self.reason='bounded_duration_finished';self.stop_event.set()
        if self.writer_fault:self.fail(self.writer_fault)

    def close(self):
        self.stop_event.set();self.nav.reset();self.goals.clear()
        if self.writer:self.writer.join(timeout=1.)
        if self.owns_control and self.lowcmd_pub is not None and not self.suppress_stop_publish:
            if self.damping is None:self.damping=motor_message(5,np.zeros(29),np.zeros(29),np.full(29,self.cfg['stop_damping_sdk']))
            for _ in range(10):
                self.lowcmd_pub.publish(self.damping);time.sleep(.02)
            self.event('damping_stop_sent',count=10)
        self.phase='STOPPED';self.status()
        try:
            trace_path=self.navigation_trace.save(self.log_path)
            if trace_path:self.event('navigation_trace_saved',path=str(trace_path),frames=len(self.navigation_trace.rows))
        except Exception as error:
            self.event('navigation_trace_save_error',reason=str(error))
        self.event('closed',motion_service_released=self.owns_control,lowcmd_messages_published=self.publish_count,
                   fault=self.writer_fault or self.fault)
        self.log_file.close()
        if self.lock_file:self.lock_file.close()
        self.destroy_node()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--enable-lowcmd',action='store_true')
    p.add_argument('--duration',type=float,default=0.,help='Optional bounded run in seconds')
    p.add_argument('--log',type=Path)
    args=p.parse_args()
    if not np.isfinite(args.duration) or args.duration<0:p.error('duration must be finite and >=0')
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=None;exit_code=0
    try:
        node=Controller(args)
        def stop(signum,frame):node.reason='signal_stop';node.stop_event.set()
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        node.run()
        if node.fault or node.writer_fault:exit_code=1
    except Exception as error:
        exit_code=1
        if node:node.fail(str(error))
        else:print('startup_failed:'+str(error),flush=True)
    finally:
        if node:node.close()
        if rclpy.ok():rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__=='__main__':main()
