#!/usr/bin/env python3
"""End-to-end mocked hardware tests. Forces loopback-only DDS domain 221."""
import os
os.environ['ROS_DOMAIN_ID']='221'
# Cyclone/Humble adds lo automatically with LOCALHOST_ONLY=1, conflicting with
# the explicit interface. Select lo exactly once; no physical NIC is enabled.
os.environ['ROS_LOCALHOST_ONLY']='0'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" /></Interfaces></General></Domain></CycloneDDS>'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger
from rclpy.parameter import Parameter
from unitree_api.msg import Request,Response
from unitree_hg.msg import LowState,LowCmd
from transformer_sru_lidar_adapter.msg import LidarTensor
from core import ROOT,CONTRACT
from transport import lowcmd_crc

HERE=Path(__file__).resolve().parent


class Fixture(Node):
    def __init__(self):
        super().__init__('transformer_sru_lidar_adapter')
        self.declare_parameter('extrinsics_verified',True)
        self.declare_parameter('tensor_output_topic','/transformer_sru/lidar')
        self.lowpub=self.create_publisher(LowState,'/lowstate',qos_profile_sensor_data)
        self.odpub=self.create_publisher(Odometry,'/dog_odom',qos_profile_sensor_data)
        self.lipub=self.create_publisher(LidarTensor,'/transformer_sru/lidar',qos_profile_sensor_data)
        self.goalpub=self.create_publisher(PointStamped,'/transformer_sru/goal',10)
        self.rpcpub=self.create_publisher(Response,'/api/motion_switcher/response',10)
        self.create_subscription(Request,'/api/motion_switcher/request',self.rpc,10)
        self.create_subscription(String,'/transformer_sru/status',self.on_status,10)
        self.create_subscription(LowCmd,'/lowcmd',self.on_cmd,qos_profile_sensor_data)
        self.mode='normal';self.releases=0;self.calls=[];self.status={};self.statuses=[]
        self.commands=[];self.bad_crc=0;self.position=np.array([0.,0.,.8]);self.lidar_enabled=True
        self.tick=0;self.odom_old=True;self.low_enabled=True;self.last_slow=0.
        self.odom_frozen=False;self.frozen_stamp=None;self.lidar_verified=False;self.buttons=0
        self.yaw=0.
        self.q=np.empty(29,np.float32);self.q[np.array(CONTRACT['policy_to_sdk'])]=CONTRACT['default_q_policy']
        elevation,az=np.meshgrid(np.deg2rad(np.linspace(-45,0,16)),np.deg2rad(np.linspace(-60,60,41)),indexing='ij')
        points=np.stack([np.cos(elevation)*np.cos(az),np.cos(elevation)*np.sin(az),np.sin(elevation)],-1)*.5
        self.xyz=points.astype(np.float32).ravel().tolist()
        self.create_timer(.005,self.publish_inputs)

    def rpc(self,msg):
        self.calls.append(int(msg.header.identity.api_id))
        api=int(msg.header.identity.api_id);r=Response();r.header.identity=msg.header.identity;r.header.status.code=0
        if api==1003:self.mode='';self.releases+=1
        if api not in (1001,1003):r.header.status.code=1
        r.data=json.dumps({'name':self.mode,'form':'0'}) if api==1001 else '{}';self.rpcpub.publish(r)

    def on_status(self,msg):
        self.status=json.loads(msg.data);self.statuses.append(self.status)

    def on_cmd(self,msg):
        if msg.crc!=lowcmd_crc(msg):self.bad_crc+=1
        self.commands.append((time.monotonic(),float(msg.motor_cmd[0].kp),float(msg.motor_cmd[0].kd)))
        # Perfect target follower for protocol testing only; NOT a physics model.
        if msg.motor_cmd[0].kp>0:self.q=np.array([m.q for m in msg.motor_cmd[:29]],np.float32)

    def publish_inputs(self):
        self.tick+=5
        if self.low_enabled:
            m=LowState();m.mode_machine=5;m.mode_pr=0;m.tick=self.tick;m.imu_state.quaternion=[1.,0.,0.,0.]
            m.wireless_remote[2]=self.buttons & 255;m.wireless_remote[3]=(self.buttons>>8) & 255
            for i in range(29):m.motor_state[i].q=float(self.q[i]);m.motor_state[i].dq=0.
            self.lowpub.publish(m)
        now=self.get_clock().now();steady=time.monotonic()
        o=Odometry();o.header.stamp=now.to_msg();o.header.frame_id='odom';o.child_frame_id='robot_center'
        if self.odom_old:o.header.stamp.sec-=2300
        if self.odom_frozen:
            if self.frozen_stamp is None:self.frozen_stamp=o.header.stamp
            o.header.stamp=self.frozen_stamp
        else:self.frozen_stamp=None
        o.pose.pose.orientation.w=float(np.cos(self.yaw/2));o.pose.pose.orientation.z=float(np.sin(self.yaw/2))
        o.pose.pose.position.x,o.pose.pose.position.y,o.pose.pose.position.z=map(float,self.position)
        self.odpub.publish(o)
        if steady-self.last_slow>=.1:
            self.last_slow=steady
            if self.lidar_enabled:
                lidar=LidarTensor();ns=now.nanoseconds-100_000_000
                lidar.header.stamp.sec=ns//10**9;lidar.header.stamp.nanosec=ns%10**9
                lidar.header.frame_id='transformer_sru_virtual_lidar_yaw';lidar.rows=16;lidar.columns=41
                lidar.normalization_distance_m=5.;lidar.occupied_cells=656;lidar.input_points=656;lidar.xyz=self.xyz
                lidar.extrinsics_verified=self.lidar_verified
                lidar.imu_received_stamp=lidar.header.stamp;lidar.imu_cloud_offset_s=0.;self.lipub.publish(lidar)


def main():
    print('ISOLATED: domain=221, sole interface=lo; all robot endpoints below are mocks',flush=True)
    output=ROOT/'experiments/real_deploy_validation'/time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    rclpy.init();node=Fixture();processes=[];files=[];checks=[]
    def wait(predicate,timeout=6):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.001)
            if predicate():return True
        return False
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        print(name,condition,flush=True)
        if not condition:raise AssertionError(name+': '+str(node.status))
    def start(name,control=False):
        node.status={};node.statuses=[]
        f=(output/(name+'.stdout')).open('w');files.append(f)
        args=[sys.executable,str(HERE/'controller.py'),'--duration','60','--log',str(output/(name+'.jsonl'))]
        if control:args.append('--enable-lowcmd')
        proc=subprocess.Popen(args,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT);processes.append(proc)
        return proc
    def stage(action, expected=True):
        client=node.create_client(Trigger,'/transformer_sru/'+action)
        check(action+'_service_ready',wait(client.service_is_ready))
        future=client.call_async(Trigger.Request())
        check(action+'_response',wait(future.done) and future.result().success is expected)
        node.destroy_client(client)
    def goal():
        f=(output/('goal_'+str(len(files))+'.stdout')).open('w');files.append(f)
        proc=subprocess.Popen([sys.executable,str(HERE/'goal.py'),'--forward','5'],env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
        processes.append(proc);check('goal_tool_exits_success',wait(lambda:proc.poll() is not None,10) and proc.returncode==0)
    def double_a():
        node.buttons=0;wait(lambda:False,.08)
        node.buttons=256;wait(lambda:False,.08)
        node.buttons=0;wait(lambda:False,.08)
        node.buttons=256;wait(lambda:False,.08)
        node.buttons=0
    try:
        check('isolated_domain_empty',wait(lambda:node.count_publishers('/lowcmd')==0,2))
        ro=start('readonly')
        check('readonly_READY',wait(lambda:node.status.get('phase')=='READY'))
        check('old_header_clock_does_not_block_fresh_odom',node.status['odom_age_s']<.3 and node.status['odom_header_clock_difference_s']>2200)
        check('unverified_lidar_is_explicit_in_readonly',node.status['lidar_extrinsics_verified'] is False)
        check('requested_scale_loaded',node.status.get('low_action_scale')==.25)
        check('readonly_no_lowcmd_no_rpc',not node.commands and not node.calls and node.count_publishers('/lowcmd')==0)
        double_a()
        check('readonly_remote_A_rejected',wait(lambda:node.status.get('last_goal_error')=='remote_goal_requires_active_policy'))
        goal();check('readonly_navigation_infers',wait(lambda:node.status.get('nav_steps',0)>=3))
        saved_stamp=node.status['goal_stamp_ns']
        bad=PointStamped();bad.header.frame_id='wrong_frame';bad.header.stamp=node.get_clock().now().to_msg();bad.point.x=2.
        node.goalpub.publish(bad)
        check('bad_goal_frame_rejected',wait(lambda:node.status.get('last_goal_error')=='goal_frame_mismatch'))
        check('bad_goal_does_not_replace_active_target',node.status['goal_stamp_ns']==saved_stamp)
        first_goal=list(node.status['goal_world'])
        for _ in range(10):node.position[0]+=.1;wait(lambda:False,.05)
        check('goal_stays_fixed',wait(lambda:node.status.get('goal_world')==first_goal))
        while np.linalg.norm(node.position[:2]-np.array(first_goal[:2]))>.15:
            delta=np.array(first_goal[:2])-node.position[:2];node.position[:2]+=.1*delta/np.linalg.norm(delta);wait(lambda:False,.05)
        check('arrival_latches',wait(lambda:node.status.get('phase')=='ARRIVED',4))
        check('arrival_zero_command',node.status['command']==[0.,0.,0.])
        goal();node.lidar_enabled=False
        check('stale_lidar_fault',wait(lambda:node.status.get('phase')=='FAULT'))
        check('fault_zero_command',node.status['command']==[0.,0.,0.])
        node.lidar_enabled=True
        wait(lambda:False,.5)
        check('fault_not_auto_rearmed',node.status.get('phase')=='FAULT')
        stop_client=node.create_client(Trigger,'/transformer_sru/stop')
        check('stop_service_discovered',wait(lambda:stop_client.service_is_ready()))
        stopped=stop_client.call_async(Trigger.Request())
        check('stop_service_acknowledged',wait(stopped.done) and stopped.result().success)
        check('readonly_shutdown',wait(lambda:ro.poll() is not None));node.destroy_client(stop_client)
        trace_path=output/'readonly.navigation.npz'
        check('navigation_full_trace_saved',trace_path.exists())
        from core import Navigation
        nav_replay=Navigation()
        with np.load(trace_path) as trace:
            check('navigation_trace_has_obs_h_c',trace['obs'].shape[1:]==(1984,) and trace['h_in'].shape[1:]==(1,1,512))
            maximum=0.
            for i in range(len(trace['obs'])):
                raw=nav_replay.session.run(None,{'obs':trace['obs'][i:i+1],
                    'h_in':trace['h_in'][i],'c_in':trace['c_in'][i]})[0][0]
                maximum=max(maximum,float(np.max(abs(raw-trace['raw_action'][i]))))
            check('navigation_trace_exact_output_replay',maximum<1e-6)
        check('readonly_never_published_motors',not node.commands and not node.calls)
        wait(lambda:False,.5)
        node.position[:]=[0,0,.8]
        node.yaw=np.pi/2
        competitor=node.create_publisher(LowCmd,'/lowcmd',qos_profile_sensor_data)
        occupied=start('occupied_lowcmd_control',True)
        check('existing_lowcmd_blocks_control',wait(lambda:occupied.poll() is not None,6) and occupied.returncode!=0)
        check('occupied_does_not_release_or_publish',not node.calls and not node.commands)
        node.destroy_publisher(competitor)
        check('competitor_removed',wait(lambda:node.count_publishers('/lowcmd')==0))
        node.low_enabled=False
        missing=start('missing_lowstate_control',True)
        check('missing_lowstate_blocks_control',wait(lambda:missing.poll() is not None,12) and missing.returncode!=0)
        check('missing_lowstate_no_release_or_motors',not node.calls and not node.commands)
        node.low_enabled=True;node.odom_frozen=True;node.lidar_enabled=False
        node.set_parameters([Parameter('extrinsics_verified',value=False)])
        control=start('control_staged',True)
        check('control_enters_PASSIVE_without_navigation_inputs',wait(lambda:node.status.get('phase')=='PASSIVE',6))
        wait(lambda:False,.7)
        check('no_automatic_prepare_or_policy',node.status['phase']=='PASSIVE' and node.status['low_steps']==0 and all(kp==0 for _,kp,_ in node.commands))
        check('mock_motion_service_released',node.releases==1)
        double_a()
        check('passive_remote_A_rejected',wait(lambda:node.status.get('last_goal_error')=='remote_goal_requires_active_policy') and node.status['goal_world'] is None)
        stage('start_policy',False)
        # Exercise the SDK L2+Up edge; held buttons must not restart preparation.
        node.buttons=(1<<5)|(1<<12)
        check('remote_prepare_enters_FIX_STAND',wait(lambda:node.status.get('phase')=='FIX_STAND'))
        wait(lambda:False,.3);node.buttons=0
        check('no_policy_during_prepare',node.status['low_steps']==0 and node.status['nav_steps']==0)
        check('interpolation_completed',wait(lambda:node.status.get('phase')=='FIX_STAND_READY',5))
        wait(lambda:False,.6)
        check('fixstand_holds_until_explicit_start',node.status['phase']=='FIX_STAND_READY' and node.status['low_steps']==0)
        from startup import STARTUP
        np.testing.assert_allclose(node.q,STARTUP['fixstand_q_sdk'],atol=1e-6)
        stage('start_policy')
        check('zero_policy_ready_without_lidar',wait(lambda:node.status.get('phase')=='ZERO_VELOCITY',5))
        check('zero_command_before_goal',node.status['command']==[0.,0.,0.] and node.status['nav_steps']==0)
        node.odom_frozen=False;node.lidar_enabled=True;node.lidar_verified=False
        wait(lambda:False,.6)
        def direct_goal():
            msg=PointStamped();msg.header.frame_id='odom';msg.header.stamp=node.get_clock().now().to_msg()
            msg.point.x=float(node.position[0]+5);msg.point.z=.5;node.goalpub.publish(msg)
        direct_goal()
        check('unverified_lidar_blocks_navigation_not_standing',wait(lambda:'lidar_extrinsics_unverified' in node.status.get('last_goal_error','')) and node.status['phase']=='ZERO_VELOCITY')
        check('blocked_goal_did_not_run_navigation',node.status['nav_steps']==0)
        double_a()
        check('remote_A_retains_calibration_check',wait(lambda:'lidar_extrinsics_unverified' in node.status.get('last_goal_error','')) and node.status['goal_world'] is None)
        node.lidar_verified=True
        wait(lambda:False,.5)
        double_a()
        check('remote_A_starts_navigation',wait(lambda:node.status.get('phase')=='NAVIGATING' and node.status.get('nav_steps',0)>=2))
        expected=np.array([node.position[0],node.position[1]+7.,.5])
        check('remote_A_seven_metres_along_yaw',np.allclose(node.status['goal_world'],expected,atol=1e-5))
        saved_stamp=node.status['goal_stamp_ns']
        node.buttons=256;wait(lambda:False,.7)
        check('held_A_does_not_replace_goal',node.status['goal_stamp_ns']==saved_stamp)
        node.buttons=0
        node.position[0]+=.1;wait(lambda:False,.6)
        check('remote_A_goal_fixed_after_movement',np.allclose(node.status['goal_world'],expected,atol=1e-5))
        double_a()
        check('second_A_pair_replaces_goal',wait(lambda:node.status.get('goal_stamp_ns',0)>saved_stamp))
        check('second_A_pair_uses_current_position',np.allclose(node.status['goal_world'],expected+[.1,0,0],atol=1e-5))
        goal();check('mock_navigation_to_unitree',wait(lambda:node.status.get('nav_steps',0)>=2,4))
        stage('damping')
        check('damping_clears_goal_and_velocity',wait(lambda:node.status.get('phase')=='PASSIVE' and node.status.get('goal_world') is None and node.status['command']==[0.,0.,0.]))
        wait(lambda:False,.2);before=node.status['low_steps'];wait(lambda:False,.6)
        check('damping_does_not_advance_low_policy',node.status['low_steps']==before)
        check('damping_motor_stream',all(kp==0 and kd==3 for _,kp,kd in node.commands[-50:]))
        direct_goal()
        check('goal_rejected_in_PASSIVE',wait(lambda:node.status.get('last_goal_error')=='controller_not_ready:PASSIVE'))
        stage('start_policy',False);stage('prepare')
        check('second_prepare_ready',wait(lambda:node.status.get('phase')=='FIX_STAND_READY',5))
        # Exercise R1+X edge for the second policy entry.
        node.buttons=1|(1<<10)
        check('remote_policy_start',wait(lambda:node.status.get('phase')=='ZERO_VELOCITY'))
        node.buttons=0
        check('second_zero_ready',wait(lambda:node.status.get('phase')=='ZERO_VELOCITY',4))
        check('old_goal_not_resumed',node.status['goal_world'] is None and node.status['command']==[0.,0.,0.])
        goal();before=node.status['nav_steps']
        check('navigation_after_reentry',wait(lambda:node.status.get('nav_steps',0)>before))
        node.lidar_verified=False
        check('calibration_loss_during_navigation_stops_control',wait(lambda:control.poll() is not None,4) and control.returncode!=0)
        check('calibration_fault_damping_sent',any(kp==0 and kd==3 for _,kp,kd in node.commands[-20:]))
        check('motor_publisher_removed_after_calibration_fault',wait(lambda:node.count_publishers('/lowcmd')==0))
        # Repeated stamped odometry cannot stay alive merely by arriving often.
        node.lidar_verified=True
        from core import Unitree
        node.q=Unitree().default_sdk.copy()
        control2=start('control_frozen_odom',True)
        check('second_control_PASSIVE',wait(lambda:node.status.get('phase')=='PASSIVE',6))
        stage('prepare');check('third_prepare_ready',wait(lambda:node.status.get('phase')=='FIX_STAND_READY',5))
        stage('start_policy');check('third_zero_ready',wait(lambda:node.status.get('phase')=='ZERO_VELOCITY',4))
        goal();check('third_navigation',wait(lambda:node.status.get('nav_steps',0)>=2))
        node.odom_frozen=True
        check('repeated_odom_stops_active_navigation',wait(lambda:control2.poll() is not None,4) and control2.returncode!=0)
        check('frozen_odom_publisher_removed',wait(lambda:node.count_publishers('/lowcmd')==0))
        node.odom_frozen=False
        # LowState watchdog must also protect Passive, with no navigation running.
        control3=start('passive_lowstate_loss',True)
        check('third_control_PASSIVE',wait(lambda:node.status.get('phase')=='PASSIVE',6))
        node.low_enabled=False
        check('passive_exits_on_lowstate_loss',wait(lambda:control3.poll() is not None,4) and control3.returncode!=0)
        check('damping_sent_on_fault',any(kp==0 and kd==3 for _,kp,kd in node.commands[-20:]))
        check('no_motor_publisher_after_exit',wait(lambda:node.count_publishers('/lowcmd')==0,3))
        node.low_enabled=True;node.buttons=0
        node.q=Unitree().default_sdk.copy()
        control4=start('select_stop_control',True)
        check('select_control_PASSIVE',wait(lambda:node.status.get('phase')=='PASSIVE',6))
        stage('prepare')
        check('select_prepare_ready',wait(lambda:node.status.get('phase')=='FIX_STAND_READY',5))
        stage('start_policy')
        check('select_zero_ready',wait(lambda:node.status.get('phase')=='ZERO_VELOCITY',4))
        goal();check('select_navigation_active',wait(lambda:node.status.get('nav_steps',0)>=2))
        node.buttons=8;wait(lambda:False,.65)
        check('held_select_does_not_exit',control4.poll() is None)
        node.buttons=0;wait(lambda:False,.1)
        node.buttons=8|256;wait(lambda:False,.1)
        check('first_select_click_does_not_exit',control4.poll() is None)
        node.buttons=0;wait(lambda:False,.1)
        node.buttons=8|256
        check('select_double_click_exits',wait(lambda:control4.poll() is not None,4) and control4.returncode==0)
        node.buttons=0
        check('select_stop_reason',node.status.get('reason')=='remote_select_double_click_stop')
        check('select_stop_clears_goal_and_command',node.status.get('goal_world') is None and node.status['command']==[0.,0.,0.])
        events=[json.loads(line) for line in (output/'select_stop_control.jsonl').read_text().splitlines()]
        check('select_has_priority_over_simultaneous_A',not any(e['event']=='goal_accepted' and e.get('source')=='remote_A_double_click' for e in events))
        check('select_stop_damping_sent',all(kp==0 and kd==3 for _,kp,kd in node.commands[-10:]))
        check('select_stop_removes_publisher',wait(lambda:node.count_publishers('/lowcmd')==0))
        check('all_mock_motor_crcs_match',node.bad_crc==0)
        (output/'result.json').write_text(json.dumps({'passed':True,'domain':221,'interface':'lo','checks':checks,
                    'mock_motor_messages':len(node.commands),'hardware_motor_messages':0},indent=2))
        print('RESULT',output,flush=True)
    finally:
        for proc in processes:
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
        for proc in processes:
            try:proc.wait(timeout=4)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for f in files:f.close()
        (output/'checks.json').write_text(json.dumps(checks,indent=2))
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
