#!/usr/bin/env python3
"""Publish a single odom-anchored relative goal and wait for its acknowledgement."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import yaml
from core import relative_goal, check_receipt_age

def stamp_ns(stamp):return int(stamp.sec)*10**9+int(stamp.nanosec)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward',type=float,default=5.)
    parser.add_argument('--left',type=float,default=0.)
    parser.add_argument('--height',type=float,default=.5,help='Fixed goal height in odom frame, metres; training used 0.5')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=parser.parse_args()
    if not np.isfinite([args.forward,args.left,args.height]).all() or not .2<=np.hypot(args.forward,args.left)<=10:
        parser.error('goal distance must be 0.2..10m and all values finite')
    cfg=yaml.safe_load(args.config.read_text());rclpy.init();node=rclpy.create_node('transformer_sru_goal_once')
    latest={};publisher=node.create_publisher(PointStamped,'/transformer_sru/goal',10)
    def odom(msg):
        stamp=stamp_ns(msg.header.stamp)
        if stamp<=0 or ('odom' in latest and stamp<=stamp_ns(latest['odom'].header.stamp)):return
        latest.update(odom=msg,odom_at=time.monotonic())
    def status(msg):latest.update(status=json.loads(msg.data),status_at=time.monotonic())
    node.create_subscription(Odometry,cfg['odom_topic'],odom,qos_profile_sensor_data)
    node.create_subscription(String,'/transformer_sru/status',status,10)
    try:
        end=time.monotonic()+8
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.05)
            if 'odom' in latest and 'status' in latest and publisher.get_subscription_count()==1:break
        else:raise RuntimeError('need odometry, controller status, and exactly one goal subscriber')
        msg=latest['odom'];st=latest['status'];now=time.monotonic()
        check_receipt_age(latest['odom_at'],now,cfg['odom_timeout_s'],'odom')
        if now-latest['status_at']>1 or st['phase'] not in ('READY','ZERO_VELOCITY','NAVIGATING','SETTLING','ARRIVED'):
            raise RuntimeError('controller_not_ready:'+str(st))
        if st['mode']=='control' and st.get('navigation_input_error'):
            raise RuntimeError('navigation_inputs_not_ready:'+st['navigation_input_error'])
        if msg.header.frame_id!=cfg['odom_frame'] or msg.child_frame_id!=cfg['odom_child_frame']:
            raise RuntimeError('odometry_frame_mismatch')
        position=msg.pose.pose.position;q=msg.pose.pose.orientation
        world=relative_goal([position.x,position.y,position.z],[q.w,q.x,q.y,q.z],args.forward,args.left,args.height)
        goal=PointStamped();goal.header.frame_id=cfg['odom_frame'];goal.header.stamp=node.get_clock().now().to_msg()
        goal.point.x,goal.point.y,goal.point.z=map(float,world)
        stamp=stamp_ns(goal.header.stamp);publisher.publish(goal)
        end=time.monotonic()+3
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.05)
            if latest['status'].get('goal_stamp_ns')==stamp:
                print(json.dumps({'accepted':True,'controller_mode':latest['status']['mode'],
                                  'fixed_goal_odom_xyz_m':world.tolist(),'goal_stamp_ns':stamp}),flush=True)
                return
        raise RuntimeError('goal_not_acknowledged:'+str(latest.get('status')))
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
