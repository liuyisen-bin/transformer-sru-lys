#!/usr/bin/env python3
"""Floor completion through the ROS node; synthetic sensor data on loopback only."""
import os
os.environ['ROS_DOMAIN_ID']='215'
os.environ['ROS_LOCALHOST_ONLY']='1'
os.environ['RMW_IMPLEMENTATION']='rmw_fastrtps_cpp'
os.environ.pop('CYCLONEDDS_URI',None)
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,PointField
from unitree_hg.msg import IMUState
from transformer_sru_lidar_adapter.msg import LidarTensor

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'deploy/transformer_sru_real'))
from lidar_projection import LidarProjection
from core import rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    projection=LidarProjection();rclpy.init()
    node=rclpy.create_node('floor_completion_test')
    ip=node.create_publisher(IMUState,'/floor_test/imu',qos_profile_sensor_data)
    cp=node.create_publisher(PointCloud2,'/floor_test/cloud',qos_profile_sensor_data)
    main_frames=[];observed_frames=[]
    node.create_subscription(LidarTensor,'/floor_test/main',main_frames.append,qos_profile_sensor_data)
    node.create_subscription(LidarTensor,'/floor_test/observed',observed_frames.append,qos_profile_sensor_data)
    def spin(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.005)
    def cloud(points):
        msg=PointCloud2();msg.header.stamp=node.get_clock().now().to_msg();msg.header.frame_id='livox_frame'
        msg.width=len(points);msg.height=1;msg.point_step=12;msg.row_step=12*len(points)
        msg.fields=[PointField(name=name,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,name in enumerate('xyz')]
        msg.data=np.asarray(points,dtype='<f4').tobytes();return msg
    params={'input_topic':'/floor_test/cloud','imu_topic':'/floor_test/imu',
            'tensor_output_topic':'/floor_test/main','observed_tensor_topic':'/floor_test/observed',
            'cloud_output_topic':'/floor_test/organized','status_topic':'/floor_test/status',
            'flat_floor_completion_enabled':'true','extrinsics_verified':'false',
            'lidar_to_torso_rpy':'[0.0, 0.04014257279586953, 0.0]'}
    command=[str(args.node.resolve()),'--ros-args']
    for k,v in params.items():command+=['-p',k+':='+v]
    cases=[]
    with (args.output/'node.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            end=time.monotonic()+10
            while time.monotonic()<end and (not ip.get_subscription_count() or not cp.get_subscription_count()):spin(.05)
            assert ip.get_subscription_count() and cp.get_subscription_count(),'DDS discovery timeout'
            spin(.3)
            for height,q in [(1.2,[1.,0.,0.,0.]),(1.45,[.9949874371,.08,-.06,0.])]:
                q=np.array(q);q/=np.linalg.norm(q)
                x,y=np.meshgrid(np.linspace(-2.4,2.4,41),np.linspace(-2.4,2.4,41))
                ground=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,-height)))
                raw=ground@rotation(q)@projection.mount
                obstacle=projection.rays[20]*.7
                raw=np.vstack((raw,obstacle,np.zeros((50,3)))).astype(np.float32)
                imu=IMUState();imu.quaternion=q.astype(np.float32).tolist();ip.publish(imu);spin(.02)
                count=len(main_frames);msg=cloud(raw);cp.publish(msg);spin(.12)
                assert len(main_frames)==count+1 and len(observed_frames)==len(main_frames)
                main,observed=main_frames[-1],observed_frames[-1]
                assert main.header.stamp==msg.header.stamp==observed.header.stamp
                assert not main.extrinsics_verified and not observed.extrinsics_verified
                assert abs(main.floor_height_m-height)<1e-6 and main.floor_fit_inlier_points>1000
                a=np.array(main.xyz,np.float32);b=np.array(observed.xyz,np.float32)
                expected,filled=projection.complete_floor(b,main.torso_quaternion_wxyz,main.floor_height_m)
                np.testing.assert_array_equal(a,expected)
                mask=np.any(b.reshape(656,3)!=0,axis=1)
                np.testing.assert_array_equal(a.reshape(656,3)[mask],b.reshape(656,3)[mask])
                assert mask[20] and np.linalg.norm(a.reshape(656,3)[20])<.15
                assert main.floor_filled_cells==filled>0 and observed.floor_filled_cells==0
                assert main.occupied_cells==observed.occupied_cells+filled
                assert main.observed_cells==observed.observed_cells==observed.occupied_cells
                cases.append(dict(case='changing_height_and_pose',height_m=height,observed=main.observed_cells,filled=filled,passed=True))
            # An isolated real obstacle is insufficient evidence to invent a floor.
            ip.publish(imu);spin(.02);cp.publish(cloud([obstacle]));spin(.12)
            assert main_frames[-1].floor_fit_inlier_points==0 and main_frames[-1].floor_filled_cells==0
            np.testing.assert_array_equal(main_frames[-1].xyz,observed_frames[-1].xyz)
            cases.append(dict(case='unavailable_floor_keeps_observed_only',passed=True))
            # Missing returns must not become a synthetic full floor frame.
            ip.publish(imu);spin(.02);count=len(main_frames);cp.publish(cloud(np.zeros((500,3))));spin(.12)
            assert len(main_frames)==count
            cases.append(dict(case='all_zero_cloud_remains_rejected',passed=True))
        finally:
            proc.send_signal(signal.SIGINT);proc.wait(timeout=5)
            node.destroy_node();rclpy.shutdown()
    report=dict(passed=True,cases=cases,robot_domain_used=False)
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
