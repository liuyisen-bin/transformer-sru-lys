#!/usr/bin/env python3
"""Isolated synthetic ROS test. Publishes ONLY synthetic cloud/IMU topics on loopback."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

# Set before importing ROS. Never join a robot DDS domain for this test.
os.environ['ROS_DOMAIN_ID'] = '214'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
os.environ.pop('CYCLONEDDS_URI', None)
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from diagnostic_msgs.msg import DiagnosticArray
from unitree_hg.msg import IMUState
from transformer_sru_lidar_adapter.msg import LidarTensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or Path(tempfile.mkdtemp(prefix='transformer_livox_smoke_'))
    out.mkdir(parents=True, exist_ok=True)
    results = []
    rclpy.init()
    node = rclpy.create_node('synthetic_lidar_contract_test')
    imu_pub = node.create_publisher(IMUState, '/adapter_test/imu', qos_profile_sensor_data)
    cloud_pub = node.create_publisher(PointCloud2, '/adapter_test/cloud', qos_profile_sensor_data)
    tensors, clouds, statuses = [], [], []
    subs = [
        node.create_subscription(LidarTensor, '/adapter_test/tensor', tensors.append, qos_profile_sensor_data),
        node.create_subscription(PointCloud2, '/adapter_test/organized', clouds.append, qos_profile_sensor_data),
        node.create_subscription(DiagnosticArray, '/adapter_test/status', statuses.append, 10),
    ]
    def spin(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(.01, max(0., deadline-time.monotonic())))
    def imu(q=(1., 0., 0., 0.)):
        m=IMUState();m.quaternion=list(q);imu_pub.publish(m);spin(.025)
    elev,az=np.meshgrid(np.deg2rad(np.linspace(-45,0,16)),np.deg2rad(np.linspace(-60,60,41)),indexing='ij')
    rays=np.stack((np.cos(elev)*np.cos(az),np.cos(elev)*np.sin(az),np.sin(elev)),axis=-1).reshape(-1,3)
    points=rays*2.
    mount=.04014257279586953
    rot=np.array([[np.cos(mount),0,np.sin(mount)],[0,1,0],[-np.sin(mount),0,np.cos(mount)]])
    expected=points@rot.T/5.
    def cloud(points=points, age=0., frame='livox_frame', big=False, organized=False):
        m=PointCloud2();stamp=node.get_clock().now().nanoseconds-int(age*1e9)
        m.header.stamp.sec=stamp//10**9;m.header.stamp.nanosec=stamp%10**9;m.header.frame_id=frame
        m.height=16 if organized else 1;m.width=len(points)//m.height
        m.fields=[PointField(name=name,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,name in enumerate('xyz')]
        m.is_bigendian=big;m.point_step=12;m.row_step=m.width*12+(8 if organized else 0);m.is_dense=False
        data=bytearray()
        for row in np.asarray(points).reshape(m.height,m.width,3):
            for p in row:data.extend(struct.pack(('>' if big else '<')+'fff',*p))
            if organized:data.extend(b'\x00'*8)
        m.data=bytes(data);return m
    def dropped(label, msg, fresh_imu=True):
        if fresh_imu:imu()
        # Refresh non-stale messages after waiting for IMU; intentionally old ones stay old.
        count=len(tensors);cloud_pub.publish(msg);spin(.08)
        assert len(tensors)==count, label+' unexpectedly published'
        results.append({'case':label,'passed':True})
    def accepted(label,msg,reference=expected,verified=True):
        count=len(tensors);cloud_pub.publish(msg);spin(.12)
        assert len(tensors)==count+1,label+' did not publish one tensor'
        t=tensors[-1];a=np.asarray(t.xyz).reshape(656,3)
        assert (t.rows,t.columns,len(t.xyz))==(16,41,1968)
        assert t.occupied_cells==656 and t.normalization_distance_m==5.
        assert t.header.stamp==msg.header.stamp
        assert t.header.frame_id=='transformer_sru_virtual_lidar_yaw'
        assert abs(t.imu_cloud_offset_s)<=.05+1e-6
        assert t.extrinsics_verified is verified
        np.testing.assert_allclose(a,reference,atol=2e-7,rtol=1e-6)
        organized=clouds[-1];assert (organized.height,organized.width)==(16,41)
        assert organized.header.stamp==msg.header.stamp
        np.testing.assert_allclose(np.frombuffer(bytes(organized.data),dtype='<f4').reshape(656,3),a,atol=0,rtol=0)
        results.append({'case':label,'passed':True,'max_xyz_error':float(np.max(np.abs(a-reference)))})
        return t
    log=(out/'node.log').open('w')
    params=['input_topic:=/adapter_test/cloud','imu_topic:=/adapter_test/imu',
        'tensor_output_topic:=/adapter_test/tensor','cloud_output_topic:=/adapter_test/organized',
        'status_topic:=/adapter_test/status','extrinsics_verified:=true',
        f'lidar_to_torso_rpy:=[0.0, {mount}, 0.0]']
    command=[str(args.node),'--ros-args']
    for p in params:command+=['-p',p]
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+12
        while time.monotonic()<deadline and (cloud_pub.get_subscription_count()==0 or imu_pub.get_subscription_count()==0):
            if proc.poll() is not None:raise RuntimeError('adapter exited: '+(out/'node.log').read_text())
            spin(.1)
        assert cloud_pub.get_subscription_count()>0 and imu_pub.get_subscription_count()>0,'DDS discovery timeout'
        spin(.3)
        dropped('missing_imu',cloud(),False)
        imu();valid=cloud();t=accepted('full_grid',valid)
        np.save(out/'lidar_xyz.npy',np.asarray(t.xyz,dtype=np.float32))
        imu();accepted('organized_big_endian_with_row_padding',cloud(big=True,organized=True))
        imu();accepted('exact_5m_float32_transport',cloud(points=rays*5.),expected*2.5)
        imu();dirty=np.vstack((points,points*1.5,np.zeros((100,3)),[np.nan,0,0],[np.inf,0,0]))
        t=accepted('nearest_echo_and_invalid_raw_points',cloud(points=dirty))
        assert t.rejected_points>=102
        for _ in range(14):imu()
        accepted('historical_frame_with_fresh_imu_stream',cloud(age=.15))
        imu();last=cloud();accepted('new_frame',last)
        dropped('repeated_stamp',last)
        dropped('stale_cloud',cloud(age=1.))
        dropped('future_cloud',cloud(age=-1.))
        bad=cloud();bad.header.stamp.sec=0;bad.header.stamp.nanosec=0
        dropped('zero_stamp',bad)
        dropped('wrong_frame',cloud(frame='wrong_frame'))
        bad=cloud();bad.data=bytes(bad.data[:-4]);dropped('truncated_data',bad)
        bad=cloud();bad.fields[1].offset=0;dropped('overlapping_fields',bad)
        spin(.15);dropped('stale_imu',cloud(),False)
        imu((0.,0.,0.,0.));dropped('invalid_quaternion',cloud(),False)
        imu();dropped('all_empty_cloud',cloud(points=np.zeros((10,3))),False)
        imu();accepted('recovery_after_new_valid_inputs',cloud())
        spin(1.1)
        assert statuses, 'no diagnostics published'
        assert statuses[-1].status[0].level!=0,'stale inputs must not remain OK'
        results.append({'case':'diagnostic_becomes_stale','passed':True})
        proc.terminate();proc.wait(timeout=5)
        # Unverified calibration still permits tagged preview data, not control.
        params=[p for p in params if not p.startswith('extrinsics_verified')]
        command=[str(args.node),'--ros-args']
        for p in params:command+=['-p',p]
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        spin(1.5);imu();accepted('unverified_extrinsics_preview',cloud(),verified=False)
        dropped('unverified_preview_still_rejects_old_cloud',cloud(age=1.))
        (out/'results.json').write_text(json.dumps({'passed':True,'cases':results,'ros_domain_id':214,'localhost_only':True},indent=2))
        print(json.dumps({'passed':True,'cases':len(results),'output':str(out)},indent=2))
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    main()
