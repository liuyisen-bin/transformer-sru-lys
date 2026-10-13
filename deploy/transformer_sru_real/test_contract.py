"""Independent comparisons against training source, ONNX/Torch, and SDK CRC."""
import ast
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from core import CONTRACT, ROOT, Navigation, Unitree, GoalTracker, check_age, check_receipt_age, tick_advanced, relative_goal, target_diagnostic
from transport import lowcmd_crc,motor_message

ROS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in
                    ('rclpy', 'unitree_hg', 'unitree_api', 'transformer_sru_lidar_adapter'))


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source=ROOT/'training/source/engineai_rl_lab/engineai_rl_lab/tasks/navigation/mdp/navigation/actions/navigation_se2_actions_unitree.py'
        tree=ast.parse(cls.source.read_text());k=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UnitreeNavigationSE2Action')
        methods=[n for n in k.body if isinstance(n,ast.FunctionDef) and n.name in ('process_actions','_current_terms','_build_policy_observations')]
        ns={'torch':torch,'UNITREE_POLICY_HISTORY_LENGTH':5,'UNITREE_POLICY_INPUT_DIM':480}
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(cls.source),'exec'),ns)
        cls.Ref=type('TrainingReference',(),{n.name:ns[n.name] for n in methods})

    def reference(self):
        b=self.Ref();b._raw_navigation_actions=torch.zeros(1,3);b._velocity_commands=torch.zeros(1,3)
        b._policy_bias=torch.zeros(1,3);b._policy_scaling=torch.ones(1,3)
        b._command_min=torch.tensor([CONTRACT['command_min']]);b._command_max=torch.tensor([CONTRACT['command_max']])
        b._per_env_per_dim_low_pass_alpha=torch.tensor([CONTRACT['command_alpha']]);b.cfg=SimpleNamespace(enable_low_pass_filter=True)
        b._joint_ids=torch.tensor(CONTRACT['policy_to_sdk']);b._last_policy_action=torch.zeros(1,29)
        b._history_initialized=torch.zeros(1,dtype=torch.bool)
        b._TERM_DIMS=dict(base_ang_vel=3,projected_gravity=3,velocity_commands=3,joint_pos_rel=29,joint_vel_rel=29,last_action=29)
        b._term_history={k:torch.zeros(1,5,v) for k,v in b._TERM_DIMS.items()};b._asset=SimpleNamespace(data=SimpleNamespace())
        return b

    def test_full_low_history_and_actions_match_training(self):
        low=Unitree();ref=self.reference();rng=np.random.default_rng(43)
        path=ROOT/'training/source/engineai_rl_lab/engineai_rl_lab/tasks/navigation/g1/unitree_velocity_policy.py'
        spec=importlib.util.spec_from_file_location('low_reference',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.UnitreeVelocityPolicyBatch(ROOT/CONTRACT['low_model'],'cpu')
        ref._asset.data.default_joint_pos=torch.from_numpy(low.default_sdk[None])
        errors=[]
        with torch.inference_mode():
            for _ in range(80):
                q=low.default_sdk+rng.normal(0,.03,29).astype(np.float32);dq=rng.normal(0,.2,29).astype(np.float32)
                gyro=rng.normal(0,.1,3).astype(np.float32);gravity=np.array([.01,.02,-.99975],np.float32)
                raw=rng.normal(0,1,3).astype(np.float32);ref.process_actions(torch.from_numpy(raw[None]))
                command=ref._velocity_commands.numpy()[0]
                ref._asset.data.joint_pos=torch.from_numpy(q[None]);ref._asset.data.joint_vel=torch.from_numpy(dq[None])
                ref._asset.data.root_ang_vel_b=torch.from_numpy(gyro[None]);ref._asset.data.projected_gravity_b=torch.from_numpy(gravity[None])
                target=low.step(q,dq,gyro,gravity,command)
                reference_obs=ref._build_policy_observations(torch.tensor([0]))
                np.testing.assert_allclose(low.last_obs,reference_obs.numpy(),atol=2e-6,rtol=1e-6)
                action=model(reference_obs);ref._last_policy_action=action.clone()
                expected=np.empty(29,np.float32);expected[low.order]=low.default+.25*action.numpy()[0]
                errors.append(float(np.max(abs(target-expected))))
                np.testing.assert_allclose(target,expected,atol=3e-6,rtol=2e-6)
        print('80-step low-level max target error:',max(errors))

    def test_navigation_matches_native_actor_and_mapping(self):
        sys.path.insert(0,str(ROOT/'tools/readonly_24999'))
        from export_onnx import load_actor
        actor,_=load_actor(ROOT/'checkpoints/model_24999.pt');actor.memory_a.hidden_states=None
        nav=Navigation();ref=self.reference()
        # Seeded valid sensor-shaped inputs keep this regression independent of
        # private field recordings while exercising carried recurrent state.
        rng=np.random.default_rng(24999)
        elevation,azimuth=np.meshgrid(np.deg2rad(np.linspace(-45,0,16)),
                                     np.deg2rad(np.linspace(-60,60,41)),indexing='ij')
        rays=np.stack([np.cos(elevation)*np.cos(azimuth),
                       np.cos(elevation)*np.sin(azimuth),np.sin(elevation)],-1).reshape(656,3)
        with torch.inference_mode():
            for _ in range(64):
                points=rays*rng.uniform(.1,1.,(656,1))
                points[rng.random(656)<.2]=0
                command=nav.step(rng.normal(0,.05,3),rng.normal(0,.05,3),
                                 np.array([0,0,-1]),np.array([5,0,-.3]),points.ravel())
                raw=actor.act_inference(torch.from_numpy(nav.last_obs))
                ref.process_actions(raw)
                np.testing.assert_allclose(nav.previous,raw.numpy()[0],atol=3e-6,rtol=1e-6)
                np.testing.assert_allclose(command,ref._velocity_commands.numpy()[0],atol=3e-6,rtol=1e-6)
        nav.reset();self.assertFalse(nav.h.any());self.assertFalse(nav.command.any())

    def test_fixed_goal_and_arrival(self):
        q=[np.sqrt(.5),0,0,np.sqrt(.5)]
        goal=relative_goal([1,2,.8],q,5,0,.5)
        np.testing.assert_allclose(goal,[1,7,.5],atol=1e-6)
        tracker=GoalTracker();tracker.set(goal,0)
        self.assertEqual(tracker.update([1,6.7,.8],.5),'SETTLING')
        self.assertEqual(tracker.update([1,6.8,.8],1.51),'ARRIVED')
        self.assertEqual(tracker.update([3,2,.8],3),'ARRIVED')
        np.testing.assert_allclose(tracker.goal,[1,7,.5],atol=1e-6)
        tracker.set([5,0,.5],4);self.assertEqual(tracker.update([0,0,.8],64),'TIMEOUT')

    def test_tick_repeat_reorder_wrap(self):
        self.assertTrue(tick_advanced(None,1));self.assertTrue(tick_advanced(0xffffffff,0))
        self.assertFalse(tick_advanced(2,2));self.assertFalse(tick_advanced(2,1))

    def test_same_clock_lidar_and_goal_checks(self):
        check_age(1_000_000_000,5,1_100_000_000,5.01,.3,'test')
        with self.assertRaisesRegex(ValueError,'source_age'):check_age(1_000_000_000,5,2_000_000_000,5.01,.3,'test')
        with self.assertRaisesRegex(ValueError,'source_age'):check_age(2_000_000_000,5,1_000_000_000,5.01,.3,'test')
        with self.assertRaisesRegex(ValueError,'receipt_stale'):check_age(1_000_000_000,4,1_100_000_000,5.01,.3,'test')

    def test_odom_receipt_liveness(self):
        check_receipt_age(5,5.01,.3,'odom')
        for at,now in [(4,5.01),(6,5.01),(float('nan'),5.01)]:
            with self.assertRaisesRegex(ValueError,'odom_receipt_stale'):
                check_receipt_age(at,now,.3,'odom')

    def test_configured_scale_changes_only_joint_residuals(self):
        import yaml
        cfg=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        baseline=Unitree();changed=Unitree(action_scale=.7)
        self.assertEqual(cfg['low_action_scale'],.25)
        rng=np.random.default_rng(81)
        for _ in range(20):
            q=baseline.default_sdk+rng.normal(0,.1,29);dq=rng.normal(0,.2,29)
            gyro=rng.normal(0,.1,3);g=np.array([.1,0,-np.sqrt(.99)]);cmd=np.zeros(3)
            original=baseline.step(q,dq,gyro,g,cmd);actual=changed.step(q,dq,gyro,g,cmd)
            np.testing.assert_array_equal(changed.last_obs,baseline.last_obs)
            np.testing.assert_array_equal(changed.previous,baseline.previous)
            expected=baseline.default_sdk+(original-baseline.default_sdk)*2.8
            np.testing.assert_allclose(actual,expected,atol=3e-7,rtol=1e-6)
        for value in (float('nan'),float('inf'),0,-.7):
            with self.assertRaises(ValueError):Unitree(action_scale=value)

    def test_target_diagnostics(self):
        low=Unitree();q=low.default_sdk;dq=np.zeros(29)
        self.assertEqual(target_diagnostic(q,q,dq),'')
        out=q.copy();out[0]=9
        self.assertIn('target_joint_limit',target_diagnostic(out,q,dq))
        out=q.copy();out[0]+=.9
        self.assertIn('estimated_pd_torque_limit',target_diagnostic(out,q,dq))
        with self.assertRaises(ValueError):target_diagnostic(q*np.nan,q,dq)

    @unittest.skipUnless(ROS_AVAILABLE, 'Requires ROS and built Unitree/adapter messages')
    def test_target_diagnostics_do_not_block_or_clip_snapshot(self):
        # Invoke the actual controller compute path without ROS or a publisher.
        from controller import Controller
        low=Unitree();q=low.default_sdk.copy()
        for offset,expected in ((9.,'target_joint_limit'),(.9,'estimated_pd_torque_limit')):
            target=q.copy();target[0]+=offset
            fake=SimpleNamespace(
                input_error=lambda full=True:'',lowstate={'q':q,'dq':np.zeros(29),
                    'quat':np.array([1.,0.,0.,0.]),'gyro':np.zeros(3),'mode_machine':5,'tick':1},
                goals=GoalTracker(),fault='',owns_control=True,
                startup=SimpleNamespace(observe=lambda st,now:None,position_command=lambda st,now:None,phase='ZERO_VELOCITY'),
                nav=SimpleNamespace(reset=lambda:None,command=np.zeros(3),previous=np.zeros(3)),
                low=SimpleNamespace(step=lambda *args:target),low_count=0,cfg={'low_action_scale':.7},
                event=lambda *args,**kwargs:None,
                fail=lambda reason:self.fail('unexpected stop: '+reason))
            Controller.compute(fake,time.monotonic())
            self.assertIn(expected,fake.target_diagnostic_text)
            np.testing.assert_array_equal([m.q for m in fake.snapshot[1].motor_cmd[:29]],target)
            self.assertEqual(fake.phase,'ZERO_VELOCITY')

    @unittest.skipUnless(ROS_AVAILABLE and os.environ.get('SRU_UNITREE_CRC_LIBRARY'),
                         'Requires ROS messages and SRU_UNITREE_CRC_LIBRARY')
    def test_crc_matches_official_native_sdk(self):
        sdk=Path(os.environ['SRU_UNITREE_CRC_LIBRARY'])
        lib=ctypes.CDLL(str(sdk));lib.crc32_core.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint32];lib.crc32_core.restype=ctypes.c_uint32
        rng=np.random.default_rng(17)
        for _ in range(20):
            msg=motor_message(5,rng.normal(0,.2,29),rng.uniform(0,100,29),rng.uniform(0,10,29))
            values=[msg.mode_pr,msg.mode_machine]
            for motor in msg.motor_cmd:values.extend([motor.mode,motor.q,motor.dq,motor.tau,motor.kp,motor.kd,motor.reserve])
            values.extend(msg.reserve);values.append(0)
            packed=struct.pack('<2B2x'+'B3x5fI'*35+'5I',*values)
            words=(ctypes.c_uint32*251).from_buffer_copy(packed)
            self.assertEqual(lowcmd_crc(msg),int(lib.crc32_core(words,250)))
            for motor in msg.motor_cmd[29:]:self.assertEqual(motor.mode,0)

    @unittest.skipUnless(ROS_AVAILABLE, 'Requires ROS and built Unitree/adapter messages')
    def test_writer_watchdogs_work_without_compute_loop(self):
        from controller import Controller
        for problem in ('lowstate','target','deadline'):
            published=[];now=time.monotonic()
            fake=SimpleNamespace(stop_event=threading.Event(),low_at=now,ramp_start=now,
                snapshot=(now,'target'),cfg={'lowstate_timeout_s':.1,'command_watchdog_s':.08,'writer_deadline_s':.02},
                last_publish_at=0.,max_publish_gap=0.,publish_count=0,
                lowcmd_pub=SimpleNamespace(publish=published.append),damping='damping',suppress_stop_publish=False,writer_fault='')
            if problem=='lowstate':fake.low_at=now-1
            elif problem=='target':fake.snapshot=(now-1,'target')
            else:fake.last_publish_at=now-1
            Controller.write_loop(fake)
            self.assertTrue(fake.stop_event.is_set());self.assertEqual(published,['damping'])
            self.assertIn(problem,fake.writer_fault)


if __name__=='__main__':unittest.main(verbosity=2)
