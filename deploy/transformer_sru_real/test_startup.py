"""Lifecycle regression against the official FSM configuration, without I/O."""
from pathlib import Path
import hashlib
import os
import unittest
import numpy as np
import yaml
from core import Unitree, target_diagnostic
from startup import Startup, STARTUP, remote_transition, SelectDoubleClick, ButtonDoubleClick


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.fsm=Startup()
        self.state={'q':self.fsm.q.copy(),'dq':np.zeros(29),'quat':np.array([1.,0.,0.,0.])}

    @unittest.skipUnless(os.environ.get('SRU_UNITREE_FSM_CONFIG'),
                         'Optional comparison requires original Unitree FSM configuration')
    def test_official_pose_gains_and_timing(self):
        path=Path(os.environ['SRU_UNITREE_FSM_CONFIG'])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),STARTUP['source_sha256'])
        source=yaml.safe_load(path.read_text())['FSM']
        np.testing.assert_array_equal(self.fsm.q,np.asarray(source['FixStand']['qs'][-1],np.float32))
        np.testing.assert_array_equal(self.fsm.kp,source['FixStand']['kp'])
        np.testing.assert_array_equal(self.fsm.kd,source['FixStand']['kd'])
        np.testing.assert_array_equal(self.fsm.passive_kd,source['Passive']['kd'])
        self.assertEqual(STARTUP['fixstand_duration_s'],source['FixStand']['ts'][-1])
        self.assertNotEqual(float(self.fsm.q[15]),float(Unitree().default_sdk[15]))

    def test_order_interpolation_and_explicit_policy(self):
        with self.assertRaisesRegex(ValueError,'FIX_STAND'):self.fsm.start_policy(self.state,0.)
        initial=self.state['q']+.01;self.fsm.prepare(initial,1.)
        target,kp,kd=self.fsm.position_command(self.state,2.5)
        np.testing.assert_allclose(target,(initial+self.fsm.q)/2)
        self.fsm.observe(self.state,4.);self.assertEqual(self.fsm.phase,'FIX_STAND_READY')
        self.fsm.observe(self.state,40.);self.assertEqual(self.fsm.phase,'FIX_STAND_READY')
        self.fsm.start_policy(self.state,40.)
        self.assertEqual(self.fsm.phase,'ZERO_VELOCITY')
        self.fsm.passive()
        with self.assertRaises(ValueError):self.fsm.start_policy(self.state,42.)

    def test_measured_error_is_diagnostic_without_timeout(self):
        self.fsm.prepare(self.state['q'],0.)
        self.state['q'][0]+=.1890934
        self.state['dq'][0]=1.
        angle=np.deg2rad(20.)
        self.state['quat']=np.array([np.cos(angle/2),np.sin(angle/2),0.,0.])
        self.fsm.observe(self.state,3.)
        self.assertAlmostEqual(self.fsm.q_error,.1890934,places=6)
        self.assertEqual(self.fsm.phase,'FIX_STAND_READY')
        self.fsm.observe(self.state,60.)
        self.fsm.start_policy(self.state,60.)
        self.assertEqual(self.fsm.phase,'ZERO_VELOCITY')

    def test_operator_can_switch_during_interpolation(self):
        self.fsm.prepare(self.state['q'],0.)
        self.fsm.start_policy(self.state,.1)
        self.assertEqual(self.fsm.phase,'ZERO_VELOCITY')
        self.assertIsNone(self.fsm.position_command(self.state,.1))

    def test_damping_has_no_position_target_but_torque_is_reported(self):
        self.state['q'][0]=10.
        q,kp,kd=self.fsm.position_command(self.state,0.)
        self.assertEqual(target_diagnostic(q,q,np.zeros(29),kp,kd),'')
        dq=np.zeros(29);dq[20]=3.
        self.assertIn('estimated_pd_torque_limit',target_diagnostic(q,q,dq,kp,kd))

    def test_remote_edges_and_stop_priority(self):
        prepare=(1<<5)|(1<<12);policy=1|(1<<10);stop=(1<<5)|(1<<9)
        self.assertIsNone(remote_transition(None,prepare))
        self.assertEqual(remote_transition(1<<5,prepare),'prepare')
        self.assertIsNone(remote_transition(prepare,prepare))
        self.assertEqual(remote_transition(1,policy),'start_policy')
        self.assertEqual(remote_transition(None,stop),'stop')
        self.assertEqual(remote_transition(0,stop|prepare),'stop')


class SelectStopTest(unittest.TestCase):
    def test_single_and_held_press_do_not_stop(self):
        d=SelectDoubleClick()
        for buttons,now in [(0,0.),(8,.1),(8,.2),(8,.6),(0,.7)]:
            self.assertFalse(d.update(buttons,now))

    def test_double_click_and_latch(self):
        d=SelectDoubleClick()
        for buttons,now in [(0,0.),(8,.1),(0,.2)]:self.assertFalse(d.update(buttons,now))
        self.assertTrue(d.update(8,.4))
        self.assertTrue(d.update(0,10.))

    def test_slow_clicks_restart_window(self):
        d=SelectDoubleClick()
        for buttons,now in [(0,0.),(8,.1),(0,.2),(8,.7),(0,.8)]:
            self.assertFalse(d.update(buttons,now))
        self.assertTrue(d.update(8,1.0))

    def test_initial_held_button_is_not_first_click(self):
        d=SelectDoubleClick()
        for buttons,now in [(8,0.),(8,.1),(0,.2),(8,.3),(0,.4)]:
            self.assertFalse(d.update(buttons,now))
        self.assertTrue(d.update(8,.5))

    def test_select_bit_and_boundary(self):
        d=SelectDoubleClick()
        for buttons,now in [(0,0.),(4,.05),(0,.1),(4,.15),(0,.2),(8,1.),(0,1.1)]:
            self.assertFalse(d.update(buttons,now))
        self.assertTrue(d.update(8,1.5))


class AGoalClickTest(unittest.TestCase):
    def test_pairs_pulse_without_hold_or_triple_retrigger(self):
        d=ButtonDoubleClick(8)
        sequence=[(0,0.,False),(256,.1,False),(0,.2,False),(256,.3,True),
                  (256,.4,False),(0,.5,False),(256,.6,False),(0,.7,False),(256,.8,True)]
        for buttons,now,expected in sequence:self.assertEqual(d.update(buttons,now),expected)

    def test_initial_hold_wrong_button_and_expired_click(self):
        d=ButtonDoubleClick(8)
        for buttons,now in [(256,0.),(0,.1),(8,.2),(0,.3),(8,.4),(0,.5),
                            (256,1.),(0,1.1),(256,1.6),(0,1.7)]:
            self.assertFalse(d.update(buttons,now))
        self.assertTrue(d.update(256,1.8))

    def test_reset_drops_partial_pair(self):
        d=ButtonDoubleClick(8)
        d.update(0,0.);d.update(256,.1);d.reset()
        for buttons,now in [(256,.2),(0,.25),(256,.3),(0,.4)]:
            self.assertFalse(d.update(buttons,now))
        self.assertTrue(d.update(256,.5))


if __name__=='__main__':unittest.main(verbosity=2)
