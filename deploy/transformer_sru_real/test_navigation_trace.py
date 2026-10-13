"""Replay captured obs/h/c exactly through the deployment ONNX."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import Navigation
from navigation_trace import NavigationTrace


class NavigationTraceTest(unittest.TestCase):
    def test_exact_recurrent_replay_and_no_mutation(self):
        nav=Navigation();trace=NavigationTrace()
        lidar=np.zeros(1968,np.float32);lidar[0:3]=[.2,0.,-.2]
        for i in range(3):
            h_in,c_in=nav.h.copy(),nav.c.copy()
            nav.step([.01*i,0.,0.],[0.,0.,.02*i],[0.,0.,-1.],[7.,0.,-.3],lidar)
            trace.append(nav,h_in,c_in,{'nav_step':i+1,'goal_stamp_ns':42})
        captured=trace.rows[-1][5].copy()
        nav.reset()
        np.testing.assert_array_equal(trace.rows[-1][5],captured)
        with tempfile.TemporaryDirectory() as folder:
            path=trace.save(Path(folder)/'run.jsonl');z=np.load(path)
            self.assertEqual(z['obs'].shape,(3,1984))
            self.assertEqual(z['h_in'].shape,(3,1,1,512))
            np.testing.assert_array_equal(z['h_in'][0],np.zeros((1,1,512)))
            np.testing.assert_array_equal(z['h_in'][1],z['h_out'][0])
            self.assertEqual(json.loads(z['metadata_json'][0])['goal_stamp_ns'],42)
            for i in range(3):
                action,h,c=nav.session.run(None,{'obs':z['obs'][i:i+1],'h_in':z['h_in'][i],'c_in':z['c_in'][i]})
                np.testing.assert_array_equal(action[0],z['raw_action'][i])
                np.testing.assert_array_equal(h,z['h_out'][i])
                np.testing.assert_array_equal(c,z['c_out'][i])
        self.assertIsNone(NavigationTrace().save(Path('/unused/no_frames.jsonl')))


if __name__=='__main__':unittest.main()
