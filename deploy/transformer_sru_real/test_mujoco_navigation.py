"""Offline integration checks for the released policy's goal and sensor contract."""
import unittest
import numpy as np
from mujoco_navigation import Simulation


class NavigationTest(unittest.TestCase):
    def test_policy_lifecycle_goal_frame_and_ray_contract(self):
        s = Simulation(.25)
        with self.assertRaises(ValueError): s.goal()
        s.prepare()
        for _ in range(160): s.step()
        self.assertEqual(s.phase, 'FIX_STAND_READY')
        s.start_policy()
        for _ in range(100): s.step()
        self.assertEqual(s.nav_steps, 0)
        np.testing.assert_array_equal(s.command, np.zeros(3))
        origin = s.data.qpos[:3].copy()
        s.goal()
        self.assertAlmostEqual(np.linalg.norm((s.goals.goal-origin)[:2]), 7., places=5)
        self.assertAlmostEqual(s.goals.goal[2], .5)
        s.step()
        obs = s.obs[0]
        self.assertEqual(obs.shape, (1984,))
        self.assertTrue(np.isfinite(obs).all())
        np.testing.assert_array_equal(obs[9:12], np.zeros(3))
        self.assertGreater(obs[12], .99)  # initial goal direction is forward
        self.assertGreater(s.hits, 0)    # no obstacle still means floor returns
        self.assertLessEqual(np.linalg.norm(obs[16:].reshape(-1, 3), axis=1).max(), 1.00001)
        self.assertEqual(s.nav_steps, 1)
        s.stopped = True; t = s.data.time; s.step()
        self.assertEqual(s.data.time, t)
        with self.assertRaises(ValueError): s.goal()
        s.reset()
        self.assertEqual(s.phase, 'PASSIVE')
        self.assertEqual(s.nav_steps, 0)

if __name__ == '__main__': unittest.main()
