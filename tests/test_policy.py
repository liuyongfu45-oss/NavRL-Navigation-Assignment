"""Tests for the baseline policies."""
import unittest

import numpy as np

from navrl.environment import MultiRobotNavEnv
from navrl.policy import APFPolicy, RandomPolicy
from navrl.utils import load_config


class TestPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_apf_action_is_bounded_and_finite(self):
        env = MultiRobotNavEnv(self.config, "S2_static", seed=0)
        obs = env.reset()
        policy = APFPolicy()
        for i in range(env.n_robots):
            action = policy.act(obs, env, i)
            self.assertEqual(action.shape, (2,))
            self.assertTrue(np.isfinite(action).all())
            self.assertLessEqual(abs(action[0]), env.v_lim + 1e-6)
            self.assertLessEqual(abs(action[1]), env.omega_lim + 1e-6)

    def test_apf_progresses_toward_goal(self):
        env = MultiRobotNavEnv(self.config, "S2_static", seed=0)
        obs = env.reset()
        policy = APFPolicy()
        robot = env.robots[0]
        start_dist = np.linalg.norm(robot.goal - robot.pos)
        for _ in range(30):
            actions = np.array([policy.act(obs, env, i)
                                for i in range(env.n_robots)])
            obs, _, dones, _ = env.step(actions)
            if env.all_done():
                break
        end_dist = np.linalg.norm(robot.goal - robot.pos)
        self.assertLess(end_dist, start_dist)

    def test_random_policy_actions(self):
        env = MultiRobotNavEnv(self.config, "S1_dynamic", seed=0)
        obs = env.reset()
        policy = RandomPolicy(seed=3)
        action = policy.act(obs, env, 0)
        self.assertTrue(np.isfinite(action).all())
        self.assertLessEqual(abs(action[0]), env.v_lim + 1e-6)


if __name__ == "__main__":
    unittest.main()
