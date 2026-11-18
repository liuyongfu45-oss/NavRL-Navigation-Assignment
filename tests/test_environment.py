"""Tests for the navigation environment."""
import unittest

import numpy as np

from navrl.environment import MultiRobotNavEnv
from navrl.utils import load_config


class TestEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def _env(self, key="S3_mixed", seed=0):
        return MultiRobotNavEnv(self.config, key, seed=seed)

    def test_reset_observation_shapes(self):
        env = self._env()
        obs = env.reset()
        self.assertEqual(obs["internal"].shape, (3, 8))
        self.assertEqual(obs["goal_dir"].shape, (3, 3))
        self.assertEqual(obs["rays"].shape, (3, 36, 4))
        self.assertEqual(obs["dynamic"].shape, (3, 5, 10))
        self.assertEqual(obs["flat"].shape, (3, 205))

    def test_step_advances_simulation(self):
        env = self._env()
        env.reset()
        actions = np.zeros((3, 2))
        obs, rewards, dones, info = env.step(actions)
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(dones.shape, (3,))
        self.assertTrue(np.isfinite(rewards).all())
        self.assertIn("step", info)

    def test_goal_reached_when_close(self):
        env = self._env()
        env.reset()
        robot = env.robots[0]
        robot.pos = robot.goal.copy()
        _, rewards, dones, info = env.step(np.zeros((3, 2)))
        self.assertTrue(robot.reached)
        self.assertTrue(dones[0])
        self.assertGreater(rewards[0], 0.0)
        self.assertEqual(info["robots"][0]["reached"], True)

    def test_collision_detected(self):
        env = self._env()
        env.reset()
        robot = env.robots[0]
        obstacle = env.obstacles[0]
        robot.pos = obstacle.pos.copy()
        _, rewards, dones, info = env.step(np.zeros((3, 2)))
        self.assertTrue(robot.collided)
        self.assertTrue(dones[0])
        self.assertLess(rewards[0], 0.0)

    def test_dynamic_obstacle_moves(self):
        env = self._env("S1_dynamic")
        env.reset()
        dynamic = [o for o in env.obstacles if o.kind == "dynamic"]
        before = dynamic[0].pos.copy()
        env.step(np.zeros((3, 2)))
        after = dynamic[0].pos
        self.assertFalse(np.allclose(before, after))

    def test_multi_robot_variant_builds(self):
        env = self._env("assignment_multi_robot")
        self.assertEqual(env.n_robots, 6)
        self.assertEqual(len(env.obstacles), 15)
        obs = env.reset()
        self.assertEqual(obs["flat"].shape[0], 6)

    def test_actions_are_clipped(self):
        env = self._env()
        env.reset()
        actions = np.array([[99.0, -99.0] for _ in range(3)])
        env.step(actions)
        for robot in env.robots:
            self.assertLessEqual(abs(robot.v), env.v_lim + 1e-6)
            self.assertLessEqual(abs(robot.omega), env.omega_lim + 1e-6)


if __name__ == "__main__":
    unittest.main()
