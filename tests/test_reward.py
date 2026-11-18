"""Tests for the reward function and reward table."""
import unittest

import numpy as np

from navrl.environment import RobotState
from navrl.reward import REWARD_TABLE, RewardConfig, compute_reward


def _robot(v=0.0, omega=0.0, last_v=0.0, last_omega=0.0) -> RobotState:
    return RobotState(
        pos=np.array([0.0, 0.0]), goal=np.array([5.0, 0.0]),
        theta=0.0, v=v, omega=omega, last_v=last_v, last_omega=last_omega,
    )


class TestReward(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig()

    def test_goal_bonus_is_positive(self):
        robot = _robot()
        total, comp = compute_reward(
            robot, prev_dist=5.0, dist=0.2, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=True,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertEqual(comp["goal"], self.cfg.goal_bonus)
        self.assertGreater(total, 0.0)

    def test_collision_penalty_is_negative(self):
        robot = _robot()
        total, comp = compute_reward(
            robot, prev_dist=5.0, dist=4.8, static_clearance=0.1,
            dynamic_clearance=0.1, collided=True, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertEqual(comp["collision"], self.cfg.collision_penalty)
        self.assertLess(total, 0.0)

    def test_timeout_penalty(self):
        robot = _robot()
        _, comp = compute_reward(
            robot, prev_dist=5.0, dist=4.8, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=False,
            timeout=True, cfg=self.cfg, dt=0.1)
        self.assertEqual(comp["timeout"], self.cfg.timeout_penalty)

    def test_progress_positive_when_closer(self):
        robot = _robot()
        _, comp = compute_reward(
            robot, prev_dist=5.0, dist=4.0, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertAlmostEqual(comp["progress"], 1.0)

    def test_progress_negative_when_farther(self):
        robot = _robot()
        _, comp = compute_reward(
            robot, prev_dist=4.0, dist=5.0, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertLess(comp["progress"], 0.0)

    def test_safety_positive_when_clear(self):
        robot = _robot()
        _, comp = compute_reward(
            robot, prev_dist=5.0, dist=5.0, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertGreater(comp["static_safety"], 0.0)
        self.assertGreater(comp["dynamic_safety"], 0.0)

    def test_safety_negative_when_too_close(self):
        robot = _robot()
        _, comp = compute_reward(
            robot, prev_dist=5.0, dist=5.0, static_clearance=0.1,
            dynamic_clearance=0.1, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertLess(comp["static_safety"], 0.0)
        self.assertLess(comp["dynamic_safety"], 0.0)

    def test_smoothness_penalises_command_change(self):
        robot = _robot(v=1.0, omega=0.5, last_v=0.0, last_omega=0.0)
        _, comp = compute_reward(
            robot, prev_dist=5.0, dist=5.0, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertLess(comp["smoothness"], 0.0)

    def test_stall_penalty_when_idle_far_from_goal(self):
        robot = _robot(v=0.0)
        _, comp = compute_reward(
            robot, prev_dist=5.0, dist=5.0, static_clearance=5.0,
            dynamic_clearance=5.0, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertEqual(comp["stall"], -self.cfg.stall_penalty)

    def test_components_sum_to_total(self):
        robot = _robot(v=0.5, omega=0.2, last_v=0.1, last_omega=0.0)
        total, comp = compute_reward(
            robot, prev_dist=5.0, dist=4.6, static_clearance=0.8,
            dynamic_clearance=2.0, collided=False, reached=False,
            timeout=False, cfg=self.cfg, dt=0.1)
        self.assertAlmostEqual(total, sum(comp.values()))

    def test_reward_table_is_complete(self):
        required = {"term", "sign", "condition", "formula", "magnitude",
                    "behavior"}
        signs = {"Positive", "Negative",
                 "Positive when closer, negative when farther",
                 "Positive when far, negative when close",
                 "Positive / negative"}
        self.assertGreater(len(REWARD_TABLE), 5)
        for row in REWARD_TABLE:
            self.assertTrue(required.issubset(row.keys()))
            self.assertIn(row["sign"], signs)
            self.assertTrue(row["term"])
            self.assertTrue(row["behavior"])


if __name__ == "__main__":
    unittest.main()
