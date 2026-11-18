"""Tests for scenario layout construction."""
import unittest

from navrl.scenarios import build_layout
from navrl.utils import load_config


class TestScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_s1_dynamic_layout(self):
        layout = build_layout(self.config, "S1_dynamic", seed=1)
        self.assertEqual(layout.n_robots, 3)
        self.assertEqual(layout.n_static, 0)
        self.assertEqual(layout.n_dynamic, 14)
        self.assertEqual(len(layout.obstacles), 14)

    def test_s2_static_layout(self):
        layout = build_layout(self.config, "S2_static", seed=1)
        self.assertEqual(layout.n_robots, 3)
        self.assertEqual(layout.n_static, 12)
        self.assertEqual(layout.n_dynamic, 0)

    def test_s3_mixed_layout(self):
        layout = build_layout(self.config, "S3_mixed", seed=1)
        self.assertEqual(layout.n_robots, 3)
        self.assertEqual(layout.n_static, 7)
        self.assertEqual(layout.n_dynamic, 8)
        self.assertEqual(len(layout.obstacles), 15)

    def test_assignment_multi_robot_variant(self):
        layout = build_layout(self.config, "assignment_multi_robot", seed=2)
        self.assertEqual(layout.n_robots, 6)
        self.assertEqual(len(layout.obstacles), 15)
        self.assertEqual(layout.starts.shape, (6, 2))
        self.assertEqual(layout.goals.shape, (6, 2))

    def test_unknown_scenario_raises(self):
        with self.assertRaises(KeyError):
            build_layout(self.config, "does_not_exist")

    def test_layout_is_deterministic(self):
        a = build_layout(self.config, "S3_mixed", seed=7)
        b = build_layout(self.config, "S3_mixed", seed=7)
        for oa, ob in zip(a.obstacles, b.obstacles):
            self.assertTrue((oa.pos == ob.pos).all())


if __name__ == "__main__":
    unittest.main()
