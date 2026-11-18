"""Tests for the actor-critic network and PPO agent (requires PyTorch)."""
import unittest

import numpy as np

from navrl.network import NetworkConfig, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from navrl.network import ActorCritic, PPOAgent, count_parameters


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.cfg = NetworkConfig()
        self.model = ActorCritic(self.cfg)
        self.batch = 4

    def _dummy(self):
        b = self.batch
        return (
            torch.randn(b, self.cfg.internal_dim),
            torch.randn(b, self.cfg.goal_dim),
            torch.randn(b, 1, self.cfg.static_rays, self.cfg.ray_bins),
            torch.randn(b, self.cfg.dynamic_nearest, self.cfg.dynamic_features),
        )

    def test_forward_shapes(self):
        internal, goal_dir, rays, dynamic = self._dummy()
        alpha_beta, value = self.model(internal, goal_dir, rays, dynamic)
        self.assertEqual(alpha_beta.shape, (self.batch, 2 * self.cfg.action_dim))
        self.assertEqual(value.shape, (self.batch,))

    def test_action_sample_in_unit_interval(self):
        internal, goal_dir, rays, dynamic = self._dummy()
        sample, log_prob, value = self.model.act(internal, goal_dir, rays, dynamic)
        self.assertTrue((sample >= 0).all() and (sample <= 1).all())
        self.assertEqual(log_prob.shape, (self.batch,))
        self.assertEqual(value.shape, (self.batch,))

    def test_parameter_count_positive(self):
        self.assertGreater(count_parameters(self.model), 1000)

    def test_ppo_update_runs(self):
        agent = PPOAgent(self.cfg)
        b = self.batch
        batch = {
            "internal": np.random.randn(b, self.cfg.internal_dim),
            "goal_dir": np.random.randn(b, self.cfg.goal_dim),
            "rays": np.random.randn(b, self.cfg.static_rays, self.cfg.ray_bins),
            "dynamic": np.random.randn(b, self.cfg.dynamic_nearest,
                                       self.cfg.dynamic_features),
            "actions": np.random.rand(b, self.cfg.action_dim),
            "log_prob": np.random.randn(b),
            "returns": np.random.randn(b),
            "advantages": np.random.randn(b),
        }
        losses = agent.update(batch)
        for key in ("actor_loss", "critic_loss", "entropy", "total_loss"):
            self.assertIn(key, losses)
            self.assertTrue(np.isfinite(losses[key]))


if __name__ == "__main__":
    unittest.main()
