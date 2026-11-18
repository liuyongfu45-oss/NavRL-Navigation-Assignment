"""Neural network architecture and PPO agent.

The learned controller follows the NavRL actor-critic design:

* A small 2-D CNN encodes the static ray matrix.
* An MLP encodes the nearest dynamic-obstacle matrix.
* A shared MLP fuses internal state, goal direction and the two embeddings.
* The actor head outputs a Beta distribution over bounded velocities; the
  critic head estimates the state value.

PyTorch is imported lazily: the environment, reward and metrics modules do
not require it. If PyTorch is not installed, :data:`TORCH_AVAILABLE` is
``False`` and the network tests are skipped.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - exercised only without torch
    nn = Any  # type: ignore


@dataclass
class NetworkConfig:
    """Hyperparameters of the actor-critic network."""

    internal_dim: int = 8
    goal_dim: int = 3
    static_rays: int = 36
    ray_bins: int = 4
    dynamic_nearest: int = 5
    dynamic_features: int = 10
    static_embed: int = 64
    dynamic_embed: int = 32
    hidden1: int = 128
    hidden2: int = 128
    hidden3: int = 64
    action_dim: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 5e-4
    clip_ratio: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


if TORCH_AVAILABLE:

    class StaticRayEncoder(nn.Module):
        """2-D CNN over the ``(n_rays, ray_bins)`` static ray matrix."""

        def __init__(self, cfg: NetworkConfig):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            flatten = 16 * cfg.static_rays * cfg.ray_bins
            self.fc = nn.Linear(flatten, cfg.static_embed)

        def forward(self, rays: "torch.Tensor") -> "torch.Tensor":
            x = F.relu(self.conv1(rays))
            x = F.relu(self.conv2(x))
            x = x.flatten(start_dim=1)
            return F.relu(self.fc(x))

    class DynamicEncoder(nn.Module):
        """MLP over the nearest dynamic-obstacle matrix."""

        def __init__(self, cfg: NetworkConfig):
            super().__init__()
            flat = cfg.dynamic_nearest * cfg.dynamic_features
            self.fc1 = nn.Linear(flat, 64)
            self.fc2 = nn.Linear(64, cfg.dynamic_embed)

        def forward(self, dynamic: "torch.Tensor") -> "torch.Tensor":
            x = dynamic.flatten(start_dim=1)
            x = F.relu(self.fc1(x))
            return F.relu(self.fc2(x))

    class ActorCritic(nn.Module):
        """Actor-critic network with a Beta action head."""

        def __init__(self, cfg: NetworkConfig):
            super().__init__()
            self.cfg = cfg
            self.static_encoder = StaticRayEncoder(cfg)
            self.dynamic_encoder = DynamicEncoder(cfg)
            fused = (cfg.internal_dim + cfg.goal_dim + cfg.static_embed
                     + cfg.dynamic_embed)
            self.shared = nn.Sequential(
                nn.Linear(fused, cfg.hidden1), nn.ReLU(),
                nn.Linear(cfg.hidden1, cfg.hidden2), nn.ReLU(),
                nn.Linear(cfg.hidden2, cfg.hidden3), nn.ReLU(),
            )
            self.actor_head = nn.Linear(cfg.hidden3, 2 * cfg.action_dim)
            self.critic_head = nn.Linear(cfg.hidden3, 1)

        def forward(self, internal: "torch.Tensor", goal_dir: "torch.Tensor",
                    rays: "torch.Tensor",
                    dynamic: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            static_emb = self.static_encoder(rays)
            dynamic_emb = self.dynamic_encoder(dynamic)
            fused = torch.cat([internal, goal_dir, static_emb, dynamic_emb], dim=1)
            h = self.shared(fused)
            alpha_beta = self.actor_head(h)
            value = self.critic_head(h).squeeze(-1)
            return alpha_beta, value

        def distribution(self, internal, goal_dir, rays, dynamic):
            alpha_beta, value = self.forward(internal, goal_dir, rays, dynamic)
            alpha_beta = alpha_beta.reshape(-1, self.cfg.action_dim, 2)
            alpha = F.softplus(alpha_beta[..., 0]) + 1e-3
            beta = F.softplus(alpha_beta[..., 1]) + 1e-3
            dist = torch.distributions.Beta(alpha, beta)
            return dist, value

        def act(self, internal, goal_dir, rays, dynamic,
                deterministic: bool = False):
            dist, value = self.distribution(internal, goal_dir, rays, dynamic)
            if deterministic:
                sample = dist.mean
            else:
                sample = dist.sample()
            log_prob = dist.log_prob(sample).sum(-1)
            return sample, log_prob, value


    def count_parameters(model: nn.Module) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    class PPOAgent:
        """Minimal PPO controller operating on the actor-critic network."""

        def __init__(self, cfg: NetworkConfig):
            self.cfg = cfg
            self.model = ActorCritic(cfg)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        @staticmethod
        def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(arr, dtype=torch.float32)

        def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            internal = self._to_tensor(obs["internal"])
            goal_dir = self._to_tensor(obs["goal_dir"])
            rays = self._to_tensor(obs["rays"]).unsqueeze(1)
            dynamic = self._to_tensor(obs["dynamic"])
            sample, log_prob, value = self.model.act(
                internal, goal_dir, rays, dynamic, deterministic)
            return (sample.detach().numpy(), log_prob.detach().numpy(),
                    value.detach().numpy())

        def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
            """One PPO update over a collected rollout batch."""
            cfg = self.cfg
            internal = self._to_tensor(batch["internal"])
            goal_dir = self._to_tensor(batch["goal_dir"])
            rays = self._to_tensor(batch["rays"]).unsqueeze(1)
            dynamic = self._to_tensor(batch["dynamic"])
            actions = self._to_tensor(batch["actions"])
            old_log_prob = self._to_tensor(batch["log_prob"])
            returns = self._to_tensor(batch["returns"])
            advantages = self._to_tensor(batch["advantages"])

            dist, value = self.model.distribution(internal, goal_dir, rays, dynamic)
            new_log_prob = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            critic_loss = F.huber_loss(value, returns)
            entropy = dist.entropy().sum(-1).mean()
            loss = (actor_loss + cfg.value_coef * critic_loss
                    - cfg.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            return {
                "actor_loss": float(actor_loss.detach()),
                "critic_loss": float(critic_loss.detach()),
                "entropy": float(entropy.detach()),
                "total_loss": float(loss.detach()),
            }

else:  # pragma: no cover

    def count_parameters(model: Any) -> int:
        return -1
