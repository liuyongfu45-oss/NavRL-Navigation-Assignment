"""PPO training loop for the NavRL actor-critic network.

This module requires PyTorch. The rest of the package (environment, reward,
metrics, APF baseline) runs without it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .environment import MultiRobotNavEnv
from .network import NetworkConfig, PPOAgent, TORCH_AVAILABLE
from .utils import REPO_ROOT, load_config


def _scale_action(normalized: np.ndarray, env: MultiRobotNavEnv) -> np.ndarray:
    """Map Beta samples in [0, 1] to bounded ``[v, omega]`` commands."""
    v = 2.0 * normalized[:, 0] * env.v_lim - env.v_lim
    omega = 2.0 * normalized[:, 1] * env.omega_lim - env.omega_lim
    return np.column_stack([v, omega])


def _compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                 gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards, dtype=float)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train(config: Optional[Dict[str, Any]] = None, scenario_key: str = "S3_mixed",
          iterations: int = 20, rollout_steps: int = 120, seed: int = 0,
          checkpoint_path: Optional[str] = None) -> Dict[str, List[float]]:
    """Train the PPO agent on one scenario.

    Parameters
    ----------
    iterations:
        Number of PPO update rounds.
    rollout_steps:
        Environment steps collected per round.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training. "
                           "Install it with `pip install torch`.")

    config = config or load_config()
    net_cfg = NetworkConfig()
    agent = PPOAgent(net_cfg)
    env = MultiRobotNavEnv(config, scenario_key, seed=seed)

    history: Dict[str, List[float]] = {"return": [], "success": [],
                                        "actor_loss": [], "critic_loss": []}

    for it in range(iterations):
        obs = env.reset()
        batch: Dict[str, List[Any]] = {
            "internal": [], "goal_dir": [], "rays": [], "dynamic": [],
            "actions": [], "log_prob": [], "rewards": [], "values": [],
            "dones": [],
        }
        episode_return = 0.0
        successes = 0

        for _ in range(rollout_steps):
            norm_action, log_prob, value = agent.act(obs)
            actions = _scale_action(norm_action, env)
            obs, rewards, dones, info = env.step(actions)

            batch["internal"].append(obs["internal"])
            batch["goal_dir"].append(obs["goal_dir"])
            batch["rays"].append(obs["rays"])
            batch["dynamic"].append(obs["dynamic"])
            batch["actions"].append(norm_action)
            batch["log_prob"].append(log_prob)
            batch["rewards"].append(rewards)
            batch["values"].append(value)
            batch["dones"].append(dones.astype(float))

            episode_return += float(rewards.mean())
            successes += sum(r.reached for r in env.robots)
            if env.all_done():
                obs = env.reset()

        rewards_arr = np.concatenate(batch["rewards"])
        values_arr = np.concatenate(batch["values"])
        dones_arr = np.concatenate(batch["dones"])
        advantages, returns = _compute_gae(rewards_arr, values_arr, dones_arr,
                                          net_cfg.gamma, net_cfg.gae_lambda)

        train_batch = {
            "internal": np.concatenate(batch["internal"]),
            "goal_dir": np.concatenate(batch["goal_dir"]),
            "rays": np.concatenate(batch["rays"]),
            "dynamic": np.concatenate(batch["dynamic"]),
            "actions": np.concatenate(batch["actions"]),
            "log_prob": np.concatenate(batch["log_prob"]),
            "returns": returns,
            "advantages": (advantages - advantages.mean()) /
                          (advantages.std() + 1e-8),
        }
        losses = agent.update(train_batch)

        history["return"].append(episode_return)
        history["success"].append(successes / max(1, env.n_robots))
        history["actor_loss"].append(losses["actor_loss"])
        history["critic_loss"].append(losses["critic_loss"])

    if checkpoint_path:
        import torch
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(agent.model.state_dict(), checkpoint_path)

    return history


def plot_training(history: Dict[str, List[float]], out_path: str) -> str:
    """Render the training convergence curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["return"], label="Mean return", color="#2E86C1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Episode return")
    ax2 = ax.twinx()
    ax2.plot(history["success"], label="Success rate", color="#27AE60")
    ax2.set_ylabel("Success rate")
    ax.set_title("PPO training convergence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
