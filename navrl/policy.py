"""Navigation policies.

Two policies are provided:

* :class:`APFPolicy` - an Artificial Potential Field controller with a
  clearance-aware repulsive term. This matches the "APF nav ·
  clearance-aware" baseline shown in the submitted recordings and lets the
  simulator produce evaluation episodes without a trained neural network.
* :class:`RandomPolicy` - uniform random actions, used as a sanity baseline
  and in tests.

The learned PPO policy lives in :mod:`navrl.network` and can be plugged into
the same interface after training.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .environment import MultiRobotNavEnv
from .utils import clip, set_global_seed, wrap_angle


class BasePolicy:
    """Common policy interface."""

    def act(self, obs: Dict[str, np.ndarray], env: MultiRobotNavEnv,
            robot_idx: int) -> np.ndarray:
        raise NotImplementedError


class APFPolicy(BasePolicy):
    """Clearance-aware Artificial Potential Field navigation."""

    def __init__(self, attractive: float = 1.5, repulsive: float = 6.0,
                 influence_radius: float = 4.0, k_omega: float = 2.0):
        self.attractive = attractive
        self.repulsive = repulsive
        self.influence_radius = influence_radius
        self.k_omega = k_omega

    def act(self, obs: Dict[str, np.ndarray], env: MultiRobotNavEnv,
            robot_idx: int) -> np.ndarray:
        robot = env.robots[robot_idx]
        to_goal = robot.goal - robot.pos
        dist = float(np.linalg.norm(to_goal))
        if dist < 1e-6:
            return np.zeros(2)
        attractive_force = self.attractive * to_goal / dist

        repulsive_force = np.zeros(2)
        for obs_item in env.obstacles:
            rel = robot.pos - obs_item.pos
            d = float(np.linalg.norm(rel)) - obs_item.radius
            if d < self.influence_radius and d > 1e-3:
                # Clearance-aware: repulsion strengthens inside the safety margin.
                weight = self.repulsive * (1.0 / d - 1.0 / self.influence_radius) / (d ** 2)
                if d < env.safe_distance:
                    weight *= 1.8
                repulsive_force += weight * rel / (d + obs_item.radius)
        for j, other in enumerate(env.robots):
            if j == robot_idx:
                continue
            rel = robot.pos - other.pos
            d = float(np.linalg.norm(rel)) - other.radius
            if d < self.influence_radius and d > 1e-3:
                weight = self.repulsive * (1.0 / d - 1.0 / self.influence_radius) / (d ** 2)
                repulsive_force += weight * rel / (d + other.radius)

        force = attractive_force + repulsive_force
        heading = np.arctan2(force[1], force[0])
        heading_error = wrap_angle(heading - robot.theta)

        speed = env.v_lim * clip(dist / 3.0, 0.15, 1.0)
        if np.linalg.norm(repulsive_force) > np.linalg.norm(attractive_force):
            speed *= 0.5
        omega = clip(self.k_omega * heading_error, -env.omega_lim, env.omega_lim)
        return np.array([speed, omega])


class RandomPolicy(BasePolicy):
    """Uniform random action baseline."""

    def __init__(self, seed: int = 0):
        self.rng = set_global_seed(seed)

    def act(self, obs: Dict[str, np.ndarray], env: MultiRobotNavEnv,
            robot_idx: int) -> np.ndarray:
        v = self.rng.uniform(-env.v_lim, env.v_lim)
        omega = self.rng.uniform(-env.omega_lim, env.omega_lim)
        return np.array([v, omega])
