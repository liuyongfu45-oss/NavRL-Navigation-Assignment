"""Reward function design for NavRL.

The reward combines two mechanisms:

1. **Terminal events** - fixed one-off rewards for reaching the goal,
   colliding, or timing out.
2. **Per-step shaping** - continuous dense feedback for goal progress,
   static clearance, dynamic clearance, motion smoothness and anti-stall.

The total reward at step ``t`` is::

    r_t = r_progress + r_static + r_dynamic + r_smooth + r_stall
          + r_goal + r_collision + r_timeout

The :data:`REWARD_TABLE` metadata documents every positive and negative term
and is reused by the README and the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:  # avoid circular import with environment.py
    from .environment import RobotState


@dataclass
class RewardConfig:
    """Coefficients of the reward function."""

    goal_bonus: float = 10.0
    collision_penalty: float = -10.0
    timeout_penalty: float = -10.0
    progress_weight: float = 1.0
    static_safe_reward: float = 0.15
    static_close_penalty: float = 0.6
    dynamic_safe_reward: float = 0.15
    dynamic_close_penalty: float = 0.6
    smoothness_penalty: float = 0.05
    stall_penalty: float = 0.10
    stall_speed: float = 0.05

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Reward table metadata (assignment requirement #2)
# --------------------------------------------------------------------------- #
REWARD_TABLE: list[Dict[str, str]] = [
    {
        "term": "Goal-reaching terminal reward",
        "sign": "Positive",
        "condition": "distance to goal < arrival threshold (0.4 m)",
        "formula": "+goal_bonus",
        "magnitude": "+10.0",
        "behavior": "Arrive and stop at the goal; end the episode successfully.",
    },
    {
        "term": "Collision penalty (terminal)",
        "sign": "Negative",
        "condition": "Body overlaps an obstacle, another robot, or leaves the valid region",
        "formula": "+collision_penalty",
        "magnitude": "-10.0",
        "behavior": "Avoid contact with obstacles, other robots and boundaries.",
    },
    {
        "term": "Timeout penalty (terminal)",
        "sign": "Negative",
        "condition": "step count reaches max_steps without reaching the goal",
        "formula": "+timeout_penalty",
        "magnitude": "-10.0",
        "behavior": "Discourage waiting, stalling and indefinite wandering.",
    },
    {
        "term": "Goal progress / velocity alignment",
        "sign": "Positive when closer, negative when farther",
        "condition": "Per-step change of goal distance",
        "formula": "k_p * (d_prev - d_t)",
        "magnitude": "k_p = 1.0",
        "behavior": "Move toward the goal and align heading with the target.",
    },
    {
        "term": "Static safety reward",
        "sign": "Positive when far, negative when close",
        "condition": "Minimum clearance to stationary obstacles",
        "formula": "+k_safe if d >= d_safe; else -k_close * (d_safe - d)/d_safe",
        "magnitude": "+0.15 / -0.60 per unit intrusion",
        "behavior": "Keep clearance from walls and stationary obstacles.",
    },
    {
        "term": "Dynamic safety reward",
        "sign": "Positive when far, negative when close",
        "condition": "Minimum clearance to moving obstacles and other robots",
        "formula": "+k_safe if d >= d_safe; else -k_close * (d_safe - d)/d_safe",
        "magnitude": "+0.15 / -0.60 per unit intrusion",
        "behavior": "Maintain clearance from moving robots or people.",
    },
    {
        "term": "Smoothness reward",
        "sign": "Negative",
        "condition": "Frame-to-frame change of commanded velocity",
        "formula": "-k_m * (|v_t - v_{t-1}| + |omega_t - omega_{t-1}|)",
        "magnitude": "k_m = 0.05",
        "behavior": "Discourage jerky, abrupt command changes; low-jerk motion.",
    },
    {
        "term": "Anti-stall penalty",
        "sign": "Negative",
        "condition": "Far from goal but commanded speed near zero",
        "formula": "-stall_penalty when |v| < stall_speed and d > arrive_radius",
        "magnitude": "-0.10 per stalled step",
        "behavior": "Prevent the freezing failure observed beside a goal.",
    },
]


def _safety_term(clearance: float, safe_distance: float, safe_reward: float,
                 close_penalty: float) -> float:
    if not np.isfinite(clearance):
        return safe_reward  # no obstacle nearby
    if clearance >= safe_distance:
        return safe_reward
    return -close_penalty * (safe_distance - clearance) / safe_distance


def compute_reward(robot: RobotState, prev_dist: float, dist: float,
                   static_clearance: float, dynamic_clearance: float,
                   collided: bool, reached: bool, timeout: bool,
                   cfg: RewardConfig, dt: float) -> tuple[float, Dict[str, float]]:
    """Compute the per-step reward for one robot.

    Returns
    -------
    (total_reward, components)
        ``components`` keeps every term separately so it can be logged,
        plotted and inspected by the tests.
    """
    progress = cfg.progress_weight * (prev_dist - dist)

    static = _safety_term(static_clearance, robot.radius + 0.8,
                          cfg.static_safe_reward, cfg.static_close_penalty)
    dynamic = _safety_term(dynamic_clearance, robot.radius + 0.8,
                           cfg.dynamic_safe_reward, cfg.dynamic_close_penalty)

    smoothness = -cfg.smoothness_penalty * (
        abs(robot.v - robot.last_v) + abs(robot.omega - robot.last_omega)
    )

    stalled = (not reached) and dist > robot.radius + 0.8 and abs(robot.v) < cfg.stall_speed
    stall = -cfg.stall_penalty if stalled else 0.0

    goal = cfg.goal_bonus if reached else 0.0
    collision = cfg.collision_penalty if collided else 0.0
    timeout_pen = cfg.timeout_penalty if timeout and not reached and not collided else 0.0

    components: Dict[str, float] = {
        "progress": progress,
        "static_safety": static,
        "dynamic_safety": dynamic,
        "smoothness": smoothness,
        "stall": stall,
        "goal": goal,
        "collision": collision,
        "timeout": timeout_pen,
    }
    # Terminal events override the per-step shaping: the episode ends and the
    # agent receives the terminal reward only (standard if/else terminal rule).
    if reached:
        total = goal
    elif collided:
        total = collision
    elif timeout:
        total = timeout_pen
    else:
        total = float(progress + static + dynamic + smoothness + stall)
    return float(total), components
