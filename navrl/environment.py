"""Two-dimensional multi-robot navigation environment.

The environment implements the Markov decision process used throughout the
NavRL assignment:

* State  - robot pose/velocity, relative goal and obstacle information.
* Action - bounded linear velocity ``v`` and angular velocity ``omega``.
* Reward - computed by :mod:`navrl.reward`.
* Done   - goal reached, collision, leaving the world, or timeout.

The simulator is intentionally lightweight (NumPy only) so it runs on CPU and
on machines without PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .reward import RewardConfig, compute_reward
from .scenarios import Obstacle, ScenarioLayout, build_layout
from .utils import clip, set_global_seed, wrap_angle


@dataclass
class RobotState:
    """Mutable state of one robot."""

    pos: np.ndarray
    goal: np.ndarray
    theta: float
    v: float = 0.0
    omega: float = 0.0
    radius: float = 0.35
    reached: bool = False
    collided: bool = False
    timeout: bool = False
    done: bool = False
    prev_dist: float = 0.0
    min_clearance: float = np.inf
    path_length: float = 0.0
    last_v: float = 0.0
    last_omega: float = 0.0


class MultiRobotNavEnv:
    """Multi-robot navigation environment for one configured scenario."""

    def __init__(self, config: Dict[str, Any], scenario_key: str,
                 seed: int = 0):
        self.config = config
        self.scenario_key = scenario_key
        self.seed = seed
        self.rng = set_global_seed(seed)

        world = config["world"]
        self.half = float(world["half"])
        self.dt = float(world["dt"])
        self.max_steps = int(world["max_steps"])

        rob = config["robot"]
        self.robot_radius = float(rob["radius"])
        self.v_lim = float(rob["v_lim"])
        self.omega_lim = float(rob["omega_lim"])
        self.arrive_radius = float(rob["arrive_radius"])
        self.safe_distance = float(rob["safe_distance"])

        obs = config["observation"]
        self.n_rays = int(obs["n_rays"])
        self.ray_bins = int(obs["ray_bins"])
        self.n_nearest_dynamic = int(obs["n_nearest_dynamic"])
        self.dynamic_features = int(obs["dynamic_features"])

        self.reward_cfg = RewardConfig.from_dict(config["reward"])
        self.layout: ScenarioLayout = build_layout(config, scenario_key, seed)

        self.robots: List[RobotState] = []
        self.obstacles: List[Obstacle] = []
        self.step_count = 0
        self.trajectories: List[np.ndarray] = []
        self._reset_robots()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _reset_robots(self) -> None:
        self.robots = []
        for start, goal in zip(self.layout.starts, self.layout.goals):
            direction = np.arctan2(goal[1] - start[1], goal[0] - start[0])
            dist = float(np.linalg.norm(goal - start))
            self.robots.append(
                RobotState(
                    pos=np.asarray(start, dtype=float).copy(),
                    goal=np.asarray(goal, dtype=float).copy(),
                    theta=float(direction),
                    radius=self.robot_radius,
                    prev_dist=dist,
                    min_clearance=np.inf,
                )
            )
        self.obstacles = [
            Obstacle(pos=o.pos.copy(), vel=o.vel.copy(),
                     radius=o.radius, kind=o.kind)
            for o in self.layout.obstacles
        ]
        self.step_count = 0
        self.trajectories = [r.pos.copy() for r in self.robots]

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the episode and return the initial observation."""
        self.rng = set_global_seed(self.seed)
        self._reset_robots()
        return self.observe()

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def n_robots(self) -> int:
        return len(self.robots)

    def _static_obstacles(self) -> List[Obstacle]:
        return [o for o in self.obstacles if o.kind == "static"]

    def _dynamic_obstacles(self, robot_idx: int) -> List[Tuple[Obstacle, np.ndarray]]:
        """Dynamic obstacles plus other robots (treated as moving bodies)."""
        moving: List[Tuple[Obstacle, np.ndarray]] = [
            (o, o.vel.copy()) for o in self.obstacles if o.kind == "dynamic"
        ]
        for j, other in enumerate(self.robots):
            if j != robot_idx:
                moving.append((
                    Obstacle(pos=other.pos.copy(), vel=np.array([other.v * np.cos(other.theta),
                                                                  other.v * np.sin(other.theta)]),
                             radius=other.radius, kind="robot"),
                    np.array([other.v * np.cos(other.theta),
                              other.v * np.sin(other.theta)]),
                ))
        return moving

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _ray_scan(self, robot: RobotState) -> np.ndarray:
        """Static obstacle ray scan, shape ``(n_rays, ray_bins)``."""
        rays = np.zeros((self.n_rays, self.ray_bins), dtype=float)
        bearings = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)
        static = self._static_obstacles()

        for i, bearing in enumerate(bearings):
            angle = robot.theta + bearing
            direction = np.array([np.cos(angle), np.sin(angle)])
            nearest = self.half * 2.0
            hit = 0.0
            for obs in static:
                rel = obs.pos - robot.pos
                proj = float(np.dot(rel, direction))
                if proj <= 0:
                    continue
                perp = abs(float(direction[0] * rel[1] - direction[1] * rel[0]))
                if perp <= obs.radius:
                    dist = proj - np.sqrt(max(obs.radius ** 2 - perp ** 2, 0.0))
                    if dist < nearest:
                        nearest = dist
                        hit = 1.0
            dist_norm = clip(nearest / (2 * self.half), 0.0, 1.0)
            # Four bins: normalized distance, hit flag, relative bearing,
            # clearance margin to the safety distance.
            margin = clip((nearest - self.safe_distance) / self.half, -1.0, 1.0)
            rays[i] = [dist_norm, hit, float(bearing), margin]
        return rays

    def _dynamic_features(self, robot_idx: int, robot: RobotState) -> np.ndarray:
        """Nearest moving obstacles, shape ``(n_nearest_dynamic, F)``."""
        moving = self._dynamic_obstacles(robot_idx)
        entries: List[Tuple[float, np.ndarray]] = []
        for obs, vel in moving:
            rel = obs.pos - robot.pos
            dist = float(np.linalg.norm(rel))
            entries.append((dist, np.concatenate([
                rel, vel, [dist, np.arctan2(rel[1], rel[0]), obs.radius, 0.0, 0.0, 1.0]
            ])))
        entries.sort(key=lambda item: item[0])

        matrix = np.zeros((self.n_nearest_dynamic, self.dynamic_features), dtype=float)
        robot_vel = np.array([robot.v * np.cos(robot.theta),
                              robot.v * np.sin(robot.theta)])
        for slot, (_, feat) in enumerate(entries[:self.n_nearest_dynamic]):
            rel_v = feat[2:4] - robot_vel
            feat[7] = float(np.linalg.norm(rel_v))  # relative speed
            # Time-to-collision along the relative bearing (rough estimate).
            rel_pos = feat[0:2]
            denom = float(np.dot(rel_pos, rel_v))
            ttc = -rel_pos.dot(rel_pos) / denom if denom < -1e-6 else 10.0
            feat[8] = clip(ttc, 0.0, 10.0)
            matrix[slot] = feat
        return matrix

    def observe(self) -> Dict[str, np.ndarray]:
        """Return the stacked observation for every robot."""
        internal = np.zeros((self.n_robots, 8), dtype=float)
        goal_dir = np.zeros((self.n_robots, 3), dtype=float)
        rays = np.zeros((self.n_robots, self.n_rays, self.ray_bins), dtype=float)
        dyn = np.zeros((self.n_robots, self.n_nearest_dynamic,
                        self.dynamic_features), dtype=float)

        for i, robot in enumerate(self.robots):
            rel = robot.goal - robot.pos
            dist = float(np.linalg.norm(rel))
            bearing = wrap_angle(np.arctan2(rel[1], rel[0]) - robot.theta)
            unit = rel / dist if dist > 1e-6 else np.zeros(2)
            internal[i] = [
                np.cos(robot.theta), np.sin(robot.theta),
                robot.v, robot.omega, dist,
                float(np.cos(bearing)), float(np.sin(bearing)), robot.radius,
            ]
            goal_dir[i] = [unit[0], unit[1], dist]
            rays[i] = self._ray_scan(robot)
            dyn[i] = self._dynamic_features(i, robot)

        flat = np.concatenate([
            internal, goal_dir,
            rays.reshape(self.n_robots, -1),
            dyn.reshape(self.n_robots, -1),
        ], axis=1)
        return {"internal": internal, "goal_dir": goal_dir, "rays": rays,
                "dynamic": dyn, "flat": flat}

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def _integrate(self, robot: RobotState, action: np.ndarray) -> None:
        v = clip(float(action[0]), -self.v_lim, self.v_lim)
        omega = clip(float(action[1]), -self.omega_lim, self.omega_lim)
        robot.last_v, robot.last_omega = robot.v, robot.omega
        robot.v, robot.omega = v, omega
        robot.theta = wrap_angle(robot.theta + omega * self.dt)
        step = v * self.dt * np.array([np.cos(robot.theta), np.sin(robot.theta)])
        robot.pos = robot.pos + step
        robot.path_length += float(np.linalg.norm(step))

    def _check_collision(self, robot_idx: int, robot: RobotState) -> bool:
        for obs in self.obstacles:
            if np.linalg.norm(robot.pos - obs.pos) < robot.radius + obs.radius - 0.02:
                return True
        for j, other in enumerate(self.robots):
            if j != robot_idx and np.linalg.norm(robot.pos - other.pos) < \
                    robot.radius + other.radius - 0.02:
                return True
        if np.any(np.abs(robot.pos) > self.half - robot.radius):
            return True
        return False

    def _update_dynamic_obstacles(self) -> None:
        for obs in self.obstacles:
            if obs.kind != "dynamic":
                continue
            obs.pos = obs.pos + obs.vel * self.dt
            for axis in (0, 1):
                if obs.pos[axis] > self.half - obs.radius:
                    obs.pos[axis] = self.half - obs.radius
                    obs.vel[axis] *= -1
                elif obs.pos[axis] < -self.half + obs.radius:
                    obs.pos[axis] = -self.half + obs.radius
                    obs.vel[axis] *= -1

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray],
                                                  np.ndarray, np.ndarray, Dict[str, Any]]:
        """Advance the simulation one decision step.

        Parameters
        ----------
        actions:
            Array of shape ``(n_robots, 2)`` with ``[v, omega]`` per robot.

        Returns
        -------
        observations, rewards, dones, info
        """
        actions = np.asarray(actions, dtype=float)
        rewards = np.zeros(self.n_robots, dtype=float)
        dones = np.zeros(self.n_robots, dtype=bool)
        info: Dict[str, Any] = {"robots": []}

        for i, robot in enumerate(self.robots):
            if robot.done:
                continue
            self._integrate(robot, actions[i])

            dist = float(np.linalg.norm(robot.goal - robot.pos))
            static_c, dynamic_c = self._clearances(robot)
            robot.min_clearance = min(robot.min_clearance, static_c, dynamic_c)

            reached = dist <= self.arrive_radius
            collided = self._check_collision(i, robot)
            timeout = self.step_count + 1 >= self.max_steps

            reward, components = compute_reward(
                robot=robot,
                prev_dist=robot.prev_dist,
                dist=dist,
                static_clearance=static_c,
                dynamic_clearance=dynamic_c,
                collided=collided,
                reached=reached,
                timeout=timeout,
                cfg=self.reward_cfg,
                dt=self.dt,
            )
            rewards[i] = reward
            robot.prev_dist = dist

            if reached:
                robot.reached = True
                robot.done = True
            elif collided:
                robot.collided = True
                robot.done = True
            elif timeout:
                robot.timeout = True
                robot.done = True
            dones[i] = robot.done

            info["robots"].append({
                "reached": robot.reached,
                "collided": robot.collided,
                "timeout": robot.timeout,
                "dist": dist,
                "min_clearance": robot.min_clearance,
                "components": components,
            })

        self._update_dynamic_obstacles()
        self.step_count += 1
        self.trajectories.append(np.array([r.pos.copy() for r in self.robots]))

        obs = self.observe()
        info["step"] = self.step_count
        info["time"] = self.step_count * self.dt
        info["all_done"] = bool(np.all([r.done for r in self.robots]))
        return obs, rewards, dones, info

    def _clearances(self, robot: RobotState) -> Tuple[float, float]:
        """Return (static_clearance, dynamic_clearance) for one robot.

        Other robots are counted as dynamic moving bodies.
        """
        static_clearance = np.inf
        dynamic_clearance = np.inf
        for obs in self.obstacles:
            d = float(np.linalg.norm(robot.pos - obs.pos)) - obs.radius
            if obs.kind == "static":
                static_clearance = min(static_clearance, d)
            else:
                dynamic_clearance = min(dynamic_clearance, d)
        for j, other in enumerate(self.robots):
            if other is not robot:
                d = float(np.linalg.norm(robot.pos - other.pos)) - other.radius
                dynamic_clearance = min(dynamic_clearance, d)
        return static_clearance, dynamic_clearance

    def _min_clearance(self, robot: RobotState) -> float:
        static_c, dynamic_c = self._clearances(robot)
        return min(static_c, dynamic_c)

    def all_done(self) -> bool:
        return bool(np.all([r.done for r in self.robots]))
