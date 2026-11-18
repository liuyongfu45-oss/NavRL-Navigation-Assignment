"""Scenario layout construction.

The assignment requires three navigation scenarios:

* **S1 Dynamic**  - robots move among moving obstacles.
* **S2 Static**   - robots move among stationary obstacles.
* **S3 Mixed**    - stationary and moving obstacles together.

A fourth ``assignment_multi_robot`` variant implements the literal assignment
statement: six robots and fifteen obstacles in a mixed layout.

All layouts are generated deterministically from a seed so that evaluation
results are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import set_global_seed


@dataclass
class Obstacle:
    """Runtime description of one circular obstacle."""

    pos: np.ndarray
    vel: np.ndarray
    radius: float
    kind: str  # "static" or "dynamic"


@dataclass
class ScenarioLayout:
    """Resolved layout for one scenario."""

    key: str
    title: str
    starts: np.ndarray
    goals: np.ndarray
    obstacles: List[Obstacle] = field(default_factory=list)
    video: Optional[str] = None

    @property
    def n_robots(self) -> int:
        return len(self.starts)

    @property
    def n_static(self) -> int:
        return sum(o.kind == "static" for o in self.obstacles)

    @property
    def n_dynamic(self) -> int:
        return sum(o.kind == "dynamic" for o in self.obstacles)


def _ring_positions(n: int, radius: float, angle_offset: float) -> np.ndarray:
    """Place ``n`` points evenly on a circle of given radius."""
    angles = angle_offset + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _generate_starts_goals(n_robots: int, rng: np.random.Generator,
                           half: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate opposite-side start/goal pairs for the multi-robot variant."""
    starts = _ring_positions(n_robots, radius=half * 0.78, angle_offset=np.pi)
    goals = _ring_positions(n_robots, radius=half * 0.78, angle_offset=0.0)
    # Small deterministic jitter so robots do not share an exact bearing.
    starts += rng.uniform(-1.2, 1.2, starts.shape)
    goals += rng.uniform(-1.2, 1.2, goals.shape)
    starts = np.clip(starts, -half + 1.0, half - 1.0)
    goals = np.clip(goals, -half + 1.0, half - 1.0)
    return starts, goals


def _place_obstacles(rng: np.random.Generator, n_static: int, n_dynamic: int,
                     radius: float, speed: float, half: float,
                     protected: np.ndarray) -> List[Obstacle]:
    """Randomly place obstacles away from robot starts and goals."""
    obstacles: List[Obstacle] = []
    attempts = 0
    target = n_static + n_dynamic

    def _far_enough(pos: np.ndarray) -> bool:
        if np.any(np.abs(pos) > half - radius - 0.3):
            return False
        for p in protected:
            if np.linalg.norm(pos - p) < 2.0 + radius:
                return False
        for o in obstacles:
            if np.linalg.norm(pos - o.pos) < o.radius + radius + 0.6:
                return False
        return True

    while len(obstacles) < target and attempts < 2000:
        attempts += 1
        pos = rng.uniform(-half + radius + 0.3, half - radius - 0.3, size=2)
        if not _far_enough(pos):
            continue
        idx = len(obstacles)
        if idx < n_static:
            obstacles.append(Obstacle(pos=pos, vel=np.zeros(2), radius=radius,
                                      kind="static"))
        else:
            angle = rng.uniform(0, 2 * np.pi)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            obstacles.append(Obstacle(pos=pos, vel=vel, radius=radius,
                                       kind="dynamic"))
    return obstacles


def build_layout(config: Dict[str, Any], scenario_key: str,
                 seed: int = 0) -> ScenarioLayout:
    """Build a :class:`ScenarioLayout` from the global configuration.

    Parameters
    ----------
    config:
        Parsed configuration dictionary.
    scenario_key:
        One of the keys under ``config["scenarios"]``.
    seed:
        Seed for deterministic random layout generation.
    """
    if scenario_key not in config["scenarios"]:
        raise KeyError(f"Unknown scenario '{scenario_key}'. "
                       f"Available: {list(config['scenarios'])}")

    scfg = config["scenarios"][scenario_key]
    world = config["world"]
    half = float(world["half"])
    rng = set_global_seed(seed)

    n_robots = int(scfg["robots"])
    if scfg.get("starts") is not None:
        starts = np.asarray(scfg["starts"], dtype=float)
        goals = np.asarray(scfg["goals"], dtype=float)
    else:
        starts, goals = _generate_starts_goals(n_robots, rng, half)

    protected = np.vstack([starts, goals])
    obstacles = _place_obstacles(
        rng,
        n_static=int(scfg["static_obstacles"]),
        n_dynamic=int(scfg["dynamic_obstacles"]),
        radius=float(scfg["obstacle_radius"]),
        speed=float(scfg["obstacle_speed"]),
        half=half,
        protected=protected,
    )

    return ScenarioLayout(
        key=scenario_key,
        title=str(scfg["title"]),
        starts=starts,
        goals=goals,
        obstacles=obstacles,
        video=scfg.get("video"),
    )
