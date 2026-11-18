"""Performance metrics, aggregation and reporting.

Metrics implemented (assignment requirement #3):

* Success rate - fraction of robots that reach the goal.
* Collision rate - fraction of robots that contact an obstacle/robot/boundary.
* Completion time - steps and seconds until arrival.
* Path length - cumulative travelled distance.
* Total episode reward - sum of per-step rewards.
* Minimum clearance - closest approach to any obstacle.
* Control effort - integral of squared speed.
* Smoothness / jerk - frame-to-frame command change.

Results are written as CSV tables and rendered as matplotlib figures.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .environment import MultiRobotNavEnv


@dataclass
class EpisodeRecord:
    """One robot's outcome in one evaluation episode."""

    scenario: str
    seed: int
    robot_id: int
    reached: bool
    collided: bool
    timeout: bool
    steps: int
    time_s: float
    path_length: float
    total_reward: float
    min_clearance: float
    control_effort: float
    jerk: float
    reward_curve: List[float] = field(default_factory=list)


class Recorder:
    """Collects per-step statistics while an episode runs."""

    def __init__(self, env: MultiRobotNavEnv, scenario: str, seed: int):
        self.env = env
        self.scenario = scenario
        self.seed = seed
        self.n = env.n_robots
        self.reward_sum = np.zeros(self.n)
        self.curves: List[List[float]] = [[] for _ in range(self.n)]
        self.effort = np.zeros(self.n)
        self.jerk = np.zeros(self.n)
        self.prev_v = np.zeros(self.n)
        self.prev_omega = np.zeros(self.n)
        self.finish_step = np.zeros(self.n, dtype=int)

    def step(self, rewards: np.ndarray, info: Dict[str, Any]) -> None:
        for i, robot in enumerate(self.env.robots):
            self.reward_sum[i] += float(rewards[i])
            self.curves[i].append(float(rewards[i]))
            self.effort[i] += robot.v ** 2 * self.env.dt
            self.jerk[i] += abs(robot.v - self.prev_v[i]) + \
                abs(robot.omega - self.prev_omega[i])
            self.prev_v[i] = robot.v
            self.prev_omega[i] = robot.omega
            if robot.done and self.finish_step[i] == 0:
                self.finish_step[i] = self.env.step_count + 1

    def finalize(self) -> List[EpisodeRecord]:
        records: List[EpisodeRecord] = []
        for i, robot in enumerate(self.env.robots):
            steps = int(self.finish_step[i]) or self.env.step_count
            records.append(EpisodeRecord(
                scenario=self.scenario,
                seed=self.seed,
                robot_id=i,
                reached=robot.reached,
                collided=robot.collided,
                timeout=robot.timeout,
                steps=steps,
                time_s=steps * self.env.dt,
                path_length=float(robot.path_length),
                total_reward=float(self.reward_sum[i]),
                min_clearance=float(robot.min_clearance)
                if np.isfinite(robot.min_clearance) else float(self.env.half),
                control_effort=float(self.effort[i]),
                jerk=float(self.jerk[i]),
                reward_curve=self.curves[i],
            ))
        return records


def aggregate(records: List[EpisodeRecord]) -> Dict[str, Dict[str, float]]:
    """Aggregate records per scenario.

    Returns a dictionary keyed by scenario with success/collision rates and
    mean +/- std of the continuous metrics.
    """
    by_scenario: Dict[str, List[EpisodeRecord]] = {}
    for rec in records:
        by_scenario.setdefault(rec.scenario, []).append(rec)

    summary: Dict[str, Dict[str, float]] = {}
    for scenario, items in by_scenario.items():
        n = len(items)
        reached = sum(r.reached for r in items)
        collided = sum(r.collided for r in items)
        timeout = sum(r.timeout for r in items)

        def _mean_std(key: str) -> tuple[float, float]:
            vals = np.array([getattr(r, key) for r in items], dtype=float)
            return float(vals.mean()), float(vals.std())

        time_m, time_s = _mean_std("time_s")
        path_m, path_s = _mean_std("path_length")
        reward_m, reward_s = _mean_std("total_reward")
        clear_m, clear_s = _mean_std("min_clearance")
        effort_m, effort_s = _mean_std("control_effort")
        jerk_m, jerk_s = _mean_std("jerk")

        summary[scenario] = {
            "episodes": n,
            "success_rate": reached / n,
            "collision_rate": collided / n,
            "timeout_rate": timeout / n,
            "mean_time_s": time_m, "std_time_s": time_s,
            "mean_path_m": path_m, "std_path_m": path_s,
            "mean_reward": reward_m, "std_reward": reward_s,
            "mean_clearance_m": clear_m, "std_clearance_m": clear_s,
            "mean_effort": effort_m, "std_effort": effort_s,
            "mean_jerk": jerk_m, "std_jerk": jerk_s,
        }
    return summary


# --------------------------------------------------------------------------- #
# CSV reporting
# --------------------------------------------------------------------------- #
DETAIL_FIELDS = [
    "scenario", "seed", "robot_id", "reached", "collided", "timeout",
    "steps", "time_s", "path_length", "total_reward", "min_clearance",
    "control_effort", "jerk",
]

SUMMARY_FIELDS = [
    "scenario", "episodes", "success_rate", "collision_rate", "timeout_rate",
    "mean_time_s", "std_time_s", "mean_path_m", "std_path_m",
    "mean_reward", "std_reward", "mean_clearance_m", "std_clearance_m",
    "mean_effort", "std_effort", "mean_jerk", "std_jerk",
]


def write_csv(records: List[EpisodeRecord], summary: Dict[str, Dict[str, float]],
              out_dir: str) -> tuple[str, str]:
    """Write detail and summary CSV files. Returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "episode_detail.csv")
    summary_path = os.path.join(out_dir, "scenario_summary.csv")

    with open(detail_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: getattr(rec, k) for k in DETAIL_FIELDS})

    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for scenario, row in summary.items():
            writer.writerow({"scenario": scenario, **row})

    return detail_path, summary_path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_summary(summary: Dict[str, Dict[str, float]],
                 records: List[EpisodeRecord], figures_dir: str) -> List[str]:
    """Render four comparison figures. Returns list of saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)
    scenarios = list(summary.keys())
    saved: List[str] = []
    colors = ["#2E86C1", "#E74C3C", "#27AE60", "#F39C12"]

    # Figure 1: success and collision rates.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(scenarios))
    width = 0.35
    success = [summary[s]["success_rate"] * 100 for s in scenarios]
    collision = [summary[s]["collision_rate"] * 100 for s in scenarios]
    ax.bar(x - width / 2, success, width, label="Success (%)", color=colors[0])
    ax.bar(x + width / 2, collision, width, label="Collision (%)", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Success and collision rate by scenario")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(figures_dir, "success_collision.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # Figure 2: mean reward and min clearance.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rewards = [summary[s]["mean_reward"] for s in scenarios]
    clear = [summary[s]["mean_clearance_m"] for s in scenarios]
    ax.bar(x - width / 2, rewards, width, label="Mean reward", color=colors[2])
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, clear, width, label="Mean clearance (m)", color=colors[3])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel("Mean episode reward")
    ax2.set_ylabel("Mean clearance (m)")
    ax.set_title("Reward and safety clearance by scenario")
    fig.tight_layout()
    p2 = os.path.join(figures_dir, "reward_clearance.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # Figure 3: completion time and path length.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    times = [summary[s]["mean_time_s"] for s in scenarios]
    paths = [summary[s]["mean_path_m"] for s in scenarios]
    ax.bar(x - width / 2, times, width, label="Mean time (s)", color=colors[0])
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, paths, width, label="Mean path (m)", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel("Time (s)")
    ax2.set_ylabel("Path length (m)")
    ax.set_title("Completion time and path length by scenario")
    fig.tight_layout()
    p3 = os.path.join(figures_dir, "time_path.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    # Figure 4: reward curves averaged per scenario.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, scenario in enumerate(scenarios):
        curves = [r.reward_curve for r in records if r.scenario == scenario]
        if not curves:
            continue
        length = max(len(c) for c in curves)
        padded = np.zeros((len(curves), length))
        for j, c in enumerate(curves):
            padded[j, :len(c)] = c
        mean_curve = padded.mean(axis=0)
        ax.plot(np.arange(length) * 0.1, np.cumsum(mean_curve),
                label=scenario, color=colors[idx % len(colors)])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward over time")
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(figures_dir, "reward_curves.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved.append(p4)

    return saved
