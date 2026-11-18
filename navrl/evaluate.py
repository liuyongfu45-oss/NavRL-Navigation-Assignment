"""Evaluation runner: roll out every scenario, aggregate metrics, export
CSV tables and figures.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .environment import MultiRobotNavEnv
from .metrics import EpisodeRecord, Recorder, aggregate, plot_summary, write_csv
from .policy import APFPolicy, BasePolicy
from .utils import REPO_ROOT, load_config, scenario_names


def run_episode(env: MultiRobotNavEnv, policy: BasePolicy,
                seed: int) -> List[EpisodeRecord]:
    """Run one episode and return per-robot records."""
    recorder = Recorder(env, env.scenario_key, seed)
    obs = env.reset()
    while not env.all_done():
        actions = np.array([
            policy.act(obs, env, i) for i in range(env.n_robots)
        ])
        obs, rewards, _, info = env.step(actions)
        recorder.step(rewards, info)
    return recorder.finalize()


def evaluate_all(config: Optional[Dict[str, Any]] = None, episodes: int = 30,
                 seed_base: int = 0, policy: Optional[BasePolicy] = None,
                 out_dir: Optional[str] = None) -> tuple[List[EpisodeRecord],
                                                          Dict[str, Dict[str, float]]]:
    """Evaluate all configured scenarios and write result artifacts."""
    config = config or load_config()
    policy = policy or APFPolicy()
    out_dir = out_dir or os.path.join(REPO_ROOT, "results")
    figures_dir = os.path.join(out_dir, "figures")

    records: List[EpisodeRecord] = []
    for key in scenario_names(config):
        for ep in range(episodes):
            env = MultiRobotNavEnv(config, key, seed=seed_base + ep)
            records.extend(run_episode(env, policy, seed_base + ep))

    summary = aggregate(records)
    write_csv(records, summary, out_dir)
    plot_summary(summary, records, figures_dir)
    return records, summary


def summary_markdown(summary: Dict[str, Dict[str, float]]) -> str:
    """Render the aggregate summary as a Markdown table."""
    headers = ["Scenario", "Episodes", "Success %", "Collision %",
                "Time (s)", "Path (m)", "Reward", "Clearance (m)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for scenario, row in summary.items():
        lines.append("| " + " | ".join([
            scenario,
            str(int(row["episodes"])),
            f"{row['success_rate'] * 100:.1f}",
            f"{row['collision_rate'] * 100:.1f}",
            f"{row['mean_time_s']:.2f} ± {row['std_time_s']:.2f}",
            f"{row['mean_path_m']:.2f} ± {row['std_path_m']:.2f}",
            f"{row['mean_reward']:.2f} ± {row['std_reward']:.2f}",
            f"{row['mean_clearance_m']:.2f}",
        ]) + " |")
    return "\n".join(lines)
