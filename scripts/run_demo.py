"""Run one short demonstration episode per scenario with the APF baseline.

Usage
-----
    python scripts/run_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navrl.environment import MultiRobotNavEnv  # noqa: E402
from navrl.policy import APFPolicy  # noqa: E402
from navrl.utils import load_config, scenario_names  # noqa: E402


def main() -> None:
    config = load_config()
    policy = APFPolicy()
    for key in scenario_names(config):
        env = MultiRobotNavEnv(config, key, seed=0)
        obs = env.reset()
        while not env.all_done():
            actions = np.array([policy.act(obs, env, i)
                                for i in range(env.n_robots)])
            obs, rewards, _, info = env.step(actions)
        outcomes = ["reached" if r.reached else
                    "collided" if r.collided else "timeout"
                    for r in env.robots]
        print(f"{key}: {outcomes}")


if __name__ == "__main__":
    main()
