"""NavRL - multi-robot navigation reinforcement learning assignment.

Package layout
--------------
* :mod:`navrl.environment` - 2-D multi-robot navigation simulator.
* :mod:`navrl.reward` - reward function and positive/negative reward table.
* :mod:`navrl.scenarios` - S1 dynamic, S2 static, S3 mixed layouts.
* :mod:`navrl.policy` - APF clearance-aware baseline and random policy.
* :mod:`navrl.network` - actor-critic network and PPO agent (PyTorch).
* :mod:`navrl.metrics` - evaluation metrics, CSV export and figures.
* :mod:`navrl.evaluate` - roll out every scenario and summarise results.
* :mod:`navrl.train` - PPO training loop.
"""

__version__ = "1.0.0"

from .environment import MultiRobotNavEnv, RobotState
from .reward import REWARD_TABLE, RewardConfig, compute_reward
from .scenarios import ScenarioLayout, build_layout
from .utils import load_config

__all__ = [
    "MultiRobotNavEnv", "RobotState", "REWARD_TABLE", "RewardConfig",
    "compute_reward", "ScenarioLayout", "build_layout", "load_config",
]
