"""Shared helpers: configuration loading, seeding and small math utilities.

This module has only NumPy as a dependency so the environment, reward and
metrics code can run without PyTorch installed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "scenarios.json")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the scenario configuration JSON.

    Parameters
    ----------
    path:
        Optional path to a config file. Falls back to ``configs/scenarios.json``.
    """
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def set_global_seed(seed: int) -> np.random.Generator:
    """Return a NumPy random generator seeded with ``seed``."""
    return np.random.default_rng(seed)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def scenario_names(config: Dict[str, Any]) -> list[str]:
    """Return the ordered list of scenario keys."""
    return list(config["scenarios"].keys())
