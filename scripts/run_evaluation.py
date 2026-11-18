"""Evaluate all scenarios and write CSV tables + figures.

Usage
-----
    python scripts/run_evaluation.py --episodes 30
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navrl.evaluate import evaluate_all, summary_markdown  # noqa: E402
from navrl.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NavRL scenarios")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    _, summary = evaluate_all(
        config=config, episodes=args.episodes, seed_base=args.seed_base,
        out_dir=args.out,
    )
    print(summary_markdown(summary))


if __name__ == "__main__":
    main()
