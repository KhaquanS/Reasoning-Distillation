import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_eval.config import load_config
from custom_eval.runner import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Run YAML-driven benchmark evaluations.")
    parser.add_argument("--config", required=True, help="Path to an evaluation YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_evaluation(config)


if __name__ == "__main__":
    main()

