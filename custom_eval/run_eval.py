"""
Main entry point for running evaluations.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_eval.config import load_config
from custom_eval.runner import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YAML-driven benchmark evaluations with Qwen support."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Override max samples from config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    # Override from command line
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    
    print(f"Loading configuration from: {args.config}")
    print(f"Models: {[m.name for m in config.models]}")
    print(f"Benchmarks: {config.benchmarks}")
    print(f"Output directory: {config.output_dir}")
    print(f"Max samples: {config.max_samples or 'all'}")
    print()
    
    run_evaluation(config)


if __name__ == "__main__":
    main()