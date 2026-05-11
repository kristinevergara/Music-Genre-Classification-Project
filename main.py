"""
main.py
Runs the full pipeline in order.

Usage:
    python main.py
    python main.py --data_dir ./data/genres_original
"""

import sys
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full genre classification pipeline.")
    parser.add_argument("--data_dir",    type=str, default="./data/genres_original")
    parser.add_argument("--out_dir",     type=str, default="./data")
    parser.add_argument("--models_dir",  type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    steps = [
        [sys.executable, "data_preparation.py", "--data_dir", args.data_dir, "--out_dir", args.out_dir],
        [sys.executable, "training_module.py",  "--data_dir", args.out_dir,  "--models_dir", args.models_dir],
        [sys.executable, "evaluation.py",       "--data_dir", args.out_dir,  "--models_dir", args.models_dir, "--out_dir", args.results_dir],
    ]

    for step in steps:
        print(f"\n{'='*50}")
        print(f"Running: {' '.join(step[1:])}")
        print('='*50)
        result = subprocess.run(step)
        if result.returncode != 0:
            print(f"\nFailed at: {step[1]}")
            sys.exit(result.returncode)

    print("\nPipeline complete.")
