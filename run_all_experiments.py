from __future__ import annotations

import argparse
from pathlib import Path

from exp1_data_quality import run_experiment as run_experiment_one
from exp2_algorithm_fairness import run_experiment as run_experiment_two

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行两个推荐系统治理实验")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="MovieLens 1M 数据目录；默认使用项目内置的 data/ml-1m",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="总输出目录；默认输出到项目内的 outputs",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-factors", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--fairness-power", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_output = Path(args.output_dir)

    exp1_args = argparse.Namespace(**vars(args))
    exp1_args.output_dir = str(root_output / "experiment1")
    run_experiment_one(exp1_args)
    print()

    exp2_args = argparse.Namespace(**vars(args))
    exp2_args.output_dir = str(root_output / "experiment2")
    run_experiment_two(exp2_args)
