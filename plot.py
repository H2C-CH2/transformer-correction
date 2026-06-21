import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TLCM"))

from graph import plot_4_1
from utils import load_results


def parse_args():
    parser = argparse.ArgumentParser(description="Plot graphs for experiments")
    # Plot by figure
    # parser.add_argument("--figure", type=str, choices=[""], help="")

    # Plot by experiment
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["4.1", "4.2", "4.3", "4.4", "5.1", "5.2"],
        help="Experiment performed in paper",
    )

    parser.add_argument(
        "--path",
        type=Path,
    )
    args = parser.parse_args()
    # assert figure and experiment align

    return args


def main():
    args = parse_args()

    res = load_results(args.path, "cpu")

    match args.experiment:
        case "4.1":
            plot_4_1(res["data"], res["metadata"]["cfg"], clamp=0.2)


if __name__ == "__main__":
    main()
