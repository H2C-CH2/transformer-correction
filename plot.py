import argparse
from pathlib import Path

from TLCM.graph import plot_fig_1_A5
from TLCM.utils import load_results


def parse_args():
    parser = argparse.ArgumentParser(description="Plot graphs for experiments")
    # Plot by figure
    # parser.add_argument("--figure", type=str, choices=[""], help="")

    # Plot by experiment
    parser.add_argument(
        "--exp",
        type=str,
        choices=["4.1", "4.2", "4.3", "4.4", "5.1", "5.2"],
        help="Experiment performed in paper",
    )

    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()
    # assert figure and experiment align

    return args


def main():
    args = parse_args()

    if args.path is None:
        exp_dir = Path("data") / args.exp

        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        paths = sorted(exp_dir.rglob("*.pt"))

        if not paths:
            raise FileNotFoundError(f"No .pt files found in {exp_dir}")

        results = [load_results(path, "cpu") for path in paths]

    else:
        results = [load_results(args.path, "cpu")]

    match args.exp:
        case "4.1":
            plot_fig_1_A5(
                [r["data"] for r in results],
                [r["metadata"]["cfg"] for r in results],
                clamp=0.2,
                file_name="exp1",
            )


if __name__ == "__main__":
    main()
