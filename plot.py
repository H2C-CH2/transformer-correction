import argparse
from pathlib import Path

from TLCM.graph import plot_cossims
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

    parser.add_argument("--paths", action="extend", nargs="+")
    args = parser.parse_args()
    # assert figure and experiment align

    return args


def search_folders(exp: str):
    exp_dir = Path("data") / exp

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    paths = sorted(exp_dir.rglob("*.pt"))

    if not paths:
        raise FileNotFoundError(f"No .pt files found in {exp_dir}")

    return [load_results(path, "cpu") for path in paths]


def main():
    args = parse_args()

    if args.paths is None:
        results = search_folders(args.exp)

    else:
        results = [load_results(path, "cpu") for path in args.paths]

    match args.exp:
        case "4.1":
            # r["data"]: Float[Tensor, "n_layers n_layers"]
            plot_cossims(
                [r["data"] for r in results],
                [r["metadata"]["cfg"] for r in results],
                clamp=0.2,
                file_name="fig1",
            )

        case "4.2":
            # r["data"]: Float[Tensor, "n_layers n_layers"]
            plot_cossims(
                [r["data"] for r in results],
                [r["metadata"]["cfg"] for r in results],
                clamp=0.2,
                file_name="figA9",
            )


if __name__ == "__main__":
    main()
