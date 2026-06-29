import math
from pathlib import Path

import matplotlib.pyplot as plt
from jaxtyping import Float
from torch import Tensor

from TLCM.utils import ExperimentConfig, make_path


def clean_name(name: str) -> str:
    return name.split("/", 1)[-1].replace("-", " ")


def plot_cossims(
    res: list[Float[Tensor, "n_layers n_layers"]],
    cfg: list[ExperimentConfig],
    clamp: float,
    title: list[str] | None = None,
    axis_names: list[tuple[str, str]] | None = None,  # [y-name, x-name]
    file_name: str = "",
) -> None:
    filename: Path = make_path("figures", cfg[0], file_name=file_name)
    assert len(res) == len(cfg)
    n = len(res)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axs.ravel()

    for ax, tensor, c in zip(axes, res, cfg):
        data = tensor.cpu().numpy().clip(-clamp, clamp)
        im = ax.pcolormesh(data, cmap="viridis", vmin=-clamp, vmax=clamp)

        ax.set_title(clean_name(c.model_name), pad=8)
        ax.xaxis.tick_top()
        # ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

    for i, (ax, tensor, c) in enumerate(zip(axes, res, cfg)):
        data = tensor.cpu().numpy().clip(-clamp, clamp)
        im = ax.pcolormesh(data, cmap="viridis", vmin=-clamp, vmax=clamp)

        if title is not None:
            ax.set_title(title[i], pad=8)
        else:
            ax.set_title(clean_name(c.model_name), pad=8)

        if axis_names is not None:
            ax.set_ylabel(axis_names[i][0])
            ax.set_xlabel(axis_names[i][1])

        ax.xaxis.tick_top()
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[n:]:
        ax.axis("off")

    fig.colorbar(
        im,
        ax=axes[:n],
        location="right",
        fraction=0.04,
        pad=0.02,
    )

    fig.savefig(
        filename,
        dpi=150,
        transparent=False,
    )
    print("Figure saved to:", filename)
