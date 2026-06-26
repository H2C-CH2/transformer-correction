import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.colorbar import make_axes
from torch import Tensor
from utils import Config, make_path


def clean_name(name: str) -> str:
    return name.split("/", 1)[-1].replace("-", " ")


def plot_fig_1_A5(
    res: list[Float[Tensor, "n_layers n_layers"]],
    cfg: list[Config],
    clamp: float,
    file_name: str,
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
    print("Figure 1 / 5A saved to:", filename)
