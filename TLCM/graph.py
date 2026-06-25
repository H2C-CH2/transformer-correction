import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.colorbar import make_axes
from torch import Tensor
from utils import Config, make_path


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

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols + 1, 6 * nrows))
    axes = np.array(axes).flatten()

    images = []
    for ax, tensor, c in zip(axes, res, cfg):
        data = tensor.cpu().numpy().clip(-clamp, clamp)
        im = ax.imshow(data, vmin=-clamp, vmax=clamp, origin="upper")
        images.append(im)

        ax.set_title(c.model_name, color="white")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(colors="white")

    for ax in axes[n:]:
        ax.set_visible(False)

    cbar_ax, _ = make_axes(
        axes[:n].tolist(), location="right", fraction=0.05, pad=0.03, shrink=0.8
    )
    cbar_ax.tick_params(labelcolor="white")
    fig.colorbar(images[0], cax=cbar_ax)

    fig.subplots_adjust(right=0.85)
    plt.savefig(filename, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print("Figure 1 / 5A saved to:", filename)
